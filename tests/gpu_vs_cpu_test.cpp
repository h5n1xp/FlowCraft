// Headless consistency test: runs the GPU LBM kernel (LbmSolver) and the CPU
// reference (LbmReference) on the same small sphere-in-a-box domain and checks
// that their velocity/density fields agree to within half-precision storage
// error. Run after any change to shaders/lbm_step.comp.glsl, the packed
// population layout, or the collision model.
//
// Build/run: see tests/run_gpu_vs_cpu_test.sh
#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "../FlowCraft/GpuShaders.h"
#include "../FlowCraft/LbmReference.h"
#include "../FlowCraft/LbmSolver.h"
#include "../FlowCraft/PanelSolver.h"
#include "../FlowCraft/PotentialFlowSolver.h"
#include "../FlowCraft/VoxelGrid.h"

int main() {
    if (!SDL_Init(SDL_INIT_VIDEO)) { std::printf("SDL_Init failed: %s\n", SDL_GetError()); return 2; }
    SDL_GPUDevice* dev = SDL_CreateGPUDevice(kEmbeddedShaderFormats | SDL_GPU_SHADERFORMAT_METALLIB, true, nullptr);
    if (!dev) { std::printf("No GPU device: %s\n", SDL_GetError()); return 2; }
    std::printf("GPU driver: %s\n", SDL_GetGPUDeviceDriver(dev));

    for (LatticeModel model : {LatticeModel::D3Q19, LatticeModel::D3Q15}) {
        VoxelGrid grid;
        grid.nx = 48; grid.ny = 24; grid.nz = 24; grid.cellSize = 1.0f;
        grid.solid.assign(grid.CellCount(), 0);
        grid.signedDistance.assign(grid.CellCount(), 1e30f);
        const float cx = 20, cy = 12, cz = 12, r = 5;
        for (int z = 0; z < grid.nz; ++z)
            for (int y = 0; y < grid.ny; ++y)
                for (int x = 0; x < grid.nx; ++x) {
                    float d = std::sqrt((x + .5f - cx) * (x + .5f - cx) + (y + .5f - cy) * (y + .5f - cy) +
                                        (z + .5f - cz) * (z + .5f - cz)) - r;
                    size_t i = grid.Index(x, y, z);
                    grid.signedDistance[i] = d;
                    grid.solid[i] = d < 0 ? 1 : 0;
                }

        LbmSolver::Params gp;  gp.tau = 0.6f; gp.inletVelocity = 0.05f; gp.smagorinskyCs = 0.12f;
        LbmReference::Params cp; cp.tau = gp.tau; cp.inletVelocity = gp.inletVelocity;
        cp.reverseFlow = gp.reverseFlow; cp.freeSlipY = gp.freeSlipY; cp.freeSlipZ = gp.freeSlipZ;
        cp.smagorinskyCs = gp.smagorinskyCs;

        LbmSolver gpu;
        if (!gpu.Init(dev, grid, model)) { std::printf("GPU Init failed: %s\n", gpu.LastError().c_str()); return 1; }
        gpu.Reset(gp, true);
        LbmReference cpu;
        cpu.Init(grid, model, cp, true);

        const int kSteps = 60;
        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(dev);
        gpu.Step(cmd, kSteps);
        SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
        SDL_WaitForGPUFences(dev, true, &fence, 1);
        SDL_ReleaseGPUFence(dev, fence);
        for (int i = 0; i < kSteps; ++i) cpu.Step();

        std::vector<float> g = gpu.ReadbackField(), c = cpu.ReadbackField();
        double maxRho = 0, maxU = 0, sumU = 0; size_t n = 0; bool nan = false;
        for (size_t i = 0; i < grid.CellCount(); ++i) {
            if (grid.solid[i]) continue;
            for (int k = 0; k < 4; ++k) if (!std::isfinite(g[i * 4 + k])) nan = true;
            maxRho = std::max<double>(maxRho, std::fabs(g[i * 4] - c[i * 4]));
            for (int k = 1; k < 4; ++k) {
                double e = std::fabs(g[i * 4 + k] - c[i * 4 + k]);
                maxU = std::max(maxU, e); sumU += e; ++n;
            }
        }
        std::printf("%s: max|drho|=%.5f  max|du|=%.5f  mean|du|=%.6f  (inlet u=%.3f)%s\n",
                    model == LatticeModel::D3Q19 ? "D3Q19" : "D3Q15", maxRho, maxU, sumU / double(n),
                    gp.inletVelocity, nan ? "  NaN!" : "");
        bool ok = !nan && maxRho < 0.01 && maxU < 0.01 && sumU / double(n) < 0.001;
        std::printf("  -> %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
        gpu.Shutdown();
    }

    // ── Test 3D Lifting Panel Method (PanelGPU vs PanelCPU) on an inclined wing ──
    {
        VoxelGrid grid;
        grid.nx = 48; grid.ny = 32; grid.nz = 24; grid.cellSize = 1.0f;
        grid.solid.assign(grid.CellCount(), 0);
        grid.signedDistance.assign(grid.CellCount(), 1e30f);
        // Inclined ellipsoid wing at +10 deg angle of attack (nose up at -X leading edge: dx < 0 => dz > 0)
        const float cx = 20.0f, cy = 16.0f, cz = 12.0f;
        const float cosA = std::cos(0.1745f), sinA = std::sin(0.1745f);
        for (int z = 0; z < grid.nz; ++z)
            for (int y = 0; y < grid.ny; ++y)
                for (int x = 0; x < grid.nx; ++x) {
                    float dx = (x + 0.5f) - cx;
                    float dy = (y + 0.5f) - cy;
                    float dz = (z + 0.5f) - cz;
                    float rx = dx * cosA - dz * sinA;
                    float rz = dx * sinA + dz * cosA;
                    float d = (std::sqrt((rx / 7.0f) * (rx / 7.0f) +
                                         (dy / 9.0f) * (dy / 9.0f) +
                                         (rz / 2.2f) * (rz / 2.2f)) - 1.0f) * 3.0f;
                    size_t i = grid.Index(x, y, z);
                    grid.signedDistance[i] = d;
                    grid.solid[i] = d < 0.0f ? 1 : 0;
                }

        PanelSolver::Params pp;
        pp.inletVelocity = 0.08f;
        pp.reverseFlow = false;

        PanelSolver panelGpu, panelCpu;
        if (!panelGpu.Init(dev, true, grid, pp)) {
            std::printf("PanelGPU Init failed: %s\n", panelGpu.LastError().c_str());
            return 1;
        }
        if (!panelCpu.Init(nullptr, false, grid, pp)) {
            std::printf("PanelCPU Init failed: %s\n", panelCpu.LastError().c_str());
            return 1;
        }
        panelGpu.Step(40);
        panelCpu.Step(40);

        std::vector<float> g = panelGpu.ReadbackField();
        std::vector<float> c = panelCpu.ReadbackField();
        double maxRho = 0, maxU = 0, sumU = 0; size_t n = 0; bool nan = false;
        double sumFx = 0, sumFz = 0;
        for (size_t i = 0; i < grid.CellCount(); ++i) {
            for (int k = 0; k < 4; ++k) if (!std::isfinite(g[i * 4 + k])) nan = true;
            if (grid.solid[i]) {
                sumFx += g[i * 4 + 1];
                sumFz += g[i * 4 + 3];
                continue;
            }
            maxRho = std::max<double>(maxRho, std::fabs(g[i * 4] - c[i * 4]));
            for (int k = 1; k < 4; ++k) {
                double e = std::fabs(g[i * 4 + k] - c[i * 4 + k]);
                maxU = std::max(maxU, e); sumU += e; ++n;
            }
        }
        std::printf("PanelMethod (panels=%zu, wakeStrips=%zu, LiftFz=%.5f, InducedFx=%.5f): max|drho|=%.6f  max|du|=%.6f  mean|du|=%.7f%s\n",
                    panelGpu.NumPanels(), panelGpu.NumWakeStrips(),
                    sumFz, sumFx,
                    maxRho, maxU, sumU / double(n), nan ? "  NaN!" : "");
        bool ok = !nan && panelGpu.NumPanels() > 0 && panelGpu.NumWakeStrips() > 0 &&
                  sumFz > 0.0 && sumFx > 0.0 && maxRho < 1e-4 && maxU < 1e-4;
        std::printf("  -> %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;

        // Cross-check LBM D3Q19 vs D3Q15 on the exact same +10 deg inclined wing
        {
            LbmSolver::Params lp; lp.tau = 0.6f; lp.inletVelocity = 0.05f; lp.smagorinskyCs = 0.12f;
            LbmSolver q19, q15;
            q19.Init(dev, grid, LatticeModel::D3Q19); q19.Reset(lp, true);
            q15.Init(dev, grid, LatticeModel::D3Q15); q15.Reset(lp, true);
            SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(dev);
            q19.Step(cmd, 80);
            q15.Step(cmd, 80);
            SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
            SDL_WaitForGPUFences(dev, true, &fence, 1);
            SDL_ReleaseGPUFence(dev, fence);
            LbmSolver::Stats s19 = q19.ReadbackStats();
            LbmSolver::Stats s15 = q15.ReadbackStats();
            std::printf("Wing Cross-Check D3Q19 (DragFx=%.5f, LiftFz=%.5f, avgU=%.5f) vs D3Q15 (DragFx=%.5f, LiftFz=%.5f, avgU=%.5f)\n",
                        s19.forceX, s19.forceZ, s19.avgSpeed, s15.forceX, s15.forceZ, s15.avgSpeed);
            bool okWing = !s19.hasNaN && !s15.hasNaN && s19.forceX > 0.0f && s15.forceX > 0.0f &&
                          s19.forceZ > 0.0f && s15.forceZ > 0.0f &&
                          std::fabs(s19.avgSpeed - s15.avgSpeed) < 0.005f;
            std::printf("  -> %s\n", okWing ? "PASS" : "FAIL");
            q19.Shutdown();
            q15.Shutdown();
            if (!okWing) return 1;
        }

        // Also verify live SetParams flow reversal on inclined wing pitched for -X flow (nose up at +X leading edge: dx > 0 => dz > 0)
        for (int z = 0; z < grid.nz; ++z)
            for (int y = 0; y < grid.ny; ++y)
                for (int x = 0; x < grid.nx; ++x) {
                    float dx = (x + 0.5f) - cx;
                    float dy = (y + 0.5f) - cy;
                    float dz = (z + 0.5f) - cz;
                    float rx =  dx * cosA + dz * sinA;
                    float rz = -dx * sinA + dz * cosA;
                    float d = (std::sqrt((rx / 7.0f) * (rx / 7.0f) +
                                         (dy / 9.0f) * (dy / 9.0f) +
                                         (rz / 2.2f) * (rz / 2.2f)) - 1.0f) * 3.0f;
                    size_t i = grid.Index(x, y, z);
                    grid.signedDistance[i] = d;
                    grid.solid[i] = d < 0.0f ? 1 : 0;
                }
        pp.reverseFlow = true;
        panelGpu.Init(dev, true, grid, pp);
        panelCpu.Init(nullptr, false, grid, pp);
        panelGpu.Step(40);
        panelCpu.Step(40);
        std::vector<float> gRev = panelGpu.ReadbackField();
        std::vector<float> cRev = panelCpu.ReadbackField();
        double sumFxRev = 0, sumFzRev = 0, maxURev = 0;
        for (size_t i = 0; i < grid.CellCount(); ++i) {
            if (grid.solid[i]) {
                sumFxRev += gRev[i * 4 + 1];
                sumFzRev += gRev[i * 4 + 3];
            } else {
                for (int k = 1; k < 4; ++k) maxURev = std::max<double>(maxURev, std::fabs(gRev[i * 4 + k] - cRev[i * 4 + k]));
            }
        }
        std::printf("PanelMethod ReverseFlow (LiftFz=%.5f, InducedFx=%.5f, max|du|=%.6f)\n", sumFzRev, sumFxRev, maxURev);
        bool okRev = (sumFzRev > 0.0) && (sumFxRev < 0.0) && (maxURev < 1e-4);
        std::printf("  -> %s\n", okRev ? "PASS" : "FAIL");
        if (!okRev) return 1;

        // Also verify PotentialFlowSolver against LBM D3Q19 conventions
        PotentialFlowSolver pot;
        PotentialFlowSolver::Params potParams{0.08f, false};
        pot.Init(grid, potParams);
        pot.Step(80);
        std::vector<float> potField = pot.ReadbackField();
        PotentialFlowSolver::Stats potStats = pot.ComputeStats();
        double maxStressPot = 0.0;
        for (size_t i = 0; i < grid.CellCount(); ++i) {
            if (grid.solid[i]) maxStressPot = std::max<double>(maxStressPot, potField[i * 4 + 0]);
        }
        std::printf("PotentialFlow (avgRho=%.4f, avgSpeed=%.4f, maxSpeed=%.4f, maxSolidStress=%.5f)\n",
                    potStats.avgDensity, potStats.avgSpeed, potStats.maxSpeed, maxStressPot);
        bool okPot = !potStats.hasNaN && potStats.avgSpeed > 0.04f && maxStressPot > 0.0;
        std::printf("  -> %s\n", okPot ? "PASS" : "FAIL");
        if (!okPot) return 1;

        panelGpu.Shutdown();
        panelCpu.Shutdown();
    }

    SDL_DestroyGPUDevice(dev);
    return 0;
}
