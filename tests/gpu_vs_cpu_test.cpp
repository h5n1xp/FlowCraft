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
    SDL_DestroyGPUDevice(dev);
    return 0;
}
