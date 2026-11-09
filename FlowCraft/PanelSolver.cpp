#include "PanelSolver.h"
#include "GpuShaders.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kFourPi = 4.0f * kPi;
constexpr uint32_t kThreadsX = 8, kThreadsY = 8, kThreadsZ = 4;

template <typename Body>
void ParallelForZ(int nz, Body body) {
    unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
    numThreads = std::min(numThreads, static_cast<unsigned int>(std::max(nz, 1)));
    int rangePerThread = (nz + static_cast<int>(numThreads) - 1) / static_cast<int>(numThreads);

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (unsigned int t = 0; t < numThreads; ++t) {
        int zBegin = static_cast<int>(t) * rangePerThread;
        int zEnd = std::min(zBegin + rangePerThread, nz);
        if (zBegin >= zEnd) break;
        threads.emplace_back(body, zBegin, zEnd);
    }
    for (std::thread& th : threads) th.join();
}

struct PanelEvalParamsGPU {
    uint32_t nx, ny, nz, numPanels;
    uint32_t numWakeStrips, zOffset, slabZStart, slabZCount;
    float inletVelocity, reverseFlow, wakeLength, maxSpeedClamp;
};

void UploadBuffer(SDL_GPUDevice* device, SDL_GPUBuffer* buffer, const void* data, uint32_t bytes) {
    if (!device || !buffer || bytes == 0) return;
    SDL_GPUTransferBufferCreateInfo tbInfo = {};
    tbInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    tbInfo.size = bytes;
    SDL_GPUTransferBuffer* transfer = SDL_CreateGPUTransferBuffer(device, &tbInfo);
    if (!transfer) return;

    void* mapped = SDL_MapGPUTransferBuffer(device, transfer, false);
    std::memcpy(mapped, data, bytes);
    SDL_UnmapGPUTransferBuffer(device, transfer);

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTransferBufferLocation src = {transfer, 0};
    SDL_GPUBufferRegion dst = {buffer, 0, bytes};
    SDL_UploadToGPUBuffer(copyPass, &src, &dst, false);
    SDL_EndGPUCopyPass(copyPass);

    SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (fence) {
        SDL_WaitForGPUFences(device, true, &fence, 1);
        SDL_ReleaseGPUFence(device, fence);
    }
    SDL_ReleaseGPUTransferBuffer(device, transfer);
}

// Regularised Biot-Savart velocity (per unit Gamma / 4pi) for a directed
// vortex segment P1 -> P2 with Rosenhead-Moore core radius squared `coreSq`.
// Matches `VortexSegmentUnit` in shaders/panel_eval.comp.glsl.
inline Vec3 VortexSegmentUnitCPU(const Vec3& pos, const Vec3& p1, const Vec3& p2, float coreSq) {
    Vec3 r0 = p2 - p1;
    Vec3 r1 = pos - p1;
    Vec3 r2 = pos - p2;
    Vec3 cr = Cross(r1, r2);
    float denom = Dot(cr, cr) + Dot(r0, r0) * coreSq;
    if (denom <= 1e-12f) return Vec3{0.0f, 0.0f, 0.0f};
    float term = Dot(r0, r1) / std::sqrt(Dot(r1, r1) + coreSq) -
                 Dot(r0, r2) / std::sqrt(Dot(r2, r2) + coreSq);
    return cr * (term / denom);
}

// Velocity induced at `pos` by a unit horseshoe vortex (Gamma / 4pi == 1) with
// bound segment A -> B and downstream wake legs of length `wakeLen` along
// (flowSign, 0, 0).
inline Vec3 HorseshoeUnitVelocityCPU(const Vec3& pos, const Vec3& A, const Vec3& B, float coreSq,
                                     float flowSign, float wakeLen) {
    Vec3 wakeOffset{flowSign * wakeLen, 0.0f, 0.0f};
    Vec3 Awake = A + wakeOffset;
    Vec3 Bwake = B + wakeOffset;
    return VortexSegmentUnitCPU(pos, Awake, A, coreSq) +
           VortexSegmentUnitCPU(pos, A, B, coreSq) +
           VortexSegmentUnitCPU(pos, B, Bwake, coreSq);
}

// Velocity induced at `pos` by a desingularized source singularity at `srcPos`
// with unit q = sigma * area / (4*pi) == 1 and smoothing radius squared `epsSq`.
inline Vec3 SourceUnitVelocityCPU(const Vec3& pos, const Vec3& srcPos, float epsSq) {
    Vec3 d = pos - srcPos;
    float r2 = Dot(d, d) + epsSq;
    float invR3 = 1.0f / (r2 * std::sqrt(r2));
    return d * invR3;
}

}  // namespace

void PanelSolver::BuildPanelsAndWake(const VoxelGrid& grid) {
    panels_.clear();
    wakeStrips_.clear();
    boundarySolidCells_.clear();

    struct BoundarySample {
        size_t cellIndex;
        int gx, gy, gz;
        Vec3 surfacePos;
        Vec3 normal;
        float area;
    };

    const bool hasSdf = grid.signedDistance.size() == grid.CellCount();
    const float invCellSize = grid.cellSize > 1e-6f ? (1.0f / grid.cellSize) : 1.0f;
    constexpr int kDirs[6][3] = {{+1, 0, 0}, {-1, 0, 0}, {0, +1, 0}, {0, -1, 0}, {0, 0, +1}, {0, 0, -1}};

    std::vector<BoundarySample> samples;
    samples.reserve(std::min<size_t>(cellCount_ / 8 + 64, 250000));

    int minX = nx_, maxX = -1;
    int minY = ny_, maxY = -1;
    int minZ = nz_, maxZ = -1;

    for (int z = 0; z < nz_; ++z) {
        for (int y = 0; y < ny_; ++y) {
            for (int x = 0; x < nx_; ++x) {
                size_t cell = grid.Index(x, y, z);
                if (!solid_[cell]) continue;

                Vec3 faceNormalSum{0.0f, 0.0f, 0.0f};
                int exposedFaces = 0;
                for (const auto& d : kDirs) {
                    int nx = x + d[0], ny = y + d[1], nz = z + d[2];
                    if (grid.InBounds(nx, ny, nz) && !solid_[grid.Index(nx, ny, nz)]) {
                        faceNormalSum.x += static_cast<float>(d[0]);
                        faceNormalSum.y += static_cast<float>(d[1]);
                        faceNormalSum.z += static_cast<float>(d[2]);
                        ++exposedFaces;
                    }
                }
                if (exposedFaces == 0) continue;

                boundarySolidCells_.push_back(cell);
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
                minZ = std::min(minZ, z);
                maxZ = std::max(maxZ, z);

                Vec3 n = Normalize(faceNormalSum);
                Vec3 pos{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, static_cast<float>(z) + 0.5f};

                if (hasSdf) {
                    auto sdfAt = [&](int cx, int cy, int cz) -> float {
                        cx = std::clamp(cx, 0, nx_ - 1);
                        cy = std::clamp(cy, 0, ny_ - 1);
                        cz = std::clamp(cz, 0, nz_ - 1);
                        return grid.signedDistance[grid.Index(cx, cy, cz)];
                    };
                    float phi0 = sdfAt(x, y, z);
                    float px = sdfAt(x + 1, y, z), mx = sdfAt(x - 1, y, z);
                    float py = sdfAt(x, y + 1, z), my = sdfAt(x, y - 1, z);
                    float pz = sdfAt(x, y, z + 1), mz = sdfAt(x, y, z - 1);
                    if (std::isfinite(phi0) && std::isfinite(px) && std::isfinite(mx) &&
                        std::isfinite(py) && std::isfinite(my) && std::isfinite(pz) && std::isfinite(mz)) {
                        Vec3 grad{px - mx, py - my, pz - mz};
                        float gLen = Length(grad);
                        if (gLen > 1e-5f) {
                            Vec3 nSdf = grad * (1.0f / gLen);
                            n = Normalize(nSdf * 0.8f + n * 0.2f);
                            float shift = std::clamp(phi0 * invCellSize, -0.45f, 0.45f);
                            pos = pos - n * shift;
                        }
                    }
                }

                float areaWeight = std::max(0.8f, Length(faceNormalSum));
                samples.push_back({cell, x, y, z, pos, n, areaWeight});
            }
        }
    }

    if (samples.empty() || maxX < minX) return;

    // 1. Cluster boundary samples into up to ~1152 surface panels (12 x 12 x 4
    // spatial bins x 2 Z-normal signs so upper and lower wing surfaces never
    // merge into the same panel, even on 1-voxel-thin trailing edges).
    constexpr int kBinsX = 12;
    constexpr int kBinsY = 12;
    constexpr int kBinsZ = 4;
    constexpr int kTotalBuckets = kBinsX * kBinsY * kBinsZ * 2;

    struct BucketAccum {
        Vec3 posWeighted{0.0f, 0.0f, 0.0f};
        Vec3 normalWeighted{0.0f, 0.0f, 0.0f};
        float areaSum = 0.0f;
        int count = 0;
    };
    std::vector<BucketAccum> buckets(kTotalBuckets);

    float spanX = static_cast<float>(std::max(1, maxX - minX + 1));
    float spanY = static_cast<float>(std::max(1, maxY - minY + 1));
    float spanZ = static_cast<float>(std::max(1, maxZ - minZ + 1));

    for (const BoundarySample& s : samples) {
        int bx = std::clamp(static_cast<int>(static_cast<float>(s.gx - minX) / spanX * kBinsX), 0, kBinsX - 1);
        int by = std::clamp(static_cast<int>(static_cast<float>(s.gy - minY) / spanY * kBinsY), 0, kBinsY - 1);
        int bz = std::clamp(static_cast<int>(static_cast<float>(s.gz - minZ) / spanZ * kBinsZ), 0, kBinsZ - 1);
        int nzSign = (s.normal.z >= 0.0f) ? 1 : 0;
        int bIdx = (((bx * kBinsY + by) * kBinsZ + bz) * 2) + nzSign;

        BucketAccum& b = buckets[bIdx];
        b.posWeighted = b.posWeighted + s.surfacePos * s.area;
        b.normalWeighted = b.normalWeighted + s.normal * s.area;
        b.areaSum += s.area;
        ++b.count;
    }

    for (const BucketAccum& b : buckets) {
        if (b.count == 0 || b.areaSum <= 1e-5f) continue;
        Vec3 n = Normalize(b.normalWeighted);
        if (Length(n) < 0.1f) continue;
        Vec3 p = b.posWeighted * (1.0f / b.areaSum);
        float radius = std::max(1.35f, std::sqrt(b.areaSum / kPi));
        float eps = 0.85f * radius;
        // Keep the desingularized source point inside its own boundary solid
        // voxel (at most 0.4 lattice units inward along -n) so upper/lower
        // wing panels never push their sources across a thin wing into the
        // fluid on the opposite side.
        float inwardOffset = std::min(0.4f, 0.25f * radius);
        Vec3 srcPos = p - n * inwardOffset;

        SurfacePanel panel;
        panel.collocation = p;
        panel.sourcePos = srcPos;
        panel.normal = n;
        panel.area = b.areaSum;
        panel.epsSq = eps * eps;
        panel.sigma = 0.0f;
        panels_.push_back(panel);
    }

    // 2. Build spanwise lifting strips (Horseshoe Vortex Lattice + Trailing-Edge
    // Kutta condition). Divide the model's Y-span into up to 48 strips, using
    // only lifting-surface boundary samples (|normal.z| >= 0.35) so vertical
    // tail fins and fuselage sidewalls do not skew the strip chord or camber.
    int numStrips = std::clamp((maxY - minY + 1) / 3, 6, 48);
    float stripWidth = spanY / static_cast<float>(numStrips);

    struct StripGeom {
        float xMin = 1e30f, xMax = -1e30f;
        float zAtXMin = 0.0f, zAtXMax = 0.0f;
        int countMin = 0, countMax = 0;
        float zSum = 0.0f;
        int totalCount = 0;
        Vec3 camberNormalSum{0.0f, 0.0f, 0.0f};
    };
    std::vector<StripGeom> stripGeoms(numStrips);

    for (const BoundarySample& s : samples) {
        if (std::fabs(s.normal.z) < 0.35f) continue;
        int sIdx = std::clamp(static_cast<int>((s.surfacePos.y - static_cast<float>(minY)) / stripWidth),
                              0, numStrips - 1);
        StripGeom& g = stripGeoms[sIdx];
        g.xMin = std::min(g.xMin, s.surfacePos.x);
        g.xMax = std::max(g.xMax, s.surfacePos.x);
        g.zSum += s.surfacePos.z;
        ++g.totalCount;
    }

    bool rev = params_.reverseFlow;
    float flowSign = rev ? -1.0f : 1.0f;
    for (const BoundarySample& s : samples) {
        if (std::fabs(s.normal.z) < 0.35f) continue;
        int sIdx = std::clamp(static_cast<int>((s.surfacePos.y - static_cast<float>(minY)) / stripWidth),
                              0, numStrips - 1);
        StripGeom& g = stripGeoms[sIdx];
        if (g.totalCount == 0) continue;
        float chord = std::max(1.0f, g.xMax - g.xMin);
        if (s.surfacePos.x <= g.xMin + 0.25f * chord) {
            g.zAtXMin += s.surfacePos.z;
            ++g.countMin;
        }
        if (s.surfacePos.x >= g.xMax - 0.25f * chord) {
            g.zAtXMax += s.surfacePos.z;
            ++g.countMax;
        }
        // Flip lower-surface normals (+Z-ward) so averaging upper and lower
        // skins cancels symmetric thickness slope and isolates mean camber +
        // incidence, weighted toward the aft Kutta region.
        float xNorm = rev ? ((g.xMax - s.surfacePos.x) / chord) : ((s.surfacePos.x - g.xMin) / chord);
        float kuttaWeight = 0.5f + std::clamp(xNorm, 0.0f, 1.0f);
        Vec3 upN = (s.normal.z >= 0.0f) ? s.normal : (s.normal * -1.0f);
        g.camberNormalSum = g.camberNormalSum + upN * (s.area * kuttaWeight);
    }

    for (int sIdx = 0; sIdx < numStrips; ++sIdx) {
        const StripGeom& g = stripGeoms[sIdx];
        if (g.totalCount < 3 || g.xMax <= g.xMin + 1.0f) continue;

        float yLeft = static_cast<float>(minY) + static_cast<float>(sIdx) * stripWidth;
        float yRight = yLeft + stripWidth;
        float yMid = 0.5f * (yLeft + yRight);

        float zMean = g.zSum / static_cast<float>(g.totalCount);
        float zMinEdge = g.countMin > 0 ? (g.zAtXMin / static_cast<float>(g.countMin)) : zMean;
        float zMaxEdge = g.countMax > 0 ? (g.zAtXMax / static_cast<float>(g.countMax)) : zMean;

        float xLE = rev ? g.xMax : g.xMin;
        float xTE = rev ? g.xMin : g.xMax;
        float zLE = rev ? zMaxEdge : zMinEdge;
        float zTE = rev ? zMinEdge : zMaxEdge;

        float xQuarter = xLE + 0.25f * (xTE - xLE);
        float zQuarter = zLE + 0.25f * (zTE - zLE);
        float xThreeQuarter = xLE + 0.75f * (xTE - xLE);
        float zThreeQuarter = zLE + 0.75f * (zTE - zLE);

        float dx = xTE - xLE;
        float dz = zTE - zLE;
        Vec3 chordN = Normalize(Vec3{-flowSign * dz, 0.0f, std::fabs(dx)});
        Vec3 surfCamberN = Length(g.camberNormalSum) > 1e-4f
                               ? Normalize(Vec3{g.camberNormalSum.x, 0.0f, std::max(0.1f, g.camberNormalSum.z)})
                               : chordN;
        Vec3 camberN = Normalize(chordN * 0.5f + surfCamberN * 0.5f);
        // Clamp effective camber/incidence slope to a physical pre-stall range
        // (|nx / nz| <= tan(20 deg) ≈ 0.36) so blunt noses or fuselage steps
        // cannot command unphysical stall circulation.
        constexpr float kMaxTanAlpha = 0.36f;
        camberN.x = std::clamp(camberN.x, -kMaxTanAlpha * camberN.z, kMaxTanAlpha * camberN.z);
        camberN = Normalize(camberN);

        WakeStrip ws;
        ws.boundA = Vec3{xQuarter, yLeft, zQuarter};
        ws.boundB = Vec3{xQuarter, yRight, zQuarter};
        ws.controlPt = Vec3{xThreeQuarter, yMid, zThreeQuarter};
        ws.camberNormal = camberN;
        ws.chord = std::fabs(dx);
        ws.span = stripWidth;
        // Scale the vortex core radius with the local chord so the single
        // quarter-chord bound vortex represents a distributed chordwise vortex
        // sheet rather than a concentrated line singularity punching through
        // the wing thickness.
        float coreR = std::max({1.5f, 0.5f * stripWidth, 0.28f * ws.chord});
        ws.coreSq = coreR * coreR;
        ws.gamma = 0.0f;
        wakeStrips_.push_back(ws);
    }

    influenceFlowReversed_ = params_.reverseFlow;
    RebuildInfluenceMatrices();
}

void PanelSolver::RebuildInfluenceMatrices() {
    const size_t P = panels_.size();
    const size_t W = wakeStrips_.size();
    const float flowSign = params_.reverseFlow ? -1.0f : 1.0f;
    const float wakeLen = 8.0f * static_cast<float>(std::max({nx_, ny_, nz_, 64}));

    panelToPanel_.assign(P * P, 0.0f);
    wakeToPanel_.assign(P * W, 0.0f);
    panelToWake_.assign(W * P, 0.0f);
    wakeToWake_.assign(W * W, 0.0f);

    for (size_t i = 0; i < P; ++i) {
        const Vec3& ci = panels_[i].collocation;
        const Vec3& ni = panels_[i].normal;
        float offDiagSum = 0.0f;
        for (size_t j = 0; j < P; ++j) {
            Vec3 u = SourceUnitVelocityCPU(ci, panels_[j].sourcePos, panels_[j].epsSq);
            float val = Dot(u, ni);
            panelToPanel_[i * P + j] = val;
            if (j != i) offDiagSum += std::fabs(val);
        }
        // Analytical planar Gauss-law self-influence (sigma/2 = 2*pi*q / area)
        // plus strict diagonal dominance so the thickness source relaxation is
        // unconditionally contractive.
        float planarSelf = (2.0f * kPi) / std::max(panels_[i].area, 1.0f);
        panelToPanel_[i * P + i] += planarSelf;
        panels_[i].selfDiag = std::max({panelToPanel_[i * P + i], 1.15f * offDiagSum, 1e-3f});
    }

    for (size_t k = 0; k < W; ++k) {
        const Vec3& ck = wakeStrips_[k].controlPt;
        const Vec3& nk = wakeStrips_[k].camberNormal;
        for (size_t j = 0; j < P; ++j) {
            Vec3 u = SourceUnitVelocityCPU(ck, panels_[j].sourcePos, panels_[j].epsSq);
            panelToWake_[k * P + j] = Dot(u, nk);
        }
        float offDiagWake = 0.0f;
        for (size_t s = 0; s < W; ++s) {
            Vec3 u = HorseshoeUnitVelocityCPU(ck, wakeStrips_[s].boundA, wakeStrips_[s].boundB,
                                              wakeStrips_[s].coreSq, flowSign, wakeLen);
            float val = Dot(u, nk);
            wakeToWake_[k * W + s] = val;
            if (s != k) offDiagWake += std::fabs(val);
        }
        // Weissinger 2D thin-airfoil self-influence (-4 / chord per unit
        // Gamma / 4pi) combined with the 3D horseshoe downwash and strict
        // diagonal dominance so circulation stays physically bounded.
        float twoDSelf = -4.0f / std::max(wakeStrips_[k].chord, 1.0f);
        float diag = std::min(wakeToWake_[k * W + k], twoDSelf);
        wakeStrips_[k].selfDiag = -std::max({std::fabs(diag), 1.15f * offDiagWake, 1e-3f});
    }
}

bool PanelSolver::Init(SDL_GPUDevice* device, bool useGpu, const VoxelGrid& grid, const Params& params) {
    Shutdown();
    lastError_.clear();

    device_ = useGpu ? device : nullptr;
    useGpu_ = useGpu && (device != nullptr);
    nx_ = grid.nx;
    ny_ = grid.ny;
    nz_ = grid.nz;
    cellCount_ = grid.CellCount();
    solid_ = grid.solid;
    params_ = params;

    if (cellCount_ == 0) {
        lastError_ = "Voxel grid is empty.";
        return false;
    }

    BuildPanelsAndWake(grid);

    if (useGpu_) {
        const ShaderBlob blob = FC_SHADER_BLOB(PanelEvalComp);
        ChosenShaderFormat fmt = ChooseShaderFormat(device_, blob);
        if (fmt.format == SDL_GPU_SHADERFORMAT_INVALID) {
            lastError_ = "The GPU supports neither SPIR-V nor MSL shaders.";
            return false;
        }

        SDL_GPUComputePipelineCreateInfo pipeInfo = {};
        pipeInfo.code = fmt.code;
        pipeInfo.code_size = fmt.size;
        pipeInfo.entrypoint = fmt.entrypoint;
        pipeInfo.format = fmt.format;
        pipeInfo.num_uniform_buffers = 1;
        pipeInfo.num_readonly_storage_buffers = 3;   // solidMask, panels, wakeStrips
        pipeInfo.num_readwrite_storage_buffers = 1;  // macroOut
        pipeInfo.threadcount_x = kThreadsX;
        pipeInfo.threadcount_y = kThreadsY;
        pipeInfo.threadcount_z = kThreadsZ;

        pipeline_ = SDL_CreateGPUComputePipeline(device_, &pipeInfo);
        if (!pipeline_) {
            lastError_ = std::string("Failed to create Panel GPU compute pipeline: ") + SDL_GetError();
            return false;
        }

        size_t solidBytes64 = cellCount_ * sizeof(uint32_t);
        if (solidBytes64 > static_cast<size_t>(UINT32_MAX)) {
            lastError_ = "Grid too large for a single 4GiB GPU solid-mask buffer. Switch to Panel (CPU).";
            Shutdown();
            return false;
        }

        auto makeBuffer = [&](uint32_t byteSize, SDL_GPUBufferUsageFlags usage) -> SDL_GPUBuffer* {
            SDL_GPUBufferCreateInfo info = {};
            info.usage = usage;
            info.size = std::max<uint32_t>(byteSize, 16u);
            return SDL_CreateGPUBuffer(device_, &info);
        };

        solidMaskBuffer_ = makeBuffer(static_cast<uint32_t>(solidBytes64), SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ);
        panelBuffer_ = makeBuffer(static_cast<uint32_t>(std::max<size_t>(1, panels_.size()) * sizeof(PanelGPU)),
                                  SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ);
        wakeBuffer_ = makeBuffer(static_cast<uint32_t>(std::max<size_t>(1, wakeStrips_.size()) * sizeof(HorseshoeGPU)),
                                 SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ);

        // Partition macroOut (16 bytes/cell) into Z-slabs under the 4GiB buffer cap
        constexpr size_t kBufferCapBytes = static_cast<size_t>(UINT32_MAX) - (1u << 20);
        size_t cellsPerLayer = static_cast<size_t>(nx_) * ny_;
        size_t maxCellsPerSlab = kBufferCapBytes / (4 * sizeof(float));
        size_t maxZPerSlab = cellsPerLayer > 0 ? (maxCellsPerSlab / cellsPerLayer) : 0;
        if (maxZPerSlab == 0) {
            lastError_ = "Single Z-layer exceeds 4GiB buffer cap.";
            Shutdown();
            return false;
        }

        slabs_.clear();
        for (int zStart = 0; zStart < nz_;) {
            int zCount = static_cast<int>(std::min<size_t>(maxZPerSlab, static_cast<size_t>(nz_ - zStart)));
            Slab s;
            s.zStart = zStart;
            s.zCount = zCount;
            s.cellCount = static_cast<size_t>(zCount) * cellsPerLayer;
            s.macroOut = makeBuffer(static_cast<uint32_t>(s.cellCount * 4 * sizeof(float)),
                                    SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE);
            slabs_.push_back(s);
            zStart += zCount;
        }

        bool ok = solidMaskBuffer_ && panelBuffer_ && wakeBuffer_;
        for (const Slab& s : slabs_) ok = ok && (s.macroOut != nullptr);
        if (!ok) {
            lastError_ = std::string("Failed to allocate Panel GPU buffers: ") + SDL_GetError();
            Shutdown();
            return false;
        }

        std::vector<uint32_t> solidU32(cellCount_);
        for (size_t i = 0; i < cellCount_; ++i) solidU32[i] = solid_[i] ? 1u : 0u;
        UploadBuffer(device_, solidMaskBuffer_, solidU32.data(), static_cast<uint32_t>(solidBytes64));
    }

    Reset(params);
    return true;
}

void PanelSolver::Shutdown() {
    if (device_) {
        if (pipeline_) SDL_ReleaseGPUComputePipeline(device_, pipeline_);
        if (solidMaskBuffer_) SDL_ReleaseGPUBuffer(device_, solidMaskBuffer_);
        if (panelBuffer_) SDL_ReleaseGPUBuffer(device_, panelBuffer_);
        if (wakeBuffer_) SDL_ReleaseGPUBuffer(device_, wakeBuffer_);
        for (Slab& s : slabs_) {
            if (s.macroOut) SDL_ReleaseGPUBuffer(device_, s.macroOut);
        }
        if (readbackTransfer_) SDL_ReleaseGPUTransferBuffer(device_, readbackTransfer_);
    }
    pipeline_ = nullptr;
    solidMaskBuffer_ = panelBuffer_ = wakeBuffer_ = nullptr;
    slabs_.clear();
    readbackTransfer_ = nullptr;
    readbackTransferCapacity_ = 0;
    device_ = nullptr;
}

void PanelSolver::Reset(const Params& params) {
    params_ = params;
    for (SurfacePanel& p : panels_) p.sigma = 0.0f;
    for (WakeStrip& w : wakeStrips_) w.gamma = 0.0f;
    // Run an initial surface solve so the panel and wake strengths are
    // immediately non-zero right after Initialize/Reset Solver.
    StepSurfaceOnly(12);
    if (useGpu_ && device_) {
        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_);
        EvaluateFieldRangeGPU(cmd, 0, nz_);
        SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
        if (fence) {
            SDL_WaitForGPUFences(device_, true, &fence, 1);
            SDL_ReleaseGPUFence(device_, fence);
        }
        gpuFieldDirty_ = false;
    }
}

void PanelSolver::SetParams(const Params& params) {
    bool dirChanged = (params.reverseFlow != params_.reverseFlow);
    params_ = params;
    if (dirChanged) {
        RebuildInfluenceMatrices();
    }
    gpuFieldDirty_ = true;
}

void PanelSolver::UploadSingularityBuffers() {
    const size_t P = panels_.size();
    const size_t W = wakeStrips_.size();

    panelsGpuData_.resize(std::max<size_t>(1, P));
    for (size_t i = 0; i < P; ++i) {
        const SurfacePanel& p = panels_[i];
        panelsGpuData_[i] = PanelGPU{
            p.sourcePos.x, p.sourcePos.y, p.sourcePos.z, p.sigma,
            p.normal.x,    p.normal.y,    p.normal.z,    p.epsSq
        };
    }
    if (P == 0) panelsGpuData_[0] = PanelGPU{0, 0, 0, 0, 0, 0, 1, 1.0f};

    wakeGpuData_.resize(std::max<size_t>(1, W));
    for (size_t s = 0; s < W; ++s) {
        const WakeStrip& w = wakeStrips_[s];
        wakeGpuData_[s] = HorseshoeGPU{
            w.boundA.x, w.boundA.y, w.boundA.z, w.gamma,
            w.boundB.x, w.boundB.y, w.boundB.z, w.coreSq
        };
    }
    if (W == 0) wakeGpuData_[0] = HorseshoeGPU{0, 0, 0, 0, 0, 1, 0, 1.0f};

    if (useGpu_ && device_) {
        UploadBuffer(device_, panelBuffer_, panelsGpuData_.data(),
                     static_cast<uint32_t>(panelsGpuData_.size() * sizeof(PanelGPU)));
        UploadBuffer(device_, wakeBuffer_, wakeGpuData_.data(),
                     static_cast<uint32_t>(wakeGpuData_.size() * sizeof(HorseshoeGPU)));
    }
}

void PanelSolver::StepSurfaceOnly(int sweeps) {
    const size_t P = panels_.size();
    const size_t W = wakeStrips_.size();
    if (P == 0 && W == 0) return;

    const float signedU = params_.reverseFlow ? -params_.inletVelocity : params_.inletVelocity;
    const Vec3 uInf{signedU, 0.0f, 0.0f};
    constexpr float kRelaxPanel = 0.65f;
    constexpr float kRelaxWake = 0.55f;

    int iterations = std::max(1, sweeps);
    for (int iter = 0; iter < iterations; ++iter) {
        // 1. Gauss-Seidel relaxation of desingularized surface source strengths
        //    solving the non-lifting thickness problem (uInf + u_src) . n_i = 0.
        //    Decoupling wakeToPanel_ from the thickness source solve prevents
        //    positive feedback between upper/lower wing panels and the internal
        //    bound vortex line.
        for (size_t i = 0; i < P; ++i) {
            float rhs = -Dot(uInf, panels_[i].normal);
            const float* rowP = &panelToPanel_[i * P];
            for (size_t j = 0; j < P; ++j) {
                if (j == i) continue;
                rhs -= rowP[j] * panels_[j].sigma;
            }
            float maxQ = 0.5f * params_.inletVelocity * panels_[i].area;
            float targetQ = std::clamp(rhs / panels_[i].selfDiag, -maxQ, maxQ);
            panels_[i].sigma += kRelaxPanel * (targetQ - panels_[i].sigma);
        }

        // 2. Gauss-Seidel relaxation of spanwise Kutta horseshoe circulations
        //    enforcing 3/4-chord camber tangency (uInf + u_src + u_wake) . n_camber = 0,
        //    with spanwise elliptic/Laplacian smoothing so circulation varies smoothly
        //    across adjacent strips and tapers at the wingtips.
        for (size_t k = 0; k < W; ++k) {
            float rhs = -Dot(uInf, wakeStrips_[k].camberNormal);
            const float* rowP = &panelToWake_[k * P];
            for (size_t j = 0; j < P; ++j) {
                rhs -= rowP[j] * panels_[j].sigma;
            }
            const float* rowW = &wakeToWake_[k * W];
            for (size_t s = 0; s < W; ++s) {
                if (s == k) continue;
                rhs -= rowW[s] * wakeStrips_[s].gamma;
            }
            float targetG = rhs / wakeStrips_[k].selfDiag;
            float leftG = (k > 0) ? wakeStrips_[k - 1].gamma : 0.0f;
            float rightG = (k + 1 < W) ? wakeStrips_[k + 1].gamma : 0.0f;
            targetG = 0.80f * targetG + 0.10f * (leftG + rightG);
            // Physical thin-airfoil circulation cap (|CL| <= ~1.6 ->
            // |Gamma / 4pi| <= 0.14 * uInf * chord).
            float maxGamma = 0.14f * params_.inletVelocity * wakeStrips_[k].chord;
            targetG = std::clamp(targetG, -maxGamma, maxGamma);
            wakeStrips_[k].gamma += kRelaxWake * (targetG - wakeStrips_[k].gamma);
        }
    }

    UploadSingularityBuffers();
    gpuFieldDirty_ = true;
}

void PanelSolver::DispatchSlabGPU(SDL_GPUCommandBuffer* cmd, size_t slabIndex, int localZOffset, int localZCount) {
    if (!pipeline_ || localZCount <= 0) return;
    const Slab& slab = slabs_[slabIndex];

    const float wakeLen = 8.0f * static_cast<float>(std::max({nx_, ny_, nz_, 64}));
    PanelEvalParamsGPU evalParams{
        static_cast<uint32_t>(nx_),
        static_cast<uint32_t>(ny_),
        static_cast<uint32_t>(nz_),
        static_cast<uint32_t>(panels_.size()),
        static_cast<uint32_t>(wakeStrips_.size()),
        static_cast<uint32_t>(localZOffset),
        static_cast<uint32_t>(slab.zStart),
        static_cast<uint32_t>(slab.zCount),
        params_.inletVelocity,
        params_.reverseFlow ? 1.0f : 0.0f,
        wakeLen,
        3.5f * params_.inletVelocity
    };

    uint32_t groupsX = (static_cast<uint32_t>(nx_) + kThreadsX - 1) / kThreadsX;
    uint32_t groupsY = (static_cast<uint32_t>(ny_) + kThreadsY - 1) / kThreadsY;
    uint32_t groupsZ = (static_cast<uint32_t>(localZCount) + kThreadsZ - 1) / kThreadsZ;

    SDL_PushGPUComputeUniformData(cmd, 0, &evalParams, sizeof(evalParams));

    SDL_GPUStorageBufferReadWriteBinding rwBinding = {};
    rwBinding.buffer = slab.macroOut;
    SDL_GPUComputePass* pass = SDL_BeginGPUComputePass(cmd, nullptr, 0, &rwBinding, 1);

    SDL_BindGPUComputePipeline(pass, pipeline_);
    SDL_GPUBuffer* readonly[3] = {solidMaskBuffer_, panelBuffer_, wakeBuffer_};
    SDL_BindGPUComputeStorageBuffers(pass, 0, readonly, 3);

    SDL_DispatchGPUCompute(pass, groupsX, groupsY, groupsZ);
    SDL_EndGPUComputePass(pass);
}

void PanelSolver::EvaluateFieldRangeGPU(SDL_GPUCommandBuffer* cmd, int zStart, int zCount) {
    if (!pipeline_ || zCount <= 0) return;
    int zEnd = zStart + zCount;
    for (size_t s = 0; s < slabs_.size(); ++s) {
        const Slab& slab = slabs_[s];
        int slabEnd = slab.zStart + slab.zCount;
        int overlapStart = std::max(zStart, slab.zStart);
        int overlapEnd = std::min(zEnd, slabEnd);
        if (overlapStart >= overlapEnd) continue;
        DispatchSlabGPU(cmd, s, overlapStart - slab.zStart, overlapEnd - overlapStart);
    }
}

void PanelSolver::Step(int sweeps, std::atomic<float>* progress) {
    if (progress) progress->store(0.1f, std::memory_order_relaxed);
    StepSurfaceOnly(sweeps);
    if (useGpu_ && device_) {
        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_);
        EvaluateFieldRangeGPU(cmd, 0, nz_);
        SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
        if (fence) {
            SDL_WaitForGPUFences(device_, true, &fence, 1);
            SDL_ReleaseGPUFence(device_, fence);
        }
        gpuFieldDirty_ = false;
    }
    if (progress) progress->store(1.0f, std::memory_order_relaxed);
}

void PanelSolver::EvaluateFieldCPU(std::vector<float>& outField) const {
    outField.assign(cellCount_ * 4, 0.0f);
    if (cellCount_ == 0) return;

    const float flowSign = params_.reverseFlow ? -1.0f : 1.0f;
    const float uInf = params_.inletVelocity;
    const float uInfSq = uInf * uInf;
    const float maxSpeed = 3.5f * uInf;
    const float maxSpeedSq = maxSpeed * maxSpeed;
    const float wakeLen = 8.0f * static_cast<float>(std::max({nx_, ny_, nz_, 64}));
    const size_t P = panels_.size();
    const size_t W = wakeStrips_.size();

    ParallelForZ(nz_, [&](int zBegin, int zEnd) {
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < ny_; ++y) {
                for (int x = 0; x < nx_; ++x) {
                    size_t cell = (static_cast<size_t>(z) * ny_ + y) * nx_ + x;
                    if (solid_[cell]) continue;

                    Vec3 pos{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, static_cast<float>(z) + 0.5f};
                    Vec3 vel{flowSign * uInf, 0.0f, 0.0f};

                    for (size_t i = 0; i < P; ++i) {
                        const PanelGPU& p = panelsGpuData_[i];
                        Vec3 d{pos.x - p.sx, pos.y - p.sy, pos.z - p.sz};
                        float r2 = Dot(d, d) + p.epsSq;
                        float invR3 = 1.0f / (r2 * std::sqrt(r2));
                        vel = vel + d * (p.q * invR3);
                    }

                    for (size_t s = 0; s < W; ++s) {
                        const HorseshoeGPU& h = wakeGpuData_[s];
                        Vec3 A{h.ax, h.ay, h.az};
                        Vec3 B{h.bx, h.by, h.bz};
                        Vec3 unitV = HorseshoeUnitVelocityCPU(pos, A, B, h.coreSq, flowSign, wakeLen);
                        vel = vel + unitV * h.gammaOver4Pi;
                    }

                    float speedSq = Dot(vel, vel);
                    if (speedSq > maxSpeedSq && speedSq > 0.0f) {
                        vel = vel * (maxSpeed / std::sqrt(speedSq));
                        speedSq = maxSpeedSq;
                    }
                    float pressureDeviation = 0.5f * (uInfSq - speedSq);
                    outField[cell * 4 + 0] = 1.0f + 3.0f * pressureDeviation;
                    outField[cell * 4 + 1] = vel.x;
                    outField[cell * 4 + 2] = vel.y;
                    outField[cell * 4 + 3] = vel.z;
                }
            }
        }
    });
}

void PanelSolver::FinalizeReadbackField(std::vector<float>& field) const {
    if (field.size() != cellCount_ * 4 || boundarySolidCells_.empty()) return;

    // 1. Compute Trefftz-plane / Kutta-Joukowski induced drag from the
    //    trailing-vortex wake downwash at each lifting strip plus baseline
    //    form/skin-friction drag so the solid boundary cells report realistic
    //    aerodynamic drag alongside the surface Bernoulli pressure.
    const float flowSign = params_.reverseFlow ? -1.0f : 1.0f;
    const float wakeLen = 8.0f * static_cast<float>(std::max({nx_, ny_, nz_, 64}));
    const Vec3 wakeOffset{flowSign * wakeLen, 0.0f, 0.0f};
    float inducedDragTotal = 0.0f;
    for (size_t s = 0; s < wakeStrips_.size(); ++s) {
        const WakeStrip& ws = wakeStrips_[s];
        Vec3 mid = (ws.boundA + ws.boundB) * 0.5f;
        float downwashZ = 0.0f;
        for (size_t j = 0; j < wakeStrips_.size(); ++j) {
            const Vec3& A = wakeStrips_[j].boundA;
            const Vec3& B = wakeStrips_[j].boundB;
            Vec3 Awake = A + wakeOffset;
            Vec3 Bwake = B + wakeOffset;
            float coreSq = wakeStrips_[j].coreSq;
            Vec3 uTrail = VortexSegmentUnitCPU(mid, Awake, A, coreSq) +
                          VortexSegmentUnitCPU(mid, B, Bwake, coreSq);
            downwashZ += uTrail.z * wakeStrips_[j].gamma;
        }
        float gamma = ws.gamma * kFourPi;
        inducedDragTotal += std::fabs(gamma * downwashZ) * ws.span;
    }
    float profileDragTotal =
        0.012f * (0.5f * params_.inletVelocity * params_.inletVelocity) * static_cast<float>(boundarySolidCells_.size());
    float totalDrag = inducedDragTotal + profileDragTotal;
    float inducedDragPerBoundaryCell =
        boundarySolidCells_.empty() ? 0.0f
                                    : (flowSign * totalDrag / static_cast<float>(boundarySolidCells_.size()));

    // 2. Populate exposed solid boundary cells with normal Bernoulli pressure
    //    stress (for the 3D Airframe Stress Map) and vertical/side pressure
    //    forces, while setting the streamwise force `fx` to the Trefftz-plane
    //    induced drag share (avoiding voxel-staircase leading-edge suction
    //    noise on the streamwise component).
    constexpr int kFaceDirs[6][3] = {{+1, 0, 0}, {-1, 0, 0}, {0, +1, 0}, {0, -1, 0}, {0, 0, +1}, {0, 0, -1}};
    for (size_t cell : boundarySolidCells_) {
        int x = static_cast<int>(cell % nx_);
        int y = static_cast<int>((cell / nx_) % ny_);
        int z = static_cast<int>(cell / (static_cast<size_t>(nx_) * ny_));

        float fxPressure = 0.0f, fy = 0.0f, fz = 0.0f;
        int exposedFaces = 0;
        for (const auto& d : kFaceDirs) {
            int nxf = x + d[0], nyf = y + d[1], nzf = z + d[2];
            if (nxf < 0 || nyf < 0 || nzf < 0 || nxf >= nx_ || nyf >= ny_ || nzf >= nz_) continue;
            size_t fcell = (static_cast<size_t>(nzf) * ny_ + nyf) * nx_ + nxf;
            if (solid_[fcell]) continue;
            float pDev = (field[fcell * 4 + 0] - 1.0f) * (1.0f / 3.0f);
            fxPressure -= static_cast<float>(d[0]) * pDev;
            fy -= static_cast<float>(d[1]) * pDev;
            fz -= static_cast<float>(d[2]) * pDev;
            ++exposedFaces;
        }
        float fx = inducedDragPerBoundaryCell;
        float stressMag = std::sqrt(fxPressure * fxPressure + fy * fy + fz * fz);
        field[cell * 4 + 0] = exposedFaces > 0 ? (stressMag / static_cast<float>(exposedFaces)) : 0.0f;
        field[cell * 4 + 1] = fx;
        field[cell * 4 + 2] = fy;
        field[cell * 4 + 3] = fz;
    }
}

void PanelSolver::ReadbackFieldRangeGPU(int zStart, int zCount, std::vector<float>& outField) {
    if (!pipeline_ || !device_ || zCount <= 0) return;

    if (gpuFieldDirty_) {
        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_);
        EvaluateFieldRangeGPU(cmd, 0, nz_);
        SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
        if (fence) {
            SDL_WaitForGPUFences(device_, true, &fence, 1);
            SDL_ReleaseGPUFence(device_, fence);
        }
        gpuFieldDirty_ = false;
    }

    int zEnd = zStart + zCount;
    size_t cellsPerLayer = static_cast<size_t>(nx_) * ny_;

    struct Piece {
        size_t slabIndex;
        uint32_t srcByteOffset;
        uint32_t byteSize;
        uint32_t dstByteOffset;
        size_t globalCellOffset;
    };
    std::vector<Piece> pieces;
    uint32_t totalBytes = 0;

    for (size_t s = 0; s < slabs_.size(); ++s) {
        const Slab& slab = slabs_[s];
        int slabEnd = slab.zStart + slab.zCount;
        int overlapStart = std::max(zStart, slab.zStart);
        int overlapEnd = std::min(zEnd, slabEnd);
        if (overlapStart >= overlapEnd) continue;

        int localZStart = overlapStart - slab.zStart;
        int localZCount = overlapEnd - overlapStart;
        size_t localCellOffset = static_cast<size_t>(localZStart) * cellsPerLayer;
        size_t chunkCells = static_cast<size_t>(localZCount) * cellsPerLayer;
        uint32_t byteSize = static_cast<uint32_t>(chunkCells * 4 * sizeof(float));
        uint32_t srcByteOffset = static_cast<uint32_t>(localCellOffset * 4 * sizeof(float));
        size_t globalCellOffset = (static_cast<size_t>(slab.zStart) + static_cast<size_t>(localZStart)) * cellsPerLayer;

        pieces.push_back({s, srcByteOffset, byteSize, totalBytes, globalCellOffset});
        totalBytes += byteSize;
    }
    if (pieces.empty()) return;

    if (totalBytes > readbackTransferCapacity_) {
        if (readbackTransfer_) SDL_ReleaseGPUTransferBuffer(device_, readbackTransfer_);
        SDL_GPUTransferBufferCreateInfo tbInfo = {};
        tbInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
        tbInfo.size = totalBytes;
        readbackTransfer_ = SDL_CreateGPUTransferBuffer(device_, &tbInfo);
        readbackTransferCapacity_ = readbackTransfer_ ? totalBytes : 0;
    }
    if (!readbackTransfer_) return;

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_);
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);
    for (const Piece& p : pieces) {
        SDL_GPUBufferRegion src = {slabs_[p.slabIndex].macroOut, p.srcByteOffset, p.byteSize};
        SDL_GPUTransferBufferLocation dst = {readbackTransfer_, p.dstByteOffset};
        SDL_DownloadFromGPUBuffer(copyPass, &src, &dst);
    }
    SDL_EndGPUCopyPass(copyPass);
    SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (fence) {
        SDL_WaitForGPUFences(device_, true, &fence, 1);
        SDL_ReleaseGPUFence(device_, fence);
    }

    const auto* mapped = static_cast<const uint8_t*>(SDL_MapGPUTransferBuffer(device_, readbackTransfer_, false));
    if (mapped) {
        for (const Piece& p : pieces) {
            std::memcpy(outField.data() + p.globalCellOffset * 4, mapped + p.dstByteOffset, p.byteSize);
        }
        SDL_UnmapGPUTransferBuffer(device_, readbackTransfer_);
    }
}

std::vector<float> PanelSolver::ReadbackField() {
    std::vector<float> out(cellCount_ * 4, 0.0f);
    if (cellCount_ == 0) return out;

    if (useGpu_ && pipeline_) {
        ReadbackFieldRangeGPU(0, nz_, out);
    } else {
        EvaluateFieldCPU(out);
    }
    FinalizeReadbackField(out);
    return out;
}

PanelSolver::Stats PanelSolver::ComputeStats() {
    Stats stats;
    std::vector<float> field = ReadbackField();
    if (field.empty()) return stats;

    double densitySum = 0.0, speedSum = 0.0;
    float minDensity = 1e30f, maxDensity = -1e30f, maxSpeed = 0.0f;
    size_t fluidCount = 0;

    for (size_t i = 0; i < cellCount_; ++i) {
        if (solid_[i]) continue;
        float rho = field[i * 4 + 0];
        float ux = field[i * 4 + 1];
        float uy = field[i * 4 + 2];
        float uz = field[i * 4 + 3];
        float speed = std::sqrt(ux * ux + uy * uy + uz * uz);
        if (!std::isfinite(rho) || !std::isfinite(speed)) {
            stats.hasNaN = true;
            continue;
        }
        densitySum += rho;
        speedSum += speed;
        minDensity = std::min(minDensity, rho);
        maxDensity = std::max(maxDensity, rho);
        maxSpeed = std::max(maxSpeed, speed);
        ++fluidCount;
    }

    if (fluidCount > 0) {
        stats.avgDensity = static_cast<float>(densitySum / static_cast<double>(fluidCount));
        stats.minDensity = minDensity;
        stats.maxDensity = maxDensity;
        stats.avgSpeed = static_cast<float>(speedSum / static_cast<double>(fluidCount));
        stats.maxSpeed = maxSpeed;
        stats.valid = true;
    }
    return stats;
}
