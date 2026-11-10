#include "LbmSolver.h"
#include "GpuShaders.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

// Splits [0, count) into contiguous ranges across
// std::thread::hardware_concurrency() threads — used only by
// ResetFromField's per-cell equilibrium computation below, the one CPU-
// bound (as opposed to GPU-API-bound) loop in this file costly enough to
// matter at high resolution. A flat range, not a Z-slab split like
// Voxelizer.cpp/LbmReference.cpp/PotentialFlowSolver.cpp's own
// ParallelForZ: every slab's cells here are already contiguous in their
// own local indexing, and nothing about this loop needs to reason about
// x/y/z at all, so there's no reason to carry that extra structure over.
template <typename Body>
void ParallelForRange(size_t count, Body body) {
    unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
    numThreads = std::min(numThreads, static_cast<unsigned int>(std::max<size_t>(count, 1)));
    size_t rangePerThread = (count + numThreads - 1) / numThreads;

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t begin = static_cast<size_t>(t) * rangePerThread;
        size_t end = std::min(begin + rangePerThread, count);
        if (begin >= end) break;
        threads.emplace_back(body, begin, end);
    }
    for (std::thread& th : threads) th.join();
}

// IEEE 754 binary16 (what MSL's `half` is) from a float32, round-to-
// nearest-even. Only used for Reset()'s initial-equilibrium upload — see
// pop_'s comment in LbmSolver.h for why the population buffers are half
// precision at all. LBM populations in this solver's low-Mach regime stay
// O(1) or smaller, nowhere near half's range limits, so the denormal-
// flush-to-zero and overflow-clamp paths below are unreachable in
// practice; they're here so a stray NaN/huge value degrades to a finite
// half instead of silently producing Inf and poisoning neighbouring cells
// through the kernel's streaming step on the next pass.
// Populations are stored two directions per 32-bit word (packed halves) — see
// shaders/lbm_step.comp.glsl's header for why. Q=19 -> 10 words, Q=15 -> 8.
size_t PopWordsPerCell(int Q) { return (static_cast<size_t>(Q) + 1) / 2; }

uint16_t FloatToHalf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0) return static_cast<uint16_t>(sign);  // flush denormals/too-small-for-half to zero
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);  // clamp to half's largest finite value

    uint32_t halfMantissa = mantissa >> 13;
    uint32_t roundBit = (mantissa >> 12) & 1u;
    uint32_t stickyBits = mantissa & 0xFFFu;
    if (roundBit && (stickyBits != 0 || (halfMantissa & 1u))) {
        halfMantissa += 1;
        if (halfMantissa == 0x400u) {  // mantissa rounded up into the exponent
            halfMantissa = 0;
            ++exponent;
            if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | halfMantissa);
}

// CPU-side mirror of the MSL kernel's own Equilibrium() — used only by
// Reset() to seed pop[0] (the kernel's own copy runs every Step() call
// on the GPU; this one never touches the hot path).
float EquilibriumCPU(float ex, float ey, float ez, float weight, float rho, float ux, float uy, float uz) {
    float eu = ex * ux + ey * uy + ez * uz;
    float u2 = ux * ux + uy * uy + uz * uz;
    return weight * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * u2);
}

// Mirrors LbmReference's GPU-side data layout exactly. mirrorY/mirrorZ:
// see LbmReference.h's own mirrorY_/mirrorZ_ comment — same meaning, same
// search-based construction (LbmSolver::Init()), just carried per-
// direction here instead of as separate arrays, reusing 2 of this
// struct's 3 original padding floats (same total size, so no buffer-size
// recalculation needed anywhere this struct's sizeof is used).
struct LatticeDirGPU {
    float ex, ey, ez, weight;
    int32_t opposite;
    int32_t mirrorY, mirrorZ;
    float pad0;
};

// See the Z-slab decomposition comment in LbmSolver.h and kLbmStepMSL's
// own comment below for what each new field (beyond the original
// nx/ny/nz/Q/tau/inletVelocity/zOffset/reverseFlow) means.
struct SimParamsGPU {
    uint32_t nx, ny, nz, Q;
    float tau, inletVelocity;
    uint32_t zOffset;  // StepRange's chunk start, now relative to THIS slab's own z range, not the whole domain
    float reverseFlow;  // 0 = inlet at x=0 (flow toward +X); nonzero = inlet at x=nx-1 (flow toward -X)
    uint32_t slabZStart;         // this slab's GLOBAL z start
    uint32_t slabZCount;         // this slab's own depth
    uint32_t prevCellCount;      // prev slab's own Q-plane stride (nx*ny*prevZCount); unused when slabZStart==0
    uint32_t prevLastLayerBase;  // nx*ny*(prevZCount-1): prev slab's own local base for its last layer; ditto
    uint32_t nextCellCount;      // next slab's own Q-plane stride; unused when slabZStart+slabZCount==nz
    // See LbmSolver::Params::freeSlipY/freeSlipZ — same 0.0/nonzero
    // convention as reverseFlow above, for the same reason.
    float freeSlipY, freeSlipZ;
    float smagorinskyCs;
};

// Direct port of LbmReference::Step()'s per-cell logic — same branches, same
// formulas, same boundary conditions. See LbmReference.h for the physics
// writeup; keep the two in sync if either changes.
//
// popIn/popOut store populations as `half` (IEEE 754 binary16) rather than
// float — halving the ping-pong buffers' size (the binding constraint on
// SDL_GPU's 4GiB-per-buffer cap, see Init()'s comment) roughly doubles the
// cell count a single buffer can hold. This is mixed precision, not a
// wholesale precision downgrade: every `half` value is converted to float
// immediately on load and stays float through the entire per-cell
// computation (streaming gather, macroscopic rho/u, equilibrium, BGK
// relax) — only the round-trip through global memory between steps is
// half. That protects the part that actually needs the precision: this
// solver runs at a deliberately low lattice Mach number (see
// kLatticeVelocityScale in main.cpp), so the velocity signal is a small
// perturbation on top of near-equilibrium population values, and doing
// the *arithmetic* itself in half would risk that signal being swamped by
// half's ~3 decimal digits. Storage-only precision loss is a much gentler
// failure mode — per-step quantization noise rather than compounding
// arithmetic error — standard practice for this exact regime (single-
// phase aerodynamics at low Mach commonly runs stably in half as long as
// accumulation happens in a higher-precision type, which is what this is).
//
// Z-SLAB DECOMPOSITION — what changed from the single-buffer version:
// popIn/popOut/macroOut are now THIS SLAB's own buffers, indexed with a
// LOCAL flat cell index (`cell`, stride `cellCount` = nx*ny*slabZCount —
// deliberately NOT the global cell count) rather than a global one.
// solidMask stays a single GLOBAL buffer (see LbmSolver.h's class
// comment), indexed by `globalCell` wherever a cell's actual solidity
// needs checking — both for this thread's own cell and for a streaming
// neighbour, regardless of which slab that neighbour happens to live in.
//
// The only place that distinction matters beyond solidMask is the
// streaming gather's neighbour population lookup: a lattice direction's
// ez is always in {-1,0,1} (D3Q15/D3Q19), so a neighbour's global z can
// differ from this thread's own by at most 1. Combined with slabs being a
// contiguous, gap-free partition of the domain, that means a neighbour's
// global z falls into exactly one of three cases: inside this slab (the
// common case — same buffers, same local-index convention as this cell's
// own), exactly one layer below this slab's start (lives in the PREVIOUS
// slab's buffer, at that slab's own LAST local layer), or exactly one
// layer above this slab's end (lives in the NEXT slab's buffer, at local
// layer 0). `prevPopIn`/`nextPopIn` are bound as two extra read-only
// buffers specifically for that — no copying/ghost-layer exchange needed
// at all, since every slab's buffer already lives in the same GPU address
// space; the kernel just reads straight across the boundary. At a domain
// edge (no prev/next slab), `prevPopIn`/`nextPopIn` are bound to a tiny
// dummy buffer that's provably never dereferenced: the only way to reach
// that read is for a neighbour's global z to sit one step outside this
// slab, which at a domain edge is also outside the GLOBAL domain — caught
// by the ordinary in-bounds check (bounce-back) before the prev/next
// branch is ever reached.
constexpr uint32_t kThreadsX = 8, kThreadsY = 8, kThreadsZ = 4;

void UploadBuffer(SDL_GPUDevice* device, SDL_GPUBuffer* buffer, const void* data, uint32_t bytes) {
    SDL_GPUTransferBufferCreateInfo tbInfo = {};
    tbInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    tbInfo.size = bytes;
    SDL_GPUTransferBuffer* transfer = SDL_CreateGPUTransferBuffer(device, &tbInfo);

    void* mapped = SDL_MapGPUTransferBuffer(device, transfer, false);
    std::memcpy(mapped, data, bytes);
    SDL_UnmapGPUTransferBuffer(device, transfer);

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTransferBufferLocation src = {transfer, 0};
    SDL_GPUBufferRegion dst = {buffer, 0, bytes};
    SDL_UploadToGPUBuffer(copyPass, &src, &dst, false);
    SDL_EndGPUCopyPass(copyPass);
    // Must wait for the copy to actually finish before freeing the transfer
    // buffer below — callers (Reset()/ResetFromField()) invoke this once per
    // slab in a tight loop, and without this fence the driver is free to
    // reuse/free this transfer buffer's memory for a later slab's transfer
    // buffer before the GPU has consumed this one's contents, corrupting
    // whichever slab loses the race.
    SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    SDL_WaitForGPUFences(device, true, &fence, 1);
    SDL_ReleaseGPUFence(device, fence);

    SDL_ReleaseGPUTransferBuffer(device, transfer);
}

}  // namespace

bool LbmSolver::Init(SDL_GPUDevice* device, const VoxelGrid& grid, LatticeModel model) {
    Shutdown();
    lastError_.clear();

    device_ = device;
    nx_ = grid.nx;
    ny_ = grid.ny;
    nz_ = grid.nz;
    cellCount_ = grid.CellCount();
    model_ = model;
    Q_ = GetLatticeSet(model).count;
    solidCpu_ = grid.solid;

    SDL_GPUComputePipelineCreateInfo pipelineInfo = {};
    const ShaderBlob lbmBlob = FC_SHADER_BLOB(LbmStepComp);
    ChosenShaderFormat lbmFormat = ChooseShaderFormat(device, lbmBlob);
    if (lbmFormat.format == SDL_GPU_SHADERFORMAT_INVALID) {
        lastError_ = "The GPU supports neither SPIR-V nor MSL shaders.";
        SDL_Log("LbmSolver: %s", lastError_.c_str());
        return false;
    }
    pipelineInfo.code = lbmFormat.code;
    pipelineInfo.code_size = lbmFormat.size;
    pipelineInfo.entrypoint = lbmFormat.entrypoint;
    pipelineInfo.format = lbmFormat.format;
    pipelineInfo.num_uniform_buffers = 1;
    pipelineInfo.num_readonly_storage_buffers = 6;   // lattice, solidMask, signedDistance, ownPopIn, prevPopIn, nextPopIn
    pipelineInfo.num_readwrite_storage_buffers = 2;  // popOut, macroOut
    pipelineInfo.threadcount_x = kThreadsX;
    pipelineInfo.threadcount_y = kThreadsY;
    pipelineInfo.threadcount_z = kThreadsZ;

    pipeline_ = SDL_CreateGPUComputePipeline(device_, &pipelineInfo);
    if (!pipeline_) {
        lastError_ = std::string("Failed to create the GPU compute pipeline: ") + SDL_GetError();
        SDL_Log("LbmSolver: %s", lastError_.c_str());
        return false;
    }

    // solidMask_/latticeTable_ stay single GLOBAL buffers (never split
    // into slabs — see the class comment in LbmSolver.h for why), so
    // THIS is the real ceiling the whole scheme now has: pop_'s own cap
    // is removed entirely by the slab split below, regardless of domain
    // size, but solidMask's one buffer still has to fit under 4GiB.
    size_t solidBytes64 = cellCount_ * sizeof(uint32_t);
    size_t latticeBytes64 = static_cast<size_t>(Q_) * sizeof(LatticeDirGPU);
    size_t maxGlobalBytes64 = std::max(solidBytes64, latticeBytes64);
    if (maxGlobalBytes64 > static_cast<size_t>(UINT32_MAX)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                       "Grid too large: %zu cells needs a %.2f GiB solid-mask buffer, but SDL_GPU's limit is 4GiB "
                       "per buffer. Reduce the voxel resolution.",
                       cellCount_, static_cast<double>(maxGlobalBytes64) / (1024.0 * 1024.0 * 1024.0));
        lastError_ = buf;
        SDL_Log("LbmSolver: %s", lastError_.c_str());
        return false;
    }

    // Partition the domain into Z-slabs, each sized so its own pop_
    // buffers stay comfortably under the 4GiB cap no matter how many
    // slabs that takes — see the class comment in LbmSolver.h. Sized off
    // pop_'s own per-cell byte cost (Q_*sizeof(half)) since that's always
    // >= every other per-slab buffer's (macroOut's 16 bytes/cell for any
    // Q>=8), so slabs sized for pop_ automatically keep macroOut's slabs
    // under the cap too — no separate macroOut-specific sizing needed.
    constexpr size_t kBufferCapBytes = static_cast<size_t>(UINT32_MAX) - (1u << 20);  // ~4GiB, 1MiB margin
    size_t cellsPerLayer = static_cast<size_t>(nx_) * ny_;
    size_t popBytesPerCell = PopWordsPerCell(Q_) * sizeof(uint32_t);  // two packed halves per word
    size_t maxCellsPerSlab = popBytesPerCell > 0 ? kBufferCapBytes / popBytesPerCell : 0;
    size_t maxZPerSlab = cellsPerLayer > 0 ? maxCellsPerSlab / cellsPerLayer : 0;
    if (maxZPerSlab == 0) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                       "Grid too large: a single Z-layer (%dx%d cells) alone needs more than a 4GiB population "
                       "buffer. Reduce the voxel resolution.",
                       nx_, ny_);
        lastError_ = buf;
        SDL_Log("LbmSolver: %s", lastError_.c_str());
        return false;
    }

    slabs_.clear();
    for (int zStart = 0; zStart < nz_;) {
        int zCount = static_cast<int>(std::min<size_t>(maxZPerSlab, static_cast<size_t>(nz_ - zStart)));
        Slab slab;
        slab.zStart = zStart;
        slab.zCount = zCount;
        slab.cellCount = cellsPerLayer * static_cast<size_t>(zCount);
        slabs_.push_back(slab);
        zStart += zCount;
    }

    auto makeBuffer = [&](uint32_t bytes) {
        SDL_GPUBufferCreateInfo info = {};
        info.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE;
        info.size = bytes;
        return SDL_CreateGPUBuffer(device_, &info);
    };

    uint32_t latticeBytes = static_cast<uint32_t>(latticeBytes64);
    uint32_t solidBytes = static_cast<uint32_t>(solidBytes64);
    latticeTable_ = makeBuffer(latticeBytes);
    solidMask_ = makeBuffer(solidBytes);
    signedDistance_ = makeBuffer(solidBytes);  // 1 float (4 bytes) per cell, same byte size as solidMask_
    // One cell's worth of Q planes — the minimum needed so the dummy
    // binding is a valid buffer at all; see its declaration's comment in
    // LbmSolver.h for why its contents never actually get read.
    dummyHaloBuffer_ = makeBuffer(static_cast<uint32_t>(PopWordsPerCell(Q_) * sizeof(uint32_t)));

    bool allOk = latticeTable_ != nullptr && solidMask_ != nullptr && signedDistance_ != nullptr &&
                 dummyHaloBuffer_ != nullptr;
    for (Slab& slab : slabs_) {
        uint32_t slabPopBytes = static_cast<uint32_t>(PopWordsPerCell(Q_) * slab.cellCount * sizeof(uint32_t));
        uint32_t slabMacroBytes = static_cast<uint32_t>(slab.cellCount * 4 * sizeof(float));
        slab.pop[0] = makeBuffer(slabPopBytes);
        slab.pop[1] = makeBuffer(slabPopBytes);
        slab.macroOut = makeBuffer(slabMacroBytes);
        allOk = allOk && slab.pop[0] && slab.pop[1] && slab.macroOut;
    }

    if (!allOk) {
        lastError_ = std::string("Failed to create GPU buffers: ") + SDL_GetError();
        SDL_Log("LbmSolver: %s", lastError_.c_str());
        return false;
    }

    LatticeSet set = GetLatticeSet(model_);
    std::vector<LatticeDirGPU> latticeGpu(Q_);
    for (int i = 0; i < Q_; ++i) {
        const LatticeDirection& d = set.directions[i];
        // mirrorY/mirrorZ found by search, not written by hand — see
        // LbmReference.h's own mirrorY_/mirrorZ_ comment for why this
        // always finds an exact match (both lattice tables are symmetric
        // under a single-axis sign flip by construction). O(Q^2), Q<=19,
        // done once here per direction.
        int mirrorY = d.opposite, mirrorZ = d.opposite;  // harmless fallback if a match is somehow never found
        for (int j = 0; j < Q_; ++j) {
            const LatticeDirection& dj = set.directions[j];
            if (dj.ex == d.ex && dj.ey == -d.ey && dj.ez == d.ez) mirrorY = j;
            if (dj.ex == d.ex && dj.ey == d.ey && dj.ez == -d.ez) mirrorZ = j;
        }
        latticeGpu[i] = {static_cast<float>(d.ex), static_cast<float>(d.ey), static_cast<float>(d.ez), d.weight,
                          d.opposite, mirrorY, mirrorZ, 0.0f};
    }
    UploadBuffer(device_, latticeTable_, latticeGpu.data(), latticeBytes);

    std::vector<uint32_t> solidGpu(cellCount_);
    for (size_t i = 0; i < cellCount_; ++i) solidGpu[i] = solidCpu_[i];
    UploadBuffer(device_, solidMask_, solidGpu.data(), solidBytes);

    // Upload signedDistance (or +infinity sentinels if the grid didn't populate
    // one, matching LbmReference::Init's fallback so isfinite() in the kernel
    // cleanly backs off to plain halfway bounce-back).
    if (grid.signedDistance.size() == cellCount_) {
        UploadBuffer(device_, signedDistance_, grid.signedDistance.data(), solidBytes);
    } else {
        std::vector<float> fallbackSdf(cellCount_, std::numeric_limits<float>::infinity());
        UploadBuffer(device_, signedDistance_, fallbackSdf.data(), solidBytes);
    }

    return true;
}

void LbmSolver::Shutdown() {
    if (!device_) return;
    if (latticeTable_) SDL_ReleaseGPUBuffer(device_, latticeTable_);
    if (solidMask_) SDL_ReleaseGPUBuffer(device_, solidMask_);
    if (signedDistance_) SDL_ReleaseGPUBuffer(device_, signedDistance_);
    if (dummyHaloBuffer_) SDL_ReleaseGPUBuffer(device_, dummyHaloBuffer_);
    if (readbackTransfer_) SDL_ReleaseGPUTransferBuffer(device_, readbackTransfer_);
    for (Slab& slab : slabs_) {
        if (slab.pop[0]) SDL_ReleaseGPUBuffer(device_, slab.pop[0]);
        if (slab.pop[1]) SDL_ReleaseGPUBuffer(device_, slab.pop[1]);
        if (slab.macroOut) SDL_ReleaseGPUBuffer(device_, slab.macroOut);
    }
    slabs_.clear();
    if (pipeline_) SDL_ReleaseGPUComputePipeline(device_, pipeline_);
    latticeTable_ = solidMask_ = signedDistance_ = dummyHaloBuffer_ = nullptr;
    readbackTransfer_ = nullptr;
    readbackTransferCapacityBytes_ = 0;
    pipeline_ = nullptr;
    device_ = nullptr;
}

void LbmSolver::Reset(const Params& params, bool seedAtMidSpeed) {
    params_ = params;
    if (!pipeline_) return;

    float rho = 1.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (seedAtMidSpeed) {
        float midSpeed = 0.5f * params_.inletVelocity;
        ux = params_.reverseFlow ? -midSpeed : midSpeed;
        // See LbmReference::Reset's comment — the two must stay in sync.
        float pressureDeviation = 0.5f * (params_.inletVelocity * params_.inletVelocity - midSpeed * midSpeed);
        rho = 1.0f + 3.0f * pressureDeviation;
    }

    LatticeSet set = GetLatticeSet(model_);
    std::vector<uint16_t> perDirection(static_cast<size_t>(Q_));
    for (int i = 0; i < Q_; ++i) {
        const LatticeDirection& d = set.directions[i];
        perDirection[static_cast<size_t>(i)] =
            FloatToHalf(EquilibriumCPU(static_cast<float>(d.ex), static_cast<float>(d.ey),
                                        static_cast<float>(d.ez), d.weight, rho, ux, uy, uz));
    }

    // Uniform across every cell (quiescent or mid-speed, same value
    // everywhere) and the same per-direction values regardless of slab —
    // so each slab's own pop[0] just gets this one small per-direction
    // table tiled across its own (smaller) cell count, rather than one
    // huge upload sized for the whole domain at once.
    for (Slab& slab : slabs_) {
        std::vector<uint32_t> initial(PopWordsPerCell(Q_) * slab.cellCount);
        for (size_t k = 0; k < PopWordsPerCell(Q_); ++k) {
            uint32_t lo = perDirection[2 * k];
            uint32_t hi = (2 * k + 1 < static_cast<size_t>(Q_)) ? perDirection[2 * k + 1] : 0u;
            std::fill(initial.begin() + k * slab.cellCount, initial.begin() + (k + 1) * slab.cellCount, lo | (hi << 16));
        }
        uint32_t popBytes = static_cast<uint32_t>(initial.size() * sizeof(uint32_t));
        UploadBuffer(device_, slab.pop[0], initial.data(), popBytes);
    }
    current_ = 0;
}

void LbmSolver::ResetFromField(const std::vector<float>& field) {
    if (!pipeline_ || field.size() != cellCount_ * 4) return;

    LatticeSet set = GetLatticeSet(model_);
    size_t cellsPerLayer = static_cast<size_t>(nx_) * ny_;

    // Same per-slab upload shape as Reset(), but each cell now gets its
    // own equilibrium (from `field`, GLOBAL-indexed) instead of one
    // value tiled everywhere — real per-cell floating-point work (Q_
    // Equilibrium calls), unlike Reset()'s plain std::fill, so this is
    // parallelized (flat range, within each slab's own local cell
    // indexing) rather than left single-threaded. The upload itself
    // stays on this (the calling) thread, same as Reset() — SDL_GPU
    // command buffer/transfer calls from multiple threads against one
    // device is not a pattern this app uses anywhere else.
    for (Slab& slab : slabs_) {
        std::vector<uint32_t> initial(PopWordsPerCell(Q_) * slab.cellCount);
        ParallelForRange(slab.cellCount, [&](size_t begin, size_t end) {
            for (size_t localCell = begin; localCell < end; ++localCell) {
                size_t localZ = localCell / cellsPerLayer;
                size_t withinLayer = localCell % cellsPerLayer;
                size_t globalCell = (static_cast<size_t>(slab.zStart) + localZ) * cellsPerLayer + withinLayer;
                float rho = solidCpu_[globalCell] ? 1.0f : field[globalCell * 4 + 0];
                float ux = solidCpu_[globalCell] ? 0.0f : field[globalCell * 4 + 1];
                float uy = solidCpu_[globalCell] ? 0.0f : field[globalCell * 4 + 2];
                float uz = solidCpu_[globalCell] ? 0.0f : field[globalCell * 4 + 3];
                uint16_t halves[20] = {};
                for (int i = 0; i < Q_; ++i) {
                    const LatticeDirection& d = set.directions[i];
                    halves[i] = FloatToHalf(EquilibriumCPU(static_cast<float>(d.ex), static_cast<float>(d.ey),
                                                            static_cast<float>(d.ez), d.weight, rho, ux, uy, uz));
                }
                // Two directions packed per word — see shaders/lbm_step.comp.glsl.
                for (size_t k = 0; k < PopWordsPerCell(Q_); ++k) {
                    initial[k * slab.cellCount + localCell] =
                        static_cast<uint32_t>(halves[2 * k]) | (static_cast<uint32_t>(halves[2 * k + 1]) << 16);
                }
            }
        });
        uint32_t popBytes = static_cast<uint32_t>(initial.size() * sizeof(uint32_t));
        UploadBuffer(device_, slab.pop[0], initial.data(), popBytes);
    }
    current_ = 0;
}

void LbmSolver::DispatchSlab(SDL_GPUCommandBuffer* cmd, size_t slabIndex, int localZOffset, int localZCount) {
    if (!pipeline_ || localZCount <= 0) return;
    const Slab& slab = slabs_[slabIndex];

    // Bound in place of a real neighbour at a domain edge — never
    // dereferenced there, see dummyHaloBuffer_'s comment in LbmSolver.h.
    SDL_GPUBuffer* prevPopIn = dummyHaloBuffer_;
    uint32_t prevCellCount = 0, prevLastLayerBase = 0;
    if (slabIndex > 0) {
        const Slab& prevSlab = slabs_[slabIndex - 1];
        prevPopIn = prevSlab.pop[current_];
        prevCellCount = static_cast<uint32_t>(prevSlab.cellCount);
        prevLastLayerBase = static_cast<uint32_t>(static_cast<size_t>(nx_) * ny_ * (prevSlab.zCount - 1));
    }
    SDL_GPUBuffer* nextPopIn = dummyHaloBuffer_;
    uint32_t nextCellCount = 0;
    if (slabIndex + 1 < slabs_.size()) {
        const Slab& nextSlab = slabs_[slabIndex + 1];
        nextPopIn = nextSlab.pop[current_];
        nextCellCount = static_cast<uint32_t>(nextSlab.cellCount);
    }

    SimParamsGPU simParams = {static_cast<uint32_t>(nx_),
                               static_cast<uint32_t>(ny_),
                               static_cast<uint32_t>(nz_),
                               static_cast<uint32_t>(Q_),
                               params_.tau,
                               params_.inletVelocity,
                               static_cast<uint32_t>(localZOffset),
                               params_.reverseFlow ? 1.0f : 0.0f,
                               static_cast<uint32_t>(slab.zStart),
                               static_cast<uint32_t>(slab.zCount),
                               prevCellCount,
                               prevLastLayerBase,
                               nextCellCount,
                               params_.freeSlipY ? 1.0f : 0.0f,
                               params_.freeSlipZ ? 1.0f : 0.0f,
                               params_.smagorinskyCs};

    uint32_t groupsX = (static_cast<uint32_t>(nx_) + kThreadsX - 1) / kThreadsX;
    uint32_t groupsY = (static_cast<uint32_t>(ny_) + kThreadsY - 1) / kThreadsY;
    uint32_t groupsZ = (static_cast<uint32_t>(localZCount) + kThreadsZ - 1) / kThreadsZ;

    SDL_GPUBuffer* popIn = slab.pop[current_];
    SDL_GPUBuffer* popOut = slab.pop[1 - current_];

    SDL_PushGPUComputeUniformData(cmd, 0, &simParams, sizeof(simParams));

    SDL_GPUStorageBufferReadWriteBinding rwBindings[2] = {};
    rwBindings[0].buffer = popOut;
    rwBindings[1].buffer = slab.macroOut;
    SDL_GPUComputePass* pass = SDL_BeginGPUComputePass(cmd, nullptr, 0, rwBindings, 2);

    SDL_BindGPUComputePipeline(pass, pipeline_);
    SDL_GPUBuffer* readonly[6] = {latticeTable_, solidMask_, signedDistance_, popIn, prevPopIn, nextPopIn};
    SDL_BindGPUComputeStorageBuffers(pass, 0, readonly, 6);

    SDL_DispatchGPUCompute(pass, groupsX, groupsY, groupsZ);
    SDL_EndGPUComputePass(pass);
}

// A caller's GLOBAL [zStart, zStart+zCount) range (StepSolverPaced's
// adaptive pacing chunks, in main.cpp) has no relation to the internal
// slab boundaries and may span several of them — e.g. a 256-layer pacing
// chunk straddling two 180-layer slabs. This splits that range into its
// per-slab overlaps and dispatches each separately (every slab is its
// own buffer set, so it can only ever be one compute pass's worth of
// bindings at a time) — transparent to every caller, which still only
// ever thinks in terms of the whole domain's global z range.
void LbmSolver::StepRange(SDL_GPUCommandBuffer* cmd, int zStart, int zCount) {
    if (!pipeline_ || zCount <= 0) return;
    int zEnd = zStart + zCount;
    for (size_t s = 0; s < slabs_.size(); ++s) {
        const Slab& slab = slabs_[s];
        int slabEnd = slab.zStart + slab.zCount;
        int overlapStart = std::max(zStart, slab.zStart);
        int overlapEnd = std::min(zEnd, slabEnd);
        if (overlapStart >= overlapEnd) continue;
        DispatchSlab(cmd, s, overlapStart - slab.zStart, overlapEnd - overlapStart);
    }
}

void LbmSolver::Step(SDL_GPUCommandBuffer* cmd, int numSubsteps) {
    if (!pipeline_) return;
    for (int step = 0; step < numSubsteps; ++step) {
        StepRange(cmd, 0, nz_);
        AdvanceSubstep();
    }
}

// Same multi-slab-overlap splitting StepRange uses (see its comment) —
// but unlike StepRange, this used to (before this comment was added)
// issue one FULL independent GPU round trip — its own transfer buffer,
// command buffer, submit, and fence wait — per overlapping slab. That's
// fine for a chunk that lands entirely within one slab (the common
// case), but a chunk that happens to straddle a slab boundary silently
// cost 2x (or more, for a chunk spanning several slabs) the GPU
// round-trip latency of an ordinary one — and since slab boundaries
// recur at a fixed depth, that 2x-cost chunk recurred at a fixed period
// too, while holding app.lbmMutex that the stepping thread also needs
// for every one of ITS OWN chunks (see StepSolverPaced's comment in
// main.cpp) — a textbook periodic stall, not just a slower average.
// StepRange never had this problem: DispatchSlab issues its compute
// pass straight into the caller's own command buffer, so spanning
// several slabs just means a few more (cheap) dispatch calls within the
// ONE submission StepRange's caller already does. This does the
// equivalent for a download: collect every overlapping slab's byte
// range first (pure arithmetic, no GPU calls), then transfer, submit,
// and fence-wait exactly ONCE for the whole chunk regardless of how
// many slabs it actually spans.
void LbmSolver::ReadbackFieldRange(int zStart, int zCount, std::vector<float>& outField) {
    if (!pipeline_ || zCount <= 0) return;
    int zEnd = zStart + zCount;
    size_t cellsPerLayer = static_cast<size_t>(nx_) * ny_;

    struct Piece {
        size_t slabIndex;
        uint32_t srcByteOffset;   // into that slab's own macroOut
        uint32_t byteSize;
        uint32_t dstByteOffset;   // into the one shared transfer buffer below
        size_t globalCellOffset;  // where it lands in outField
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

    // Grow-on-demand, never shrink/recreate for a smaller later call —
    // see readbackTransfer_'s own comment in LbmSolver.h for why this
    // isn't created and released every call the way it used to be.
    if (totalBytes > readbackTransferCapacityBytes_) {
        if (readbackTransfer_) SDL_ReleaseGPUTransferBuffer(device_, readbackTransfer_);
        SDL_GPUTransferBufferCreateInfo tbInfo = {};
        tbInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
        tbInfo.size = totalBytes;
        readbackTransfer_ = SDL_CreateGPUTransferBuffer(device_, &tbInfo);
        readbackTransferCapacityBytes_ = readbackTransfer_ ? totalBytes : 0;
    }
    if (!readbackTransfer_) return;

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_);
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);
    for (const Piece& piece : pieces) {
        SDL_GPUBufferRegion src = {slabs_[piece.slabIndex].macroOut, piece.srcByteOffset, piece.byteSize};
        SDL_GPUTransferBufferLocation dst = {readbackTransfer_, piece.dstByteOffset};
        SDL_DownloadFromGPUBuffer(copyPass, &src, &dst);
    }
    SDL_EndGPUCopyPass(copyPass);

    SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    SDL_WaitForGPUFences(device_, true, &fence, 1);
    SDL_ReleaseGPUFence(device_, fence);

    const uint8_t* mapped = static_cast<const uint8_t*>(SDL_MapGPUTransferBuffer(device_, readbackTransfer_, false));
    for (const Piece& piece : pieces) {
        const float* src = reinterpret_cast<const float*>(mapped + piece.dstByteOffset);
        size_t floatCount = piece.byteSize / sizeof(float);
        std::copy(src, src + floatCount, outField.begin() + static_cast<ptrdiff_t>(piece.globalCellOffset * 4));
    }
    SDL_UnmapGPUTransferBuffer(device_, readbackTransfer_);
}

bool LbmSolver::ProbeSliceHealth(int zLayer) {
    if (!pipeline_ || !device_ || nz_ <= 0) return false;
    int gz = std::clamp(zLayer, 0, nz_ - 1);
    size_t cellsPerLayer = static_cast<size_t>(nx_) * ny_;
    if (cellsPerLayer == 0) return false;

    const Slab* targetSlab = nullptr;
    int localZ = 0;
    for (const Slab& slab : slabs_) {
        if (gz >= slab.zStart && gz < slab.zStart + slab.zCount) {
            targetSlab = &slab;
            localZ = gz - slab.zStart;
            break;
        }
    }
    if (!targetSlab || !targetSlab->macroOut) return false;

    uint32_t byteSize = static_cast<uint32_t>(cellsPerLayer * 4 * sizeof(float));
    uint32_t srcByteOffset = static_cast<uint32_t>(static_cast<size_t>(localZ) * cellsPerLayer * 4 * sizeof(float));

    if (byteSize > readbackTransferCapacityBytes_) {
        if (readbackTransfer_) SDL_ReleaseGPUTransferBuffer(device_, readbackTransfer_);
        SDL_GPUTransferBufferCreateInfo tbInfo = {};
        tbInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
        tbInfo.size = byteSize;
        readbackTransfer_ = SDL_CreateGPUTransferBuffer(device_, &tbInfo);
        readbackTransferCapacityBytes_ = readbackTransfer_ ? byteSize : 0;
    }
    if (!readbackTransfer_) return false;

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_);
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUBufferRegion src = {targetSlab->macroOut, srcByteOffset, byteSize};
    SDL_GPUTransferBufferLocation dst = {readbackTransfer_, 0};
    SDL_DownloadFromGPUBuffer(copyPass, &src, &dst);
    SDL_EndGPUCopyPass(copyPass);

    SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (fence) {
        SDL_WaitForGPUFences(device_, true, &fence, 1);
        SDL_ReleaseGPUFence(device_, fence);
    }

    bool diverged = false;
    const float* mapped = static_cast<const float*>(SDL_MapGPUTransferBuffer(device_, readbackTransfer_, false));
    if (mapped) {
        size_t globalCellBase = static_cast<size_t>(gz) * cellsPerLayer;
        for (size_t i = 0; i < cellsPerLayer; ++i) {
            float v0 = mapped[i * 4 + 0];
            float v1 = mapped[i * 4 + 1];
            float v2 = mapped[i * 4 + 2];
            float v3 = mapped[i * 4 + 3];
            if (!std::isfinite(v0) || !std::isfinite(v1) || !std::isfinite(v2) || !std::isfinite(v3)) {
                diverged = true;
                break;
            }
            if (!solidCpu_[globalCellBase + i]) {
                float speedSq = v1 * v1 + v2 * v2 + v3 * v3;
                if (v0 <= 0.0f || speedSq > 4.0f) {
                    diverged = true;
                    break;
                }
            }
        }
        SDL_UnmapGPUTransferBuffer(device_, readbackTransfer_);
    }
    return diverged;
}

std::vector<float> LbmSolver::ReadbackField() {
    std::vector<float> field;
    if (!pipeline_) return field;
    field.resize(cellCount_ * 4);

    // Z-layers are contiguous within each slab's own macroOut (same
    // z-major convention as before, now per-slab) — ReadbackFieldRange
    // downloads however many of these 16-layer chunks' worth of slabs
    // a given call spans as one batched transfer, see its own comment.
    constexpr int kReadbackChunkZDepth = 16;
    for (int zStart = 0; zStart < nz_; zStart += kReadbackChunkZDepth) {
        int zCount = std::min(kReadbackChunkZDepth, nz_ - zStart);
        ReadbackFieldRange(zStart, zCount, field);
    }
    return field;
}

LbmSolver::Stats LbmSolver::ReadbackStats() {
    Stats stats;
    std::vector<float> field = ReadbackField();
    if (field.empty()) return stats;

    stats.minDensity = 1e30f;
    stats.maxDensity = -1e30f;
    size_t fluidCount = 0;
    double densitySum = 0.0, speedSum = 0.0;
    double forceXSum = 0.0, forceYSum = 0.0, forceZSum = 0.0;
    for (size_t cell = 0; cell < cellCount_; ++cell) {
        if (solidCpu_[cell]) {
            float stress = field[cell * 4 + 0];
            float fx = field[cell * 4 + 1];
            float fy = field[cell * 4 + 2];
            float fz = field[cell * 4 + 3];
            if (std::isfinite(stress) && std::isfinite(fx) && std::isfinite(fy) && std::isfinite(fz)) {
                forceXSum += fx;
                forceYSum += fy;
                forceZSum += fz;
                stats.maxStress = std::max(stats.maxStress, stress);
            }
            continue;
        }
        float rho = field[cell * 4 + 0];
        float ux = field[cell * 4 + 1];
        float uy = field[cell * 4 + 2];
        float uz = field[cell * 4 + 3];
        if (!std::isfinite(rho) || !std::isfinite(ux) || !std::isfinite(uy) || !std::isfinite(uz)) {
            stats.hasNaN = true;
            continue;
        }
        float speed = std::sqrt(ux * ux + uy * uy + uz * uz);
        stats.minDensity = std::min(stats.minDensity, rho);
        stats.maxDensity = std::max(stats.maxDensity, rho);
        stats.maxSpeed = std::max(stats.maxSpeed, speed);
        densitySum += rho;
        speedSum += speed;
        ++fluidCount;
    }
    if (fluidCount > 0) {
        stats.avgDensity = static_cast<float>(densitySum / static_cast<double>(fluidCount));
        stats.avgSpeed = static_cast<float>(speedSum / static_cast<double>(fluidCount));
    }
    stats.forceX = static_cast<float>(forceXSum);
    stats.forceY = static_cast<float>(forceYSum);
    stats.forceZ = static_cast<float>(forceZSum);
    stats.valid = true;
    return stats;
}
