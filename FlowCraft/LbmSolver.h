#pragma once

#include <SDL3/SDL.h>

#include <cstdint>
#include <string>
#include <vector>

#include "LatticeModel.h"
#include "VoxelGrid.h"

// GPU (SDL_GPU compute) LBM solver. Mirrors LbmReference's math exactly —
// see that class's header comment for the physics and boundary-condition
// design; LbmSolver.cpp's embedded MSL kernel is a direct line-by-line port
// and should be kept in sync with LbmReference.cpp if either changes.
//
// The lattice model (D3Q15/D3Q19) is runtime-selectable without any shader
// variants: the kernel loops over Q directions using a direction/weight
// table uploaded as a storage buffer, so swapping models is just
// re-uploading that table (done in Init/Reset), not recompiling anything.
//
// Z-SLAB DOMAIN DECOMPOSITION — why this class has a `slabs_` vector
// instead of one `pop_`/`macroOut_` buffer pair:
//
// SDL_GPUBufferCreateInfo::size is a plain uint32, so every individual
// SDL_GPU buffer is hard-capped at 4GiB regardless of how much memory the
// GPU actually has. The population buffers (Q*cellCount*sizeof(half)
// bytes) hit that ceiling first — around 113M cells (D3Q19) / 143M cells
// (D3Q15), i.e. a cubic resolution of roughly 480-520 — because Q (15 or
// 19) copies of the per-cell state make them the single largest buffer by
// a wide margin. macroOut (4 floats/cell) hits the same ceiling later, at
// ~268M cells (~resolution 645); solidMask (1 uint32/cell) and the Q-sized
// latticeTable never practically hit it (~1.07B cells, resolution ~1024).
//
// Rather than one global `pop_[2]`/`macroOut_` pair, the domain is split
// into Z-slabs (chosen in Init(), see its comment), each with its OWN
// pop_[2]/macroOut buffer pair sized only for that slab's own cell count —
// so no single buffer ever needs to approach the 4GiB cap no matter how
// large the whole domain gets; growing the domain just means more,
// equally-small slabs. solidMask_/signedDistance_/latticeTable_ stay single
// GLOBAL buffers, indexed by the domain's true global cell index exactly as
// before — they're nowhere near the cap even at the highest resolution
// decomposition is meant to reach (4 bytes/cell each for solidMask_ and
// signedDistance_), so splitting them would only add complexity for no
// benefit. The actual decomposition kernel-side work is in LbmSolver.cpp's
// comment on DispatchSlab/kLbmStepMSL.
//
// This is a *memory-capacity* decomposition, not a parallelism one — it
// doesn't run slabs concurrently or on separate devices, it only exists so
// each slab's own storage fits under the API's per-buffer limit. The
// public interface below is completely unchanged by this: every method
// still takes/returns GLOBAL z-ranges and indices exactly as it always
// has (Step/StepRange/ReadbackField/ReadbackFieldRange/Nx/Ny/Nz) — the
// slab boundaries are an internal implementation detail that callers
// (main.cpp's StepSolverPaced, ReadbackFieldPacedGPU, etc.) never need to
// know about, even when a single StepRange/ReadbackFieldRange call's
// range spans more than one slab (handled internally — see their .cpp
// comments).
class LbmSolver {
public:
    struct Params {
        float tau = 0.8f;
        float inletVelocity = 0.05f;
        // false: inlet at x=0, flow pushes toward +X — "Right to Left" as
        // seen from the Right view (see main.cpp's Flow direction combo).
        // false is the default — flipped from true after the grid's
        // own x/y-to-world mapping changed (grid-x is world Y now, not
        // world X — see VoxelGrid::CellCenter's comment) made the old
        // default's on-screen direction read backward. Not re-deriving
        // which literal value reads which way on screen this time (see
        // this file's own history of getting that wrong) — this is
        // set from the user's own direct report of the resulting
        // direction, not re-predicted. Everything else about the
        // solver (the interior collision/streaming, the outlet's
        // extrapolation logic) is direction-agnostic and just follows
        // whichever face is which — see the kernel's comment in
        // LbmSolver.cpp.
        bool reverseFlow = false;
        // When true (the default), that side wall uses free-slip
        // (specular/mirror) reflection: only the velocity component
        // normal to the wall is reflected, tangential flow passes
        // through unaffected — an open, wall-less domain edge rather
        // than a real tunnel wall. Defaulted on because a damped,
        // no-slip boundary layer against the domain's own edges (the
        // alternative) tends to mask exactly the turbulent structure in
        // the flow someone's using this to look at. When false, that
        // wall instead uses ordinary no-slip bounce-back — same
        // treatment as solid model geometry, forming a real boundary
        // layer against it, like a physical wind tunnel's own walls
        // would. Two independent flags (not one) since a domain can
        // reasonably want a no-slip "ground" (freeSlipZ false) alongside
        // free-slip lateral walls, or any other combination. Implemented
        // identically on both backends — see this file's own shader
        // source and LbmReference.cpp's Step() for the matching logic.
        bool freeSlipY = true;
        bool freeSlipZ = true;
        // Smagorinsky sub-grid turbulence constant (0 = off). Adds eddy viscosity so very
        // low tau (near-air viscosity) stays stable. Typical 0.10-0.18.
        float smagorinskyCs = 0.12f;
    };

    bool Init(SDL_GPUDevice* device, const VoxelGrid& grid, LatticeModel model);
    void Shutdown();

    // Set when Init() returns false, so a caller (the UI, in particular —
    // see AppState::lbmInitError) can show *why* instead of just failing
    // silently/to a log the user never sees. Empty after a successful
    // Init(). Not thread-safe on its own; callers that run Init() off the
    // UI thread already serialize access through app.lbmMutex anyway.
    const std::string& LastError() const { return lastError_; }

    // Re-seeds every slab's population buffers and (re)uploads the
    // lattice table, solid mask, and params. Call when changing the
    // model, grid, or starting a fresh run. Quiescent (rho=1, u=0
    // everywhere) unless `seedAtMidSpeed` — see LbmReference::Reset's
    // comment, which this mirrors exactly (the two must stay in sync,
    // same as everything else about these two solvers).
    void Reset(const Params& params, bool seedAtMidSpeed = false);

    // Same idea as Reset(), but seeds each cell's equilibrium from its
    // OWN (rho,ux,uy,uz) in `field` instead of one uniform value
    // broadcast everywhere — `field` is GLOBAL-indexed, 4 floats/cell,
    // the exact same layout ReadbackField() returns (so a caller can
    // feed this solver's own, or another grid's nearest-sampled, field
    // straight back in). Used for the "warm start" seed — see
    // ApplyWarmStart's comment in main.cpp. Deliberately does NOT touch
    // params_ the way Reset() does — a caller using this INSTEAD of
    // Reset() still needs its own SetParams() call first (Init() alone
    // never sets params_), or every subsequent Step() runs with
    // whatever params_ happened to default-construct to.
    void ResetFromField(const std::vector<float>& field);

    // Updates tau/inletVelocity without touching the current flow state —
    // safe to call while running.
    void SetParams(const Params& params) { params_ = params; }

    // Dispatches `numSubsteps` fused collision+streaming+BC passes. Each
    // substep is its own compute pass (ping-ponging the population
    // buffers), since SDL_GPU does not order writes between dispatches
    // within a single pass. Implemented as numSubsteps calls to
    // StepRange(cmd, 0, Nz()) + AdvanceSubstep() — see those for why a
    // caller might want to call them directly instead.
    void Step(SDL_GPUCommandBuffer* cmd, int numSubsteps);

    // One substep's dispatch, restricted to cells with GLOBAL z in
    // [zStart, zStart+zCount) — the rest of StepSolverPaced's reasoning is
    // in main.cpp: splitting a substep this way bounds a single dispatch's
    // GPU time to a fixed chunk size regardless of total grid resolution,
    // rather than scaling with it, by giving a caller that submits each
    // chunk as its own command buffer somewhere else to interleave in
    // between chunks. This range is unrelated to (and may span several
    // of) the internal Z-slabs described above the class — StepRange
    // transparently issues one dispatch per slab the range overlaps; see
    // its .cpp comment.
    //
    // Reads pop_[current] and writes pop_[1-current] for every cell in the
    // chunk, same as one iteration of Step()'s loop, but does NOT flip
    // which buffer is "current" — the whole grid shares one ping-pong
    // state, so that only makes sense once every chunk covering the full
    // Z range has been dispatched. Call AdvanceSubstep() once after that,
    // not after each chunk.
    void StepRange(SDL_GPUCommandBuffer* cmd, int zStart, int zCount);

    // Flips the ping-pong population buffer index — call once per substep,
    // after every StepRange() chunk covering the full Z range for that
    // substep has been dispatched (not once per chunk). One index shared
    // across every slab: every slab's pop[0]/pop[1] pair always ping-pongs
    // in lockstep, since they jointly represent one global substep state,
    // just split by Z range.
    void AdvanceSubstep() { current_ = 1 - current_; }

    struct Stats {
        float avgDensity = 0.0f, minDensity = 0.0f, maxDensity = 0.0f;
        float avgSpeed = 0.0f, maxSpeed = 0.0f;
        // Integrated Bouzidi momentum-exchange force components (lattice force
        // units, along grid axes: x=flow, y=lateral, z=up), peak per-cell
        // surface stress magnitude, and Centre of Pressure (world mm).
        float forceX = 0.0f, forceY = 0.0f, forceZ = 0.0f;
        float maxStress = 0.0f;
        Vec3 centerOfPressure{0.0f, 0.0f, 0.0f};
        float liftBaseZ = 0.0f;
        bool hasNaN = false;
        bool valid = false;
    };
    // Blocking: submits its own command buffer, waits for it, and reads the
    // macroscopic (rho,u) buffer back to CPU. Not for per-frame use — call
    // periodically for diagnostics/UI, not every render frame.
    Stats ReadbackStats();

    // Same readback, but returns the full per-cell field (4 floats per
    // cell: rho,ux,uy,uz) rather than aggregate stats — for building the
    // flow overlay geometry. Also blocking; call on demand, not per-frame.
    // Internally chunked along Z (see ReadbackFieldRange) for the same
    // reason Step() is: one unchunked download of the whole field is a
    // single long GPU operation, and any single long GPU operation on any
    // thread was found to stall a *different* thread's swapchain
    // acquire/present for its entire duration — see StepSolverPaced's
    // comment in main.cpp for the measurements. This was the piece that
    // chunking Step() alone didn't cover: overlayAutoUpdate's periodic
    // readback (on the solver thread) used to run as one unchunked
    // download regardless of resolution.
    std::vector<float> ReadbackField();

    // One Z-chunk of ReadbackField(), writing into the corresponding slice
    // of `outField` (must already be sized to CellCount()*4). `zStart`/
    // `zCount` are GLOBAL, exactly like StepRange's, and may likewise span
    // more than one internal slab — handled transparently (one download
    // per overlapping slab's own macroOut buffer — see the .cpp comment).
    // Exposed in case a caller wants its own pacing between chunks
    // (ReadbackField itself does not pace — downloads are far more
    // uniform/predictable in cost than compute dispatches, so a fixed
    // chunk depth was enough without needing StepSolverPaced's adaptive
    // measurement).
    void ReadbackFieldRange(int zStart, int zCount, std::vector<float>& outField);

    const std::vector<uint8_t>& SolidMask() const { return solidCpu_; }

    bool IsValid() const { return pipeline_ != nullptr; }
    LatticeModel Model() const { return model_; }
    int Nx() const { return nx_; }
    int Ny() const { return ny_; }
    int Nz() const { return nz_; }

private:
    // One Z-slab's own storage — see the class comment above for why this
    // exists at all. `zStart`/`zCount` are this slab's position within the
    // GLOBAL domain (fixed once chosen in Init(), never change); `cellCount`
    // is nx_*ny_*zCount, this slab's own Q-plane stride for pop[0]/pop[1]
    // (deliberately NOT the global cellCount_ — that's the whole point).
    struct Slab {
        int zStart = 0;
        int zCount = 0;
        size_t cellCount = 0;
        SDL_GPUBuffer* pop[2] = {nullptr, nullptr};
        SDL_GPUBuffer* macroOut = nullptr;
    };

    // Dispatches one slab's compute pass for its OWN local z-range
    // [localZOffset, localZOffset+localZCount) (i.e. GLOBAL z range
    // [slab.zStart+localZOffset, slab.zStart+localZOffset+localZCount)) —
    // see StepRange's .cpp comment for how a caller's global range is
    // split across however many slabs it overlaps before calling this.
    void DispatchSlab(SDL_GPUCommandBuffer* cmd, size_t slabIndex, int localZOffset, int localZCount);

    SDL_GPUDevice* device_ = nullptr;
    SDL_GPUComputePipeline* pipeline_ = nullptr;

    int nx_ = 0, ny_ = 0, nz_ = 0, Q_ = 0;
    size_t cellCount_ = 0;  // GLOBAL cell count: solidMask_'s size/indexing and ReadbackField()'s output size
    LatticeModel model_ = LatticeModel::D3Q19;
    Params params_;

    SDL_GPUBuffer* latticeTable_ = nullptr;    // global, Q-sized
    SDL_GPUBuffer* solidMask_ = nullptr;       // global, GLOBAL-cellCount_-sized — see the class comment on why this
                                                 // and signedDistance_ below, not the slabs, are the real resolution ceiling
    SDL_GPUBuffer* signedDistance_ = nullptr;  // global, GLOBAL-cellCount_-sized (float per cell) — mirrors
                                                 // VoxelGrid::signedDistance for SDF interpolated bounce-back in kLbmStepMSL
    // Bound in place of a missing prev/next slab's popIn at a domain edge
    // (the first slab has no prev, the last has no next) — SDL_GPU compute
    // pipelines bind a fixed number of buffers, so there's no way to bind
    // "nothing" there instead. Never actually dereferenced in that case:
    // the kernel only reads prevPopIn/nextPopIn when the streaming
    // neighbour's global z is exactly one below/above this slab's own
    // range, which by construction can't happen at a domain edge (that
    // neighbour is already out of the GLOBAL domain, caught by the
    // existing in-bounds check same as it always was) — see
    // kLbmStepMSL's comment. Sized for one cell's worth of Q planes,
    // the minimum SDL_GPU will create.
    SDL_GPUBuffer* dummyHaloBuffer_ = nullptr;
    std::vector<Slab> slabs_;

    // ReadbackFieldRange's own download transfer buffer, kept and grown
    // on demand (never shrunk, never recreated just because a later call
    // needs fewer bytes) rather than created and released on every call —
    // same reasoning as OverlayRenderer::dynamicTransfer_'s own comment:
    // this can be called many times in a tight pacing loop (see
    // ReadbackFieldPacedGPU in main.cpp), and SDL_CreateGPUTransferBuffer/
    // SDL_ReleaseGPUTransferBuffer's own allocation cost, paid every
    // single chunk, turned out to be a bigger piece of a large readback's
    // total wall time than the actual transfer — exactly the kind of
    // per-call GPU resource churn StepSolverPaced's own comment (main.cpp)
    // already identified as a stutter source in a different context. No
    // cycling/fencing concerns the way an upload buffer reused across
    // frames would have: ReadbackFieldRange already fence-waits for the
    // GPU before mapping it, every single call, so it's never still in
    // use by the time the next call wants to reuse it.
    SDL_GPUTransferBuffer* readbackTransfer_ = nullptr;
    uint32_t readbackTransferCapacityBytes_ = 0;

    int current_ = 0;  // shared ping-pong index across every slab — see AdvanceSubstep()'s comment

    std::vector<uint8_t> solidCpu_;  // kept for stats (skip solid cells without a GPU readback)
    std::string lastError_;
};
