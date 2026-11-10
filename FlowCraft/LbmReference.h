#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "LatticeModel.h"
#include "VoxelGrid.h"

// CPU reference implementation of one LBM step: fused streaming +
// regularized collision + boundary conditions ("pull" streaming — each
// cell gathers from its neighbours rather than scattering into them, so it
// parallelizes per-cell with no write races). The interior collision is
// regularized (Latt & Chopard), not plain BGK — it reconstructs the
// relaxed non-equilibrium populations from only their 2nd-order Hermite
// (stress-tensor) projection rather than relaxing the gathered
// populations directly, filtering out higher-order numerical noise that
// plain BGK would otherwise only damp slowly — see Step()'s comment for
// the derivation. This exists to validate the physics (mass conservation,
// stability, sane inlet response) with ordinary standalone tests before
// trusting the GPU compute shader, which mirrors this exact logic — see
// LbmSolver.cpp's shader source and keep the two in sync if either
// changes.
//
// Domain boundary conditions (all in lattice units, dx=dt=1), normally
// x=0 is the inlet and x=nx-1 the outlet — reverseFlow (Params) swaps
// which face is which and negates the inlet velocity, see Step()'s
// comment:
//  - inlet: fixed equilibrium at rho=1, u=(+-inletVelocity,0,0) — the
//    domain's one fixed pressure/velocity reference.
//  - outlet: convective (zero-gradient) outflow — both rho and u are
//    extrapolated from the interior neighbour and reconstructed via the
//    equilibrium distribution, nothing pinned to a fixed value. See the
//    outlet's comment in Step() for why *not* pinning density matters
//    (transient pressure waves reflecting off a fixed-density boundary).
//  - y/z domain edges and solid cells: bounce-back (no-slip) — falls out
//    of the same neighbour lookup used for ordinary streaming, since an
//    out-of-bounds or solid neighbour is treated the same way.
class LbmReference {
public:
    struct Params {
        float tau = 0.8f;            // relaxation time, >0.5 required for stability
        float inletVelocity = 0.05f;  // lattice-unit velocity at the inlet; keep well under ~0.1 (low Mach)
        bool reverseFlow = false;   // see LbmSolver::Params::reverseFlow — mirrors it exactly, including the default
        // See LbmSolver::Params::freeSlipY/freeSlipZ — mirrors them
        // exactly, including the default. Implemented here in Step()'s
        // domain-edge branch; LbmSolver.cpp's shader has the matching
        // GPU-side logic.
        bool freeSlipY = true;
        bool freeSlipZ = true;
        float smagorinskyCs = 0.12f;  // see LbmSolver::Params::smagorinskyCs
    };

    // `seedAtMidSpeed` forwarded straight to Reset() — see its comment.
    void Init(const VoxelGrid& grid, LatticeModel model, const Params& params, bool seedAtMidSpeed = false);

    // Quiescent start (rho=1, u=0 everywhere, inlet pushes flow in from
    // there) unless `seedAtMidSpeed` — then every cell starts at rho and
    // u consistent with half the inlet speed (direction per
    // params_.reverseFlow) via the same Bernoulli relation
    // PotentialFlowSolver's pressure uses, rather than every cell having
    // to accelerate from rest: a faster route to a representative-looking
    // state, at the cost of not being a physically real initial condition
    // (nothing produced it via the actual dynamics) — so early steps
    // after a mid-speed seed are relaxing away an artificial start, not
    // showing genuine transient behaviour, the same caveat a quiescent
    // start's own early transient already has, just a different one.
    void Reset(bool seedAtMidSpeed = false);

    // Same idea as Reset(), but seeds each cell's equilibrium from its
    // OWN (rho,ux,uy,uz) in `field` instead of one uniform value
    // broadcast everywhere — `field` is GLOBAL-indexed, 4 floats/cell,
    // the exact same layout ReadbackField() returns. Used for the "warm
    // start" seed — see ApplyWarmStart's comment in main.cpp. Unlike
    // LbmSolver::ResetFromField, there's no params_ to worry about here —
    // Init() already set it before this would ever be called.
    void ResetFromField(const std::vector<float>& field);

    // Updates tau/inletVelocity without touching the current flow state —
    // mirrors LbmSolver::SetParams, so main.cpp can treat both backends
    // the same way.
    void SetParams(const Params& params) { params_ = params; }

    // Fused collision+streaming+BC for every cell, one lattice timestep.
    // Parallelized across std::thread::hardware_concurrency() threads,
    // split by Z-range: safe because every cell only *reads* from the
    // previous step's population buffer and only *writes* its own slot in
    // the next one (the same "pull" structure the GPU kernel uses, which
    // is why that one parallelizes trivially across SIMD lanes too) — no
    // two threads ever touch the same memory.
    //
    // `progress`, when non-null, is updated as each thread finishes a
    // Z-layer (so at most nz_ stores total, negligible next to the
    // per-cell work) with (substepIndex + thisCallFraction) /
    // substepCount — i.e. this one Step() call's own 0..1 progress,
    // rescaled to its slot within a caller's larger substeps batch (see
    // StepReferenceLocked in main.cpp). Multiple threads finish Z-layers
    // in no particular order, so the stored value can briefly tick
    // backward by a layer or two — fine for a progress bar, not for
    // anything that needs monotonicity. Defaults (substepIndex=0,
    // substepCount=1) mean "just this call's own fraction," for a caller
    // that isn't batching several Step() calls into one reported range.
    void Step(std::atomic<float>* progress = nullptr, int substepIndex = 0, int substepCount = 1);

    void GetMacroscopic(int x, int y, int z, float& rho, float& ux, float& uy, float& uz) const;

    // Every cell's (rho,ux,uy,uz), in the same 4-floats-per-cell layout
    // LbmSolver::ReadbackField() uses — lets main.cpp's overlay/stats code
    // stay backend-agnostic. Also parallelized, same reasoning as Step().
    std::vector<float> ReadbackField() const;

    struct Stats {
        float avgDensity = 0.0f, minDensity = 0.0f, maxDensity = 0.0f;
        float avgSpeed = 0.0f, maxSpeed = 0.0f;
        float forceX = 0.0f, forceY = 0.0f, forceZ = 0.0f;
        float maxStress = 0.0f;
        double totalMass = 0.0;
        bool hasNaN = false;
    };
    // Also parallelized (per-thread partial stats merged at the end) —
    // ComputeStats was never free, and the whole point of a CPU backend
    // is handling grids too large for a single GPU buffer, so this can
    // run over a *lot* of cells.
    Stats ComputeStats() const;

    int Nx() const { return nx_; }
    int Ny() const { return ny_; }
    int Nz() const { return nz_; }
    bool HasDiverged() const { return hasDiverged_.load(std::memory_order_relaxed); }

private:
    static constexpr int kMaxQ = 19;

    size_t CellIndex(int x, int y, int z) const { return (static_cast<size_t>(z) * ny_ + y) * nx_ + x; }
    size_t PopIndex(int dir, size_t cell) const { return static_cast<size_t>(dir) * cellCount_ + cell; }
    bool InBounds(int x, int y, int z) const { return x >= 0 && y >= 0 && z >= 0 && x < nx_ && y < ny_ && z < nz_; }
    bool IsSolid(int x, int y, int z) const { return InBounds(x, y, z) && solid_[CellIndex(x, y, z)] != 0; }
    void MacroscopicFrom(const std::vector<float>& f, size_t cell, float& rho, float& ux, float& uy, float& uz) const;

    // Interpolated (Bouzidi/Mei-Luo-Shyy) bounce-back for a fluid cell
    // `cell` whose direction-`i` neighbor (grid position nxp,nyp,nzp) is
    // solid model geometry — replaces the plain halfway bounce-back
    // (`fIn[opposite][cell]`) with a value that places the wall at its
    // true SDF-derived position along that link instead of always
    // exactly halfway, correcting the voxel-staircase for a *static*
    // immersed boundary. Falls back to the plain scheme wherever the SDF
    // data needed isn't usable (see its own .cpp comment) — never worse
    // than the scheme this replaces, only better when valid data is
    // available. Only ever called for a genuinely-solid in-bounds
    // neighbor, never a domain edge (see Step()'s own comment on why
    // those two cases are no longer handled identically).
    float InterpolatedBounceBack(const std::vector<float>& fIn, int x, int y, int z, size_t cell, int i, int nxp,
                                   int nyp, int nzp) const;
    void SolidCellForce(const std::vector<float>& fIn, int x, int y, int z, size_t cell, float& outStress,
                        float& outFx, float& outFy, float& outFz) const;

    int nx_ = 0, ny_ = 0, nz_ = 0, Q_ = 0;
    size_t cellCount_ = 0;
    Params params_;
    std::vector<LatticeDirection> dirs_;
    std::vector<uint8_t> solid_;
    // Mirrors VoxelGrid::signedDistance — see InterpolatedBounceBack.
    // Empty when the grid Init() was given didn't have one (e.g. a grid
    // built before this existed), in which case every bounce-back falls
    // back to the plain halfway scheme (see Step()'s own check).
    std::vector<float> signedDistance_;
    // mirrorY_[i]/mirrorZ_[i]: the direction with the same (ex,ez)/(ex,ey)
    // but ey/ez negated — i.e. direction i's own reflection across a
    // Y-normal/Z-normal plane. Built once in Init() (see its .cpp) by
    // searching dirs_ for the matching velocity, not hand-written: both
    // lattice tables (LatticeModel.h) are symmetric under a single-axis
    // sign flip by construction, so the search always finds an exact
    // match. Used by Step()'s domain-edge branch for freeSlipY/freeSlipZ
    // — composing mirrorY_[mirrorZ_[i]] gives the reflection across both
    // planes at once (a domain Y-Z edge/corner with both walls free-slip),
    // without needing a third precomputed table.
    std::vector<int> mirrorY_, mirrorZ_;
    std::vector<float> f_[2];
    int current_ = 0;
    std::atomic<bool> hasDiverged_{false};
};
