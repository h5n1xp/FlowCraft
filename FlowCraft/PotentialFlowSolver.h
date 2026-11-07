#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "VoxelGrid.h"

// A deliberately "naive" solver — not a simplified LBM, a different and
// much cheaper model entirely: steady, inviscid, incompressible potential
// flow (∇²φ = 0, velocity u = ∇φ, pressure from Bernoulli). It is not a
// substitute for LbmSolver/LbmReference's actual physics — it has no
// viscosity and cannot produce separation or a low-pressure wake behind a
// bluff body (the classic d'Alembert's-paradox limitation of potential
// flow theory: pressure recovers to ambient at the rear instead of
// staying low), so flow around and ahead of the model is a reasonable
// approximation while flow behind it is not. What it buys for that: one
// scalar field (φ) per cell instead of LBM's 2×(15 or 19), so roughly
// 30-75x less memory per cell — a grid LbmSolver can't even fit in a
// single 4GiB GPU buffer fits here with room to spare, and even
// LbmReference's CPU path (same system-RAM-only ceiling, but much more
// memory-hungry per cell) runs out of practical resolution long before
// this does.
//
// Solved by red-black (checkerboard) Gauss-Seidel SOR, in place — no
// ping-pong buffer needed, unlike LBM's. Within one color, every cell's
// write target is only ever read (as a neighbour) by the *other* color in
// that same pass, since one step in any axis flips (x+y+z)'s parity — so
// a whole color's cells are mutually independent and safe to update
// across threads with no synchronization, the same reasoning that lets
// LbmReference::Step() parallelize by Z-slab (see its comment).
//
// Boundary conditions, in the same lattice-like unit convention (dx=1,
// reference inlet speed = kLatticeVelocityScale) LbmSolver/LbmReference
// use, so results from this solver plug into the exact same readback
// format, UI, and unit conversions (LatticeSpeedToMmPerSecond etc.) those
// two already have:
//  - inlet (x = nx-1 if reverseFlow, else x = 0): Dirichlet, φ pinned to
//    (signed inlet speed) * x every sweep — the domain's one fixed
//    reference, same role LBM's inlet equilibrium plays.
//  - outlet (the opposite face): no fixed condition at all — a boundary
//    cell's "missing" neighbour past the domain edge is substituted with
//    its own value (see NeighborPhi), which is the standard way to get a
//    zero-gradient (open, non-reflecting) condition out of a relaxation
//    solve.
//  - solid cells and the y/z domain edges: the same self-substitution —
//    zero normal derivative, i.e. no flow through a wall or into the
//    model, falling out of NeighborPhi exactly like LbmReference's
//    bounce-back falls out of its own neighbour lookup.
class PotentialFlowSolver {
public:
    struct Params {
        float inletVelocity = 0.1f;  // lattice-unit reference speed — see kLatticeVelocityScale in main.cpp
        bool reverseFlow = false;    // mirrors LbmSolver::Params::reverseFlow exactly, including the default
    };

    // Allocates the φ field and copies the solid mask, then calls Reset().
    void Init(const VoxelGrid& grid, const Params& params);

    // Re-seeds φ to a uniform-flow linear ramp (φ = signed inlet speed *
    // x) everywhere — a reasonable starting guess that's already exact
    // far from the model, so the relaxation only has to correct the
    // region actually near it.
    void Reset(const Params& params);

    // Updates inletVelocity/reverseFlow without resetting φ — safe to
    // call while running. The inlet's Dirichlet value is recomputed from
    // the new params every sweep (see Step()), so the boundary condition
    // takes effect immediately; the rest of the domain relaxes toward
    // the new solution over subsequent sweeps, the same gradual-
    // convergence behavior LbmReference::SetParams has.
    void SetParams(const Params& params) { params_ = params; }

    // Runs `sweeps` full red-black SOR passes (two half-passes each),
    // parallelized by Z-slab across std::thread::hardware_concurrency()
    // threads per half-pass. Much cheaper per sweep than one LBM substep
    // (a 6-neighbour average vs. a full collide+stream over 15-19
    // directions), so affording far more of them per UI tick is normal.
    // `progress`, when non-null, is updated after each half-pass with
    // how many of the `sweeps * 2` half-passes this call has completed
    // so far — unlike LbmReference::Step(), this already loops over all
    // of `sweeps` internally, so there's no separate substepIndex/Count
    // to rescale into (a caller never needs to batch several Step()
    // calls into one reported range the way StepReferenceLocked does).
    void Step(int sweeps, std::atomic<float>* progress = nullptr);

    // (rho, ux, uy, uz) per cell, same 4-floats-per-cell layout and same
    // unit convention LbmSolver::ReadbackField()/LbmReference::ReadbackField()
    // use — ux/uy/uz from central differences of φ (via NeighborPhi, so a
    // cell against a wall/solid/domain edge gets exactly zero normal
    // velocity there), rho encoding a Bernoulli-derived pressure deviation
    // (rho = 1 + 3*pressureDeviation, matching FieldOverlay::SampleFieldValue's
    // (rho-1)/3 reconstruction) rather than anything the solve itself
    // tracks — potential flow has no density field of its own, this is
    // purely for display compatibility with every existing overlay/
    // particle code path. Parallelized the same way Step() is.
    std::vector<float> ReadbackField() const;

    struct Stats {
        float avgDensity = 0.0f, minDensity = 0.0f, maxDensity = 0.0f;
        float avgSpeed = 0.0f, maxSpeed = 0.0f;
        float forceX = 0.0f, forceY = 0.0f, forceZ = 0.0f;
        float maxStress = 0.0f;
        bool hasNaN = false;
    };
    Stats ComputeStats() const;

    int Nx() const { return nx_; }
    int Ny() const { return ny_; }
    int Nz() const { return nz_; }

private:
    size_t CellIndex(int x, int y, int z) const { return (static_cast<size_t>(z) * ny_ + y) * nx_ + x; }
    bool InBounds(int x, int y, int z) const { return x >= 0 && y >= 0 && z >= 0 && x < nx_ && y < ny_ && z < nz_; }
    bool IsSolid(int x, int y, int z) const { return InBounds(x, y, z) && solid_[CellIndex(x, y, z)] != 0; }
    // A fluid cell's own φ in place of a neighbour that's solid or past
    // the domain's y/z edge — the correct zero-flux (no flow through a
    // wall or into the model) ghost value, since the wall sits exactly
    // on the face between the two, making them symmetric about it. Only
    // for y/z neighbours and any in-bounds solid hit — see NeighborPhiX
    // for why the x direction needs different treatment at the domain's
    // x edges specifically.
    float NeighborPhi(int x, int y, int z, size_t selfCell) const {
        if (!InBounds(x, y, z) || IsSolid(x, y, z)) return phi_[selfCell];
        return phi_[CellIndex(x, y, z)];
    }

    // x-direction neighbour lookup. An in-bounds solid hit (the object)
    // is a true wall, so it mirrors just like NeighborPhi. Out-of-bounds
    // in x can only happen at the outlet face — the inlet is Dirichlet-
    // pinned and Step()'s SOR sweep never asks this for a neighbour past
    // it (see its inletX skip) — and mirroring there, the way a real
    // wall needs, would be wrong: it forces the velocity this cell
    // computes to zero (φ(ghost)=φ(self) means no gradient = no flow),
    // which is exactly backwards for an *open* boundary that flow should
    // pass through freely. Linear extrapolation from this cell and its
    // interior neighbour (ghost = 2·φ(self) − φ(interior neighbour one
    // further in) instead gives zero *second* derivative at the
    // boundary — flow continues through at whatever rate it already has
    // instead of being clamped to zero.
    float NeighborPhiX(int x, int y, int z, size_t selfCell) const {
        if (InBounds(x, y, z)) return IsSolid(x, y, z) ? phi_[selfCell] : phi_[CellIndex(x, y, z)];
        int selfX = (x < 0) ? 0 : nx_ - 1;
        int step = (x < 0) ? 1 : -1;
        int interiorX = selfX + step;
        if (!InBounds(interiorX, y, z) || IsSolid(interiorX, y, z)) return phi_[selfCell];  // degenerate fallback
        return 2.0f * phi_[selfCell] - phi_[CellIndex(interiorX, y, z)];
    }

    int nx_ = 0, ny_ = 0, nz_ = 0;
    size_t cellCount_ = 0;
    Params params_;
    std::vector<uint8_t> solid_;
    std::vector<float> phi_;
};
