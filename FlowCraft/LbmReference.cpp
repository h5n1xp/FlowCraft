#include "LbmReference.h"

#include <algorithm>
#include <cmath>
#include <thread>

namespace {

float Equilibrium(const LatticeDirection& dir, float rho, float ux, float uy, float uz) {
    float eu = dir.ex * ux + dir.ey * uy + dir.ez * uz;
    float u2 = ux * ux + uy * uy + uz * uz;
    return dir.weight * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * u2);
}

// Runs `body(zBegin, zEnd, threadIndex)` across
// std::thread::hardware_concurrency() threads, each a contiguous,
// disjoint Z-slab numbered 0..N-1 — the split every parallel method below
// uses. threadIndex lets a caller (ComputeStats) give each thread its own
// slot to accumulate into with no shared-state synchronization, without
// having to re-derive which slot a given zBegin belongs to. A fresh set
// of threads per call (rather than a persistent pool) is deliberate
// simplicity: at any grid size worth using a CPU backend for, thread
// creation is negligible next to the per-cell work each one does.
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
        threads.emplace_back(body, zBegin, zEnd, t);
    }
    for (std::thread& th : threads) th.join();
}

}  // namespace

void LbmReference::MacroscopicFrom(const std::vector<float>& f, size_t cell, float& rho, float& ux, float& uy,
                                    float& uz) const {
    rho = 0.0f;
    ux = 0.0f;
    uy = 0.0f;
    uz = 0.0f;
    for (int i = 0; i < Q_; ++i) {
        float fi = f[PopIndex(i, cell)];
        rho += fi;
        ux += fi * dirs_[i].ex;
        uy += fi * dirs_[i].ey;
        uz += fi * dirs_[i].ez;
    }
    if (rho > 1e-6f) {
        ux /= rho;
        uy /= rho;
        uz /= rho;
    }
}

// Bouzidi/Mei-Luo-Shyy linear interpolated bounce-back. (x,y,z)/`cell` is
// the fluid node; (nxp,nyp,nzp) its direction-`i` neighbor, already known
// solid. `i` points FROM that neighbor TOWARD (x,y,z) (see Step()'s own
// streaming loop — fpre[i] is "what arrives at `cell` in direction i"),
// so `opposite(i)` is the direction pointing from (x,y,z) BACK toward the
// solid neighbor, i.e. toward the wall.
//
// q is the fraction of the link from (x,y,z) to the solid neighbor at
// which the true surface sits, taken from the two ends' signed distances
// (positive in fluid, negative in solid) assuming phi varies linearly
// along the link — the standard way to extract a sub-cell wall position
// from a sampled SDF. The reconstructed population:
//
//   q <  0.5: 2q * f_opp(x,t) + (1-2q) * f_opp(x + e_i, t)
//   q >= 0.5: (1/2q) * f_opp(x,t) + (1 - 1/2q) * f_i(x,t)
//
// (all populations read from fIn, i.e. last step's post-collision values
// — "f^+" in the original papers' notation), where x + e_i is the next
// fluid node *further from* the wall along the same link (e_i, the
// direction i's own velocity, points from the solid neighbor toward
// (x,y,z) and on past it). Both branches reduce to the existing plain
// scheme (fIn[opp][cell]) at exactly q=0.5, so this is a strict
// refinement of it, not a different scheme that happens to agree at one
// point.
//
// Falls back to the plain scheme — mirroring exactly what Step() used to
// do unconditionally for every solid/out-of-bounds neighbor — whenever
// the data this needs isn't usable: no SDF at all (signedDistance_
// empty), either end's value non-finite (the ±infinity "never near any
// triangle" sentinel, or an honestly corrupt value), a degenerate/
// inverted pair (phi(x) - phi(neighbor) not meaningfully positive — it
// must be, since x is fluid (phi>0) and the neighbor is solid (phi<0),
// so this only trips on bad data), or (q<0.5 branch only) the "further"
// node needed for interpolation isn't itself a valid in-bounds fluid
// cell (a thin, sub-cell-scale feature — nothing sensible to interpolate
// from there, so this backs off to the safe, unconditionally-stable
// scheme rather than extrapolating from garbage).
float LbmReference::InterpolatedBounceBack(const std::vector<float>& fIn, int x, int y, int z, size_t cell, int i,
                                              int nxp, int nyp, int nzp) const {
    int opp = dirs_[i].opposite;
    float plain = fIn[PopIndex(opp, cell)];
    if (signedDistance_.empty()) return plain;

    size_t solidCell = CellIndex(nxp, nyp, nzp);
    float phiFluid = signedDistance_[cell];
    float phiSolid = signedDistance_[solidCell];
    if (!std::isfinite(phiFluid) || !std::isfinite(phiSolid)) return plain;
    float denom = phiFluid - phiSolid;
    if (!(denom > 1e-4f)) return plain;
    float q = std::clamp(phiFluid / denom, 0.0f, 1.0f);

    if (q < 0.5f) {
        int ffx = x + dirs_[i].ex, ffy = y + dirs_[i].ey, ffz = z + dirs_[i].ez;
        if (!InBounds(ffx, ffy, ffz) || IsSolid(ffx, ffy, ffz)) return plain;
        size_t ffCell = CellIndex(ffx, ffy, ffz);
        return 2.0f * q * plain + (1.0f - 2.0f * q) * fIn[PopIndex(opp, ffCell)];
    }
    float invTwoQ = 1.0f / (2.0f * q);
    return invTwoQ * plain + (1.0f - invTwoQ) * fIn[PopIndex(i, cell)];
}

// Computes the Wen et al. / Ladd momentum-exchange force and link-weight-
// normalized surface stress on a solid cell (x, y, z) from all adjacent fluid
// neighbors, matching kLbmStepMSL's solid-cell macroOut branch in LbmSolver.cpp.
void LbmReference::SolidCellForce(const std::vector<float>& fIn, int x, int y, int z, size_t /*cell*/,
                                   float& outStress, float& outFx, float& outFy, float& outFz) const {
    float fx = 0.0f, fy = 0.0f, fz = 0.0f, wSum = 0.0f;
    for (int i = 1; i < Q_; ++i) {
        int nxf = x + dirs_[i].ex, nyf = y + dirs_[i].ey, nzf = z + dirs_[i].ez;
        if (!InBounds(nxf, nyf, nzf) || IsSolid(nxf, nyf, nzf)) continue;
        size_t fluidCell = CellIndex(nxf, nyf, nzf);
        int opp = dirs_[i].opposite;
        float plain = fIn[PopIndex(opp, fluidCell)];
        float fBounced = InterpolatedBounceBack(fIn, nxf, nyf, nzf, fluidCell, i, x, y, z);
        float dp = (plain + fBounced) - 2.0f * dirs_[i].weight;
        fx -= static_cast<float>(dirs_[i].ex) * dp;
        fy -= static_cast<float>(dirs_[i].ey) * dp;
        fz -= static_cast<float>(dirs_[i].ez) * dp;
        wSum += dirs_[i].weight;
    }
    float forceMag = std::sqrt(fx * fx + fy * fy + fz * fz);
    outStress = (wSum > 0.0f) ? (forceMag / (6.0f * wSum)) : 0.0f;
    outFx = fx;
    outFy = fy;
    outFz = fz;
}

void LbmReference::Init(const VoxelGrid& grid, LatticeModel model, const Params& params, bool seedAtMidSpeed) {
    nx_ = grid.nx;
    ny_ = grid.ny;
    nz_ = grid.nz;
    cellCount_ = grid.CellCount();
    params_ = params;

    LatticeSet set = GetLatticeSet(model);
    Q_ = set.count;
    dirs_.assign(set.directions, set.directions + set.count);

    solid_ = grid.solid;
    // Only valid when it's the same size as solid_ — an empty or
    // mismatched-size grid.signedDistance (e.g. from a VoxelGrid built
    // before this field existed) means every bounce-back below falls
    // back to the plain halfway scheme instead of indexing into it.
    signedDistance_ = grid.signedDistance.size() == cellCount_ ? grid.signedDistance : std::vector<float>{};

    // See mirrorY_/mirrorZ_'s own comment — found by search, not written
    // by hand, since both lattice tables are symmetric under a single-
    // axis sign flip by construction (an O(Q^2) search, done once here,
    // Q<=19, negligible).
    mirrorY_.assign(static_cast<size_t>(Q_), 0);
    mirrorZ_.assign(static_cast<size_t>(Q_), 0);
    for (int i = 0; i < Q_; ++i) {
        for (int j = 0; j < Q_; ++j) {
            if (dirs_[j].ex == dirs_[i].ex && dirs_[j].ey == -dirs_[i].ey && dirs_[j].ez == dirs_[i].ez) {
                mirrorY_[i] = j;
            }
            if (dirs_[j].ex == dirs_[i].ex && dirs_[j].ey == dirs_[i].ey && dirs_[j].ez == -dirs_[i].ez) {
                mirrorZ_[i] = j;
            }
        }
    }

    f_[0].assign(static_cast<size_t>(Q_) * cellCount_, 0.0f);
    f_[1].assign(static_cast<size_t>(Q_) * cellCount_, 0.0f);
    current_ = 0;

    Reset(seedAtMidSpeed);
}

void LbmReference::Reset(bool seedAtMidSpeed) {
    float rho = 1.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (seedAtMidSpeed) {
        float midSpeed = 0.5f * params_.inletVelocity;
        ux = params_.reverseFlow ? -midSpeed : midSpeed;
        // Bernoulli-consistent pressure for that speed, same convention
        // FieldOverlay/PotentialFlowSolver use — see this function's
        // header comment.
        float pressureDeviation = 0.5f * (params_.inletVelocity * params_.inletVelocity - midSpeed * midSpeed);
        rho = 1.0f + 3.0f * pressureDeviation;
    }
    // Uniform across every cell, so this is embarrassingly parallel — same
    // ParallelForZ split Step() uses below, just applied to the seed too;
    // at high resolution this was a single-threaded pass over every cell
    // with no benefit from it, unlike everything else in this class.
    size_t cellsPerLayer = static_cast<size_t>(nx_) * ny_;
    ParallelForZ(nz_, [&](int zBegin, int zEnd, unsigned int /*threadIndex*/) {
        size_t cellBegin = static_cast<size_t>(zBegin) * cellsPerLayer;
        size_t cellEnd = static_cast<size_t>(zEnd) * cellsPerLayer;
        for (size_t cell = cellBegin; cell < cellEnd; ++cell) {
            for (int i = 0; i < Q_; ++i) f_[0][PopIndex(i, cell)] = Equilibrium(dirs_[i], rho, ux, uy, uz);
        }
    });
    current_ = 0;
}

void LbmReference::ResetFromField(const std::vector<float>& field) {
    if (field.size() != cellCount_ * 4) return;
    // Same parallel-by-Z-slab shape as Reset() above, just reading each
    // cell's own (rho,ux,uy,uz) from `field` (GLOBAL-indexed, same
    // layout ReadbackField() returns) instead of one value closed over
    // from this call's arguments.
    ParallelForZ(nz_, [&](int zBegin, int zEnd, unsigned int /*threadIndex*/) {
        size_t cellsPerLayer = static_cast<size_t>(nx_) * ny_;
        size_t cellBegin = static_cast<size_t>(zBegin) * cellsPerLayer;
        size_t cellEnd = static_cast<size_t>(zEnd) * cellsPerLayer;
        for (size_t cell = cellBegin; cell < cellEnd; ++cell) {
            float rho = solid_[cell] ? 1.0f : field[cell * 4 + 0];
            float ux = solid_[cell] ? 0.0f : field[cell * 4 + 1];
            float uy = solid_[cell] ? 0.0f : field[cell * 4 + 2];
            float uz = solid_[cell] ? 0.0f : field[cell * 4 + 3];
            for (int i = 0; i < Q_; ++i) f_[0][PopIndex(i, cell)] = Equilibrium(dirs_[i], rho, ux, uy, uz);
        }
    });
    current_ = 0;
}

void LbmReference::Step(std::atomic<float>* progress, int substepIndex, int substepCount) {
    const std::vector<float>& fIn = f_[current_];
    std::vector<float>& fOut = f_[1 - current_];
    std::atomic<int> doneZ{0};

    // Flow direction: normally the inlet is at x=0 and flow pushes toward
    // +X; reverseFlow swaps which face is the inlet and negates the inlet
    // velocity, so flow pushes toward -X instead — see
    // LbmSolver::Params::reverseFlow (the two must stay in sync).
    // Everything else (the interior collision/streaming, the outlet's
    // extrapolation-from-the-interior-neighbour logic below) just follows
    // whichever face is which.
    int inletX = params_.reverseFlow ? nx_ - 1 : 0;
    int outletX = params_.reverseFlow ? 0 : nx_ - 1;
    float inletUx = params_.reverseFlow ? -params_.inletVelocity : params_.inletVelocity;

    ParallelForZ(nz_, [&](int zBegin, int zEnd, unsigned int /*threadIndex*/) {
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < ny_; ++y) {
                for (int x = 0; x < nx_; ++x) {
                    size_t cell = CellIndex(x, y, z);

                    if (solid_[cell]) {
                        for (int i = 0; i < Q_; ++i) fOut[PopIndex(i, cell)] = dirs_[i].weight;
                        continue;
                    }

                    if (x == inletX) {
                        float rho = 1.0f, ux = inletUx, uy = 0.0f, uz = 0.0f;
                        for (int i = 0; i < Q_; ++i) fOut[PopIndex(i, cell)] = Equilibrium(dirs_[i], rho, ux, uy, uz);
                        continue;
                    }

                    if (x == outletX) {
                        // Convective (zero-gradient) outflow: both density
                        // *and* velocity are extrapolated from the
                        // interior neighbour, rather than pinning density
                        // to a fixed reference (which instead acts as a
                        // reflecting wall for transient pressure
                        // disturbances: a density wave arriving at the
                        // boundary gets discarded rather than allowed to
                        // leave, and by conservation that discontinuity
                        // generates a wave reflected back upstream —
                        // observed, at low tau, to grow until the apparent
                        // flow direction reverses, then diverges).
                        //
                        // Pure extrapolation (kOutletRelax == 1) fixes
                        // that, but removes the domain's only density
                        // anchor at the far side — observed, over tens of
                        // thousands of steps, to let the outlet slowly
                        // climb from 1.0 toward a new equilibrium (a real,
                        // legitimate transient given this boundary's
                        // physics, not runaway/unstable — it does
                        // converge — just slow enough, and far enough
                        // from equilibrium at the timescales someone would
                        // actually watch, to visibly stretch the overlay's
                        // color range and make everything else look
                        // artificially "blue" by comparison, since that
                        // range always spans the whole field's current
                        // min/max). kOutletRelax blends a small pull back
                        // toward the ambient reference into the
                        // extrapolated value: strong enough, over
                        // thousands of steps, to bound that drift to
                        // something small — but weak enough within any
                        // single step that a genuine transient pressure
                        // wave (which crosses the boundary far faster than
                        // this pull acts) still passes through almost as
                        // freely as pure extrapolation, so the reflection
                        // fix above doesn't regress.
                        //
                        // Copying the whole neighbour *distribution*
                        // (an earlier attempt, before even this) is a
                        // different, stricter condition than extrapolating
                        // just the macroscopic rho/u here, and collapses
                        // into a de facto sealed wall once the gradient
                        // between the two cells flattens out in steady
                        // state — that failure mode doesn't apply to this.
                        constexpr float kOutletRelax = 0.999f;
                        int neighborX = params_.reverseFlow ? x + 1 : x - 1;  // one step back toward the inlet side
                        float nrho, nux, nuy, nuz;
                        MacroscopicFrom(fIn, CellIndex(neighborX, y, z), nrho, nux, nuy, nuz);
                        float rho = nrho > 1e-6f ? 1.0f + kOutletRelax * (nrho - 1.0f) : 1.0f;
                        for (int i = 0; i < Q_; ++i)
                            fOut[PopIndex(i, cell)] = Equilibrium(dirs_[i], rho, nux, nuy, nuz);
                        continue;
                    }

                    float fpre[kMaxQ];
                    {
                        for (int i = 0; i < Q_; ++i) {
                            int nxp = x - dirs_[i].ex, nyp = y - dirs_[i].ey, nzp = z - dirs_[i].ez;
                            bool inBounds = InBounds(nxp, nyp, nzp);
                            if (inBounds && !IsSolid(nxp, nyp, nzp)) {
                                fpre[i] = fIn[PopIndex(i, CellIndex(nxp, nyp, nzp))];
                            } else if (inBounds) {
                                // Solid model geometry — curved/
                                // interpolated bounce-back using the SDF
                                // (falls back to plain bounce-back itself
                                // wherever that data isn't usable — see
                                // its own comment).
                                fpre[i] = InterpolatedBounceBack(fIn, x, y, z, cell, i, nxp, nyp, nzp);
                            } else {
                                // Domain edge (Y/Z bounds), not model
                                // geometry — no SDF to curve against, and
                                // there's no reason to: unlike the model's
                                // surface, this wall sits exactly on a
                                // grid plane by construction, so plain
                                // halfway bounce-back is already exact
                                // here, not an approximation to improve on.
                                //
                                // freeSlipY/freeSlipZ (see Params' own
                                // comment) swap that no-slip reflection for
                                // a free-slip (specular/mirror) one on
                                // whichever of Y/Z this particular
                                // direction actually ran off the edge on —
                                // only the velocity component normal to
                                // that wall gets reflected, not the whole
                                // thing, so tangential flow isn't damped
                                // by a wall that's not supposed to be
                                // there. A direction that ran off BOTH Y
                                // and Z at once (a domain edge-line/corner
                                // cell) only gets the free-slip treatment
                                // when *both* relevant walls are free-slip
                                // — if either one is genuinely no-slip,
                                // this falls back to the ordinary full
                                // reflection rather than inventing a
                                // mixed condition for an edge case with no
                                // obviously-correct answer.
                                bool outY = nyp < 0 || nyp >= ny_;
                                bool outZ = nzp < 0 || nzp >= nz_;
                                bool mirrorOk = (!outY || params_.freeSlipY) && (!outZ || params_.freeSlipZ);
                                if ((outY || outZ) && mirrorOk) {
                                    int dir = i;
                                    if (outY) dir = mirrorY_[dir];
                                    if (outZ) dir = mirrorZ_[dir];
                                    fpre[i] = fIn[PopIndex(dir, cell)];
                                } else {
                                    fpre[i] = fIn[PopIndex(dirs_[i].opposite, cell)];
                                }
                            }
                        }
                    }

                    float rho = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
                    for (int i = 0; i < Q_; ++i) {
                        rho += fpre[i];
                        ux += fpre[i] * dirs_[i].ex;
                        uy += fpre[i] * dirs_[i].ey;
                        uz += fpre[i] * dirs_[i].ez;
                    }
                    if (rho > 1e-6f) {
                        ux /= rho;
                        uy /= rho;
                        uz /= rho;
                    }

                    // Regularized collision (Latt & Chopard): rather than
                    // relaxing fpre directly (plain BGK), project its
                    // non-equilibrium part fpre-feq onto the 2nd-order
                    // Hermite basis — the stress tensor Pi — and
                    // reconstruct a *filtered* non-equilibrium from just
                    // that 6-component tensor. Any higher-order content
                    // fpre picked up from streaming/bounce-back (a
                    // "ghost mode" — numerical noise with no hydrodynamic
                    // meaning at this lattice's order) has no Pi
                    // component and so is discarded outright, instead of
                    // only being damped gradually by ordinary BGK
                    // relaxation over many steps. This is what improves
                    // stability at low tau — see kOutletRelax's comment
                    // above for a case (the outlet) where exactly this
                    // kind of spurious noise, before this existed, could
                    // grow instead of decay. Reduces to plain BGK exactly
                    // when fpre's non-equilibrium part already lies
                    // entirely within that 2nd-order space, which is the
                    // common case away from sharp features — so this is a
                    // filter, not a different physical model.
                    constexpr float kCs2 = 1.0f / 3.0f;  // lattice speed-of-sound squared, dx=dt=1
                    float pixx = 0.0f, piyy = 0.0f, pizz = 0.0f, pixy = 0.0f, pixz = 0.0f, piyz = 0.0f;
                    for (int i = 0; i < Q_; ++i) {
                        float feq = Equilibrium(dirs_[i], rho, ux, uy, uz);
                        float fneq = fpre[i] - feq;
                        fpre[i] = feq;  // fpre's raw value is never needed again — cache feq here for the loop below
                        float ex = dirs_[i].ex, ey = dirs_[i].ey, ez = dirs_[i].ez;
                        pixx += ex * ex * fneq;
                        piyy += ey * ey * fneq;
                        pizz += ez * ez * fneq;
                        pixy += ex * ey * fneq;
                        pixz += ex * ez * fneq;
                        piyz += ey * ez * fneq;
                    }

                    // Smagorinsky LES sub-grid viscosity (Hou et al.) — see the
                    // matching block in LbmSolver.cpp's kernel; keep in sync.
                    const float kSmagorinskyCs = params_.smagorinskyCs;
                    float piMag = std::sqrt(pixx * pixx + piyy * piyy + pizz * pizz +
                                            2.0f * (pixy * pixy + pixz * pixz + piyz * piyz));
                    float tauEff = 0.5f * (params_.tau + std::sqrt(params_.tau * params_.tau +
                                       18.0f * 1.41421356f * kSmagorinskyCs * kSmagorinskyCs * piMag / std::max(rho, 1e-6f)));
                    float omega = 1.0f / tauEff;
                    for (int i = 0; i < Q_; ++i) {
                        float ex = dirs_[i].ex, ey = dirs_[i].ey, ez = dirs_[i].ez;
                        float qxx = ex * ex - kCs2, qyy = ey * ey - kCs2, qzz = ez * ez - kCs2;
                        float qxy = ex * ey, qxz = ex * ez, qyz = ey * ez;
                        float fneqReg = (dirs_[i].weight * 4.5f) * (qxx * pixx + qyy * piyy + qzz * pizz +
                                                                     2.0f * qxy * pixy + 2.0f * qxz * pixz +
                                                                     2.0f * qyz * piyz);
                        fOut[PopIndex(i, cell)] = fpre[i] + (1.0f - omega) * fneqReg;  // fpre[i] holds feq here
                    }
                }
            }
            if (progress) {
                int done = ++doneZ;
                progress->store((static_cast<float>(substepIndex) + static_cast<float>(done) / static_cast<float>(nz_)) /
                                     static_cast<float>(substepCount),
                                 std::memory_order_relaxed);
            }
        }
    });

    current_ = 1 - current_;
}

void LbmReference::GetMacroscopic(int x, int y, int z, float& rho, float& ux, float& uy, float& uz) const {
    MacroscopicFrom(f_[current_], CellIndex(x, y, z), rho, ux, uy, uz);
}

std::vector<float> LbmReference::ReadbackField() const {
    std::vector<float> field(cellCount_ * 4);
    const std::vector<float>& fCur = f_[current_];
    ParallelForZ(nz_, [&](int zBegin, int zEnd, unsigned int /*threadIndex*/) {
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < ny_; ++y) {
                for (int x = 0; x < nx_; ++x) {
                    size_t cell = CellIndex(x, y, z);
                    if (solid_[cell]) {
                        SolidCellForce(fCur, x, y, z, cell, field[cell * 4 + 0], field[cell * 4 + 1],
                                       field[cell * 4 + 2], field[cell * 4 + 3]);
                    } else {
                        float rho, ux, uy, uz;
                        GetMacroscopic(x, y, z, rho, ux, uy, uz);
                        field[cell * 4 + 0] = rho;
                        field[cell * 4 + 1] = ux;
                        field[cell * 4 + 2] = uy;
                        field[cell * 4 + 3] = uz;
                    }
                }
            }
        }
    });
    return field;
}

LbmReference::Stats LbmReference::ComputeStats() const {
    struct Partial {
        float minDensity = 1e30f, maxDensity = -1e30f, maxSpeed = 0.0f, maxStress = 0.0f;
        double densitySum = 0.0, speedSum = 0.0, totalMass = 0.0;
        double forceXSum = 0.0, forceYSum = 0.0, forceZSum = 0.0;
        size_t fluidCount = 0;
        bool hasNaN = false;
    };

    // Sized generously (hardware_concurrency(), an upper bound on how many
    // threads ParallelForZ will actually spawn) — each thread writes only
    // its own slot (given directly as threadIndex), so this needs no lock.
    std::vector<Partial> partials(std::max(1u, std::thread::hardware_concurrency()));
    const std::vector<float>& fCur = f_[current_];

    ParallelForZ(nz_, [&](int zBegin, int zEnd, unsigned int threadIndex) {
        Partial& p = partials[threadIndex];
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < ny_; ++y) {
                for (int x = 0; x < nx_; ++x) {
                    size_t cell = CellIndex(x, y, z);
                    if (solid_[cell]) {
                        float stress, fx, fy, fz;
                        SolidCellForce(fCur, x, y, z, cell, stress, fx, fy, fz);
                        if (std::isfinite(stress) && std::isfinite(fx) && std::isfinite(fy) && std::isfinite(fz)) {
                            p.forceXSum += fx;
                            p.forceYSum += fy;
                            p.forceZSum += fz;
                            p.maxStress = std::max(p.maxStress, stress);
                        }
                        continue;
                    }
                    float rho, ux, uy, uz;
                    GetMacroscopic(x, y, z, rho, ux, uy, uz);
                    if (!std::isfinite(rho) || !std::isfinite(ux) || !std::isfinite(uy) || !std::isfinite(uz)) {
                        p.hasNaN = true;
                        continue;
                    }
                    float speed = std::sqrt(ux * ux + uy * uy + uz * uz);
                    p.minDensity = std::min(p.minDensity, rho);
                    p.maxDensity = std::max(p.maxDensity, rho);
                    p.maxSpeed = std::max(p.maxSpeed, speed);
                    p.densitySum += rho;
                    p.speedSum += speed;
                    p.totalMass += rho;
                    ++p.fluidCount;
                }
            }
        }
    });

    Stats s;
    s.minDensity = 1e30f;
    s.maxDensity = -1e30f;
    size_t fluidCount = 0;
    double densitySum = 0.0, speedSum = 0.0;
    double forceXSum = 0.0, forceYSum = 0.0, forceZSum = 0.0;
    for (const Partial& p : partials) {
        s.hasNaN = s.hasNaN || p.hasNaN;
        s.maxStress = std::max(s.maxStress, p.maxStress);
        forceXSum += p.forceXSum;
        forceYSum += p.forceYSum;
        forceZSum += p.forceZSum;
        if (p.fluidCount == 0) continue;
        s.minDensity = std::min(s.minDensity, p.minDensity);
        s.maxDensity = std::max(s.maxDensity, p.maxDensity);
        s.maxSpeed = std::max(s.maxSpeed, p.maxSpeed);
        densitySum += p.densitySum;
        speedSum += p.speedSum;
        s.totalMass += p.totalMass;
        fluidCount += p.fluidCount;
    }
    s.forceX = static_cast<float>(forceXSum);
    s.forceY = static_cast<float>(forceYSum);
    s.forceZ = static_cast<float>(forceZSum);
    if (fluidCount > 0) {
        s.avgDensity = static_cast<float>(densitySum / static_cast<double>(fluidCount));
        s.avgSpeed = static_cast<float>(speedSum / static_cast<double>(fluidCount));
    } else {
        s.minDensity = 0.0f;
        s.maxDensity = 0.0f;
    }
    return s;
}
