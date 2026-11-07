#include "PotentialFlowSolver.h"

#include <algorithm>
#include <cmath>
#include <thread>

namespace {

// Same split/parallelize-by-Z-slab pattern as LbmReference.cpp's own
// ParallelForZ (duplicated rather than shared across the two files, same
// reasoning as that one's comment: a fresh set of threads per call is
// negligible next to the per-cell work each does).
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

// Over-relaxation factor for the red-black SOR sweep: ω=1 is plain
// Gauss-Seidel (slow but unconditionally stable for this kind of
// discrete Laplace system); 1.5 converges noticeably faster while
// staying well clear of the instability that can show up as ω approaches
// 2 — "naive" doesn't need to mean "untuned", and a diverging solver
// would be a worse first impression than a slightly slower one.
constexpr float kOmega = 1.5f;

}  // namespace

void PotentialFlowSolver::Init(const VoxelGrid& grid, const Params& params) {
    nx_ = grid.nx;
    ny_ = grid.ny;
    nz_ = grid.nz;
    cellCount_ = grid.CellCount();
    solid_ = grid.solid;
    phi_.assign(cellCount_, 0.0f);
    Reset(params);
}

void PotentialFlowSolver::Reset(const Params& params) {
    params_ = params;
    float signedU = params_.reverseFlow ? -params_.inletVelocity : params_.inletVelocity;
    // Every cell's value depends only on its own x, no cross-cell
    // dependency at all — same ParallelForZ split Step()'s SOR sweep
    // uses below, just applied to the seed too (previously single-
    // threaded, unlike everything else in this class).
    ParallelForZ(nz_, [&](int zBegin, int zEnd) {
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < ny_; ++y) {
                for (int x = 0; x < nx_; ++x) {
                    phi_[CellIndex(x, y, z)] = signedU * static_cast<float>(x);
                }
            }
        }
    });
}

void PotentialFlowSolver::Step(int sweeps, std::atomic<float>* progress) {
    if (cellCount_ == 0) return;
    float signedU = params_.reverseFlow ? -params_.inletVelocity : params_.inletVelocity;
    int inletX = params_.reverseFlow ? nx_ - 1 : 0;
    int halfPassesDone = 0;
    int totalHalfPasses = std::max(sweeps, 0) * 2;

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        // Pin the inlet plane's Dirichlet value fresh every sweep — cheap
        // (O(ny*nz), a single face) and means a SetParams() change takes
        // effect on the very next Step() call, not after however long it
        // takes the relaxation to overwrite a stale seed.
        for (int z = 0; z < nz_; ++z) {
            for (int y = 0; y < ny_; ++y) {
                phi_[CellIndex(inletX, y, z)] = signedU * static_cast<float>(inletX);
            }
        }

        for (int color = 0; color < 2; ++color) {
            ParallelForZ(nz_, [&](int zBegin, int zEnd) {
                for (int z = zBegin; z < zEnd; ++z) {
                    for (int y = 0; y < ny_; ++y) {
                        for (int x = 0; x < nx_; ++x) {
                            if (x == inletX) continue;  // Dirichlet-fixed above, never relaxed
                            if ((x + y + z) % 2 != color) continue;
                            size_t cell = CellIndex(x, y, z);
                            if (solid_[cell]) continue;

                            float sum = NeighborPhiX(x + 1, y, z, cell) + NeighborPhiX(x - 1, y, z, cell) +
                                        NeighborPhi(x, y + 1, z, cell) + NeighborPhi(x, y - 1, z, cell) +
                                        NeighborPhi(x, y, z + 1, cell) + NeighborPhi(x, y, z - 1, cell);
                            float gaussSeidel = sum / 6.0f;
                            phi_[cell] += kOmega * (gaussSeidel - phi_[cell]);
                        }
                    }
                }
            });
            if (progress) {
                ++halfPassesDone;
                progress->store(static_cast<float>(halfPassesDone) / static_cast<float>(totalHalfPasses),
                                 std::memory_order_relaxed);
            }
        }
    }
}

std::vector<float> PotentialFlowSolver::ReadbackField() const {
    std::vector<float> out(cellCount_ * 4, 0.0f);
    if (cellCount_ == 0) return out;
    float uInfSq = params_.inletVelocity * params_.inletVelocity;

    ParallelForZ(nz_, [&](int zBegin, int zEnd) {
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < ny_; ++y) {
                for (int x = 0; x < nx_; ++x) {
                    size_t cell = CellIndex(x, y, z);
                    if (solid_[cell]) continue;
                    float ux = (NeighborPhiX(x + 1, y, z, cell) - NeighborPhiX(x - 1, y, z, cell)) * 0.5f;
                    float uy = (NeighborPhi(x, y + 1, z, cell) - NeighborPhi(x, y - 1, z, cell)) * 0.5f;
                    float uz = (NeighborPhi(x, y, z + 1, cell) - NeighborPhi(x, y, z - 1, cell)) * 0.5f;
                    float speedSq = ux * ux + uy * uy + uz * uz;
                    // Bernoulli: p + 0.5*speed^2 = const along a streamline
                    // (reference density 1, matching LBM's rho≈1 ambient
                    // convention) — faster flow reads as lower pressure,
                    // exactly like the real (if wake-free) potential-flow
                    // result around a cylinder/aerofoil.
                    float pressureDeviation = 0.5f * (uInfSq - speedSq);
                    out[cell * 4 + 0] = 1.0f + 3.0f * pressureDeviation;
                    out[cell * 4 + 1] = ux;
                    out[cell * 4 + 2] = uy;
                    out[cell * 4 + 3] = uz;
                }
            }
        }
    });

    // Populate solid boundary cells with normal Bernoulli pressure force
    // (same (stress, fx, fy, fz) layout LbmSolver/LbmReference write for
    // solid cells) so the Airframe Stress Map and force readout work on
    // the Naive backend too.
    constexpr int kFaceDirs[6][3] = {{+1, 0, 0}, {-1, 0, 0}, {0, +1, 0}, {0, -1, 0}, {0, 0, +1}, {0, 0, -1}};
    ParallelForZ(nz_, [&](int zBegin, int zEnd) {
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < ny_; ++y) {
                for (int x = 0; x < nx_; ++x) {
                    size_t cell = CellIndex(x, y, z);
                    if (!solid_[cell]) continue;
                    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
                    int exposedFaces = 0;
                    for (const auto& d : kFaceDirs) {
                        int nxf = x + d[0], nyf = y + d[1], nzf = z + d[2];
                        if (!InBounds(nxf, nyf, nzf) || IsSolid(nxf, nyf, nzf)) continue;
                        size_t fcell = CellIndex(nxf, nyf, nzf);
                        float pDev = (out[fcell * 4 + 0] - 1.0f) * (1.0f / 3.0f);
                        fx -= static_cast<float>(d[0]) * pDev;
                        fy -= static_cast<float>(d[1]) * pDev;
                        fz -= static_cast<float>(d[2]) * pDev;
                        ++exposedFaces;
                    }
                    float forceMag = std::sqrt(fx * fx + fy * fy + fz * fz);
                    out[cell * 4 + 0] = exposedFaces > 0 ? (forceMag / static_cast<float>(exposedFaces)) : 0.0f;
                    out[cell * 4 + 1] = fx;
                    out[cell * 4 + 2] = fy;
                    out[cell * 4 + 3] = fz;
                }
            }
        }
    });
    return out;
}

PotentialFlowSolver::Stats PotentialFlowSolver::ComputeStats() const {
    Stats stats;
    if (cellCount_ == 0) return stats;
    float uInfSq = params_.inletVelocity * params_.inletVelocity;

    double densitySum = 0.0, speedSum = 0.0;
    float minDensity = 1e30f, maxDensity = -1e30f, maxSpeed = 0.0f;
    size_t fluidCount = 0;
    for (int z = 0; z < nz_; ++z) {
        for (int y = 0; y < ny_; ++y) {
            for (int x = 0; x < nx_; ++x) {
                size_t cell = CellIndex(x, y, z);
                if (solid_[cell]) continue;
                float ux = (NeighborPhiX(x + 1, y, z, cell) - NeighborPhiX(x - 1, y, z, cell)) * 0.5f;
                float uy = (NeighborPhi(x, y + 1, z, cell) - NeighborPhi(x, y - 1, z, cell)) * 0.5f;
                float uz = (NeighborPhi(x, y, z + 1, cell) - NeighborPhi(x, y, z - 1, cell)) * 0.5f;
                float speed = std::sqrt(ux * ux + uy * uy + uz * uz);
                float density = 1.0f + 3.0f * (0.5f * (uInfSq - speed * speed));
                if (!std::isfinite(density) || !std::isfinite(speed)) {
                    stats.hasNaN = true;
                    continue;
                }
                densitySum += density;
                speedSum += speed;
                minDensity = std::min(minDensity, density);
                maxDensity = std::max(maxDensity, density);
                maxSpeed = std::max(maxSpeed, speed);
                ++fluidCount;
            }
        }
    }
    if (fluidCount > 0) {
        stats.avgDensity = static_cast<float>(densitySum / static_cast<double>(fluidCount));
        stats.minDensity = minDensity;
        stats.maxDensity = maxDensity;
        stats.avgSpeed = static_cast<float>(speedSum / static_cast<double>(fluidCount));
        stats.maxSpeed = maxSpeed;
    }
    return stats;
}
