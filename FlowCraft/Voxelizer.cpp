#include "Voxelizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace {

// How many threads ParallelForZ below will actually spawn for a domain
// `nz` layers deep — pulled out on its own (rather than left buried
// inside ParallelForZ, as it was before the parallel flood fill below
// was added) because the flood fill needs to know this SAME number
// up front, outside of ParallelForZ itself, to size its per-thread
// bookkeeping (localComponentCount/globalBase — see Voxelize's comment)
// and to independently recompute each slab's exact [zBegin, zEnd) for
// the boundary-stitching pass. The two must never disagree, so both
// ParallelForZ and the flood fill call this one function rather than
// each hardcoding the same formula.
//
// This takes a second pass specifically because a naive
// min(hardware_concurrency(), nz) is NOT always the number of threads
// ParallelForZ's loop actually ends up spawning: rangePerThread is
// ceil(nz/candidateThreads), and when nz doesn't divide evenly that
// rounding can make the later slabs' zBegin run past nz entirely, which
// ParallelForZ's loop silently handles by breaking out early (it was
// never relying on the exact thread count for anything before now). A
// caller using the naive candidate count instead of the real one would
// then compute boundary-stitching Z coordinates for slabs that don't
// exist — out-of-bounds indexing, not just a missed optimization.
// Recomputing the thread count FROM that same rangePerThread (one more
// ceiling division) is self-consistent: feeding the result back through
// identical math reproduces the same rangePerThread and thread count
// again, so this and ParallelForZ's own internal computation always
// agree, and the loop inside ParallelForZ never actually needs to break
// early anymore.
unsigned int ChooseThreadCount(int nz) {
    int clampedNz = std::max(nz, 1);
    unsigned int candidateThreads = std::max(1u, std::thread::hardware_concurrency());
    candidateThreads = std::min(candidateThreads, static_cast<unsigned int>(clampedNz));
    int rangePerThread = (clampedNz + static_cast<int>(candidateThreads) - 1) / static_cast<int>(candidateThreads);
    return static_cast<unsigned int>((clampedNz + rangePerThread - 1) / rangePerThread);
}

// Same split-by-Z-slab-across-ChooseThreadCount()-threads pattern as
// LbmReference.cpp/PotentialFlowSolver.cpp's own ParallelForZ (duplicated
// rather than shared across files, same reasoning as theirs: a fresh set
// of threads per call is negligible next to the per-cell/per-triangle
// work each one does). threadIndex lets BuildPreviewMesh (and the flood
// fill below) give each thread its own slot to build/count into with no
// shared-state synchronization.
template <typename Body>
void ParallelForZ(int nz, Body body) {
    unsigned int numThreads = ChooseThreadCount(nz);
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

// The exact [zBegin, zEnd) ParallelForZ(nz, ...) will hand to slab
// `threadIndex` of `numSlabs` — same formula as ParallelForZ's own loop,
// called out separately so the flood fill's boundary-stitching pass (run
// AFTER ParallelForZ has already finished and joined) can recompute
// precisely which two slabs meet at which Z, without ParallelForZ having
// to report anything back itself.
int SlabZBegin(int nz, unsigned int numSlabs, unsigned int threadIndex) {
    int rangePerThread = (nz + static_cast<int>(numSlabs) - 1) / static_cast<int>(numSlabs);
    return static_cast<int>(threadIndex) * rangePerThread;
}
int SlabZEnd(int nz, unsigned int numSlabs, unsigned int threadIndex) {
    int rangePerThread = (nz + static_cast<int>(numSlabs) - 1) / static_cast<int>(numSlabs);
    return std::min(SlabZBegin(nz, numSlabs, threadIndex) + rangePerThread, nz);
}

// Closest point on triangle (a,b,c) to point p. Standard Voronoi-region
// algorithm (Ericson, "Real-Time Collision Detection" 5.1.5) — chosen over
// a full triangle-box SAT test because it's much less error-prone to get
// right, at the cost of being a slightly conservative (over-inclusive)
// solid classification: see Voxelize()'s rasterization comment.
Vec3 ClosestPointOnTriangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    Vec3 ap = p - a;
    float d1 = Dot(ab, ap);
    float d2 = Dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    Vec3 bp = p - b;
    float d3 = Dot(ab, bp);
    float d4 = Dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + ab * v;
    }

    Vec3 cp = p - c;
    float d5 = Dot(ab, cp);
    float d6 = Dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + ac * w;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

int FloorToInt(float v) { return static_cast<int>(std::floor(v)); }
int CeilToInt(float v) { return static_cast<int>(std::ceil(v)); }

}  // namespace

VoxelGrid Voxelizer::Voxelize(const std::vector<Vec3>& positions, const std::vector<uint32_t>& indices,
                               const Params& params, std::atomic<float>* progress) {
    auto report = [&](float f) {
        if (progress) progress->store(f, std::memory_order_relaxed);
    };
    report(0.0f);

    VoxelGrid grid;
    if (positions.empty() || indices.size() < 3) return grid;

    Bounds bounds;
    for (const Vec3& p : positions) bounds.Expand(p);
    Vec3 size = bounds.Size();
    Vec3 pad{size.x * params.paddingFactor.x, size.y * params.paddingFactor.y, size.z * params.paddingFactor.z};
    Vec3 domainMin = bounds.min - pad;
    Vec3 domainMax = bounds.max + pad;
    Vec3 domainSize = domainMax - domainMin;

    float longestAxis = std::max({domainSize.x, domainSize.y, domainSize.z});
    int resolution = std::max(params.targetResolution, 4);
    float cellSize = std::max(longestAxis / static_cast<float>(resolution), 1e-6f);

    grid.cellSize = cellSize;
    grid.origin = domainMin;
    // nx/ny swapped from the world size they're built from — see
    // VoxelGrid::CellCenter's comment: grid-x spans world Y, grid-y
    // spans world X.
    grid.nx = std::max(CeilToInt(domainSize.y / cellSize), 1);
    grid.ny = std::max(CeilToInt(domainSize.x / cellSize), 1);
    grid.nz = std::max(CeilToInt(domainSize.z / cellSize), 1);
    grid.solid.assign(grid.CellCount(), 0);
    // Running *unsigned* nearest-triangle distance, in world units — sign
    // (fluid=+, solid=-) is applied once solid/fluid classification is
    // final, below. Starts at +infinity: a cell no triangle's rasterization
    // loop ever visits (far from the surface) keeps it, which doubles as
    // the "no meaningful SDF here" sentinel LbmReference's interpolated
    // bounce-back checks for via std::isfinite.
    grid.signedDistance.assign(grid.CellCount(), std::numeric_limits<float>::infinity());

    // Rasterize: mark every cell whose center lies within the circumscribed
    // radius of a cube (half the space-diagonal) of the nearest point on
    // any triangle. That radius is a necessary bound for true cell-triangle
    // overlap, so this can't leave gaps a genuinely intersected cell would
    // fall through — it can only mark a thin margin of extra cells beyond
    // the triangle's literal footprint, which is harmless for a boundary
    // shell (and for the flood fill below, which only cares that the shell
    // has no holes). The same pass also tracks each visited cell's own
    // nearest-triangle distance (a running min across every triangle that
    // visits it — not gated on conservativeRadius, since the SDF needs
    // the true nearest distance, not just "is it within the shell"), which
    // is why this no longer skips a cell once grid.solid is already set:
    // a later triangle visiting that same cell can still turn out closer,
    // and skipping it would silently leave the SDF reflecting an earlier,
    // farther triangle instead of the actual nearest one. That's a real
    // cost over the plain solid/fluid classification this used to be —
    // every nearby triangle gets checked against every cell regardless of
    // its current classification, rather than stopping at the first hit.
    //
    // Parallelized by Z-slab (same pattern as LbmReference/
    // PotentialFlowSolver's own Step()): this is almost always the
    // dominant cost of Voxelize, since it's real floating-point geometry
    // work (ClosestPointOnTriangle) per nearby cell, per triangle, rather
    // than a cheap per-cell scan like the flood fill or finalize loop
    // below. Every thread scans every triangle (triMin/triMax/the X/Y
    // extent are cheap to recompute, and worth it to avoid a separate
    // triangle-bucketing pass), but clips the Z range it actually
    // rasterizes into to its own disjoint slab [zBegin, zEnd) — each
    // thread only ever writes grid.solid/signedDistance cells inside its
    // own slab, so there's no data race despite every thread touching the
    // same underlying arrays.
    float conservativeRadius = cellSize * 0.8660254f;  // sqrt(3)/2
    float edgeTieTol = cellSize * 1e-3f;
    size_t triangleCount = indices.size() / 3;
    {
        // Tracks |Dot(center - closest, unitNormal)| for the triangle that
        // currently owns grid.signedDistance[idx]. At a sharp edge or vertex
        // (such as a thin wing trailing edge), adjacent faces share the exact
        // same closest point on that edge and therefore produce equal unsigned
        // distances (up to floating-point rounding), while one face's plane
        // normal points toward the query cell and the other's points away.
        // Breaking ties by the larger |Dot(diff, unitNormal)| selects the
        // exact sign of the angle-bisector pseudonormal (n1 + n2), avoiding
        // arbitrary sign flips behind trailing edges. Scoped to this block so
        // the buffer is freed before `label` is allocated for the flood fill.
        std::vector<float> bestNormalProj(grid.CellCount(), -1.0f);
        ParallelForZ(grid.nz, [&](int zBegin, int zEnd, unsigned int /*threadIndex*/) {
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                if ((i / 3) % 256 == 0) {
                    report(0.5f * static_cast<float>(i / 3) / static_cast<float>(std::max<size_t>(triangleCount, 1)));
                }

                const Vec3& v0 = positions[indices[i]];
                const Vec3& v1 = positions[indices[i + 1]];
                const Vec3& v2 = positions[indices[i + 2]];

                Vec3 rawNormal = Cross(v1 - v0, v2 - v0);
                float normalLen = Length(rawNormal);
                if (normalLen <= 1e-12f) continue;
                Vec3 unitNormal = rawNormal * (1.0f / normalLen);

                Vec3 triMin{std::min({v0.x, v1.x, v2.x}), std::min({v0.y, v1.y, v2.y}), std::min({v0.z, v1.z, v2.z})};
                Vec3 triMax{std::max({v0.x, v1.x, v2.x}), std::max({v0.y, v1.y, v2.y}), std::max({v0.z, v1.z, v2.z})};

                // minX/maxX (grid-x, which is world Y) built from the
                // triangle's world-Y extent, and minY/maxY (grid-y, world X)
                // from its world-X extent — swapped to match CellCenter's
                // mapping (see VoxelGrid.h's comment), so this still bounds
                // the exact same world-space region the triangle actually
                // occupies.
                int minX = std::max(FloorToInt((triMin.y - conservativeRadius - grid.origin.y) / cellSize), 0);
                int minY = std::max(FloorToInt((triMin.x - conservativeRadius - grid.origin.x) / cellSize), 0);
                int minZ = std::max(FloorToInt((triMin.z - conservativeRadius - grid.origin.z) / cellSize), 0);
                int maxX = std::min(CeilToInt((triMax.y + conservativeRadius - grid.origin.y) / cellSize), grid.nx - 1);
                int maxY = std::min(CeilToInt((triMax.x + conservativeRadius - grid.origin.x) / cellSize), grid.ny - 1);
                int maxZ = std::min(CeilToInt((triMax.z + conservativeRadius - grid.origin.z) / cellSize), grid.nz - 1);

                int zLo = std::max(minZ, zBegin);
                int zHi = std::min(maxZ, zEnd - 1);
                if (zLo > zHi) continue;  // this triangle doesn't reach this thread's slab at all

                // Within the conservativeRadius shell, sign comes from the
                // triangle geometry (with shared-edge ties broken by
                // |Dot(diff, unitNormal)|) rather than grid.solid, because
                // grid.solid's conservativeRadius margin is deliberately
                // over-inclusive so the flood fill below gets a watertight
                // 6-connected shell. Cells outside conservativeRadius are
                // reconciled against the flood-fill classification after Step 3.
                for (int z = zLo; z <= zHi; ++z) {
                    for (int y = minY; y <= maxY; ++y) {
                        for (int x = minX; x <= maxX; ++x) {
                            size_t idx = grid.Index(x, y, z);
                            Vec3 center = grid.CellCenter(x, y, z);
                            Vec3 closest = ClosestPointOnTriangle(center, v0, v1, v2);
                            Vec3 diff = center - closest;
                            float dist = Length(diff);
                            float normalDot = Dot(diff, unitNormal);
                            float absNormalDot = std::fabs(normalDot);
                            float signedDist = normalDot < 0.0f ? -dist : dist;
                            float bestDist = std::fabs(grid.signedDistance[idx]);
                            if (dist + edgeTieTol < bestDist) {
                                grid.signedDistance[idx] = signedDist;
                                bestNormalProj[idx] = absNormalDot;
                            } else if (dist <= bestDist + edgeTieTol) {
                                if (absNormalDot > bestNormalProj[idx]) {
                                    float minDist = std::min(dist, bestDist);
                                    grid.signedDistance[idx] = normalDot < 0.0f ? -minDist : minDist;
                                    bestNormalProj[idx] = absNormalDot;
                                } else if (dist < bestDist) {
                                    grid.signedDistance[idx] = grid.signedDistance[idx] < 0.0f ? -dist : dist;
                                }
                            }
                            if (dist <= conservativeRadius) grid.solid[idx] = 1;
                        }
                    }
                }
            }
        });
    }

    // Find every fluid cell reachable from a domain corner (guaranteed
    // fluid: the padding keeps it outside the model) — anything left
    // unreached and not already part of the shell is enclosed by it,
    // i.e. solid interior. A plain single-seed flood fill can't safely
    // be split by Z-slab the way rasterization above was: reachability
    // through the grid is a genuinely global property (a cell's path
    // back to the seed can wind through any other slab), so one thread
    // naively owning one slab doesn't have enough information on its
    // own. Connected-component labelling + union-find across slabs does:
    //
    //  1. Each thread flood-fills ENTIRELY WITHIN its own slab only,
    //     labelling every distinct fluid component it finds there (a
    //     slab can easily contain more than one — e.g. the true exterior
    //     plus an isolated interior pocket). Fully independent, no
    //     cross-thread communication at all.
    //  2. One cheap *sequential* pass unions the labels of any two
    //     touching fluid cells across each of the (numSlabs-1) internal
    //     Z boundaries — this is where two slabs' labellings actually
    //     meet, and it's only nx*ny cells per boundary, not the whole
    //     grid.
    //  3. A final parallel pass marks every cell solid unless its
    //     label's resolved root matches the seed cell's.
    //
    // Each cell's own labelling is thread-local, so distinct slabs can
    // (and do) reuse the same small label values — Step 2 below turns
    // these into one globally dense id range before any unioning happens.
    report(0.5f);
    constexpr uint32_t kNoLabel = UINT32_MAX;
    std::vector<uint32_t> label(grid.CellCount(), kNoLabel);
    unsigned int numSlabs = ChooseThreadCount(grid.nz);
    std::vector<uint32_t> localComponentCount(numSlabs, 0);

    ParallelForZ(grid.nz, [&](int zBegin, int zEnd, unsigned int threadIndex) {
        std::vector<size_t> stack;
        uint32_t nextLocalId = 0;

        auto tryVisit = [&](int x, int y, int z, uint32_t componentId) {
            if (x < 0 || y < 0 || x >= grid.nx || y >= grid.ny || z < zBegin || z >= zEnd) return;
            size_t idx = grid.Index(x, y, z);
            if (grid.solid[idx] || label[idx] != kNoLabel) return;
            label[idx] = componentId;
            stack.push_back(idx);
        };

        for (int z = zBegin; z < zEnd; ++z) {
            // z/nz is each cell's actual global position, not a per-
            // thread counter, so this stays a meaningful progress
            // fraction even though several threads report concurrently —
            // same reasoning as the finalize/BuildPreviewMesh passes.
            report(0.5f + 0.2f * static_cast<float>(z) / static_cast<float>(std::max(grid.nz, 1)));
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    size_t start = grid.Index(x, y, z);
                    if (grid.solid[start] || label[start] != kNoLabel) continue;

                    uint32_t componentId = nextLocalId++;
                    label[start] = componentId;
                    stack.push_back(start);
                    while (!stack.empty()) {
                        size_t cur = stack.back();
                        stack.pop_back();
                        size_t cellsPerLayerLocal = static_cast<size_t>(grid.nx) * grid.ny;
                        int cz = static_cast<int>(cur / cellsPerLayerLocal);
                        int rem = static_cast<int>(cur % cellsPerLayerLocal);
                        int cy = rem / grid.nx;
                        int cx = rem % grid.nx;

                        tryVisit(cx - 1, cy, cz, componentId);
                        tryVisit(cx + 1, cy, cz, componentId);
                        tryVisit(cx, cy - 1, cz, componentId);
                        tryVisit(cx, cy + 1, cz, componentId);
                        tryVisit(cx, cy, cz - 1, componentId);
                        tryVisit(cx, cy, cz + 1, componentId);
                    }
                }
            }
        }
        localComponentCount[threadIndex] = nextLocalId;
    });

    // Step 2a: turn each thread's own locally-dense [0, localCount)
    // component ids into one globally dense [0, totalComponents) range —
    // a per-thread base offset (prefix sum of every earlier thread's own
    // count) added to every cell's label. Still just integer addition
    // into a plain array, no hashing/dictionary lookups anywhere in this
    // whole scheme (deliberately — a hash lookup per cell here would
    // likely cost more than the flood fill it's replacing).
    std::vector<uint32_t> globalBase(numSlabs, 0);
    uint32_t totalComponents = 0;
    for (unsigned int t = 0; t < numSlabs; ++t) {
        globalBase[t] = totalComponents;
        totalComponents += localComponentCount[t];
    }
    size_t cellsPerLayer = static_cast<size_t>(grid.nx) * grid.ny;
    ParallelForZ(grid.nz, [&](int zBegin, int zEnd, unsigned int threadIndex) {
        uint32_t base = globalBase[threadIndex];
        for (int z = zBegin; z < zEnd; ++z) {
            size_t rowStart = static_cast<size_t>(z) * cellsPerLayer;
            for (size_t j = 0; j < cellsPerLayer; ++j) {
                size_t cell = rowStart + j;
                if (!grid.solid[cell]) label[cell] += base;
            }
        }
    });

    // Step 2b: union-find over just the components (typically a tiny
    // number next to cellCount), plain-array-backed for the same reason
    // as above. Path compression is safe here — this whole step is
    // single-threaded.
    std::vector<uint32_t> parent(totalComponents);
    for (uint32_t i = 0; i < totalComponents; ++i) parent[i] = i;
    auto find = [&](uint32_t a) {
        while (parent[a] != a) a = parent[a];
        return a;
    };
    auto unite = [&](uint32_t a, uint32_t b) {
        uint32_t ra = find(a), rb = find(b);
        if (ra != rb) parent[ra] = rb;
    };
    for (unsigned int t = 0; t + 1 < numSlabs; ++t) {
        int zLow = SlabZEnd(grid.nz, numSlabs, t) - 1;
        int zHigh = SlabZBegin(grid.nz, numSlabs, t + 1);
        for (int y = 0; y < grid.ny; ++y) {
            for (int x = 0; x < grid.nx; ++x) {
                size_t cellLow = grid.Index(x, y, zLow);
                size_t cellHigh = grid.Index(x, y, zHigh);
                if (!grid.solid[cellLow] && !grid.solid[cellHigh]) unite(label[cellLow], label[cellHigh]);
            }
        }
    }

    std::vector<uint32_t> resolvedRoot(totalComponents);
    for (uint32_t i = 0; i < totalComponents; ++i) resolvedRoot[i] = find(i);
    // Cell (0,0,0) is guaranteed fluid *in the normal case* (padding
    // keeps the seed corner outside the model) — but with zero padding
    // (or a degenerate/empty grid) it can end up solid like any other
    // cell, same as the old flood fill's tryPush(0,0,0) could already
    // silently no-op on it. `hasSeed` false reproduces that case's
    // effective result exactly: nothing was ever reachable, so every
    // fluid cell ends up enclosed — rather than indexing resolvedRoot
    // with kNoLabel, which (unlike the old reachedExterior flag check)
    // would be a genuine out-of-bounds access.
    uint32_t seedLabel = label[grid.Index(0, 0, 0)];
    bool hasSeed = seedLabel != kNoLabel;
    uint32_t seedRoot = hasSeed ? resolvedRoot[seedLabel] : 0;

    // Step 3: finalize — same shape as the old reachedExterior-driven
    // pass, just testing resolvedRoot/seedRoot (a plain array lookup)
    // instead of a shared reached flag.
    report(0.75f);
    ParallelForZ(grid.nz, [&](int zBegin, int zEnd, unsigned int /*threadIndex*/) {
        for (int z = zBegin; z < zEnd; ++z) {
            report(0.75f + 0.1f * static_cast<float>(z) / static_cast<float>(std::max(grid.nz, 1)));
            size_t base = static_cast<size_t>(z) * cellsPerLayer;
            for (size_t j = 0; j < cellsPerLayer; ++j) {
                size_t i = base + j;
                if (!grid.solid[i] && (!hasSeed || resolvedRoot[label[i]] != seedRoot)) grid.solid[i] = 1;
            }
        }
    });

    // Reconcile signedDistance outside the conservativeRadius boundary shell
    // against the flood-fill classification:
    //  - Every cell with !grid.solid[i] lies outside the watertight
    //    conservativeRadius shell and was reached from the exterior seed, so
    //    it is strictly in the fluid (true signed distance > +conservativeRadius).
    //    Forcing +d corrects any wake/exterior cells that fell inside the 3D
    //    AABB of a skewed triangle whose back-plane faced them.
    //  - Every cell with grid.solid[i] and d > conservativeRadius was marked
    //    solid by the interior flood fill rather than the surface shell, so
    //    it is strictly inside the solid interior (true signed distance <
    //    -conservativeRadius, including flipping untouched +infinity cells to
    //    -infinity).
    // Cells within the conservativeRadius surface shell keep their exact
    // triangle-derived sub-cell signed distance.
    ParallelForZ(grid.nz, [&](int zBegin, int zEnd, unsigned int /*threadIndex*/) {
        size_t base = static_cast<size_t>(zBegin) * cellsPerLayer;
        size_t end = static_cast<size_t>(zEnd) * cellsPerLayer;
        for (size_t i = base; i < end; ++i) {
            float d = std::fabs(grid.signedDistance[i]);
            if (!grid.solid[i]) {
                grid.signedDistance[i] = d;
            } else if (d > conservativeRadius) {
                grid.signedDistance[i] = -d;
            }
        }
    });

    report(0.85f);
    return grid;
}

Mesh Voxelizer::BuildPreviewMesh(const VoxelGrid& grid, std::atomic<float>* progress) {
    auto report = [&](float f) {
        if (progress) progress->store(f, std::memory_order_relaxed);
    };

    Mesh mesh;
    mesh.name = "Voxel Preview";

    struct Face {
        int dx, dy, dz;      // neighbour offset this face faces
        Vec3 normal;
        Vec3 corners[4];     // unit-cube-local corners, CCW as seen from outside
    };
    const Face faces[6] = {
        {+1, 0, 0, {1, 0, 0}, {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}}},
        {-1, 0, 0, {-1, 0, 0}, {{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 0, 0}}},
        {0, +1, 0, {0, 1, 0}, {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}}},
        {0, -1, 0, {0, -1, 0}, {{0, 0, 1}, {0, 0, 0}, {1, 0, 0}, {1, 0, 1}}},
        {0, 0, +1, {0, 0, 1}, {{1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 0, 1}}},
        {0, 0, -1, {0, 0, -1}, {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}},
    };

    // Parallelized by Z-slab, same pattern as Voxelize's own two passes
    // above: each thread builds its own LOCAL mesh (own vertex/index/
    // normal vectors, own bounds) over its disjoint Z range — accumulating
    // straight into a single shared `mesh` isn't safe here the way the
    // grid.solid writes above are, since every push_back needs its own
    // thread-local base vertex index, not a globally-shared one. The
    // partial meshes are concatenated afterward, offsetting each one's
    // indices by how many vertices came before it — the one place this
    // needs to not just be "run it per-slab", unlike the two passes above.
    unsigned int numThreads = ChooseThreadCount(grid.nz);
    std::vector<Mesh> partials(numThreads);

    ParallelForZ(grid.nz, [&](int zBegin, int zEnd, unsigned int threadIndex) {
        Mesh& local = partials[threadIndex];
        for (int z = zBegin; z < zEnd; ++z) {
            // z/nz is each cell's actual global position, not a per-
            // thread counter, so this stays meaningful even with several
            // threads reporting concurrently — see the finalize loop
            // above in Voxelize for the same reasoning.
            report(0.85f + 0.15f * static_cast<float>(z) / static_cast<float>(std::max(grid.nz, 1)));
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    if (!grid.IsSolid(x, y, z)) continue;
                    Vec3 minCorner = grid.CellMinCorner(x, y, z);

                    for (const Face& f : faces) {
                        if (grid.IsSolid(x + f.dx, y + f.dy, z + f.dz)) continue;

                        uint32_t base = static_cast<uint32_t>(local.vertices.size());
                        for (const Vec3& c : f.corners) {
                            // f.corners/f.normal are grid-local (x,y,z) —
                            // swap x/y here too, same as CellMinCorner
                            // above, before treating them as world-space
                            // offsets/directions. Missing this (an
                            // earlier version of this swap did) warps
                            // every cube's face into the wrong shape —
                            // the model voxelized hollow/with gaps, since
                            // adjacent cells' faces no longer actually
                            // met.
                            local.vertices.push_back(minCorner + Vec3(c.y, c.x, c.z) * grid.cellSize);
                            local.normals.push_back(Vec3{f.normal.y, f.normal.x, f.normal.z});
                            local.bounds.Expand(local.vertices.back());
                        }
                        local.indices.push_back(base + 0);
                        local.indices.push_back(base + 1);
                        local.indices.push_back(base + 2);
                        local.indices.push_back(base + 0);
                        local.indices.push_back(base + 2);
                        local.indices.push_back(base + 3);
                    }
                }
            }
        }
    });

    size_t totalVerts = 0, totalIndices = 0;
    for (const Mesh& local : partials) {
        totalVerts += local.vertices.size();
        totalIndices += local.indices.size();
    }
    mesh.vertices.reserve(totalVerts);
    mesh.normals.reserve(totalVerts);
    mesh.indices.reserve(totalIndices);
    for (const Mesh& local : partials) {
        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.insert(mesh.vertices.end(), local.vertices.begin(), local.vertices.end());
        mesh.normals.insert(mesh.normals.end(), local.normals.begin(), local.normals.end());
        for (uint32_t idx : local.indices) mesh.indices.push_back(base + idx);
        mesh.bounds.Expand(local.bounds.min);
        mesh.bounds.Expand(local.bounds.max);
    }

    report(1.0f);
    return mesh;
}
