#include "FieldOverlay.h"
#include "MarchingCubesTables.h"

#include <algorithm>
#include <cmath>
#include <thread>

Vec3 FieldOverlay::Colormap(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    struct Stop {
        float t;
        Vec3 color;
    };
    static const Stop kStops[5] = {
        {0.00f, {0.10f, 0.10f, 0.85f}},  // blue
        {0.25f, {0.10f, 0.80f, 0.85f}},  // cyan
        {0.50f, {0.15f, 0.85f, 0.15f}},  // green
        {0.75f, {0.95f, 0.90f, 0.10f}},  // yellow
        {1.00f, {0.90f, 0.10f, 0.10f}},  // red
    };
    for (int i = 0; i < 4; ++i) {
        if (t <= kStops[i + 1].t) {
            float localT = (t - kStops[i].t) / (kStops[i + 1].t - kStops[i].t);
            return kStops[i].color + (kStops[i + 1].color - kStops[i].color) * localT;
        }
    }
    return kStops[4].color;
}

void FieldOverlay::AppendMesh(OverlayMesh& dst, const OverlayMesh& src) {
    uint32_t base = static_cast<uint32_t>(dst.vertices.size());
    dst.vertices.insert(dst.vertices.end(), src.vertices.begin(), src.vertices.end());
    dst.indices.reserve(dst.indices.size() + src.indices.size());
    for (uint32_t idx : src.indices) dst.indices.push_back(idx + base);
}

float FieldOverlay::SampleFieldValue(const std::vector<float>& field, size_t cell, FieldColorMode mode) {
    float rho = field[cell * 4 + 0];
    float ux = field[cell * 4 + 1];
    float uy = field[cell * 4 + 2];
    float uz = field[cell * 4 + 3];
    if (mode == FieldColorMode::Speed) return std::sqrt(ux * ux + uy * uy + uz * uz);
    constexpr float kCs2 = 1.0f / 3.0f;  // lattice speed-of-sound squared
    return kCs2 * (rho - 1.0f);          // pressure deviation from ambient
}

namespace {

// Same split-by-Z-slab-per-thread pattern as LbmReference.cpp's and
// PotentialFlowSolver.cpp's own copies of this (duplicated locally again
// rather than shared across files, same reasoning as those: a fresh set
// of threads per call is negligible next to the per-cell work each one
// does). `body(zBegin, zEnd, threadIndex)` — threadIndex lets a caller
// (ComputeRange, ComputeXSliceMeans) give each thread its own slot to
// accumulate into with no shared-state synchronization, the same way
// LbmReference::ComputeStats() does.
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

// How many threads ParallelForZ(nz, ...) will actually spawn — callers
// that keep a per-thread accumulation slot (indexed by threadIndex) need
// to size that storage to this, not to hardware_concurrency() directly,
// since a small nz caps the thread count below that.
unsigned int ParallelForZThreadCount(int nz) {
    return std::min(std::max(1u, std::thread::hardware_concurrency()), static_cast<unsigned int>(std::max(nz, 1)));
}

// `xMeans`, when given, subtracts that X-slice's mean from Pressure (see
// FieldOverlay::ComputeXSliceMeans) — a no-op for Speed, or when null.
float FieldValueMaybeDetrended(const std::vector<float>& field, size_t cell, int x, FieldColorMode mode,
                                const std::vector<float>* xMeans) {
    float v = FieldOverlay::SampleFieldValue(field, cell, mode);
    if (xMeans && mode == FieldColorMode::Pressure) v -= (*xMeans)[x];
    return v;
}

// A soft circular sprite facing the camera (camRight/camUp span its
// plane) — a fan of kSegments triangles meeting at a center vertex, which
// carries the real (caller-computed, pressure-window-driven) alpha, with
// every rim vertex pinned to alpha 0. Ordinary per-triangle vertex-alpha
// interpolation then gives a smooth radial falloff from full strength at
// the center to fully transparent at the rim — a "fuzzy ball" rather than
// a hard-edged disc — with no shader changes needed (OverlayRenderer's
// transparent pipeline already just uses each vertex's own alpha
// directly). Always reads as a soft circle facing the viewer regardless
// of camera angle, the same billboarding SmokeTrailSystem::BuildMesh
// already uses for trail segments — standard technique for this look.
void AddFuzzyBillboard(OverlayMesh& mesh, const Vec3& center, float radius, const Vec3& camRight, const Vec3& camUp,
                        const Vec3& color, float alpha) {
    constexpr int kSegments = 12;
    constexpr float kTwoPi = 6.28318530718f;
    uint32_t centerIdx = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({center, color, alpha});
    for (int i = 0; i < kSegments; ++i) {
        float theta = kTwoPi * static_cast<float>(i) / static_cast<float>(kSegments);
        Vec3 rim = center + camRight * (std::cos(theta) * radius) + camUp * (std::sin(theta) * radius);
        mesh.vertices.push_back({rim, color, 0.0f});
    }
    uint32_t firstRim = centerIdx + 1;
    for (int i = 0; i < kSegments; ++i) {
        mesh.indices.push_back(centerIdx);
        mesh.indices.push_back(firstRim + static_cast<uint32_t>(i));
        mesh.indices.push_back(firstRim + static_cast<uint32_t>((i + 1) % kSegments));
    }
}

}  // namespace

// Parallelized by Z-slab (see ParallelForZ) — this is called up to three
// times by every single auto-update/Build Overlay (raw Speed, raw
// Pressure, detrended Pressure), unconditionally regardless of overlay
// style, so at high resolution it was a real, measured contributor to
// the "periodic pause" the GPU-readback chunking fix alone didn't touch
// (that fix only addressed the GPU download; this was pure single-
// threaded CPU work afterward, on the same thread, paying for itself
// again on every update instead of using the machine's other cores).
void FieldOverlay::ComputeRange(const VoxelGrid& grid, const std::vector<float>& field, FieldColorMode mode,
                                 float& outMin, float& outMax, bool& outHasNaN, const std::vector<float>* xMeans) {
    outHasNaN = false;
    if (field.size() != grid.CellCount() * 4) {
        outMin = outMax = 0.0f;
        return;
    }

    unsigned int numThreads = ParallelForZThreadCount(grid.nz);
    std::vector<float> partialLo(numThreads, 1e30f), partialHi(numThreads, -1e30f);
    std::vector<uint8_t> partialHasNaN(numThreads, 0);

    ParallelForZ(grid.nz, [&](int zBegin, int zEnd, unsigned int threadIndex) {
        float lo = 1e30f, hi = -1e30f;
        bool hasNaN = false;
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    size_t cell = grid.Index(x, y, z);
                    if (grid.solid[cell]) continue;
                    float v = FieldValueMaybeDetrended(field, cell, x, mode, xMeans);
                    if (!std::isfinite(v)) {
                        hasNaN = true;
                        continue;
                    }
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
            }
        }
        partialLo[threadIndex] = lo;
        partialHi[threadIndex] = hi;
        partialHasNaN[threadIndex] = hasNaN ? 1 : 0;
    });

    float lo = 1e30f, hi = -1e30f;
    for (unsigned int t = 0; t < numThreads; ++t) {
        lo = std::min(lo, partialLo[t]);
        hi = std::max(hi, partialHi[t]);
        if (partialHasNaN[t]) outHasNaN = true;
    }
    if (!(hi > lo)) hi = lo + 1e-6f;
    outMin = lo;
    outMax = hi;
}

// Also parallelized, same reasoning as ComputeRange — each thread
// accumulates into its own nx-sized partial sums/counts (indexed by
// threadIndex) rather than sharing one, since every Z-slab spans the
// full X range and would otherwise race on the same sums[x]/counts[x].
std::vector<float> FieldOverlay::ComputeXSliceMeans(const VoxelGrid& grid, const std::vector<float>& field,
                                                      FieldColorMode mode) {
    std::vector<float> means(std::max(grid.nx, 0), 0.0f);
    if (field.size() != grid.CellCount() * 4 || grid.nx <= 0) return means;

    unsigned int numThreads = ParallelForZThreadCount(grid.nz);
    std::vector<std::vector<double>> partialSums(numThreads, std::vector<double>(grid.nx, 0.0));
    std::vector<std::vector<size_t>> partialCounts(numThreads, std::vector<size_t>(grid.nx, 0));

    ParallelForZ(grid.nz, [&](int zBegin, int zEnd, unsigned int threadIndex) {
        std::vector<double>& sums = partialSums[threadIndex];
        std::vector<size_t>& counts = partialCounts[threadIndex];
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    size_t cell = grid.Index(x, y, z);
                    if (grid.solid[cell]) continue;
                    float v = FieldOverlay::SampleFieldValue(field, cell, mode);
                    if (!std::isfinite(v)) continue;
                    sums[x] += v;
                    counts[x] += 1;
                }
            }
        }
    });

    std::vector<double> sums(grid.nx, 0.0);
    std::vector<size_t> counts(grid.nx, 0);
    for (unsigned int t = 0; t < numThreads; ++t) {
        for (int x = 0; x < grid.nx; ++x) {
            sums[x] += partialSums[t][x];
            counts[x] += partialCounts[t][x];
        }
    }
    for (int x = 0; x < grid.nx; ++x) {
        means[x] = counts[x] > 0 ? static_cast<float>(sums[x] / counts[x]) : 0.0f;
    }
    return means;
}

// Every cell in the slice plane gets its own vertex (at its cell center,
// not skipped even when solid), and adjacent cells share theirs — unlike
// the independent flat-colored quad-per-fluid-cell mesh this replaced.
// That's what lets a solid-adjacent-to-fluid pair of vertices, one at
// alpha 0 and one at the slice's normal opacity, render as a smooth
// gradient across the triangle between them (ordinary GPU vertex-alpha
// interpolation) instead of a hard per-cell cutoff. Alpha is a smoothstep
// of the SDF (VoxelGrid::signedDistance) over one cellSize centered on
// the true surface, so that gradient's midpoint sits at the IBM-corrected
// boundary's actual sub-cell position rather than at the voxel face —
// the softened edge genuinely tracks the smoothed physics now, not just
// a blur for its own sake. Falls back to the old hard solid/fluid alpha
// cutoff wherever there's no SDF to use (grid.signedDistance empty, e.g.
// a VoxelGrid built before it existed).
OverlayMesh FieldOverlay::BuildSliceMeshFromRange(const VoxelGrid& grid, const std::vector<float>& field,
                                                   SliceAxis axis, int index, FieldColorMode mode, float lo,
                                                   float hi, const std::vector<float>* xMeans) {
    OverlayMesh mesh;
    if (field.size() != grid.CellCount() * 4) return mesh;

    float invRange = 1.0f / (hi - lo);
    constexpr Vec3 kNaNColor{1.0f, 0.1f, 1.0f};  // flat magenta — never produced by Colormap, so unmistakable
    // A flat-opaque slice reads as a harsh, featureless wall, especially
    // composited on top of the unvoxelized model (overlayShowModel) —
    // just enough transparency to soften that without meaningfully
    // hurting legibility of the slice's own colors. Rendered through the
    // same alpha-blended pipeline Voxel Cloud/Particles use (see
    // main.cpp's render dispatch) rather than the opaque one Slice used
    // to use, since that's what actually honors per-vertex alpha — the
    // opaque pipeline's fragment shader ignores it and always outputs 1.0.
    constexpr float kSliceAlpha = 0.85f;
    bool hasSdf = grid.signedDistance.size() == grid.CellCount();

    // SliceAxis::X/Y name GRID axes, and grid-x/grid-y are world Y/X
    // respectively (see VoxelGrid::CellCenter's comment) — iCount/jCount
    // below are always grid-index counts, never assumed to be nx/ny in
    // that order, so this stays correct for either case. SliceAxis::Z is
    // unaffected (grid-z is still world Z).
    int iCount, jCount, fixedIndex;
    switch (axis) {
        case SliceAxis::Z:
            iCount = grid.nx;
            jCount = grid.ny;
            fixedIndex = std::clamp(index, 0, grid.nz - 1);
            break;
        case SliceAxis::Y:
            iCount = grid.nx;
            jCount = grid.nz;
            fixedIndex = std::clamp(index, 0, grid.ny - 1);
            break;
        case SliceAxis::X:
        default:
            iCount = grid.ny;
            jCount = grid.nz;
            fixedIndex = std::clamp(index, 0, grid.nx - 1);
            break;
    }
    if (iCount <= 0 || jCount <= 0) return mesh;

    auto cellCoord = [&](int i, int j, int& x, int& y, int& z) {
        switch (axis) {
            case SliceAxis::Z:
                x = i;
                y = j;
                z = fixedIndex;
                break;
            case SliceAxis::Y:
                x = i;
                y = fixedIndex;
                z = j;
                break;
            case SliceAxis::X:
            default:
                x = fixedIndex;
                y = i;
                z = j;
                break;
        }
    };

    mesh.vertices.resize(static_cast<size_t>(iCount) * static_cast<size_t>(jCount));
    for (int j = 0; j < jCount; ++j) {
        for (int i = 0; i < iCount; ++i) {
            int x, y, z;
            cellCoord(i, j, x, y, z);
            size_t cell = grid.Index(x, y, z);
            // Solid cells store their surface momentum-exchange force in
            // field[cell*4 + 0..3], not fluid (rho, u) — clamp them to 0
            // here so the SDF-smoothstepped boundary transition on a
            // slice plane blends toward zero speed / ambient pressure at
            // the wall rather than reading the solid cell's force vector.
            float v = grid.solid[cell] ? 0.0f : FieldValueMaybeDetrended(field, cell, x, mode, xMeans);
            Vec3 color = std::isfinite(v) ? Colormap((v - lo) * invRange) : kNaNColor;

            float alpha;
            if (hasSdf) {
                float sdf = grid.signedDistance[cell];
                if (!grid.solid[cell]) sdf = std::fabs(sdf);
                if (std::isfinite(sdf)) {
                    // t=0 at half a cell inside solid, t=1 at half a cell
                    // into fluid, smoothstepped in between — the ramp's
                    // own midpoint (t=0.5) is exactly the true surface.
                    float t = std::clamp(sdf / grid.cellSize + 0.5f, 0.0f, 1.0f);
                    t = t * t * (3.0f - 2.0f * t);
                    alpha = kSliceAlpha * t;
                } else {
                    alpha = sdf > 0.0f ? kSliceAlpha : 0.0f;  // sentinel: deep fluid or deep solid
                }
            } else {
                alpha = grid.solid[cell] ? 0.0f : kSliceAlpha;
            }

            mesh.vertices[static_cast<size_t>(j) * static_cast<size_t>(iCount) + static_cast<size_t>(i)] = {
                grid.CellCenter(x, y, z), color, alpha};
        }
    }

    mesh.indices.reserve(static_cast<size_t>(iCount - 1) * static_cast<size_t>(jCount - 1) * 6);
    for (int j = 0; j + 1 < jCount; ++j) {
        for (int i = 0; i + 1 < iCount; ++i) {
            uint32_t v00 = static_cast<uint32_t>(j * iCount + i);
            uint32_t v10 = static_cast<uint32_t>(j * iCount + i + 1);
            uint32_t v11 = static_cast<uint32_t>((j + 1) * iCount + i + 1);
            uint32_t v01 = static_cast<uint32_t>((j + 1) * iCount + i);
            mesh.indices.push_back(v00);
            mesh.indices.push_back(v10);
            mesh.indices.push_back(v11);
            mesh.indices.push_back(v00);
            mesh.indices.push_back(v11);
            mesh.indices.push_back(v01);
        }
    }

    return mesh;
}

int FieldOverlay::ChooseVoxelCloudStride(const VoxelGrid& grid, size_t targetVoxelCount) {
    size_t cellCount = grid.CellCount();
    if (cellCount <= targetVoxelCount) return 1;
    double ratio = static_cast<double>(cellCount) / static_cast<double>(std::max<size_t>(targetVoxelCount, 1));
    return std::max(1, static_cast<int>(std::ceil(std::cbrt(ratio))));
}

OverlayMesh FieldOverlay::BuildVoxelCloudMesh(const VoxelGrid& grid, const std::vector<float>& field, float speedLo,
                                               float speedHi, float pressureAlphaLo, float pressureAlphaHi,
                                               int stride, const Vec3& camRight, const Vec3& camUp,
                                               const std::vector<float>* pressureAlphaXMeans,
                                               bool highlightNegativePressure) {
    OverlayMesh mesh;
    if (field.size() != grid.CellCount() * 4 || stride < 1) return mesh;

    float invSpeedRange = 1.0f / (speedHi - speedLo);
    float invAlphaRange = 1.0f / (pressureAlphaHi - pressureAlphaLo);
    constexpr Vec3 kNaNColor{1.0f, 0.1f, 1.0f};  // same convention as the slice: flat magenta, unmistakable
    // The alpha ceiling: at or above pressureAlphaHi a cube still reads as
    // translucent, not solid — this is a "feel for the shape" view, not
    // meant to look opaque anywhere. The floor is 0 outright (not some
    // small-but-nonzero kAlphaMin, like an earlier version of this had) —
    // the whole point of pressureAlphaLo being caller-chosen is to let
    // "ambient, undisturbed fluid" hit exactly invisible rather than a
    // permanent faint haze. NaN cells are a fixed, higher alpha instead
    // of going through this mapping at all — divergence should always be
    // obviously visible, never faded out just because the (meaningless)
    // value it computes to happens to fall below the alpha floor.
    constexpr float kAlphaMax = 0.15f;
    constexpr float kNaNAlpha = 0.4f;
    // See highlightNegativePressure's header comment: a below-ambient
    // cell can otherwise fade to fully invisible under the normal alpha
    // window, hiding exactly the suction/low-pressure signal someone
    // turned this on to see.
    constexpr Vec3 kNegativePressureColor{0.25f, 0.25f, 0.25f};
    constexpr float kNegativePressureAlpha = 0.3f;

    for (int bz = 0; bz < grid.nz; bz += stride) {
        int sizeZ = std::min(stride, grid.nz - bz);
        int cz = std::min(bz + stride / 2, grid.nz - 1);
        for (int by = 0; by < grid.ny; by += stride) {
            int sizeY = std::min(stride, grid.ny - by);
            int cy = std::min(by + stride / 2, grid.ny - 1);
            for (int bx = 0; bx < grid.nx; bx += stride) {
                int sizeX = std::min(stride, grid.nx - bx);
                int cx = std::min(bx + stride / 2, grid.nx - 1);

                size_t sampleCell = grid.Index(cx, cy, cz);
                if (grid.solid[sampleCell]) continue;

                float speed = FieldOverlay::SampleFieldValue(field, sampleCell, FieldColorMode::Speed);
                float pressure = FieldValueMaybeDetrended(field, sampleCell, cx, FieldColorMode::Pressure,
                                                           pressureAlphaXMeans);
                bool finite = std::isfinite(speed) && std::isfinite(pressure);

                Vec3 color;
                float alpha;
                float sizeScale = 1.0f;
                if (!finite) {
                    color = kNaNColor;
                    alpha = kNaNAlpha;
                    sizeScale = 1.0f;
                } else if (highlightNegativePressure && pressure < 0.0f) {
                    color = kNegativePressureColor;
                    alpha = kNegativePressureAlpha;
                    float suctionRef = std::max(std::fabs(pressureAlphaLo), 1e-4f);
                    float suctionT = std::clamp(-pressure / suctionRef, 0.25f, 1.0f);
                    sizeScale = 0.30f + 0.85f * suctionT;
                } else {
                    float speedT = std::clamp((speed - speedLo) * invSpeedRange, 0.0f, 1.0f);
                    float pressureT = std::clamp((pressure - pressureAlphaLo) * invAlphaRange, 0.0f, 1.0f);
                    if (pressureT <= 1e-4f) continue;
                    color = Colormap(speedT);
                    // Both circle size (radius) and soft center opacity scale
                    // with pressure so higher-pressure disturbances render as
                    // larger fuzzy circles while ambient/low-pressure regions
                    // shrink away.
                    sizeScale = 0.20f + 1.15f * pressureT;
                    alpha = (0.35f + 0.65f * pressureT) * kAlphaMax;
                }

                // Routed through grid.CellMinCorner() rather than
                // hand-rolling origin+index*cellSize (an earlier version
                // of this did) — that bypassed CellMinCorner's own
                // grid-x-is-world-Y/grid-y-is-world-X mapping (see its
                // comment in VoxelGrid.h), so blocks ended up positioned
                // using the wrong world axis for their grid-x/grid-y
                // offset. `size` needs the same swap applied directly
                // (sizeX/sizeY count grid cells along grid-x/grid-y,
                // i.e. world Y/X, not world X/Y) since there's no
                // single-axis-size helper to route through instead.
                Vec3 minCorner = grid.CellMinCorner(bx, by, bz);
                Vec3 size{static_cast<float>(sizeY) * grid.cellSize, static_cast<float>(sizeX) * grid.cellSize,
                          static_cast<float>(sizeZ) * grid.cellSize};
                Vec3 center = minCorner + size * 0.5f;
                // A circle can't tile a cubic grid with no gaps the way
                // the cube this replaced did — kOverlapFactor sizes the
                // base radius a bit past the block's own half-extent, and
                // `sizeScale` (driven by pressure) scales each fuzzy
                // circle's radius relative to that base size.
                constexpr float kOverlapFactor = 1.25f;
                float baseRadius = (size.x + size.y + size.z) / 3.0f * 0.5f * kOverlapFactor;
                float radius = baseRadius * sizeScale;
                AddFuzzyBillboard(mesh, center, radius, camRight, camUp, color, alpha);
            }
        }
    }

    return mesh;
}

FieldOverlay::AerodynamicForces FieldOverlay::ComputeAerodynamicForces(const VoxelGrid& grid,
                                                                        const std::vector<float>& field) {
    AerodynamicForces out;
    if (field.size() != grid.CellCount() * 4 || grid.CellCount() == 0) return out;

    unsigned int numThreads = ParallelForZThreadCount(grid.nz);
    struct Partial {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        double liftWeightSum = 0.0, liftWx = 0.0, liftWy = 0.0, liftWz = 0.0;
        double forceWeightSum = 0.0, forceWx = 0.0, forceWy = 0.0, forceWz = 0.0;
        double geomCx = 0.0, geomCy = 0.0, geomCz = 0.0;
        size_t solidCount = 0;
        int maxSolidZ = -1;
        float lo = 1e30f, hi = -1e30f;
        bool hasNaN = false;
    };
    std::vector<Partial> partials(numThreads);

    ParallelForZ(grid.nz, [&](int zBegin, int zEnd, unsigned int threadIndex) {
        Partial& p = partials[threadIndex];
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    size_t cell = grid.Index(x, y, z);
                    if (!grid.solid[cell]) continue;
                    Vec3 pos = grid.CellCenter(x, y, z);
                    p.geomCx += pos.x;
                    p.geomCy += pos.y;
                    p.geomCz += pos.z;
                    ++p.solidCount;
                    p.maxSolidZ = std::max(p.maxSolidZ, z);

                    float stress = field[cell * 4 + 0];
                    float fx = field[cell * 4 + 1];
                    float fy = field[cell * 4 + 2];
                    float fz = field[cell * 4 + 3];
                    if (!std::isfinite(stress) || !std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(fz)) {
                        p.hasNaN = true;
                        continue;
                    }
                    p.fx += fx;
                    p.fy += fy;
                    p.fz += fz;

                    double wLift = std::fabs(static_cast<double>(fz));
                    p.liftWeightSum += wLift;
                    p.liftWx += wLift * pos.x;
                    p.liftWy += wLift * pos.y;
                    p.liftWz += wLift * pos.z;

                    double wForce = std::sqrt(static_cast<double>(fx) * fx + static_cast<double>(fy) * fy +
                                              static_cast<double>(fz) * fz);
                    p.forceWeightSum += wForce;
                    p.forceWx += wForce * pos.x;
                    p.forceWy += wForce * pos.y;
                    p.forceWz += wForce * pos.z;

                    // Only surface solid cells (which actually touch fluid)
                    // contribute to the stress color range; deep interior
                    // solid cells have 0 links and aren't part of the skin.
                    if (stress > 0.0f) {
                        p.lo = std::min(p.lo, stress);
                        p.hi = std::max(p.hi, stress);
                    }
                }
            }
        }
    });

    double fxSum = 0.0, fySum = 0.0, fzSum = 0.0;
    double liftWSum = 0.0, liftWx = 0.0, liftWy = 0.0, liftWz = 0.0;
    double forceWSum = 0.0, forceWx = 0.0, forceWy = 0.0, forceWz = 0.0;
    double geomCx = 0.0, geomCy = 0.0, geomCz = 0.0;
    size_t solidCount = 0;
    int maxSolidZ = -1;
    float lo = 1e30f, hi = -1e30f;
    for (unsigned int t = 0; t < numThreads; ++t) {
        fxSum += partials[t].fx;
        fySum += partials[t].fy;
        fzSum += partials[t].fz;
        liftWSum += partials[t].liftWeightSum;
        liftWx += partials[t].liftWx;
        liftWy += partials[t].liftWy;
        liftWz += partials[t].liftWz;
        forceWSum += partials[t].forceWeightSum;
        forceWx += partials[t].forceWx;
        forceWy += partials[t].forceWy;
        forceWz += partials[t].forceWz;
        geomCx += partials[t].geomCx;
        geomCy += partials[t].geomCy;
        geomCz += partials[t].geomCz;
        solidCount += partials[t].solidCount;
        maxSolidZ = std::max(maxSolidZ, partials[t].maxSolidZ);
        lo = std::min(lo, partials[t].lo);
        hi = std::max(hi, partials[t].hi);
        if (partials[t].hasNaN) out.hasNaN = true;
    }
    if (lo > hi) {
        lo = 0.0f;
        hi = 1e-6f;
    } else {
        // Anchor the low end of the stress color scale at 0 so unloaded /
        // quiescent skin reads as deep blue (0 Pa) rather than stretching
        // minuscule background variations across the whole spectrum.
        lo = 0.0f;
        if (!(hi > lo)) hi = lo + 1e-6f;
    }
    out.forceX = static_cast<float>(fxSum);
    out.forceY = static_cast<float>(fySum);
    out.forceZ = static_cast<float>(fzSum);
    out.minStress = lo;
    out.maxStress = hi;

    if (liftWSum > 1e-9) {
        out.centerOfPressure = Vec3{static_cast<float>(liftWx / liftWSum), static_cast<float>(liftWy / liftWSum),
                                    static_cast<float>(liftWz / liftWSum)};
    } else if (forceWSum > 1e-9) {
        out.centerOfPressure = Vec3{static_cast<float>(forceWx / forceWSum), static_cast<float>(forceWy / forceWSum),
                                    static_cast<float>(forceWz / forceWSum)};
    } else if (solidCount > 0) {
        double inv = 1.0 / static_cast<double>(solidCount);
        out.centerOfPressure =
            Vec3{static_cast<float>(geomCx * inv), static_cast<float>(geomCy * inv), static_cast<float>(geomCz * inv)};
    }

    // Locate the upper surface of the model directly above the Centre of
    // Pressure (inverting CellCenter's world-X <-> grid-y, world-Y <-> grid-x
    // mapping) so the Lift arrow is anchored cleanly above the top skin at
    // (cp.x, cp.y), falling back to the model's overall top Z if (cp.x, cp.y)
    // lies over a hollow opening.
    int cpGridX = std::clamp(static_cast<int>(std::floor((out.centerOfPressure.y - grid.origin.y) / grid.cellSize)), 0,
                             std::max(grid.nx - 1, 0));
    int cpGridY = std::clamp(static_cast<int>(std::floor((out.centerOfPressure.x - grid.origin.x) / grid.cellSize)), 0,
                             std::max(grid.ny - 1, 0));
    int colTopZ = -1;
    for (int dy = -2; dy <= 2; ++dy) {
        int gy = std::clamp(cpGridY + dy, 0, std::max(grid.ny - 1, 0));
        for (int dx = -2; dx <= 2; ++dx) {
            int gx = std::clamp(cpGridX + dx, 0, std::max(grid.nx - 1, 0));
            for (int z = grid.nz - 1; z >= 0; --z) {
                if (grid.solid[grid.Index(gx, gy, z)]) {
                    colTopZ = std::max(colTopZ, z);
                    break;
                }
            }
        }
    }
    int effectiveTopZ = (colTopZ >= 0) ? colTopZ : std::max(maxSolidZ, 0);
    float topWorldZ = grid.origin.z + (static_cast<float>(effectiveTopZ) + 1.0f) * grid.cellSize;
    out.liftArrowBase = Vec3{out.centerOfPressure.x, out.centerOfPressure.y, topWorldZ};

    return out;
}

OverlayMesh FieldOverlay::BuildStressMapMesh(const VoxelGrid& grid, const std::vector<float>& field, float stressLo,
                                              float stressHi) {
    OverlayMesh mesh;
    if (field.size() != grid.CellCount() * 4 || grid.CellCount() == 0) return mesh;

    float invRange = 1.0f / std::max(stressHi - stressLo, 1e-9f);
    constexpr Vec3 kNaNColor{1.0f, 0.1f, 1.0f};
    const Vec3 kLightDir = Normalize(Vec3{0.45f, -0.55f, 0.70f});
    bool hasSdf = grid.signedDistance.size() == grid.CellCount();

    struct Face {
        int dx, dy, dz;
        Vec3 normal;
        int corners[4][3];  // unit-cube-local integer corner offsets, CCW from outside
    };
    static constexpr Face kFaces[6] = {
        {+1, 0, 0, {1, 0, 0}, {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}}},
        {-1, 0, 0, {-1, 0, 0}, {{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 0, 0}}},
        {0, +1, 0, {0, 1, 0}, {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}}},
        {0, -1, 0, {0, -1, 0}, {{0, 0, 1}, {0, 0, 0}, {1, 0, 0}, {1, 0, 1}}},
        {0, 0, +1, {0, 0, 1}, {{1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 0, 1}}},
        {0, 0, -1, {0, 0, -1}, {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}},
    };

    // Evaluates a shared grid corner (vx, vy, vz):
    // 1. Averages surface stress across the up to 8 surrounding cells that
    //    are exposed boundary solid cells, so adjacent faces sharing this
    //    corner interpolate smoothly without per-voxel stairstep jumps.
    // 2. When the SDF is available and finite around this corner, projects
    //    the vertex along the SDF gradient onto the true zero-isosurface
    //    (clamped to +-0.45*cellSize so cells never invert), keeping the
    //    mesh 100% watertight because the projected position depends only
    //    on (vx, vy, vz).
    auto evalCorner = [&](int vx, int vy, int vz, float fallbackStress, const Vec3& faceWorldNormal, Vec3& outPos,
                          Vec3& outNormal, float& outStress, bool& outFinite) {
        float stressSum = 0.0f;
        int stressCount = 0;
        outFinite = std::isfinite(fallbackStress);

        float sdf[2][2][2];
        bool allSdfFinite = hasSdf && vx >= 1 && vx < grid.nx && vy >= 1 && vy < grid.ny && vz >= 1 && vz < grid.nz;

        for (int oz = 0; oz < 2; ++oz) {
            int cz = vz - 1 + oz;
            for (int oy = 0; oy < 2; ++oy) {
                int cy = vy - 1 + oy;
                for (int ox = 0; ox < 2; ++ox) {
                    int cx = vx - 1 + ox;
                    if (!grid.InBounds(cx, cy, cz)) {
                        allSdfFinite = false;
                        continue;
                    }
                    size_t c = grid.Index(cx, cy, cz);
                    if (allSdfFinite) {
                        float phi = grid.signedDistance[c];
                        if (std::isfinite(phi)) {
                            sdf[oz][oy][ox] = phi;
                        } else {
                            allSdfFinite = false;
                        }
                    }
                    if (grid.solid[c]) {
                        float s = field[c * 4 + 0];
                        if (!std::isfinite(s)) {
                            outFinite = false;
                        } else if (s > 0.0f) {
                            stressSum += s;
                            ++stressCount;
                        }
                    }
                }
            }
        }

        outStress = stressCount > 0 ? (stressSum / static_cast<float>(stressCount)) : fallbackStress;
        outPos = grid.CellMinCorner(vx, vy, vz);
        outNormal = faceWorldNormal;

        if (allSdfFinite) {
            float phiCorner = 0.125f * (sdf[0][0][0] + sdf[0][0][1] + sdf[0][1][0] + sdf[0][1][1] +
                                        sdf[1][0][0] + sdf[1][0][1] + sdf[1][1][0] + sdf[1][1][1]);
            float gx = (sdf[0][0][1] - sdf[0][0][0]) + (sdf[0][1][1] - sdf[0][1][0]) +
                       (sdf[1][0][1] - sdf[1][0][0]) + (sdf[1][1][1] - sdf[1][1][0]);
            float gy = (sdf[0][1][0] - sdf[0][0][0]) + (sdf[0][1][1] - sdf[0][0][1]) +
                       (sdf[1][1][0] - sdf[1][0][0]) + (sdf[1][1][1] - sdf[1][0][1]);
            float gz = (sdf[1][0][0] - sdf[0][0][0]) + (sdf[1][0][1] - sdf[0][0][1]) +
                       (sdf[1][1][0] - sdf[0][1][0]) + (sdf[1][1][1] - sdf[0][1][1]);
            // Swap gx/gy to world space (grid-x is world Y, grid-y is world X —
            // see VoxelGrid::CellCenter).
            Vec3 gradWorld{gy, gx, gz};
            float gradLen = Length(gradWorld);
            if (gradLen > 1e-5f) {
                Vec3 nSdf = gradWorld * (1.0f / gradLen);
                float maxShift = 0.45f * grid.cellSize;
                float shift = std::clamp(phiCorner, -maxShift, maxShift);
                outPos = outPos - nSdf * shift;
                outNormal = Normalize(faceWorldNormal * 0.3f + nSdf * 0.7f);
            }
        }
    };

    unsigned int numThreads = ParallelForZThreadCount(grid.nz);
    std::vector<OverlayMesh> partials(numThreads);

    ParallelForZ(grid.nz, [&](int zBegin, int zEnd, unsigned int threadIndex) {
        OverlayMesh& local = partials[threadIndex];
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    if (!grid.IsSolid(x, y, z)) continue;
                    size_t cell = grid.Index(x, y, z);
                    float cellStress = field[cell * 4 + 0];

                    for (const Face& f : kFaces) {
                        int nx = x + f.dx, ny = y + f.dy, nz = z + f.dz;
                        if (grid.IsSolid(nx, ny, nz)) continue;

                        Vec3 faceWorldNormal{f.normal.y, f.normal.x, f.normal.z};
                        uint32_t base = static_cast<uint32_t>(local.vertices.size());
                        for (int cIdx = 0; cIdx < 4; ++cIdx) {
                            int vx = x + f.corners[cIdx][0];
                            int vy = y + f.corners[cIdx][1];
                            int vz = z + f.corners[cIdx][2];

                            Vec3 pos, normal;
                            float stress;
                            bool finite;
                            evalCorner(vx, vy, vz, cellStress, faceWorldNormal, pos, normal, stress, finite);

                            Vec3 color;
                            if (!finite) {
                                color = kNaNColor;
                            } else {
                                float t = std::clamp((stress - stressLo) * invRange, 0.0f, 1.0f);
                                Vec3 baseCol = Colormap(t);
                                float ndotl = std::max(0.0f, Dot(normal, kLightDir));
                                float hemi = 0.5f + 0.5f * normal.z;
                                float shade = 0.68f + 0.24f * ndotl + 0.08f * hemi;
                                color = baseCol * shade;
                            }
                            local.vertices.push_back({pos, color, 1.0f});
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

    for (const OverlayMesh& local : partials) {
        AppendMesh(mesh, local);
    }
    return mesh;
}

OverlayMesh FieldOverlay::BuildIsoSurfaceMesh(const VoxelGrid& grid, const std::vector<float>& field,
                                                FieldColorMode mode, float isoValue, float lo, float hi,
                                                float alpha, bool encloseGreater, int stride,
                                                const std::vector<float>* xMeans) {
    OverlayMesh mesh;
    if (field.size() != grid.CellCount() * 4 || grid.nx < 2 || grid.ny < 2 || grid.nz < 2) return mesh;
    stride = std::max(1, stride);

    float invRange = 1.0f / std::max(hi - lo, 1e-6f);
    float tColor = std::clamp((isoValue - lo) * invRange, 0.0f, 1.0f);
    Vec3 baseColor = Colormap(tColor);
    const Vec3 kLightDir = Normalize(Vec3{0.45f, -0.55f, 0.70f});

    auto sampleVal = [&](int x, int y, int z) -> float {
        x = std::clamp(x, 0, grid.nx - 1);
        y = std::clamp(y, 0, grid.ny - 1);
        z = std::clamp(z, 0, grid.nz - 1);
        size_t c = grid.Index(x, y, z);
        if (grid.solid[c]) {
            return encloseGreater ? -1e30f : 1e30f;
        }
        float v = FieldValueMaybeDetrended(field, c, x, mode, xMeans);
        if (!std::isfinite(v)) {
            return encloseGreater ? -1e30f : 1e30f;
        }
        return v;
    };

    auto gradWorld = [&](int x, int y, int z) -> Vec3 {
        float gx = (sampleVal(x + stride, y, z) - sampleVal(x - stride, y, z)) * 0.5f;
        float gy = (sampleVal(x, y + stride, z) - sampleVal(x, y - stride, z)) * 0.5f;
        float gz = (sampleVal(x, y, z + stride) - sampleVal(x, y, z - stride)) * 0.5f;
        // Swap grid-x (world-Y) and grid-y (world-X)
        return Vec3{gy, gx, gz};
    };

    unsigned int numThreads = ParallelForZThreadCount(grid.nz - 1);
    std::vector<OverlayMesh> partials(numThreads);

    ParallelForZ(grid.nz - 1, [&](int zBegin, int zEnd, unsigned int threadIndex) {
        OverlayMesh& local = partials[threadIndex];
        int zStartAligned = (zBegin / stride) * stride;
        for (int z = zStartAligned; z < zEnd; z += stride) {
            if (z < zBegin) continue;
            int zNext = std::min(z + stride, grid.nz - 1);
            if (zNext == z) continue;

            for (int y = 0; y + 1 < grid.ny; y += stride) {
                int yNext = std::min(y + stride, grid.ny - 1);
                for (int x = 0; x + 1 < grid.nx; x += stride) {
                    int xNext = std::min(x + stride, grid.nx - 1);

                    int cx[8] = {x, xNext, xNext, x, x, xNext, xNext, x};
                    int cy[8] = {y, y, yNext, yNext, y, y, yNext, yNext};
                    int cz[8] = {z, z, z, z, zNext, zNext, zNext, zNext};

                    float val[8];
                    int cubeIndex = 0;
                    for (int k = 0; k < 8; ++k) {
                        val[k] = sampleVal(cx[k], cy[k], cz[k]);
                        bool inside = encloseGreater ? (val[k] >= isoValue) : (val[k] <= isoValue);
                        if (inside) cubeIndex |= (1 << k);
                    }

                    int edgeMask = MarchingCubes::kEdgeTable[cubeIndex];
                    if (edgeMask == 0) continue;

                    OverlayVertex edgeVerts[12];
                    for (int e = 0; e < 12; ++e) {
                        if (!(edgeMask & (1 << e))) continue;
                        int c0 = MarchingCubes::kEdgeCorners[e][0];
                        int c1 = MarchingCubes::kEdgeCorners[e][1];
                        float v0 = val[c0];
                        float v1 = val[c1];
                        float denom = v1 - v0;
                        float t = (std::fabs(denom) > 1e-12f) ? std::clamp((isoValue - v0) / denom, 0.0f, 1.0f) : 0.5f;

                        Vec3 p0 = grid.CellCenter(cx[c0], cy[c0], cz[c0]);
                        Vec3 p1 = grid.CellCenter(cx[c1], cy[c1], cz[c1]);
                        Vec3 pos = p0 + (p1 - p0) * t;

                        Vec3 g0 = gradWorld(cx[c0], cy[c0], cz[c0]);
                        Vec3 g1 = gradWorld(cx[c1], cy[c1], cz[c1]);
                        Vec3 grad = g0 + (g1 - g0) * t;
                        float glen = Length(grad);
                        Vec3 normal = (glen > 1e-5f) ? (grad * (1.0f / glen)) : Vec3{0.0f, 0.0f, 1.0f};
                        if (!encloseGreater) normal = normal * -1.0f;

                        float ndotl = std::fabs(Dot(normal, kLightDir));
                        float hemi = 0.5f + 0.5f * std::fabs(normal.z);
                        float shade = 0.62f + 0.28f * ndotl + 0.10f * hemi;
                        edgeVerts[e] = {pos, baseColor * shade, alpha};
                    }

                    for (int i = 0; MarchingCubes::kTriTable[cubeIndex][i] != -1; i += 3) {
                        int e0 = MarchingCubes::kTriTable[cubeIndex][i + 0];
                        int e1 = MarchingCubes::kTriTable[cubeIndex][i + 1];
                        int e2 = MarchingCubes::kTriTable[cubeIndex][i + 2];
                        uint32_t base = static_cast<uint32_t>(local.vertices.size());
                        local.vertices.push_back(edgeVerts[e0]);
                        local.vertices.push_back(edgeVerts[e1]);
                        local.vertices.push_back(edgeVerts[e2]);
                        local.indices.push_back(base + 0);
                        local.indices.push_back(base + 1);
                        local.indices.push_back(base + 2);
                    }
                }
            }
        }
    });

    for (const OverlayMesh& local : partials) {
        AppendMesh(mesh, local);
    }
    return mesh;
}

OverlayMesh FieldOverlay::BuildRaymarchMesh(const VoxelGrid& grid, const std::vector<float>& field,
                                            FieldColorMode mode, float lo, float hi,
                                            const Vec3& camPos, const Vec3& camForward,
                                            const Vec3& camRight, const Vec3& camUp,
                                            int sliceCount, float densityScale, float cutoff,
                                            float gamma, const std::vector<float>* xMeans) {
    OverlayMesh mesh;
    if (field.size() != grid.CellCount() * 4 || grid.nx < 2 || grid.ny < 2 || grid.nz < 2) return mesh;

    Vec3 minBox = grid.CellMinCorner(0, 0, 0);
    Vec3 maxBox = minBox + Vec3{static_cast<float>(grid.ny) * grid.cellSize,
                                static_cast<float>(grid.nx) * grid.cellSize,
                                static_cast<float>(grid.nz) * grid.cellSize};

    static const int kBoxCorners[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };
    Vec3 corners[8];
    for (int i = 0; i < 8; ++i) {
        corners[i] = Vec3{
            kBoxCorners[i][0] ? maxBox.x : minBox.x,
            kBoxCorners[i][1] ? maxBox.y : minBox.y,
            kBoxCorners[i][2] ? maxBox.z : minBox.z
        };
    }

    static const int kBoxEdges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };

    Vec3 fwd = Normalize(camForward);
    Vec3 right = Normalize(camRight);
    Vec3 up = Normalize(camUp);

    float dMin = 1e30f;
    float dMax = -1e30f;
    for (int i = 0; i < 8; ++i) {
        float d = Dot(corners[i] - camPos, fwd);
        dMin = std::min(dMin, d);
        dMax = std::max(dMax, d);
    }
    if (dMax <= dMin + 1e-4f) return mesh;
    dMin = std::max(dMin, 0.05f);
    if (dMax <= dMin + 1e-4f) return mesh;

    sliceCount = std::clamp(sliceCount, 8, 192);
    float sliceThickness = (dMax - dMin) / static_cast<float>(sliceCount);
    float invRange = 1.0f / std::max(hi - lo, 1e-6f);

    auto sampleContinuous = [&](const Vec3& p, float& outVal, float& outSolid) -> bool {
        float gx = (p.y - minBox.y) / grid.cellSize - 0.5f;
        float gy = (p.x - minBox.x) / grid.cellSize - 0.5f;
        float gz = (p.z - minBox.z) / grid.cellSize - 0.5f;

        int x0 = static_cast<int>(std::floor(gx));
        int y0 = static_cast<int>(std::floor(gy));
        int z0 = static_cast<int>(std::floor(gz));
        if (x0 < 0 || x0 >= grid.nx - 1 || y0 < 0 || y0 >= grid.ny - 1 || z0 < 0 || z0 >= grid.nz - 1) {
            outVal = 0.0f;
            outSolid = 0.0f;
            return false;
        }
        float fx = gx - static_cast<float>(x0);
        float fy = gy - static_cast<float>(y0);
        float fz = gz - static_cast<float>(z0);

        float valSum = 0.0f;
        float solidSum = 0.0f;
        for (int dz = 0; dz <= 1; ++dz) {
            float wz = dz ? fz : (1.0f - fz);
            for (int dy = 0; dy <= 1; ++dy) {
                float wy = dy ? fy : (1.0f - fy);
                for (int dx = 0; dx <= 1; ++dx) {
                    float wx = dx ? fx : (1.0f - fx);
                    float w = wx * wy * wz;
                    size_t c = grid.Index(x0 + dx, y0 + dy, z0 + dz);
                    if (grid.solid[c]) {
                        solidSum += w;
                    } else {
                        float v = FieldValueMaybeDetrended(field, c, x0 + dx, mode, xMeans);
                        if (std::isfinite(v)) {
                            valSum += w * v;
                        }
                    }
                }
            }
        }
        outVal = valSum;
        outSolid = solidSum;
        return true;
    };

    auto evalTransfer = [&](const Vec3& p) -> OverlayVertex {
        OverlayVertex vert{};
        vert.pos = p;
        float val = 0.0f, solid = 0.0f;
        if (!sampleContinuous(p, val, solid) || solid > 0.45f) {
            vert.color = Vec3{0.0f, 0.0f, 0.0f};
            vert.alpha = 0.0f;
            return vert;
        }
        float t = std::clamp((val - lo) * invRange, 0.0f, 1.0f);
        if (t < cutoff) {
            vert.color = Colormap(t);
            vert.alpha = 0.0f;
            return vert;
        }
        float tEff = (t - cutoff) / std::max(1.0f - cutoff, 1e-4f);
        float density = std::pow(tEff, gamma);
        float alpha = 1.0f - std::exp(-densityScale * density * sliceThickness * 20.0f);
        alpha = std::clamp(alpha, 0.0f, 0.85f);
        vert.color = Colormap(t);
        vert.alpha = alpha;
        return vert;
    };

    unsigned int numThreads = ParallelForZThreadCount(sliceCount);
    std::vector<OverlayMesh> partials(numThreads);

    ParallelForZ(sliceCount, [&](int sBegin, int sEnd, unsigned int threadIndex) {
        OverlayMesh& local = partials[threadIndex];
        struct IntersectPt {
            Vec3 pos;
            float angle = 0.0f;
        };

        for (int s = sBegin; s < sEnd; ++s) {
            // Slices ordered from back (dMax) to front (dMin)
            float d_s = dMax - (static_cast<float>(s) + 0.5f) * sliceThickness;

            std::vector<IntersectPt> poly;
            poly.reserve(6);
            for (int e = 0; e < 12; ++e) {
                Vec3 pA = corners[kBoxEdges[e][0]];
                Vec3 pB = corners[kBoxEdges[e][1]];
                float distA = Dot(pA - camPos, fwd) - d_s;
                float distB = Dot(pB - camPos, fwd) - d_s;
                if ((distA > 0.0f) != (distB > 0.0f)) {
                    float t = distA / (distA - distB);
                    poly.push_back({pA + (pB - pA) * t, 0.0f});
                }
            }
            if (poly.size() < 3) continue;

            Vec3 centroid{0.0f, 0.0f, 0.0f};
            for (const auto& pt : poly) centroid = centroid + pt.pos;
            centroid = centroid * (1.0f / static_cast<float>(poly.size()));

            for (auto& pt : poly) {
                Vec3 diff = pt.pos - centroid;
                float u = Dot(diff, right);
                float v = Dot(diff, up);
                pt.angle = std::atan2(v, u);
            }
            std::sort(poly.begin(), poly.end(), [](const IntersectPt& a, const IntersectPt& b) {
                return a.angle < b.angle;
            });

            constexpr int kSubdiv = 5;
            int polyCount = static_cast<int>(poly.size());
            for (int i = 0; i < polyCount; ++i) {
                int next = (i + 1) % polyCount;
                const Vec3& p0 = centroid;
                const Vec3& p1 = poly[i].pos;
                const Vec3& p2 = poly[next].pos;

                OverlayVertex triVerts[(kSubdiv + 1) * (kSubdiv + 2) / 2];
                int idx = 0;
                for (int r = 0; r <= kSubdiv; ++r) {
                    for (int c = 0; c <= r; ++c) {
                        float w0 = 1.0f - static_cast<float>(r) / static_cast<float>(kSubdiv);
                        float w1 = static_cast<float>(r - c) / static_cast<float>(kSubdiv);
                        float w2 = static_cast<float>(c) / static_cast<float>(kSubdiv);
                        Vec3 pt = p0 * w0 + p1 * w1 + p2 * w2;
                        triVerts[idx++] = evalTransfer(pt);
                    }
                }

                auto vertIndex = [&](int r, int c) -> int {
                    return r * (r + 1) / 2 + c;
                };

                for (int r = 0; r < kSubdiv; ++r) {
                    for (int c = 0; c <= r; ++c) {
                        int v0 = vertIndex(r, c);
                        int v1 = vertIndex(r + 1, c);
                        int v2 = vertIndex(r + 1, c + 1);

                        if (triVerts[v0].alpha > 0.0f || triVerts[v1].alpha > 0.0f || triVerts[v2].alpha > 0.0f) {
                            uint32_t base = static_cast<uint32_t>(local.vertices.size());
                            local.vertices.push_back(triVerts[v0]);
                            local.vertices.push_back(triVerts[v1]);
                            local.vertices.push_back(triVerts[v2]);
                            local.indices.push_back(base + 0);
                            local.indices.push_back(base + 1);
                            local.indices.push_back(base + 2);
                        }

                        if (c < r) {
                            int v3 = vertIndex(r, c + 1);
                            if (triVerts[v0].alpha > 0.0f || triVerts[v2].alpha > 0.0f || triVerts[v3].alpha > 0.0f) {
                                uint32_t base = static_cast<uint32_t>(local.vertices.size());
                                local.vertices.push_back(triVerts[v0]);
                                local.vertices.push_back(triVerts[v2]);
                                local.vertices.push_back(triVerts[v3]);
                                local.indices.push_back(base + 0);
                                local.indices.push_back(base + 1);
                                local.indices.push_back(base + 2);
                            }
                        }
                    }
                }
            }
        }
    });

    for (const OverlayMesh& local : partials) {
        AppendMesh(mesh, local);
    }
    return mesh;
}
