#pragma once

#include <cstdint>
#include <vector>

#include "Mesh.h"

// A regular cubic-cell grid over a world-space (millimetre) domain,
// classifying each cell as solid (inside or on the model's surface) or
// fluid. This is the geometry input the LBM solver will read its boundary
// condition from — one lattice site per fluid cell, bounce-back at solid
// cells.
struct VoxelGrid {
    int nx = 0, ny = 0, nz = 0;
    float cellSize = 1.0f;  // mm, uniform cubic cells
    Vec3 origin{0.0f, 0.0f, 0.0f};  // world-space position of cell (0,0,0)'s min corner

    std::vector<uint8_t> solid;  // nx*ny*nz, 1 = solid, 0 = fluid

    // Signed distance to the model's true surface, in world units (mm):
    // positive in fluid, negative in solid, magnitude = exact distance to
    // the nearest point on the triangle mesh — see Voxelizer::Voxelize's
    // own comment for how it's computed (reusing the same closest-
    // triangle query that already classifies solid/fluid). Only
    // meaningful near the surface — cells far from any triangle keep the
    // sentinel ±infinity (sign still correct) they're initialized to,
    // since nothing ever queries an SDF value that far from a boundary.
    // Same size/indexing as `solid`; empty on a VoxelGrid built before
    // this existed (a default-constructed one, or one from an older code
    // path that never populated it) — a consumer must treat an empty
    // vector as "no SDF available" and fall back accordingly, never index
    // into it assuming it matches `solid`'s size.
    std::vector<float> signedDistance;

    size_t CellCount() const { return static_cast<size_t>(nx) * ny * nz; }

    size_t Index(int x, int y, int z) const { return (static_cast<size_t>(z) * ny + y) * nx + x; }

    bool InBounds(int x, int y, int z) const { return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz; }

    bool IsSolid(int x, int y, int z) const { return InBounds(x, y, z) && solid[Index(x, y, z)] != 0; }

    // Grid-x maps to world Y and grid-y to world X — swapped from the
    // "obvious" identity mapping so that the solver's own inlet/outlet
    // logic (always grid-x, in LbmReference/LbmSolver/PotentialFlowSolver
    // — e.g. LbmReference::Step()'s inletX/outletX) ends up blowing along
    // world Y without those solver files needing to know or care: they
    // only ever read/write grid-index space, and this is the one place
    // that index space meets world space. Grid-z still maps to world Z
    // (up), unchanged. Everything downstream that calls CellCenter/
    // CellMinCorner (FieldOverlay, SmokeTrail, the overlay slice/scale
    // ruler code, Voxelizer's own BuildPreviewMesh) picks up the new
    // mapping for free; nothing else needed to change.
    Vec3 CellCenter(int x, int y, int z) const {
        return origin + Vec3((y + 0.5f) * cellSize, (x + 0.5f) * cellSize, (z + 0.5f) * cellSize);
    }

    Vec3 CellMinCorner(int x, int y, int z) const {
        return origin + Vec3(y * cellSize, x * cellSize, z * cellSize);
    }
};
