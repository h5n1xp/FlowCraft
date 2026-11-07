#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "Mesh.h"
#include "VoxelGrid.h"

// Converts a triangle mesh into a solid/fluid VoxelGrid for the LBM solver,
// and builds a lightweight cube-soup Mesh for previewing that grid.
class Voxelizer {
public:
    struct Params {
        // Voxels along the longest axis of the padded domain. Cell size is
        // derived from this plus the domain extent, so it adapts to
        // whatever physical size the (scaled) model currently is.
        int targetResolution = 640;
        // Extra empty domain added around the model's extent on each side,
        // as a fraction of that extent, one value per WORLD axis — x is
        // lateral, y is the solver's actual flow axis (see VoxelGrid's
        // CellCenter comment for why that's grid-x, not world-x), z is
        // up, so e.g. a wider y than x/z gives more upstream/downstream
        // room (for the wake, or just less wall-blockage along the flow)
        // without paying the same extra cell count sideways where it's
        // less useful. Still symmetric per axis (same padding on both
        // faces of that axis) — asymmetric (e.g. more downstream than
        // upstream) would need separate lo/hi values, not implemented
        // here.
        Vec3 paddingFactor{0.3f, 1.0f, 0.5f};
    };

    // `positions`/`indices` must already be in world space (millimetres,
    // flow-aligned) — i.e. the mesh's vertices transformed by its current
    // orientation/scale, not the STL's native coordinates. The grid only
    // means anything relative to the fixed flow axis once it's expressed
    // in that same space.
    //
    // `progress`, if non-null, is written to from whatever thread calls
    // this (never read back by it) with a 0..0.85 fraction as the three
    // internal phases (rasterize/flood-fill/finalize) advance — the
    // remaining 0.85..1.0 is BuildPreviewMesh's. Safe to poll from another
    // thread concurrently since it's only ever written here, never read.
    static VoxelGrid Voxelize(const std::vector<Vec3>& positions, const std::vector<uint32_t>& indices,
                               const Params& params, std::atomic<float>* progress = nullptr);

    // One flat-shaded quad per exposed face (a solid cell touching a fluid
    // or out-of-bounds neighbour) — not one cube per solid cell, since
    // fully-interior cells are never visible and would just bloat the
    // preview mesh. Positions are already world-space, matching the grid.
    // `progress` continues Voxelize's 0.85..1.0 range if given — see that
    // function's comment.
    static Mesh BuildPreviewMesh(const VoxelGrid& grid, std::atomic<float>* progress = nullptr);
};
