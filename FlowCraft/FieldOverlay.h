#pragma once

#include <cstdint>
#include <vector>

#include "Mesh.h"
#include "VoxelGrid.h"

// Builds a colored cross-section through the solver's velocity/pressure
// field for visualization — the classic CFD "slice contour" plot. Not lit
// (the color IS the data), so it's rendered by OverlayRenderer rather than
// the normal-shaded MeshRenderer.
struct OverlayVertex {
    Vec3 pos;
    Vec3 color;
    float alpha = 1.0f;  // only meaningful to OverlayRenderer::RenderTransparent — see BuildVoxelCloudMesh
};

struct OverlayMesh {
    std::vector<OverlayVertex> vertices;
    std::vector<uint32_t> indices;
};

enum class SliceAxis { X, Y, Z };
enum class FieldColorMode { Speed, Pressure };

namespace FieldOverlay {

// Maps t in [0,1] to an RGB color via a 5-stop blue->cyan->green->yellow->red
// gradient — standard for CFD contour plots (low values cool, high warm).
Vec3 Colormap(float t);

// Appends src's geometry onto dst, offsetting src's indices by dst's
// vertex count at the time of the call — for combining several
// independently-built meshes (e.g. multiple simultaneous slice planes, or
// their confined particle fields) into one upload/draw call.
void AppendMesh(OverlayMesh& dst, const OverlayMesh& src);

// The raw per-cell value `mode` selects from the 4-floats-per-cell layout
// (Speed: velocity magnitude; Pressure: lattice-unit deviation from
// ambient, rho=1). Exposed rather than kept file-internal so other
// consumers of that same layout — e.g. SmokeTrailSystem's point
// samples — use the one definition instead of duplicating it.
float SampleFieldValue(const std::vector<float>& field, size_t cell, FieldColorMode mode);

// `field` is 4 floats per cell (rho,ux,uy,uz), laid out as VoxelGrid
// addresses it — i.e. exactly what LbmSolver::ReadbackField() returns.
//
// Split in two because they have very different costs: ComputeRange is
// O(every cell in the grid) — it has to be, since the color scale is
// deliberately computed from the *entire* field, not just the visible
// slice, so the same color means the same thing as the user moves the
// slice around. BuildSliceMeshFromRange, given that range, is only
// O(the visible slice's area) — moving the slice or changing its axis
// doesn't change the range at all (that only depends on the field and
// the color mode), so it never needs to re-scan the whole volume. A
// caller with a cached range from a previous ComputeRange call (same
// field, same mode) should go straight to BuildSliceMeshFromRange —
// cheap enough to call synchronously, even at high resolution, which is
// what makes moving the slice feel instant instead of kicking off
// another background task. BuildSliceMesh at the bottom is both steps
// together, for when there's no cached range yet (a fresh field just
// arrived, or the color mode changed, which does change the range).
//
// `outHasNaN` is set if any cell's value was non-finite (the solver has
// diverged) — a property of the whole field, not the slice, so it's only
// reported by ComputeRange/BuildSliceMesh. Individual NaN cells within a
// slice are still painted a flat, distinct magenta by
// BuildSliceMeshFromRange regardless — that's a per-cell check, not tied
// to the aggregate scan — instead of being run through Colormap, which
// otherwise maps a NaN input to solid red (clamp()/comparisons against
// NaN are all false, so it falls through every stop and hits the last
// one): without that, a diverged solver and a legitimately high-value
// field would look identical.
// `xMeans`, when given, is subtracted from each cell's own X-slice value
// before it's used — see ComputeXSliceMeans below. Only affects Pressure;
// passing it alongside FieldColorMode::Speed is a harmless no-op.
void ComputeRange(const VoxelGrid& grid, const std::vector<float>& field, FieldColorMode mode, float& outMin,
                   float& outMax, bool& outHasNaN, const std::vector<float>* xMeans = nullptr);

OverlayMesh BuildSliceMeshFromRange(const VoxelGrid& grid, const std::vector<float>& field, SliceAxis axis, int index,
                                     FieldColorMode mode, float lo, float hi, const std::vector<float>* xMeans = nullptr);

inline OverlayMesh BuildSliceMesh(const VoxelGrid& grid, const std::vector<float>& field, SliceAxis axis, int index,
                                   FieldColorMode mode, float& outMin, float& outMax, bool& outHasNaN,
                                   const std::vector<float>* xMeans = nullptr) {
    ComputeRange(grid, field, mode, outMin, outMax, outHasNaN, xMeans);
    return BuildSliceMeshFromRange(grid, field, axis, index, mode, outMin, outMax, xMeans);
}

// Per-X-slice mean of `mode`'s value (over every finite, non-solid cell at
// that X), one entry per X index. Flow always runs along the grid's X axis
// (reverseFlow just swaps which end is inlet — see LbmSolver::Params'
// comment), so a channel with no-slip Y/Z walls needs a pressure gradient
// along X just to overcome viscous friction, the same way a pipe does —
// a real effect, but one that can dwarf a model's own local
// stagnation/wake signature at ordinary settings. Subtracting each cell's
// own X-slice mean (via the `xMeans` parameter above) removes that
// lengthwise gradient, leaving the local deviation. O(cellCount), same
// cost tier as ComputeRange.
std::vector<float> ComputeXSliceMeans(const VoxelGrid& grid, const std::vector<float>& field, FieldColorMode mode);

// A second, alternative visualization (alongside the slice): the whole
// volume as semi-transparent fuzzy billboards (a soft circular sprite per
// block, facing the camera — full alpha at its center fading smoothly to
// 0 at its rim, see AddFuzzyBillboard in the .cpp), meant to be drawn with
// alpha blending (see OverlayRenderer::RenderTransparent) so looking
// through several layers gives a rough sense of the field's 3D shape —
// thicker/denser regions along a sightline build up more opacity, thinner
// ones stay faint, and the soft edges read as a cloud/mist rather than a
// grid of boxes. `camRight`/`camUp` orient that billboard — pass the
// camera's own right/up (e.g. OrbitCamera::GetBasis(), reconstructed from
// a thread-safe cached forward vector when this runs off the UI thread —
// see main.cpp's BuildMeshForStyle for why it can't just read app.camera
// directly here).
//
// Color and opacity are deliberately driven by two *different* fields
// rather than both by whatever "Color by" is set to: color is always
// Speed (seeing the flow's shape is the point), and opacity is always
// Pressure (so the model's disturbance — wake, stagnation, anywhere
// pressure deviates from ambient — is what stays visible, while
// undisturbed free-flowing fluid can fade out of the way instead of
// hazing up the whole cloud). `speedLo`/`speedHi` is the color range (a
// ComputeRange(..., FieldColorMode::Speed, ...) result, same as the
// slice uses). `pressureAlphaLo`/`pressureAlphaHi` is *not* ComputeRange's
// pressure range — it's a user-chosen window: pressure at or below
// `pressureAlphaLo` renders fully transparent (alpha 0), at or above
// `pressureAlphaHi` renders at the fixed alpha ceiling (see
// FieldOverlay.cpp's kAlphaMax), linear in between. Letting the caller
// pick this window (rather than always spanning the field's actual
// pressure min/max) is what makes "make the free-flowing fluid 100%
// transparent" possible — set `pressureAlphaLo` at/above the ambient
// pressure value and everything near it drops out.
//
// Rendering one billboard per actual cell doesn't scale — a full-
// resolution grid can be tens of millions of cells. So this downsamples
// by `stride`: cells are grouped into stride^3 blocks, each rendered as
// one billboard sized from stride*cellSize (unlike the cube this
// replaced, circles can't tile a cubic grid with no gaps — see
// AddFuzzyBillboard's own comment on the radius chosen to blend
// neighbouring blocks together instead), colored/faded by sampling that
// block's own center cell rather than averaging it (cheap, and close
// enough for a "get a feel for the shape" view). A block is skipped only
// if its sampled center cell is solid — a sampled cell whose speed or
// pressure isn't finite instead renders flat magenta at a fixed, higher
// alpha (see FieldOverlay.cpp), the same anywhere-NaN convention the
// slice uses, so divergence can't accidentally fade away.
//
// Cost is O(cellCount/stride^3), not O(cellCount) — same reasoning as
// BuildSliceMeshFromRange, this takes already-known ranges rather than
// computing them, so a caller that just changed the alpha window, or
// which visualization is showing, can call this directly instead of
// paying for another ComputeRange pass.
// `pressureAlphaXMeans`, when given, detrends the pressure value used for
// alpha the same way ComputeRange's `xMeans` does — see
// ComputeXSliceMeans. The caller is expected to have computed
// `pressureAlphaLo/Hi` against that same detrended basis in that case
// (e.g. from a ComputeRange call that was also given it).
//
// `highlightNegativePressure`, when true, overrides the normal Speed/
// alpha-window mapping for any cell whose (possibly detrended) pressure
// is below zero: it renders a fixed dark grey at a fixed 30% alpha
// instead — low enough to read as a highlight rather than solid geometry,
// but well above what the alpha window would often fade a mildly-below-
// ambient cell to (possibly all the way to invisible), so suction/low-
// pressure regions — a real wake/stagnation signal — stay visible and
// distinguishable from the ordinary Speed colormap instead of just
// fading out alongside undisturbed fluid.
OverlayMesh BuildVoxelCloudMesh(const VoxelGrid& grid, const std::vector<float>& field, float speedLo, float speedHi,
                                 float pressureAlphaLo, float pressureAlphaHi, int stride, const Vec3& camRight,
                                 const Vec3& camUp, const std::vector<float>* pressureAlphaXMeans = nullptr,
                                 bool highlightNegativePressure = false);

// Picks a stride for BuildVoxelCloudMesh that keeps the rendered cube
// count near `targetVoxelCount` regardless of the grid's actual
// resolution — bounding build/upload/render cost at a fixed budget
// instead of letting it scale with cellCount.
int ChooseVoxelCloudStride(const VoxelGrid& grid, size_t targetVoxelCount);

// Integrated aerodynamic forces (summed over all solid boundary cells'
// (Fx, Fy, Fz) slots in `field`, in lattice force units along grid axes:
// x = flow, y = lateral, z = up) and the min/max link-normalized surface
// stress magnitude (slot 0 of `field` at exposed solid cells) for coloring
// the Airframe Stress Map. Parallelized with ParallelForZ, same cost tier
// as ComputeRange.
struct AerodynamicForces {
    float forceX = 0.0f;
    float forceY = 0.0f;
    float forceZ = 0.0f;
    float minStress = 0.0f;
    float maxStress = 0.0f;
    Vec3 centerOfPressure{0.0f, 0.0f, 0.0f};
    Vec3 liftArrowBase{0.0f, 0.0f, 0.0f};
    bool hasNaN = false;
};
AerodynamicForces ComputeAerodynamicForces(const VoxelGrid& grid, const std::vector<float>& field);

// Builds a watertight, SDF-smoothed 3D surface mesh of the solid airframe
// colored by local aerodynamic surface stress (via Colormap over
// [stressLo, stressHi]) with subtle directional/hemisphere shading so the
// 3D form of the airframe reads clearly alongside the stress colors.
OverlayMesh BuildStressMapMesh(const VoxelGrid& grid, const std::vector<float>& field, float stressLo, float stressHi);

// Builds a 3D Iso-surface extraction via parallel Marching Cubes over the
// fluid velocity or pressure field. Generates a smooth 3D surface mesh
// shaded with directional + hemispherical lighting where the field crosses
// `isoValue`. `encloseGreater` determines whether the surface encloses values
// >= isoValue (e.g. high-speed jets, stagnation zones) or values <= isoValue
// (e.g. low-speed aerodynamic wake bubbles, suction vortex cores).
OverlayMesh BuildIsoSurfaceMesh(const VoxelGrid& grid, const std::vector<float>& field, FieldColorMode mode,
                                float isoValue, float lo, float hi, float alpha = 0.90f,
                                bool encloseGreater = false, int stride = 1,
                                const std::vector<float>* xMeans = nullptr);

}  // namespace FieldOverlay
