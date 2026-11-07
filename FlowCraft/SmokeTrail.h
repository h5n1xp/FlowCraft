#pragma once

#include <random>
#include <vector>

#include "FieldOverlay.h"
#include "Mesh.h"
#include "VoxelGrid.h"

// A dye-injection style "smoke trail" flow visualization — the third
// Flow Overlay style alongside Slice and Voxel Cloud. Particles are
// continuously seeded at the inlet face and advected downstream by the
// solver's own velocity field (the same readback those two use), each
// one leaving a short fading trail behind it so the whole thing reads as
// streaklines/smoke rather than a sparse point cloud, the way dye or
// smoke injected into a real wind tunnel does.
//
// Simulated and meshed entirely on whichever thread calls Step()/
// BuildMesh() — intended to be the UI thread, once per rendered frame.
// That's deliberate, not an oversight: unlike Slice/Voxel Cloud's
// O(cellCount) field scans (which is why those run on a background
// thread and only rebuild on demand — see BuildOverlayMeshFromField in
// main.cpp), this is O(particleCount * trailLength) and samples the
// already-cached field rather than re-scanning it, cheap enough to
// afford every frame.
class SmokeTrailSystem {
public:
    // Confines every particle to a single plane perpendicular to `axis`,
    // at `index` (same meaning as the Slice overlay style's own slice
    // axis/index — see FieldOverlay::BuildSliceMeshFromRange) — used by
    // the Slice style's "Also show particles" option (main.cpp) to show
    // the smoke trail restricted to the current slice plane instead of
    // the whole 3D domain, with no colored slice mesh drawn alongside it.
    // Respawn() seeds directly onto the plane (fixing whichever of x/y/z
    // `axis` names and letting the other two vary, or randomize, exactly
    // as the unconfined case does); Step() zeroes the velocity component
    // along `axis` before integrating, so a particle's coordinate on that
    // axis never changes after it's seeded — equivalent to (but cheaper
    // than) advecting in full 3D and snapping back onto the plane every
    // step, since the two other coordinates end up identical either way.
    struct PlaneConfinement {
        SliceAxis axis;
        int index;
    };

    // (Re)seeds every particle, each as a single point with no trail yet
    // — the stream fills in over the next `trailLength` calls to Step().
    // With `confine` null, scatters across the WHOLE domain (every axis
    // randomized, same as ScatterRandom below) rather than only at the
    // inlet face, so the very first frame already looks like a mature,
    // already-flowing stream instead of every particle starting flush
    // against the inlet and needing many seconds of Step() calls to
    // spread out and fill the domain — see Reset()'s .cpp comment for
    // why that mattered enough to fix. `reverseFlow` is accepted but
    // unused for this reason (kept for signature stability; it still
    // matters to Respawn()'s ongoing inlet re-entry, called from Step()).
    // With `confine` non-null, seeds directly onto that plane instead
    // (every axis *except* confine->axis randomized) — see
    // PlaneConfinement's comment. Call whenever the grid, particle/trail
    // counts, or confinement change; cheap enough (O(particleCount)) to
    // call freely.
    void Reset(const VoxelGrid& grid, bool reverseFlow, int particleCount, int trailLength,
               const PlaneConfinement* confine = nullptr);

    // Advances every particle by `dtSeconds` of real (wall-clock) time —
    // internally capped at 0.1s regardless of what's passed in, so a
    // stalled frame's own much-larger elapsed time can't advect a
    // particle far enough out of the grid in one step to make it respawn
    // at the inlet on the very next call; see this method's .cpp comment
    // for why that's what "the stream resets right after a pause" was
    // actually caused by — sampling velocity (nearest-cell, the same
    // convention BuildVoxelCloudMesh's sampling uses) from `field` — the
    // 4-floats-per-cell layout LbmSolver::ReadbackField() returns — at
    // the particle's current position, and converting it to a real mm/s
    // displacement via `mmPerSecPerLatticeUnit` (pass
    // cellSize/LatticeDtSeconds(app) from main.cpp — equivalent to
    // LatticeSpeedToMmPerSecond(app, 1.0f), since that conversion is
    // linear in the lattice speed). A particle that leaves the grid
    // entirely, or wanders into a solid cell normally respawns at the
    // inlet face (or, confined, back onto the plane — see Reset()) with a
    // fresh random position and an empty trail — what keeps the stream
    // continuous rather than thinning out as particles reach the outlet.
    // `suspendRespawn`, when true, skips that respawn instead: the
    // particle just vanishes (empty trail, inactive) and stays that way
    // until a later Step() call with `suspendRespawn` false respawns it
    // again. Holding it true for a short real-time window (see main.cpp's
    // "Puff" button) stops new particles entering the domain while
    // whatever's already in flight keeps advecting and draining out the
    // outlet — the whole stream thins to nothing and then refills once
    // respawning resumes, reading as a single puff of smoke rather than a
    // continuous plume. `confine` — see PlaneConfinement's comment; must
    // match whatever Reset() was last called with, or newly-advected
    // points will snap onto a different plane than already-seeded ones
    // sit on.
    void Step(const VoxelGrid& grid, const std::vector<float>& field, bool reverseFlow,
              float mmPerSecPerLatticeUnit, float dtSeconds, bool suspendRespawn = false,
              const PlaneConfinement* confine = nullptr);

    // Instantly jumps every particle to a single fresh point (clearing
    // its trail, same as Reset()) at a uniformly random non-solid cell
    // across the WHOLE domain — or, with `confine` non-null, across the
    // whole plane it names — rather than only at the inlet face/plane
    // edge the way Reset()/Respawn() do. Call on demand (see main.cpp's
    // "Scatter particles" button) for an immediate full-extent shuffle of
    // the visualization, instead of waiting several Step() calls for
    // particles that only enter at the inlet to refill the domain.
    void ScatterRandom(const VoxelGrid& grid, const PlaneConfinement* confine = nullptr);

    // Slides every already-seeded particle (and its whole trail history,
    // not just its newest point) sideways onto a new plane position along
    // the SAME axis — overwriting every trail point's coordinate on
    // `confine.axis` to the new plane's world coordinate, leaving their
    // other two coordinates (and so the pattern/distribution they already
    // formed) untouched. For dragging the Slice position slider while
    // "Also show particles" is on: a full Reset() there would dump every
    // particle back to a single fresh point at the plane's edge, which
    // reads as the whole stream snapping back to the inlet on every drag
    // step — jarring, and exactly what made the slice position hard to
    // judge by eye. This instead reads as the existing particle cloud
    // simply following the plane as it moves, same as the colored
    // cross-section itself would if it were still being drawn. Only valid
    // when `confine.axis` matches whatever Reset()/ShiftPlane() last
    // confined to — changing the axis itself still needs a full Reset(),
    // since there's no sensible way to carry a particle's position across
    // to a differently-oriented plane.
    void ShiftPlane(const VoxelGrid& grid, const PlaneConfinement& confine);

    // Builds the trail geometry: each particle's consecutive trail points
    // become a short strip of small quads, one per segment, each facing
    // the camera by using `camForward` (the view's own forward vector —
    // GetBasis().forward) crossed with that segment's own direction as
    // its width axis, so every strip reads edge-on-correct from whatever
    // angle the camera is currently at rather than only from straight on.
    // Each segment is colored by Speed at its newer endpoint and faded by
    // position along the trail (full at the head, zero at the tail —
    // what makes it read as a streak rather than a rigid ribbon of
    // uniform opacity) — and, when `pressureAlphaEnabled` is true, ALSO
    // by Pressure there, mapped through the same [pressureAlphaLo,
    // pressureAlphaHi] window Voxel Cloud's alpha uses (see
    // FieldOverlay::BuildVoxelCloudMesh), multiplied into the same fade —
    // so the model's own wake/stagnation disturbance stays visible while
    // flow through undisturbed ambient fluid fades out, the same
    // convention. When false (see AppState::smokeTrailPressureAlphaEnabled's
    // comment for why that's the default), every segment instead gets a
    // fixed, fully-opaque-at-the-ceiling alpha before the trail-position
    // fade, same as if its pressure sample always fell at or above
    // `pressureAlphaHi`. `pressureXMeans`, when non-null, detrends that
    // pressure sample the same way FieldOverlay::ComputeXSliceMeans does
    // elsewhere (see AppState::detrendPressure in main.cpp) — read
    // regardless of `pressureAlphaEnabled`, since highlightNegativePressure
    // below still needs an actual (possibly detrended) pressure value.
    // `highlightNegativePressure`, when true, overrides a below-zero-
    // pressure segment's color/alpha to a fixed dark grey at 30% —
    // same convention and reasoning as FieldOverlay::BuildVoxelCloudMesh's
    // own parameter of the same name, and independent of
    // `pressureAlphaEnabled`: a different visual signal, not part of the
    // alpha window.
    OverlayMesh BuildMesh(const VoxelGrid& grid, const std::vector<float>& field, const Vec3& camForward,
                          float quadHalfSize, float pressureLo, float pressureHi, float pressureAlphaLo,
                          float pressureAlphaHi, const std::vector<float>* pressureXMeans,
                          bool highlightNegativePressure, bool pressureAlphaEnabled) const;

private:
    // trail.back() is the current (newest) position, trail.front() the
    // oldest still kept — capped at trailLength (see Step()).
    struct Particle {
        std::vector<Vec3> trail;
    };

    void Respawn(const VoxelGrid& grid, bool reverseFlow, const PlaneConfinement* confine, Particle& p);

    // Shared retry-avoiding-solid-cells logic behind both Respawn() and
    // ScatterRandom(): any of fixedX/fixedY/fixedZ that isn't -1 is held
    // fixed, the rest are drawn uniformly at random from the grid's full
    // extent on that axis. Falls back to the fixed coordinates (or the
    // domain's rough center, for an unconfined axis) if all 8 attempts
    // land on a solid cell — same trade-off the original single-site
    // Respawn() always made, just shared rather than duplicated.
    void PickCell(const VoxelGrid& grid, int fixedX, int fixedY, int fixedZ, int& outX, int& outY, int& outZ);

    std::vector<Particle> particles_;
    int trailLength_ = 10;
    std::mt19937 rng_{12345};
};
