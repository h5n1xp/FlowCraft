#include "SmokeTrail.h"

#include <algorithm>
#include <cmath>

namespace {

// Nearest-cell lookup for a world-space point — same sampling convention
// BuildVoxelCloudMesh uses (sample the containing cell's own value, no
// interpolation). Returns false (coordinates unset) if the point has
// drifted outside the grid.
// Adds a random offset within the cell's own footprint along whichever
// grid axes are free (not pinned to an inlet/confine plane), instead of
// every particle landing on the literal cell center — otherwise every
// particle that picks the same (grid-y,grid-z) lands on EXACTLY the
// same point, which reads as particles clumping into a handful of
// discrete lines/blocks rather than spreading continuously. Most
// visible along whichever world axis maps to the grid dimension with
// the fewest cells (e.g. a model much longer than it is wide only
// offers a few dozen distinct lateral slots at typical resolutions —
// not a bug in how many cells there are, just that snapping exactly to
// each one's center made that cell count visible). The jitter is
// computed in grid-local terms, then run through the same x/y swap
// grid.CellCenter() itself applies (see VoxelGrid.h), so it lands in
// the correct world component regardless of which world axis each
// grid axis maps to.
Vec3 JitteredCellCenter(const VoxelGrid& grid, int x, int y, int z, bool xFixed, bool yFixed, bool zFixed,
                         std::mt19937& rng) {
    std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);
    float jx = xFixed ? 0.0f : jitter(rng) * grid.cellSize;
    float jy = yFixed ? 0.0f : jitter(rng) * grid.cellSize;
    float jz = zFixed ? 0.0f : jitter(rng) * grid.cellSize;
    return grid.CellCenter(x, y, z) + Vec3{jy, jx, jz};
}

bool CellAt(const VoxelGrid& grid, const Vec3& pos, int& outX, int& outY, int& outZ) {
    // Inverse of CellCenter's own mapping (VoxelGrid.h) — grid-x comes
    // from world Y and grid-y from world X, so converting a world
    // position back to grid indices needs the same swap, not a direct
    // per-component divide. Missing this meant outX/outY were being
    // bounds-checked against the WRONG dimension (nx, sized for the
    // long flow extent, against the narrow lateral coordinate, and vice
    // versa) — on an elongated domain the mismatched bound was wide
    // enough that a particle could travel far past the real edge of the
    // grid without InBounds ever noticing, never triggering Step()'s
    // respawn.
    Vec3 local = pos - grid.origin;
    outX = static_cast<int>(std::floor(local.y / grid.cellSize));
    outY = static_cast<int>(std::floor(local.x / grid.cellSize));
    outZ = static_cast<int>(std::floor(local.z / grid.cellSize));
    return grid.InBounds(outX, outY, outZ);
}

// Trilinearly interpolates field's velocity at a continuous world
// position, rather than nearest-cell sampling it the way CellAt's own
// lookup (used for exit/respawn checks, not this) does. Nearest-cell
// sampling makes velocity piecewise-constant across each cell — invisible
// at typical/high resolution (cells small enough nobody notices), but at
// low resolution a whole row of particles crossing the same big cell all
// move by the identical displacement that step, in lockstep with the
// grid's own cell boundaries — reading as visible banding aligned with
// the grid rather than smooth flow. Blending between the up to 8
// surrounding cells instead removes that. Clamped to the grid's valid
// range at the domain edges (no extrapolation past it); a solid cell
// among the 8 is read as-is, not special-cased — the solver already
// seeds every solid cell's own velocity at exactly 0 (see e.g.
// LbmReference::Step's solid branch), which is the physically correct
// value to blend toward approaching a no-slip wall anyway.
Vec3 SampleVelocityInterpolated(const VoxelGrid& grid, const std::vector<float>& field, const Vec3& pos) {
    Vec3 local = pos - grid.origin;
    // Same inverse mapping CellAt uses, but continuous (not floored) and
    // offset by -0.5 so an exact cell center lands on an exact integer —
    // matching CellCenter's own "+0.5" cell-center placement — rather
    // than on a cell's min corner.
    float gxf = local.y / grid.cellSize - 0.5f;
    float gyf = local.x / grid.cellSize - 0.5f;
    float gzf = local.z / grid.cellSize - 0.5f;

    int gx0 = static_cast<int>(std::floor(gxf));
    int gy0 = static_cast<int>(std::floor(gyf));
    int gz0 = static_cast<int>(std::floor(gzf));
    float tx = gxf - static_cast<float>(gx0);
    float ty = gyf - static_cast<float>(gy0);
    float tz = gzf - static_cast<float>(gz0);

    auto sample = [&](int x, int y, int z) -> Vec3 {
        size_t cell = grid.Index(std::clamp(x, 0, grid.nx - 1), std::clamp(y, 0, grid.ny - 1),
                                   std::clamp(z, 0, grid.nz - 1));
        // Solid cells store their surface momentum-exchange force in
        // field[cell*4 + 0..3], not fluid velocity — their true physical
        // velocity at a no-slip wall is zero.
        if (grid.solid[cell]) return Vec3{0.0f, 0.0f, 0.0f};
        // field's (ux,uy,uz) are grid-axis components — swap to world,
        // same convention the plain nearest-cell lookup this replaces
        // used (see Step()'s own comment on that swap).
        return Vec3{field[cell * 4 + 2], field[cell * 4 + 1], field[cell * 4 + 3]};
    };

    Vec3 c000 = sample(gx0, gy0, gz0), c100 = sample(gx0 + 1, gy0, gz0);
    Vec3 c010 = sample(gx0, gy0 + 1, gz0), c110 = sample(gx0 + 1, gy0 + 1, gz0);
    Vec3 c001 = sample(gx0, gy0, gz0 + 1), c101 = sample(gx0 + 1, gy0, gz0 + 1);
    Vec3 c011 = sample(gx0, gy0 + 1, gz0 + 1), c111 = sample(gx0 + 1, gy0 + 1, gz0 + 1);

    Vec3 c00 = c000 + (c100 - c000) * tx;
    Vec3 c10 = c010 + (c110 - c010) * tx;
    Vec3 c01 = c001 + (c101 - c001) * tx;
    Vec3 c11 = c011 + (c111 - c011) * tx;

    Vec3 c0 = c00 + (c10 - c00) * ty;
    Vec3 c1 = c01 + (c11 - c01) * ty;

    return c0 + (c1 - c0) * tz;
}

// Trilinearly interpolates fluid pressure deviation (rho - 1) / 3 at a
// continuous world position, optionally subtracting the per-X-slice mean
// when channel pressure detrending is enabled.
float SamplePressureInterpolated(const VoxelGrid& grid, const std::vector<float>& field, const Vec3& pos,
                                 const std::vector<float>* pressureXMeans) {
    Vec3 local = pos - grid.origin;
    float gxf = local.y / grid.cellSize - 0.5f;
    float gyf = local.x / grid.cellSize - 0.5f;
    float gzf = local.z / grid.cellSize - 0.5f;

    int gx0 = static_cast<int>(std::floor(gxf));
    int gy0 = static_cast<int>(std::floor(gyf));
    int gz0 = static_cast<int>(std::floor(gzf));
    float tx = gxf - static_cast<float>(gx0);
    float ty = gyf - static_cast<float>(gy0);
    float tz = gzf - static_cast<float>(gz0);

    auto sampleP = [&](int x, int y, int z) -> float {
        int cx = std::clamp(x, 0, grid.nx - 1);
        int cy = std::clamp(y, 0, grid.ny - 1);
        int cz = std::clamp(z, 0, grid.nz - 1);
        size_t cell = grid.Index(cx, cy, cz);
        if (grid.solid[cell]) return 0.0f;
        float p = FieldOverlay::SampleFieldValue(field, cell, FieldColorMode::Pressure);
        if (pressureXMeans && cx < static_cast<int>(pressureXMeans->size())) {
            p -= (*pressureXMeans)[cx];
        }
        return p;
    };

    float c000 = sampleP(gx0, gy0, gz0), c100 = sampleP(gx0 + 1, gy0, gz0);
    float c010 = sampleP(gx0, gy0 + 1, gz0), c110 = sampleP(gx0 + 1, gy0 + 1, gz0);
    float c001 = sampleP(gx0, gy0, gz0 + 1), c101 = sampleP(gx0 + 1, gy0, gz0 + 1);
    float c011 = sampleP(gx0, gy0 + 1, gz0 + 1), c111 = sampleP(gx0 + 1, gy0 + 1, gz0 + 1);

    float c00 = c000 + (c100 - c000) * tx;
    float c10 = c010 + (c110 - c010) * tx;
    float c01 = c001 + (c101 - c001) * tx;
    float c11 = c011 + (c111 - c011) * tx;

    float c0 = c00 + (c10 - c00) * ty;
    float c1 = c01 + (c11 - c01) * ty;

    return c0 + (c1 - c0) * tz;
}

}  // namespace

void SmokeTrailSystem::PickCell(const VoxelGrid& grid, int fixedX, int fixedY, int fixedZ, int& outX, int& outY,
                                  int& outZ) {
    std::uniform_int_distribution<int> xd(0, std::max(grid.nx - 1, 0));
    std::uniform_int_distribution<int> yd(0, std::max(grid.ny - 1, 0));
    std::uniform_int_distribution<int> zd(0, std::max(grid.nz - 1, 0));
    outX = fixedX >= 0 ? fixedX : grid.nx / 2;
    outY = fixedY >= 0 ? fixedY : grid.ny / 2;
    outZ = fixedZ >= 0 ? fixedZ : grid.nz / 2;
    for (int attempt = 0; attempt < 8; ++attempt) {
        int tx = fixedX >= 0 ? fixedX : xd(rng_);
        int ty = fixedY >= 0 ? fixedY : yd(rng_);
        int tz = fixedZ >= 0 ? fixedZ : zd(rng_);
        if (!grid.IsSolid(tx, ty, tz)) {
            outX = tx;
            outY = ty;
            outZ = tz;
            return;
        }
    }
}

void SmokeTrailSystem::Respawn(const VoxelGrid& grid, bool reverseFlow, const PlaneConfinement* confine,
                                 Particle& p) {
    // Unconfined: x is the inlet face, (y,z) random. Confined: whichever
    // of x/y/z `confine->axis` names is fixed at the plane's index
    // instead, the other two behave exactly as unconfined (one of them
    // still the inlet face, unless axis itself IS x, in which case
    // there's no inlet face left to speak of — the whole plane already
    // sits at a single x — and both y and z are random).
    int fixedX = confine && confine->axis == SliceAxis::X ? std::clamp(confine->index, 0, grid.nx - 1)
                                                            : (reverseFlow ? grid.nx - 1 : 0);
    int fixedY = confine && confine->axis == SliceAxis::Y ? std::clamp(confine->index, 0, grid.ny - 1) : -1;
    int fixedZ = confine && confine->axis == SliceAxis::Z ? std::clamp(confine->index, 0, grid.nz - 1) : -1;

    int x, y, z;
    PickCell(grid, fixedX, fixedY, fixedZ, x, y, z);
    p.trail.clear();
    // x is always pinned here (the inlet face, or the confine plane if
    // confine->axis==X) — only y/z are ever free, and only when not
    // separately pinned by confine.
    p.trail.push_back(JitteredCellCenter(grid, x, y, z, true, fixedY < 0, fixedZ < 0, rng_));
}

void SmokeTrailSystem::ScatterRandom(const VoxelGrid& grid, const PlaneConfinement* confine) {
    // Unlike Respawn(), x is never pinned to the inlet face here — every
    // axis `confine` doesn't fix is randomized across the cell's full
    // extent, for an instant shuffle across the whole domain/plane.
    int fixedX = confine && confine->axis == SliceAxis::X ? std::clamp(confine->index, 0, grid.nx - 1) : -1;
    int fixedY = confine && confine->axis == SliceAxis::Y ? std::clamp(confine->index, 0, grid.ny - 1) : -1;
    int fixedZ = confine && confine->axis == SliceAxis::Z ? std::clamp(confine->index, 0, grid.nz - 1) : -1;

    for (Particle& p : particles_) {
        int x, y, z;
        PickCell(grid, fixedX, fixedY, fixedZ, x, y, z);
        p.trail.clear();
        p.trail.push_back(JitteredCellCenter(grid, x, y, z, fixedX >= 0, fixedY >= 0, fixedZ >= 0, rng_));
    }
}

void SmokeTrailSystem::ShiftPlane(const VoxelGrid& grid, const PlaneConfinement& confine) {
    // Routed through grid.CellCenter() rather than hand-rolling
    // origin+index*cellSize per axis (an earlier version of this did,
    // straight from grid.origin.x for SliceAxis::X) — that bypassed
    // CellCenter's own grid-x-is-world-Y/grid-y-is-world-X mapping (see
    // its comment in VoxelGrid.h) entirely, so a SliceAxis::X plane
    // (fixed grid-x) was being nailed to a WORLD-X coordinate instead of
    // the world-Y one it actually sits at. Passing 0 for the two axes
    // that don't matter is fine: CellCenter's x/y/z outputs each depend
    // on exactly one of the three grid indices.
    switch (confine.axis) {
        case SliceAxis::X: {
            int clamped = std::clamp(confine.index, 0, grid.nx - 1);
            float coord = grid.CellCenter(clamped, 0, 0).y;
            for (Particle& p : particles_)
                for (Vec3& v : p.trail) v.y = coord;
            break;
        }
        case SliceAxis::Y: {
            int clamped = std::clamp(confine.index, 0, grid.ny - 1);
            float coord = grid.CellCenter(0, clamped, 0).x;
            for (Particle& p : particles_)
                for (Vec3& v : p.trail) v.x = coord;
            break;
        }
        case SliceAxis::Z: {
            int clamped = std::clamp(confine.index, 0, grid.nz - 1);
            float coord = grid.CellCenter(0, 0, clamped).z;
            for (Particle& p : particles_)
                for (Vec3& v : p.trail) v.z = coord;
            break;
        }
    }
}

void SmokeTrailSystem::Reset(const VoxelGrid& grid, bool reverseFlow, int particleCount, int trailLength,
                              const PlaneConfinement* confine) {
    trailLength_ = std::max(trailLength, 2);
    particles_.assign(static_cast<size_t>(std::max(particleCount, 0)), Particle{});
    // The initial population is scattered across the WHOLE flow axis (see
    // ScatterRandom) rather than seeded at the inlet face via Respawn()
    // the way every *ongoing* mid-stream respawn still is — a real
    // continuous dye/smoke source would already have filled the domain
    // by the time anyone's looking at it, so this makes the very first
    // frame already read as a mature, fully-developed stream instead of
    // every particle starting flush against the inlet and needing many
    // seconds to visibly spread out and fill the domain (worse still at
    // the default 0.1x smokeTrailSpeedScale than it was at 1.0x). Reading
    // as "particles bunched on one side, then correct after a few
    // iterations" was this exact transient, not a bug in Step() itself —
    // reverseFlow is unused here as a result: it only matters for
    // Respawn()'s inlet re-entry, meaningless once x is random too.
    (void)reverseFlow;
    ScatterRandom(grid, confine);
}

void SmokeTrailSystem::Step(const VoxelGrid& grid, const std::vector<float>& field, bool reverseFlow,
                             float mmPerSecPerLatticeUnit, float dtSeconds, bool suspendRespawn,
                             const PlaneConfinement* confine) {
    if (field.size() != grid.CellCount() * 4 || dtSeconds <= 0.0f) return;

    // Caps how far a single Step() call can advect a particle, regardless
    // of how large `dtSeconds` is. Without this, a stalled frame (the UI
    // thread blocked for however long by anything — a slow readback, a
    // busy task, a one-off OS scheduling hiccup) feeds its own whole real
    // elapsed duration in as dtSeconds on the very next frame, so `next`
    // below can land far outside the grid in one step. That by itself
    // isn't visible yet — but THIS frame's out-of-bounds `next` becomes
    // NEXT frame's `cur`, which fails the in-bounds check just below and
    // respawns at the inlet — reading as the whole stream suddenly
    // resetting right after a pause, even though nothing here ever
    // intended a reset. Clamping instead makes a stall read as the
    // particles briefly "catching up" by one bounded-size step, same as
    // an ordinary frame, however long the real gap actually was.
    constexpr float kMaxDtSeconds = 0.1f;
    dtSeconds = std::min(dtSeconds, kMaxDtSeconds);

    for (Particle& p : particles_) {
        if (p.trail.empty()) {
            if (!suspendRespawn) Respawn(grid, reverseFlow, confine, p);
            continue;
        }
        Vec3 cur = p.trail.back();

        int cx, cy, cz;
        if (!CellAt(grid, cur, cx, cy, cz) || grid.IsSolid(cx, cy, cz)) {
            if (suspendRespawn) {
                p.trail.clear();  // vanish instead of respawning — see Step()'s header comment
            } else {
                Respawn(grid, reverseFlow, confine, p);
            }
            continue;
        }

        // Trilinearly interpolated, not the cell's own single value — see
        // SampleVelocityInterpolated's comment for why (visible grid-
        // aligned banding at low resolution otherwise). (cx,cy,cz)/cell
        // above are only for the exit/respawn check just above, not used
        // for the velocity itself.
        Vec3 vel = SampleVelocityInterpolated(grid, field, cur);
        // Zeroing the confined axis's velocity component before
        // integrating keeps `cur`'s (already-on-the-plane, inductively —
        // see Respawn()) coordinate on that axis exactly unchanged, i.e.
        // `next` stays on the same plane too — cheaper than, and
        // equivalent to, advecting in full 3D and snapping back after.
        // confine->axis is a SliceAxis, i.e. a GRID axis (SliceAxis::X
        // means fixed grid-x, world Y) — so it zeroes the world
        // component CellCenter maps THAT grid axis to, not the
        // identically-named world component.
        if (confine) {
            switch (confine->axis) {
                case SliceAxis::X: vel.y = 0.0f; break;
                case SliceAxis::Y: vel.x = 0.0f; break;
                case SliceAxis::Z: vel.z = 0.0f; break;
            }
        }
        Vec3 next = cur + vel * (mmPerSecPerLatticeUnit * dtSeconds);

        p.trail.push_back(next);
        if (static_cast<int>(p.trail.size()) > trailLength_) p.trail.erase(p.trail.begin());
    }
}

OverlayMesh SmokeTrailSystem::BuildMesh(const VoxelGrid& grid, const std::vector<float>& field,
                                          const Vec3& camForward, float quadHalfSize, float pressureLo,
                                          float pressureHi, float pressureAlphaLo, float pressureAlphaHi,
                                          const std::vector<float>* pressureXMeans,
                                          bool highlightNegativePressure, bool pressureAlphaEnabled) const {
    (void)highlightNegativePressure;
    OverlayMesh mesh;
    if (field.size() != grid.CellCount() * 4) return mesh;

    float invPressureRange = 1.0f / std::max(pressureHi - pressureLo, 1e-9f);
    float invAlphaRange = 1.0f / std::max(pressureAlphaHi - pressureAlphaLo, 1e-9f);
    constexpr Vec3 kNaNColor{1.0f, 0.1f, 1.0f};  // same convention as Slice/Voxel Cloud
    constexpr float kNaNAlpha = 0.6f;
    constexpr float kAlphaMax = 0.85f;

    struct TrailNode {
        Vec3 pos;
        Vec3 offset;
        Vec3 color;
        float alpha;
        bool valid;
    };
    std::vector<TrailNode> nodes;

    for (const Particle& p : particles_) {
        size_t n = p.trail.size();
        if (n < 2) continue;

        nodes.resize(n);
        for (size_t k = 0; k < n; ++k) {
            const Vec3& pos = p.trail[k];
            int cx, cy, cz;
            if (!CellAt(grid, pos, cx, cy, cz) || grid.IsSolid(cx, cy, cz)) {
                nodes[k].valid = false;
                continue;
            }

            // Smooth tangent along the trail for continuous ribbon joints
            Vec3 tangent = (k == 0)       ? (p.trail[1] - p.trail[0])
                           : (k == n - 1) ? (p.trail[n - 1] - p.trail[n - 2])
                                          : (p.trail[k + 1] - p.trail[k - 1]);
            float tanLen = Length(tangent);
            if (tanLen < 1e-7f) {
                nodes[k].valid = false;
                continue;
            }
            tangent = tangent * (1.0f / tanLen);
            Vec3 widthDir = Normalize(Cross(camForward, tangent));
            if (Length(widthDir) < 1e-6f) {
                nodes[k].valid = false;
                continue;
            }

            // k == 0 is the oldest tail (thin & transparent); k == n - 1 is
            // the leading head (wider & opaque) so flow direction and speed
            // (segment length) are immediately legible.
            float ageT = static_cast<float>(k) / static_cast<float>(n - 1);
            float widthScale = 0.15f + 0.85f * ageT;

            float pressure = SamplePressureInterpolated(grid, field, pos, pressureXMeans);
            Vec3 color;
            float alpha;
            if (!std::isfinite(pressure)) {
                color = kNaNColor;
                alpha = kNaNAlpha * ageT;
            } else {
                float pressureT = std::clamp((pressure - pressureLo) * invPressureRange, 0.0f, 1.0f);
                color = FieldOverlay::Colormap(pressureT);
                float alphaT =
                    pressureAlphaEnabled ? std::clamp((pressure - pressureAlphaLo) * invAlphaRange, 0.0f, 1.0f) : 1.0f;
                // Smooth quadratic fade from 0 at the tail to kAlphaMax at the head
                alpha = alphaT * kAlphaMax * (ageT * ageT);
            }

            nodes[k] = TrailNode{pos, widthDir * (quadHalfSize * widthScale), color, alpha, true};
        }

        for (size_t i = 1; i < n; ++i) {
            const TrailNode& a = nodes[i - 1];
            const TrailNode& b = nodes[i];
            if (!a.valid || !b.valid) continue;
            if (a.alpha <= 1e-4f && b.alpha <= 1e-4f) continue;

            uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back({a.pos - a.offset, a.color, a.alpha});
            mesh.vertices.push_back({a.pos + a.offset, a.color, a.alpha});
            mesh.vertices.push_back({b.pos + b.offset, b.color, b.alpha});
            mesh.vertices.push_back({b.pos - b.offset, b.color, b.alpha});
            mesh.indices.push_back(base + 0);
            mesh.indices.push_back(base + 1);
            mesh.indices.push_back(base + 2);
            mesh.indices.push_back(base + 0);
            mesh.indices.push_back(base + 2);
            mesh.indices.push_back(base + 3);
        }

        // Add a pointed leading-edge arrow tip at the newest head node so
        // the direction of travel is unmistakable even on short trails.
        const TrailNode& head = nodes[n - 1];
        const TrailNode& prev = nodes[n - 2];
        if (head.valid && prev.valid && head.alpha > 1e-3f) {
            Vec3 headDir = Normalize(head.pos - prev.pos);
            float stepLen = Length(head.pos - prev.pos);
            if (stepLen > 1e-6f) {
                float tipLen = std::min(quadHalfSize * 2.2f, stepLen * 0.9f);
                Vec3 tipPos = head.pos + headDir * tipLen;
                Vec3 headFlange = head.offset * 1.35f;
                uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
                mesh.vertices.push_back({head.pos - headFlange, head.color, head.alpha});
                mesh.vertices.push_back({head.pos + headFlange, head.color, head.alpha});
                mesh.vertices.push_back({tipPos, head.color, head.alpha});
                mesh.indices.push_back(base + 0);
                mesh.indices.push_back(base + 1);
                mesh.indices.push_back(base + 2);
            }
        }
    }

    return mesh;
}
