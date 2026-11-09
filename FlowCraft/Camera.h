#pragma once

#include <algorithm>
#include <cmath>

#include "FcMath.h"

// Simple trackball-style orbit camera: drag to rotate around a fixed
// target, scroll to zoom. Good enough to inspect a loaded model from any
// angle, which is the near-term goal; a free-fly camera can replace this
// later once the CFD overlay needs it.
class OrbitCamera {
public:
    void Frame(const Bounds& bounds) {
        target = bounds.Center();
        sceneRadius = std::max(Length(bounds.Size()) * 0.5f, 0.001f);
        distance = sceneRadius * 2.5f;
        minDistance = sceneRadius * 0.05f;
        maxDistance = sceneRadius * 20.0f;
    }

    void OnDrag(float dxPixels, float dyPixels) {
        constexpr float kSensitivity = 0.005f;
        float dx = invertX ? -dxPixels : dxPixels;
        float dy = invertY ? -dyPixels : dyPixels;
        yaw -= dx * kSensitivity;
        pitch = std::clamp(pitch + dy * kSensitivity, -1.5f, 1.5f);
    }

    void OnScroll(float amount) {
        float effectiveAmount = invertScrollZoom ? -amount : amount;
        distance = std::clamp(distance * std::pow(0.9f, effectiveAmount), minDistance, maxDistance);
    }

    // Z-up, X-forward world convention (see main.cpp's kYawAxis/kPitchAxis
    // comment) — pitch is the Z (vertical) component; yaw=0 faces along X
    // (so X is the depth/"out of screen" axis at the default angle,
    // matching Fusion 360's own convention for an RC-model body frame:
    // X=forward, Y=lateral, Z=up), sweeping toward Y as yaw increases.
    // Was Y-up with X/Y swapped in the horizontal terms before this;
    // yaw/pitch's own meaning (sweep angle / elevation angle) is
    // unchanged, only which world axis each one actually moves.
    Vec3 Eye() const {
        return target + Vec3(std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), std::sin(pitch)) * distance;
    }

    // Builds the view matrix directly (same math Mat4::LookAt uses) rather
    // than calling that generic helper, because LookAt's own fixed
    // Cross(up, forward) ordering gives the *opposite* screen-right
    // convention from GetBasis() below — using it here would make the
    // actual rendered view a mirror image of what GetBasis() (and so the
    // orientation gizmo/scale ruler) shows. See GetBasis()'s comment for
    // why this order is the one that matches Fusion 360's own convention.
    Mat4 ViewMatrix() const {
        Vec3 eye = Eye();
        Vec3 z = Normalize(target - eye);
        Vec3 x = Normalize(Cross(z, Vec3{0.0f, 0.0f, 1.0f}));
        Vec3 y = Cross(x, z);

        Mat4 r;
        r.m[0] = x.x;
        r.m[1] = y.x;
        r.m[2] = z.x;
        r.m[3] = 0.0f;
        r.m[4] = x.y;
        r.m[5] = y.y;
        r.m[6] = z.y;
        r.m[7] = 0.0f;
        r.m[8] = x.z;
        r.m[9] = y.z;
        r.m[10] = z.z;
        r.m[11] = 0.0f;
        r.m[12] = -Dot(x, eye);
        r.m[13] = -Dot(y, eye);
        r.m[14] = -Dot(z, eye);
        r.m[15] = 1.0f;
        return r;
    }

    // Near/far clip planes sized to the current zoom level and model scale,
    // not a fixed generic range. A fixed huge range (e.g. 0.01 to 10000)
    // wastes almost all of the depth buffer's precision once the camera is
    // more than a few units out, causing z-fighting between close-together
    // triangles that gets visibly worse the farther the camera sits —
    // exactly what a fixed range produces when orbiting out to frame a
    // whole model.
    struct ClipPlanes {
        float nearZ, farZ;
    };
    ClipPlanes NearFar() const {
        float nearZ = std::max(distance - sceneRadius * 1.5f, distance * 0.01f);
        nearZ = std::max(nearZ, 1e-4f);
        float farZ = distance + sceneRadius * 1.5f;
        return {nearZ, farZ};
    }

    // Camera-space right/up/forward basis, in world-space coordinates —
    // the same basis ViewMatrix() is built from. Used by the orientation
    // gizmo to project world axes into screen space.
    struct Basis {
        Vec3 right, up, forward;
    };
    Basis GetBasis() const {
        Vec3 forward = Normalize(target - Eye());
        // Cross(forward, worldUp) rather than Cross(worldUp, forward) —
        // the argument order (and up's below, to match) is what decides
        // whether +Y reads as screen-right or screen-left at the default
        // view (forward=-X, worldUp=+Z): this order puts +Y on the
        // right, matching Fusion 360's own convention for an RC-model
        // body frame (X=forward, Y=right, Z=up) the rest of this app's
        // world axes were already set up to match. The other order
        // (still right-handed, just the opposite screen-mapping
        // convention) is what main.cpp's camera-view comments described
        // before this.
        Vec3 right = Normalize(Cross(forward, Vec3{0.0f, 0.0f, 1.0f}));
        Vec3 up = Cross(right, forward);
        return {right, up, forward};
    }

    Vec3 target{0.0f, 0.0f, 0.0f};
    // yaw=0, pitch=0 faces along X (the flow axis, with reverseFlow at
    // its own default of true — see main.cpp's "Front" View button) —
    // the default starting angle.
    float yaw = 0.0f;
    float pitch = 0.0f;
    float distance = 5.0f;
    float minDistance = 0.01f;
    float maxDistance = 1000.0f;
    float sceneRadius = 2.0f;
    bool invertX = false;
    bool invertY = false;
    bool invertScrollZoom = false;
};
