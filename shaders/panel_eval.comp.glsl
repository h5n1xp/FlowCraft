#version 450
// 3D Lifting Panel Method field-evaluation kernel (GPU backend for PanelSolver).
// Evaluates the velocity and Bernoulli pressure field induced at each fluid
// cell center by the solved desingularized surface source panels and the
// trailing-edge Kutta horseshoe vortex wake strips.
//
// SDL_GPU SPIR-V compute binding convention:
//   set 0 = read-only storage buffers
//   set 1 = read-write storage buffers
//   set 2 = uniform buffers

layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

layout(set = 2, binding = 0) uniform PanelEvalParamsBlock {
    uint nx, ny, nz, numPanels;
    uint numWakeStrips, zOffset, slabZStart, slabZCount;
    float inletVelocity, reverseFlow, wakeLength, maxSpeedClamp;
} params;

struct PanelGPU {
    float sx, sy, sz, q;       // internal source position + q = sigma * area / (4*pi)
    float nx, ny, nz, epsSq;   // outward normal + smoothing core radius squared
};

struct HorseshoeGPU {
    float ax, ay, az, gammaOver4Pi;  // bound vortex endpoint A + Gamma / (4*pi)
    float bx, by, bz, coreSq;        // bound vortex endpoint B + vortex core radius squared
};

layout(std430, set = 0, binding = 0) readonly buffer SolidBuf { uint solidMask[]; };
layout(std430, set = 0, binding = 1) readonly buffer PanelBuf { PanelGPU panels[]; };
layout(std430, set = 0, binding = 2) readonly buffer WakeBuf  { HorseshoeGPU wakeStrips[]; };
layout(std430, set = 1, binding = 0) buffer MacroBuf { float macroOut[]; };

// Regularised Biot-Savart velocity (divided by Gamma / 4pi) for a directed
// vortex segment P1 -> P2 with Rosenhead-Moore core radius squared `coreSq`.
vec3 VortexSegmentUnit(vec3 pos, vec3 p1, vec3 p2, float coreSq) {
    vec3 r0 = p2 - p1;
    vec3 r1 = pos - p1;
    vec3 r2 = pos - p2;
    vec3 cr = cross(r1, r2);
    float denom = dot(cr, cr) + dot(r0, r0) * coreSq;
    if (denom <= 1e-12) return vec3(0.0);
    float term = dot(r0, r1) * inversesqrt(dot(r1, r1) + coreSq) -
                 dot(r0, r2) * inversesqrt(dot(r2, r2) + coreSq);
    return cr * (term / denom);
}

void main() {
    uvec3 gid = gl_GlobalInvocationID;
    uint localZ = gid.z + params.zOffset;
    if (gid.x >= params.nx || gid.y >= params.ny || localZ >= params.slabZCount) return;
    uint gz = params.slabZStart + localZ;
    if (gz >= params.nz) return;

    uint cell = (localZ * params.ny + gid.y) * params.nx + gid.x;
    uint globalCell = (gz * params.ny + gid.y) * params.nx + gid.x;

    if (solidMask[globalCell] != 0u) {
        macroOut[cell * 4u + 0u] = 0.0;
        macroOut[cell * 4u + 1u] = 0.0;
        macroOut[cell * 4u + 2u] = 0.0;
        macroOut[cell * 4u + 3u] = 0.0;
        return;
    }

    float flowSign = (params.reverseFlow != 0.0) ? -1.0 : 1.0;
    vec3 pos = vec3(float(gid.x) + 0.5, float(gid.y) + 0.5, float(gz) + 0.5);
    vec3 vel = vec3(flowSign * params.inletVelocity, 0.0, 0.0);

    for (uint i = 0u; i < params.numPanels; ++i) {
        PanelGPU p = panels[i];
        vec3 d = pos - vec3(p.sx, p.sy, p.sz);
        float r2 = dot(d, d) + p.epsSq;
        float invR3 = inversesqrt(r2) / r2;
        vel += d * (p.q * invR3);
    }

    vec3 wakeOffset = vec3(flowSign * params.wakeLength, 0.0, 0.0);
    for (uint s = 0u; s < params.numWakeStrips; ++s) {
        HorseshoeGPU h = wakeStrips[s];
        vec3 A = vec3(h.ax, h.ay, h.az);
        vec3 B = vec3(h.bx, h.by, h.bz);
        vec3 Awake = A + wakeOffset;
        vec3 Bwake = B + wakeOffset;
        vec3 unitV = VortexSegmentUnit(pos, Awake, A, h.coreSq) +
                     VortexSegmentUnit(pos, A, B, h.coreSq) +
                     VortexSegmentUnit(pos, B, Bwake, h.coreSq);
        vel += unitV * h.gammaOver4Pi;
    }

    float speedSq = dot(vel, vel);
    float maxSq = params.maxSpeedClamp * params.maxSpeedClamp;
    if (speedSq > maxSq && speedSq > 0.0) {
        vel *= params.maxSpeedClamp * inversesqrt(speedSq);
        speedSq = maxSq;
    }

    float uInfSq = params.inletVelocity * params.inletVelocity;
    float pressureDeviation = 0.5 * (uInfSq - speedSq);
    macroOut[cell * 4u + 0u] = 1.0 + 3.0 * pressureDeviation;
    macroOut[cell * 4u + 1u] = vel.x;
    macroOut[cell * 4u + 2u] = vel.y;
    macroOut[cell * 4u + 3u] = vel.z;
}
