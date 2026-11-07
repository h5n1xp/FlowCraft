#version 450
// Lattice-Boltzmann step kernel (GLSL port of the original MSL kernel, kept
// in sync with LbmReference::Step() — see LbmReference.h for the physics
// writeup and LbmSolver.h for the Z-slab decomposition).
//
// PORTABILITY NOTE — packed populations: the original kernel stored
// populations as `half` in a raw `device half*` buffer. HLSL/GLSL/Vulkan
// can't portably do 16-bit loads/stores into storage buffers (it needs
// optional extensions), and neighbouring threads writing adjacent 16-bit
// values that share a 32-bit word would race. Instead, two directions of the
// SAME cell are packed into one uint32 (low half = even direction, high half
// = odd direction) via packHalf2x16/unpackHalf2x16 — core GLSL everywhere.
// Each thread owns all the words of its own cell, so there are no races, and
// the memory footprint is unchanged (Q=19 -> 10 words = 40 bytes/cell vs the
// old 38). Word `k` of cell `c` lives at index  k * cellCount + c.
//
// SDL_GPU SPIR-V compute binding convention: set 0 = read-only storage
// buffers (in the order bound), set 1 = read-write storage buffers, set 2 =
// uniform buffers.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

layout(set = 2, binding = 0) uniform SimParamsBlock {
    uint nx, ny, nz, Q;
    float tau, inletVelocity;
    uint zOffset;
    float reverseFlow;
    uint slabZStart, slabZCount;
    uint prevCellCount, prevLastLayerBase;
    uint nextCellCount;
    float freeSlipY, freeSlipZ;
    float smagorinskyCs;
} params;

struct LatticeDirGPU {
    float ex, ey, ez, weight;
    int opposite;
    int mirrorY, mirrorZ;
    float pad0;
};

layout(std430, set = 0, binding = 0) readonly buffer LatticeBuf { LatticeDirGPU lattice[]; };
layout(std430, set = 0, binding = 1) readonly buffer SolidBuf { uint solidMask[]; };
layout(std430, set = 0, binding = 2) readonly buffer SdfBuf { float signedDistance[]; };
layout(std430, set = 0, binding = 3) readonly buffer PopInBuf { uint popIn[]; };
layout(std430, set = 0, binding = 4) readonly buffer PrevPopBuf { uint prevPopIn[]; };
layout(std430, set = 0, binding = 5) readonly buffer NextPopBuf { uint nextPopIn[]; };
layout(std430, set = 1, binding = 0) buffer PopOutBuf { uint popOut[]; };
layout(std430, set = 1, binding = 1) buffer MacroBuf { float macroOut[]; };

float Equilibrium(float ex, float ey, float ez, float weight, float rho, float ux, float uy, float uz) {
    float eu = ex * ux + ey * uy + ez * uz;
    float u2 = ux * ux + uy * uy + uz * uz;
    return weight * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
}

bool IsFinite(float v) { return !isnan(v) && !isinf(v); }

// Unpack direction `dir` (of the cell whose word 0 is at `base`, word stride
// `stride`) from a packed-half word.
float UnpackPop(uint word, uint dir) {
    vec2 pair = unpackHalf2x16(word);
    return (dir & 1u) == 0u ? pair.x : pair.y;
}

float ReadOwnPop(uint dir, uint stride, uint localCell) {
    return UnpackPop(popIn[(dir >> 1) * stride + localCell], dir);
}

// Fetches a population in direction `dir` at grid coordinates (gx, gy, gz)
// where gz is already known to be within the global domain and at most 1
// Z-layer away from this thread's own slab (every lattice direction has ez in
// {-1, 0, 1}). See the Z-slab comment in LbmSolver.h for the three cases.
float ReadPopAtGlobalZ(uint dir, uint gx, uint gy, uint gz, uint cellCount) {
    if (gz >= params.slabZStart && gz < params.slabZStart + params.slabZCount) {
        uint localNz = gz - params.slabZStart;
        uint localNcell = (localNz * params.ny + gy) * params.nx + gx;
        return UnpackPop(popIn[(dir >> 1) * cellCount + localNcell], dir);
    } else if (gz + 1u == params.slabZStart) {
        uint localNcell = params.prevLastLayerBase + (gy * params.nx + gx);
        return UnpackPop(prevPopIn[(dir >> 1) * params.prevCellCount + localNcell], dir);
    } else {
        uint localNcell = gy * params.nx + gx;  // next slab's local z = 0
        return UnpackPop(nextPopIn[(dir >> 1) * params.nextCellCount + localNcell], dir);
    }
}

// Writes all Q post-collision populations of this cell, packed two per word.
void StorePops(float v[19], uint Q, uint cellCount, uint cell) {
    uint words = (Q + 1u) >> 1;
    for (uint k = 0u; k < words; ++k) {
        float lo = v[2u * k];
        float hi = (2u * k + 1u < Q) ? v[2u * k + 1u] : 0.0;
        popOut[k * cellCount + cell] = packHalf2x16(vec2(lo, hi));
    }
}

void main() {
    uvec3 gid = gl_GlobalInvocationID;
    // localZ is this slab's own z (indexes popIn/popOut/macroOut); gz is the
    // domain-wide z (indexes solidMask, and is what every boundary/neighbour
    // check below actually reasons about).
    uint localZ = gid.z + params.zOffset;
    if (gid.x >= params.nx || gid.y >= params.ny || localZ >= params.slabZCount) return;
    uint gz = params.slabZStart + localZ;
    if (gz >= params.nz) return;  // defensive — slabZStart+slabZCount<=nz by construction

    int x = int(gid.x), y = int(gid.y), z = int(gz);
    uint cellCount = params.nx * params.ny * params.slabZCount;  // THIS slab's own word stride
    uint cell = (localZ * params.ny + gid.y) * params.nx + gid.x;  // THIS slab's own local flat index
    uint globalCell = (gz * params.ny + gid.y) * params.nx + gid.x;  // solidMask is global
    uint Q = params.Q;

    float outv[19];

    if (solidMask[globalCell] != 0u) {
        for (uint i = 0u; i < Q; ++i) outv[i] = lattice[i].weight;
        StorePops(outv, Q, cellCount, cell);
        // Solid cells have no fluid (rho, u) of their own, so their 4 macroOut
        // slots record the Wen et al. / Ladd momentum-exchange force exerted on
        // this solid cell by adjacent fluid cells (using the same Bouzidi
        // SDF-interpolated bounce-back populations the fluid cells compute).
        // Subtracting the quiescent reference 2*w_i per boundary link cancels
        // ambient hydrostatic pressure, so boundary cells report gauge
        // aerodynamic stress while summing to the identical net force over any
        // closed body. Slot 0 stores |F| / (6 * sum(w_i)); slots 1..3 store the
        // un-normalized (Fx, Fy, Fz) so summing across all solid cells yields
        // the exact discrete surface integral for Lift, Drag and Side force.
        float fx = 0.0, fy = 0.0, fz = 0.0, wSum = 0.0;
        float phiSolid = signedDistance[globalCell];
        int minReachZ = (params.slabZStart > 0u) ? int(params.slabZStart) - 1 : 0;
        int maxReachZ = (params.nextCellCount > 0u) ? int(params.slabZStart + params.slabZCount)
                                                    : int(params.slabZStart + params.slabZCount) - 1;
        for (uint i = 1u; i < Q; ++i) {
            LatticeDirGPU d = lattice[i];
            int nxf = x + int(d.ex), nyf = y + int(d.ey), nzf = z + int(d.ez);
            bool inBounds = nxf >= 0 && nyf >= 0 && nzf >= 0 && nxf < int(params.nx) && nyf < int(params.ny) &&
                            nzf < int(params.nz);
            if (!inBounds) continue;
            uint globalFluidCell = (uint(nzf) * params.ny + uint(nyf)) * params.nx + uint(nxf);
            if (solidMask[globalFluidCell] != 0u) continue;

            uint opp = uint(d.opposite);
            float plain = ReadPopAtGlobalZ(opp, uint(nxf), uint(nyf), uint(nzf), cellCount);
            float fBounced = plain;
            float phiFluid = signedDistance[globalFluidCell];
            if (IsFinite(phiFluid) && IsFinite(phiSolid)) {
                float denom = phiFluid - phiSolid;
                if (denom > 1e-4) {
                    float q = clamp(phiFluid / denom, 0.0, 1.0);
                    if (q < 0.5) {
                        int ffx = nxf + int(d.ex), ffy = nyf + int(d.ey), ffz = nzf + int(d.ez);
                        bool ffInBounds = ffx >= 0 && ffy >= 0 && ffz >= minReachZ && ffx < int(params.nx) &&
                                          ffy < int(params.ny) && ffz <= maxReachZ && ffz < int(params.nz);
                        if (ffInBounds) {
                            uint globalFFcell = (uint(ffz) * params.ny + uint(ffy)) * params.nx + uint(ffx);
                            if (solidMask[globalFFcell] == 0u) {
                                float ffOpp = ReadPopAtGlobalZ(opp, uint(ffx), uint(ffy), uint(ffz), cellCount);
                                fBounced = 2.0 * q * plain + (1.0 - 2.0 * q) * ffOpp;
                            }
                        }
                    } else {
                        float invTwoQ = 1.0 / (2.0 * q);
                        float fFluidI = ReadPopAtGlobalZ(i, uint(nxf), uint(nyf), uint(nzf), cellCount);
                        fBounced = invTwoQ * plain + (1.0 - invTwoQ) * fFluidI;
                    }
                }
            }
            float dp = (plain + fBounced) - 2.0 * d.weight;
            fx -= d.ex * dp;
            fy -= d.ey * dp;
            fz -= d.ez * dp;
            wSum += d.weight;
        }
        float forceMag = sqrt(fx * fx + fy * fy + fz * fz);
        macroOut[cell * 4u + 0u] = (wSum > 0.0) ? (forceMag / (6.0 * wSum)) : 0.0;
        macroOut[cell * 4u + 1u] = fx;
        macroOut[cell * 4u + 2u] = fy;
        macroOut[cell * 4u + 3u] = fz;
        return;
    }

    // Flow direction: normally the inlet is at x=0 and the flow pushes toward
    // +X; reverseFlow swaps which face is the inlet and negates the inlet
    // velocity. Nothing else about the kernel changes.
    bool reversed = params.reverseFlow != 0.0;
    int inletX = reversed ? int(params.nx) - 1 : 0;
    int outletX = reversed ? 0 : int(params.nx) - 1;
    float inletUx = reversed ? -params.inletVelocity : params.inletVelocity;

    if (x == inletX) {
        float rho = 1.0, ux = inletUx, uy = 0.0, uz = 0.0;
        for (uint i = 0u; i < Q; ++i) {
            LatticeDirGPU d = lattice[i];
            outv[i] = Equilibrium(d.ex, d.ey, d.ez, d.weight, rho, ux, uy, uz);
        }
        StorePops(outv, Q, cellCount, cell);
        macroOut[cell * 4u + 0u] = rho;
        macroOut[cell * 4u + 1u] = ux;
        macroOut[cell * 4u + 2u] = uy;
        macroOut[cell * 4u + 3u] = uz;
        return;
    }

    if (x == outletX) {
        // Same slab, same local z as this cell — the outlet's extrapolation
        // neighbour only ever differs in x, never z, so this never crosses a
        // slab boundary.
        uint neighborX = reversed ? uint(x) + 1u : uint(x) - 1u;
        uint neighborCell = (localZ * params.ny + gid.y) * params.nx + neighborX;
        float nrho = 0.0, nux = 0.0, nuy = 0.0, nuz = 0.0;
        for (uint i = 0u; i < Q; ++i) {
            LatticeDirGPU d = lattice[i];
            float fi = ReadOwnPop(i, cellCount, neighborCell);
            nrho += fi;
            nux += fi * d.ex;
            nuy += fi * d.ey;
            nuz += fi * d.ez;
        }
        // Density is extrapolated from the interior neighbour (zero-gradient /
        // "convective") with a tiny relaxation back toward the ambient
        // reference; see the long comment on kOutletRelax in the original
        // kernel history / LbmReference.cpp for why pure pinning reflects
        // waves and pure extrapolation drifts.
        const float kOutletRelax = 0.999;
        float rho = 1.0;
        if (nrho > 1e-6) {
            nux /= nrho;
            nuy /= nrho;
            nuz /= nrho;
            rho = 1.0 + kOutletRelax * (nrho - 1.0);
        }
        for (uint i = 0u; i < Q; ++i) {
            LatticeDirGPU d = lattice[i];
            outv[i] = Equilibrium(d.ex, d.ey, d.ez, d.weight, rho, nux, nuy, nuz);
        }
        StorePops(outv, Q, cellCount, cell);
        macroOut[cell * 4u + 0u] = rho;
        macroOut[cell * 4u + 1u] = nux;
        macroOut[cell * 4u + 2u] = nuy;
        macroOut[cell * 4u + 3u] = nuz;
        return;
    }

    float fpre[19];
    for (uint i = 0u; i < Q; ++i) {
        LatticeDirGPU d = lattice[i];
        int nxp = x - int(d.ex), nyp = y - int(d.ey), nzp = z - int(d.ez);
        bool inBounds = nxp >= 0 && nyp >= 0 && nzp >= 0 && nxp < int(params.nx) && nyp < int(params.ny) &&
                        nzp < int(params.nz);
        if (inBounds) {
            uint globalNcell = (uint(nzp) * params.ny + uint(nyp)) * params.nx + uint(nxp);
            if (solidMask[globalNcell] == 0u) {
                fpre[i] = ReadPopAtGlobalZ(i, uint(nxp), uint(nyp), uint(nzp), cellCount);
                continue;
            }
            // Solid model geometry — Bouzidi/Mei-Luo-Shyy linear interpolated
            // bounce-back using the SDF, matching
            // LbmReference::InterpolatedBounceBack line-by-line (falls back to
            // plain halfway bounce-back wherever the SDF data or further-fluid
            // neighbour isn't usable).
            uint opp = uint(d.opposite);
            float plain = ReadOwnPop(opp, cellCount, cell);
            float phiFluid = signedDistance[globalCell];
            float phiSolid = signedDistance[globalNcell];
            if (IsFinite(phiFluid) && IsFinite(phiSolid)) {
                float denom = phiFluid - phiSolid;
                if (denom > 1e-4) {
                    float q = clamp(phiFluid / denom, 0.0, 1.0);
                    if (q < 0.5) {
                        int ffx = x + int(d.ex), ffy = y + int(d.ey), ffz = z + int(d.ez);
                        bool ffInBounds = ffx >= 0 && ffy >= 0 && ffz >= 0 && ffx < int(params.nx) &&
                                          ffy < int(params.ny) && ffz < int(params.nz);
                        if (ffInBounds) {
                            uint globalFFcell = (uint(ffz) * params.ny + uint(ffy)) * params.nx + uint(ffx);
                            if (solidMask[globalFFcell] == 0u) {
                                float ffOpp = ReadPopAtGlobalZ(opp, uint(ffx), uint(ffy), uint(ffz), cellCount);
                                fpre[i] = 2.0 * q * plain + (1.0 - 2.0 * q) * ffOpp;
                                continue;
                            }
                        }
                    } else {
                        float invTwoQ = 1.0 / (2.0 * q);
                        fpre[i] = invTwoQ * plain + (1.0 - invTwoQ) * ReadOwnPop(i, cellCount, cell);
                        continue;
                    }
                }
            }
            fpre[i] = plain;
            continue;
        } else {
            // Domain edge (Y and/or Z out of range — nxp is always in range
            // here, the inlet/outlet branches above already handled every
            // x==inletX/outletX cell). See LbmReference.cpp's Step() for the
            // identical logic, including why a mixed corner (only one of the
            // two relevant walls free-slip) falls back to plain bounce-back.
            bool outY = nyp < 0 || nyp >= int(params.ny);
            bool outZ = nzp < 0 || nzp >= int(params.nz);
            bool mirrorOk = (!outY || params.freeSlipY != 0.0) && (!outZ || params.freeSlipZ != 0.0);
            if (mirrorOk) {
                uint dir = i;
                if (outY) dir = uint(d.mirrorY);
                if (outZ) dir = uint(lattice[dir].mirrorZ);
                fpre[i] = ReadOwnPop(dir, cellCount, cell);
                continue;
            }
        }
        uint opp = uint(d.opposite);
        fpre[i] = ReadOwnPop(opp, cellCount, cell);  // bounce-back, this slab's own local cell
    }

    float rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
    for (uint i = 0u; i < Q; ++i) {
        LatticeDirGPU d = lattice[i];
        rho += fpre[i];
        ux += fpre[i] * d.ex;
        uy += fpre[i] * d.ey;
        uz += fpre[i] * d.ez;
    }
    if (rho > 1e-6) {
        ux /= rho;
        uy /= rho;
        uz /= rho;
    }

    // Regularized collision (Latt & Chopard) — see LbmReference.h/.cpp for the
    // derivation. Must mirror LbmReference::Step()'s interior branch exactly.
    const float kCs2 = 1.0 / 3.0;
    float pixx = 0.0, piyy = 0.0, pizz = 0.0, pixy = 0.0, pixz = 0.0, piyz = 0.0;
    for (uint i = 0u; i < Q; ++i) {
        LatticeDirGPU d = lattice[i];
        float feq = Equilibrium(d.ex, d.ey, d.ez, d.weight, rho, ux, uy, uz);
        float fneq = fpre[i] - feq;
        fpre[i] = feq;  // fpre's raw value is never needed again — cache feq here for the loop below
        pixx += d.ex * d.ex * fneq;
        piyy += d.ey * d.ey * fneq;
        pizz += d.ez * d.ez * fneq;
        pixy += d.ex * d.ey * fneq;
        pixz += d.ex * d.ez * fneq;
        piyz += d.ey * d.ez * fneq;
    }

    // Smagorinsky LES sub-grid viscosity (Hou et al.): raises the local
    // relaxation time by an eddy viscosity proportional to the magnitude of
    // the non-equilibrium stress tensor Pi, so near-air viscosities (tau just
    // above 0.5) stay stable. Must mirror LbmReference::Step() exactly.
    float cs = params.smagorinskyCs;
    float piMag = sqrt(pixx * pixx + piyy * piyy + pizz * pizz + 2.0 * (pixy * pixy + pixz * pixz + piyz * piyz));
    float tauEff = 0.5 * (params.tau + sqrt(params.tau * params.tau + 18.0 * 1.41421356 * cs * cs * piMag / max(rho, 1e-6)));
    float omega = 1.0 / tauEff;
    for (uint i = 0u; i < Q; ++i) {
        LatticeDirGPU d = lattice[i];
        float qxx = d.ex * d.ex - kCs2, qyy = d.ey * d.ey - kCs2, qzz = d.ez * d.ez - kCs2;
        float qxy = d.ex * d.ey, qxz = d.ex * d.ez, qyz = d.ey * d.ez;
        float fneqReg = (d.weight * 4.5) *
                        (qxx * pixx + qyy * piyy + qzz * pizz + 2.0 * qxy * pixy + 2.0 * qxz * pixz + 2.0 * qyz * piyz);
        outv[i] = fpre[i] + (1.0 - omega) * fneqReg;  // fpre[i] holds feq here
    }
    StorePops(outv, Q, cellCount, cell);
    macroOut[cell * 4u + 0u] = rho;
    macroOut[cell * 4u + 1u] = ux;
    macroOut[cell * 4u + 2u] = uy;
    macroOut[cell * 4u + 3u] = uz;
}
