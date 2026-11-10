#pragma once

#include <SDL3/SDL_gpu.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "VoxelGrid.h"

// 3D Lifting Panel Method (Boundary Element Method) solver with both GPU
// (SDL_GPU compute shader `shaders/panel_eval.comp.glsl`) and multithreaded
// CPU backends.
//
// Physics model:
//  1. Desingularized 3D Source Panels distributed over the model's SDF-
//     projected surface boundary (clustered into up to ~1536 surface panels,
//     preserving upper/lower surface normal separation so thin wings never
//     collapse into one sheet). Enforces the Neumann no-penetration boundary
//     condition (U_inf + u_src + u_wake) . n_i = 0 at every panel centroid.
//  2. Spanwise Lifting Strips with Trailing-Edge Kutta Wake (Horseshoe
//     Vortices): divides the spanwise (grid-Y) extent of the model into
//     lifting strips, each carrying a quarter-chord bound vortex segment
//     A -> B and two trailing wake vortex filaments shedding downstream from
//     the trailing edge to infinity. Enforces the 3/4-chord Kutta tangency
//     condition (U_inf + u_src + u_wake) . n_camber = 0 to solve for the
//     spanwise circulation distribution Gamma(y), generating real aerodynamic
//     Lift, wingtip vortices, wake downwash, and Induced Drag.
//  3. 3D Volume Field Evaluation: evaluates (rho, ux, uy, uz) at every fluid
//     cell center in the VoxelGrid (via GPU compute shader in PanelGPU mode
//     or multithreaded CPU in PanelCPU mode) and populates exposed solid
//     boundary cells with surface stress and aerodynamic force components
//     (stress, fx, fy, fz) so every existing FlowCraft visualization (Slice
//     planes, Voxel Cloud, Smoke Trail particles, Airframe Stress Map, Lift
//     Arrow, and Telemetry HUD) works identically.
class PanelSolver {
public:
    struct Params {
        float inletVelocity = 0.1f;  // lattice-unit reference speed (see kLatticeVelocityScale)
        bool reverseFlow = false;    // false = flow toward +X; true = flow toward -X
    };

    struct Stats {
        float avgDensity = 1.0f, minDensity = 1.0f, maxDensity = 1.0f;
        float avgSpeed = 0.0f, maxSpeed = 0.0f;
        bool hasNaN = false;
        bool valid = false;
    };

    struct PanelGPU {
        float sx, sy, sz, q;       // internal source position + q = sigma * area / (4*pi)
        float nx, ny, nz, epsSq;   // outward unit normal + smoothing core radius squared
    };

    struct HorseshoeGPU {
        float ax, ay, az, gammaOver4Pi;  // bound vortex endpoint A + Gamma / (4*pi)
        float bx, by, bz, coreSq;        // bound vortex endpoint B + vortex core radius squared
    };

    ~PanelSolver() { Shutdown(); }

    // Initializes the surface panels and trailing-edge Kutta wake strips from
    // `grid`. When `useGpu` is true, also creates the GPU compute pipeline
    // and Z-slab macroOut buffers on `device`.
    bool Init(SDL_GPUDevice* device, bool useGpu, const VoxelGrid& grid, const Params& params);
    void Shutdown();

    const std::string& LastError() const { return lastError_; }
    bool IsUsingGpu() const { return useGpu_; }

    void Reset(const Params& params);
    void SetParams(const Params& params);

    // Runs `sweeps` Gauss-Seidel/SOR relaxation iterations on the coupled
    // source-panel + Kutta horseshoe-vortex system. In GPU mode, also uploads
    // the updated singularity strengths and dispatches the 3D field evaluation
    // compute shader across the domain.
    void Step(int sweeps, std::atomic<float>* progress = nullptr);

    // Paced Z-chunk GPU evaluation for the background worker thread when
    // running in PanelGPU mode (mirrors LbmSolver::StepRange).
    void StepSurfaceOnly(int sweeps);
    void EvaluateFieldRangeGPU(SDL_GPUCommandBuffer* cmd, int zStart, int zCount);

    std::vector<float> ReadbackField();
    void ReadbackFieldRangeGPU(int zStart, int zCount, std::vector<float>& outField);
    void FinalizeReadbackField(std::vector<float>& field) const;

    Stats ComputeStats();
    Stats ComputeStatsFromField(const std::vector<float>& field) const;

    int Nx() const { return nx_; }
    int Ny() const { return ny_; }
    int Nz() const { return nz_; }
    size_t NumPanels() const { return panels_.size(); }
    size_t NumWakeStrips() const { return wakeStrips_.size(); }

    // Aero Polar Sweep (XFLR5-style multi-point polar generation)
    struct PolarPoint {
        float alphaDeg = 0.0f;
        float cl = 0.0f;
        float cd = 0.0f;
        float cdi = 0.0f;
        float cd0 = 0.0f;
        float cm = 0.0f;
        float ld = 0.0f;
        float liftN = 0.0f;
        float dragN = 0.0f;
        float cpX_mm = 0.0f;
    };

    struct SpanStation {
        float yGrid = 0.0f;       // spanwise grid coordinate (0 .. ny-1)
        float yMm = 0.0f;         // spanwise coordinate relative to model span center (mm)
        float chordMm = 0.0f;     // local chord length (mm)
        float clLocal = 0.0f;     // local 3D section lift coefficient at reference AoA
        float clElliptic = 0.0f;  // ideal elliptical lift distribution at same total CL
        float gammaNorm = 0.0f;   // normalized circulation Gamma / Gamma_max
    };

    struct PolarSweepResult {
        std::vector<PolarPoint> points;
        std::vector<SpanStation> spanStations;
        float spanRefAlphaDeg = 4.0f;
        float clAlphaPerDeg = 0.0f;
        float clAlphaPerRad = 0.0f;
        float alphaZeroLiftDeg = 0.0f;
        float maxLD = 0.0f;
        float alphaAtMaxLD = 0.0f;
        float minCD0 = 0.0f;
        float oswaldEfficiency = 0.85f;
        float neutralPointX_mm = 0.0f;
        float refAreaM2 = 0.0f;
        float refChordM = 0.0f;
        float refSpanM = 0.0f;
        float aspectRatio = 0.0f;
        bool valid = false;
    };

    struct SectionProfilePoint {
        float xNorm = 0.0f;      // 0 (LE) to 1 (TE)
        float xMm = 0.0f;        // chordwise coordinate from LE (mm)
        float zUpperMm = 0.0f;   // upper skin height relative to LE (mm)
        float zLowerMm = 0.0f;   // lower skin height relative to LE (mm)
        float zCamberMm = 0.0f;  // mean camber line height relative to LE (mm)
        float zChordMm = 0.0f;   // straight chord line height relative to LE (mm)
        Vec3 worldUpper{0.0f, 0.0f, 0.0f};
        Vec3 worldLower{0.0f, 0.0f, 0.0f};
    };

    struct SectionPolarResult {
        PolarSweepResult polar;  // sectional polar curves (cl, cd, cdi, cd0, cm, ld vs alpha)
        std::vector<SectionProfilePoint> profile;
        int sliceY = -1;
        float spanPosMm = 0.0f;          // lateral position relative to model span center (mm)
        float localChordMm = 0.0f;       // local section chord (mm)
        float maxThicknessRatio = 0.0f;  // (t/c)_max (e.g. 0.12 = 12%)
        float maxThicknessX = 0.0f;      // chordwise x/c of max thickness
        float maxCamberRatio = 0.0f;     // (f/c)_max (e.g. 0.02 = 2%)
        float maxCamberX = 0.0f;         // chordwise x/c of max camber
        float localTwistDeg = 0.0f;      // local geometric incidence/twist (deg)
        float reynoldsNumber = 0.0f;     // local chord Reynolds number Re_c
        float stallAlphaDeg = 14.0f;     // predicted local stall angle (deg)
        bool valid = false;
    };

    PolarSweepResult RunPolarSweep(float alphaMinDeg, float alphaMaxDeg, float alphaStepDeg,
                                  float inletVelocityMmPerS, float cellSizeMm,
                                  float refAreaM2 = 0.0f, float refChordM = 0.0f,
                                  int sweepsPerPoint = 35);

    SectionPolarResult RunSectionPolarSweep(const VoxelGrid& grid, int sliceY,
                                            float alphaMinDeg, float alphaMaxDeg, float alphaStepDeg,
                                            float inletVelocityMmPerS, bool include3DDownwash = false) const;

private:
    struct SurfacePanel {
        Vec3 collocation;   // surface collocation point (on the SDF zero-isosurface)
        Vec3 sourcePos;     // desingularized internal source location (offset inward along -normal)
        Vec3 normal;        // outward unit surface normal
        float area = 1.0f;  // panel surface area (lattice units^2)
        float epsSq = 1.5f; // smoothing core radius squared
        float sigma = 0.0f; // solved source strength per unit area
        float selfDiag = 1.0f;
    };

    struct WakeStrip {
        Vec3 boundA;        // quarter-chord bound vortex left endpoint (y_min of strip)
        Vec3 boundB;        // quarter-chord bound vortex right endpoint (y_max of strip)
        Vec3 controlPt;     // 3/4-chord Kutta tangency control point
        Vec3 camberNormal;  // unit normal to the local camber line at controlPt (pointing +Z-ish)
        float chord = 1.0f;
        float span = 1.0f;
        float coreSq = 2.0f;
        float gamma = 0.0f; // solved bound circulation Gamma
        float selfDiag = 1.0f;
    };

    struct Slab {
        int zStart = 0;
        int zCount = 0;
        size_t cellCount = 0;
        SDL_GPUBuffer* macroOut = nullptr;
    };

    struct LiftingSample {
        Vec3 surfacePos;
        Vec3 normal;
        float area = 1.0f;
    };

    void BuildPanelsAndWake(const VoxelGrid& grid);
    void BuildWakeStrips();
    void RebuildInfluenceMatrices();
    void UploadSingularityBuffers();
    void DispatchSlabGPU(SDL_GPUCommandBuffer* cmd, size_t slabIndex, int localZOffset, int localZCount);
    void EvaluateFieldCPU(std::vector<float>& outField) const;

    SDL_GPUDevice* device_ = nullptr;
    bool useGpu_ = false;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    size_t cellCount_ = 0;
    Params params_;
    bool influenceFlowReversed_ = false;
    bool gpuFieldDirty_ = true;

    std::vector<uint8_t> solid_;
    std::vector<SurfacePanel> panels_;
    std::vector<WakeStrip> wakeStrips_;
    std::vector<PanelGPU> panelsGpuData_;
    std::vector<HorseshoeGPU> wakeGpuData_;

    // Precomputed dense influence matrices for the fast iterative surface solve:
    //   panelToPanel_[i * P + j] = normal velocity at panel i per unit q_j of panel j
    //   wakeToPanel_[i * W + s]  = normal velocity at panel i per unit (Gamma_s / 4pi) of strip s
    //   panelToWake_[k * P + j]  = normal velocity at strip k controlPt per unit q_j of panel j
    //   wakeToWake_[k * W + s]   = normal velocity at strip k controlPt per unit (Gamma_s / 4pi) of strip s
    std::vector<float> panelToPanel_;
    std::vector<float> wakeToPanel_;
    std::vector<float> panelToWake_;
    std::vector<float> wakeToWake_;

    // Exposed boundary solid cell indices for fast surface stress/force computation
    std::vector<size_t> boundarySolidCells_;
    std::vector<LiftingSample> liftingSamples_;
    int minY_ = 0, maxY_ = -1;

    // GPU resources (only allocated when useGpu_ == true)
    SDL_GPUComputePipeline* pipeline_ = nullptr;
    SDL_GPUBuffer* solidMaskBuffer_ = nullptr;
    SDL_GPUBuffer* panelBuffer_ = nullptr;
    SDL_GPUBuffer* wakeBuffer_ = nullptr;
    std::vector<Slab> slabs_;
    SDL_GPUTransferBuffer* readbackTransfer_ = nullptr;
    uint32_t readbackTransferCapacity_ = 0;

    std::string lastError_;
};
