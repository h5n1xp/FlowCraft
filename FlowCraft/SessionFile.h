#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "FcMath.h"
#include "Mesh.h"

// Everything needed to restore a FlowCraft session's *configuration* —
// deliberately not a live simulation snapshot. The solver's own state
// (populations, or φ for the Naive backend) is never saved: at the
// resolutions this app targets that can be many gigabytes, the same
// reason LbmSolver's buffers themselves are the thing hitting SDL_GPU's
// 4GiB cap in the first place, and it's fully reconstructible from these
// settings anyway (Reset()/Init() already does the same quiescent-or-
// mid-speed seed either way). Loading a session restores these fields and
// the model, but leaves Voxelize/Initialize Solver as the same manual,
// explicit, progress-shown steps they already are when starting from a
// fresh STL — this file format has no opinion about when, or whether, to
// kick either off automatically.
struct SessionSettings {
    // Model transform (ModelMatrix's inputs) — modelPivot specifically,
    // not derived from the mesh, since RotateModel/scale changes read it
    // directly (see main.cpp).
    Mat4 modelOrientation = Mat4::Identity();
    float scaleMM = 1.0f;
    Vec3 modelPivot{0.0f, 0.0f, 0.0f};
    // See AppState::lockedOrientation's own comment — what Reset
    // Orientation resets to, and what a freshly-opened STL starts at.
    Mat4 lockedOrientation = Mat4::Identity();

    // Voxelization
    int voxelResolution = 640;
    Vec3 voxelPaddingFactor{0.3f, 1.0f, 0.5f};

    // Solver — the int32 fields hold SolverBackend/LatticeModel enum
    // values; SessionFile.h deliberately doesn't depend on main.cpp's
    // enum definitions (so this stays a plain, app-structure-agnostic
    // data file), so the caller casts at the boundary.
    int32_t lbmBackend = 0;
    int32_t lbmModel = 0;
    float lbmTau = 0.8f;
    bool lbmReverseFlow = false;
    // See LbmSolver::Params::freeSlipY/freeSlipZ, including why the
    // default is true.
    bool lbmFreeSlipY = true;
    bool lbmFreeSlipZ = true;
    // See LbmSolver::Params::smagorinskyCs.
    float lbmSmagorinskyCs = 0.12f;
    float lbmInletVelocityMmPerS = 10000.0f;
    int lbmSubstepsPerFrame = 4;
    bool lbmSeedMidSpeed = true;
    // See AppState::warmStartEnabled/warmStartSteps/warmStartResolutionDivisor.
    bool warmStartEnabled = true;
    int warmStartSteps = 1000;
    int warmStartResolutionDivisor = 10;
    bool buildOverlayOnRun = true;

    // Overlay / visualization
    int32_t overlayStyle = 0;
    // One independent slice plane per world axis (X/Y/Z, index-matched to
    // SliceAxis) — see AppState::slicePlanes. Replaces the single
    // overlaySliceAxis/overlaySliceIndex pair format version 5 and
    // earlier had, since Slice style can now show up to all 3 at once
    // instead of switching between them.
    bool sliceEnabled[3] = {false, true, false};  // Y on by default, matching the old single-slice default
    int sliceIndex[3] = {-1, -1, -1};
    int32_t overlayColorMode = 0;
    bool overlayAutoUpdate = true;
    int overlayUpdateIntervalMs = 300;
    int overlayAutoUpdateWarmupSteps = 100;
    bool overlayShowModel = true;
    // See AppState::sliceShowParticles — Slice-style only, layers the
    // Particles smoke trail on top of the slice plane.
    bool sliceShowParticles = false;
    float voxelAlphaPressureLo = 0.0f, voxelAlphaPressureHi = 0.0f;
    bool voxelAlphaRangeInitialized = false;
    bool detrendPressure = true;
    bool voxelNegativePressureGrey = true;

    // Particles (smoke trail)
    int smokeTrailParticleCount = 1500;
    int smokeTrailLength = 10;
    float smokeTrailSpeedScale = 1.0f;
    float smokeTrailPuffDurationSeconds = 3.0f;
    // See AppState::smokeTrailPressureAlphaEnabled.
    bool smokeTrailPressureAlphaEnabled = false;

    // UI state (activeTab is a WorkspaceTab value). UI style/font are user
    // preferences and deliberately NOT part of the session.
    int32_t activeTab = 0;
    bool showInspectorPanel = true;
    bool showTelemetryHud = true;
    bool showLiftArrow = true;

    // Camera — view angle/zoom only, not target/sceneRadius: those are
    // re-derived from OrbitCamera::Frame() once the restored mesh loads,
    // and saving them would just fight that derivation on load.
    float cameraYaw = 0.0f, cameraPitch = 0.0f, cameraDistance = 5.0f;
};

// Reads/writes a self-contained ".fcsession" file: SessionSettings above,
// plus the source STL's raw bytes embedded whole, so loading a session
// never needs to find the original STL file again — it can have moved,
// been renamed, or not exist on this machine at all.
class SessionFile {
public:
    static bool Save(const std::string& path, const SessionSettings& settings, const std::string& stlFileName,
                      const std::vector<uint8_t>& stlBytes, std::string& outError);

    struct LoadResult {
        bool ok = false;
        std::string error;
        SessionSettings settings;
        std::string stlFileName;        // the embedded STL's original filename — cosmetic (Mesh::name), not a path
        std::vector<uint8_t> stlBytes;  // feed directly to STLLoader::LoadFromMemory
    };
    static LoadResult Load(const std::string& path);
};
