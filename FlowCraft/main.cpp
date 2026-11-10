// FlowCraft: portable realtime CFD visualizer for RC aeromodellers.
//
// This first milestone covers the foundation everything else builds on:
// pick an STL file, parse it (binary or ASCII) into a shared Mesh
// data structure, and view it from any angle. The LBM solver and airflow
// overlay come next, reusing this same Mesh as their geometry input.

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <numbers>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlgpu3.h"

#include "AmigaTopazFont.h"
#include "Camera.h"
#include "FieldOverlay.h"
#include "FileDialog.h"
#include "FlowCraftIcon.h"
#include "GpuShaders.h"
#include "LatticeModel.h"
#include "LbmReference.h"
#include "LbmSolver.h"
#include "OverlayRenderer.h"
#include "PanelSolver.h"
#include "PotentialFlowSolver.h"
#include "Renderer.h"
#include "SessionFile.h"
#include "SmokeTrail.h"
#include "SystemMeters.h"
#include "STLLoader.h"
#include "VoxelGrid.h"
#include "Voxelizer.h"
#include <implot.h>

namespace {

// Fluid flow is a fixed world-space convention: it travels along +X.
// Rotation controls are defined relative to this, aircraft-style: roll
// about the flow axis itself, yaw about world-up, pitch about the
// remaining lateral axis. Keeping these fixed in world space (rather
// than the model's own, currently-rotated frame) makes repeated key
// presses behave like a turntable instead of drifting with attitude.
// Z-up world convention — X = flow, Z = up, Y = lateral — matching
// Autodesk Fusion 360's own axis convention (also Z-up) rather than the
// more common Y-up used by most other 3D tools, so an STL exported
// straight out of Fusion (the primary modeling tool for this app's RC
// audience) already sits upright on load with no manual re-orientation
// needed, the way it previously did for an already-Y-up source instead.
// Only the "which axis is up" labeling/camera math changed for this — the
// solver treats Y and Z as fully interchangeable cross-section axes
// either way (see LbmReference/PotentialFlowSolver's own Step()), and the
// flow axis (X) never moved.
constexpr Vec3 kFlowAxis{1.0f, 0.0f, 0.0f};
constexpr Vec3 kYawAxis{0.0f, 0.0f, 1.0f};
constexpr Vec3 kPitchAxis{0.0f, 1.0f, 0.0f};
// kYawRotationAxis: RotateModel's actual rotation axis for Yaw — the
// true +kYawAxis direction above, used as-is for the orientation
// gizmo's display, gives a rotation that reads backward on screen
// under OrbitCamera's Y-right/X-forward-out-of-screen convention
// (verified by projecting test points through the real ViewMatrix), so
// Yaw's control uses this negated axis instead.
//
// kPitchRotationAxis: despite the name (kept to avoid re-plumbing every
// call site), this is now the Roll control's rotation axis, and
// kFlowAxis below is Pitch's — confirmed backward by the user from the
// actual rendered app (Pitch was spinning the model flat, Roll was
// tipping it in depth) after this file's own math checked out fine in
// isolation; whatever makes the two labels feel right to a given
// mesh's own modeled axes apparently isn't fully pinned down by the
// camera/world convention alone. Swap these two assignments back (at
// every RotateModel(app, kFlowAxis/kPitchRotationAxis, ...) call for
// Pitch/Roll, not Yaw) if a differently-authored mesh ever disagrees.
constexpr Vec3 kYawRotationAxis{0.0f, 0.0f, -1.0f};
constexpr Vec3 kPitchRotationAxis{0.0f, -1.0f, 0.0f};
constexpr float kRotationStepRadians = 0.05f;  // per on-screen button press
constexpr float kQuarterTurnRadians = std::numbers::pi_v<float> / 2.0f;

// Camera view presets, sharing the same fixed world axes as the model
// orientation controls (X = flow, Z = up, Y = lateral) so "Front" always
// means looking downstream regardless of how the model is currently
// rotated. Top's pitch stops just short of +90° (matching OrbitCamera's own
// drag clamp) rather than hitting it exactly, since forward becomes
// parallel to the world-up vector at the singularity and GetBasis()'s
// Cross(worldUp, forward) degenerates to zero there.
constexpr float kTopViewPitch = 1.5f;

// The LBM solver itself always runs at this fixed, numerically-safe lattice
// inlet velocity (well under the ~0.3-0.4 instability threshold, comfortably
// low-Mach) — it is never exposed to the user. What the user actually sets
// is a real inlet speed in mm/s; that, together with the voxel grid's known
// cell size (mm), determines how much real time one lattice step represents
// (dt). This is a pure relabeling of the time axis after the fact — the
// lattice-space dynamics (and therefore the dimensionless flow pattern) are
// identical regardless of dt, so changing the mm/s value never requires
// re-running the simulation, only reinterpreting its output. See
// LatticeDtSeconds()/LatticeSpeedToMmPerSecond() below.
constexpr float kLatticeVelocityScale = 0.1f;

// The Resolution slider's own range (DrawUI) and FindMaxGpuResolution's
// search bound share this one definition rather than each hardcoding the
// same number, since the latter's whole premise is "the highest value
// *within the slider's own range* that still fits" — letting them drift
// apart would make that search silently stop short of (or overshoot) what
// the slider actually allows. 2048 is generous headroom for the CPU/Naive
// backends (system RAM only) well beyond where the GPU backend's own
// ceiling (see LbmSolver.h's class comment on solidMask_) already sits —
// CPU/Naive will still exhaust real system memory long before 2048 at
// D3Q19/float32 cell costs, same as any resolution limit here, this is
// just not an artificial API ceiling stopping them short of trying.
constexpr int kMinVoxelResolution = 16;
constexpr int kMaxVoxelResolution = 2048;

enum class ViewMode { Model, VoxelGrid, FlowOverlay };

// Voxelizing, (re)initializing the solver, and a manual overlay readback
// are all slow enough at high resolution to otherwise freeze the window
// with no feedback at all (see StartTask's comment) — this names which one
// is in flight so DrawBusyOverlay can show a relevant label.
enum class TaskKind { None, Voxelizing, InitializingSolver, BuildingOverlay };

// GPU (LbmSolver) is the fast path, but SDL_GPUBufferCreateInfo::size is a
// plain Uint32 — every SDL_GPU buffer is hard-capped at 4GiB regardless of
// available GPU memory, which LbmSolver::Init() hits (and fails cleanly on
// — see its comment) well before resolutions worth testing scaling at.
// CPU (LbmReference) has no such ceiling, only system RAM, at the cost of
// being far slower even multithreaded — see LbmReference::Step()'s
// comment. Naive (PotentialFlowSolver) is a different, much cheaper model
// entirely — not LBM at all, see its header comment — that trades
// physical accuracy (no viscosity, no wake) for roughly 30-75x less
// memory per cell, reaching resolutions neither of the other two can.
// Switching backends is handled exactly like switching lattice model:
// invalidates the current solver, needs re-Init.
enum class SolverBackend { GPU = 0, CPU = 1, Naive = 2, PanelGPU = 3, PanelCPU = 4 };

inline bool IsLbmBackend(SolverBackend b) {
    return b == SolverBackend::GPU || b == SolverBackend::CPU;
}
inline bool IsPanelBackend(SolverBackend b) {
    return b == SolverBackend::PanelGPU || b == SolverBackend::PanelCPU;
}
inline bool IsGpuBackend(SolverBackend b) {
    return b == SolverBackend::GPU || b == SolverBackend::PanelGPU;
}

// Helper for UI: maps SolverBackend <-> (solverTypeIndex, deviceIndex) where
// solverTypeIndex: 0 = LBM, 1 = Panel Method, 2 = Potential Flow
// deviceIndex:     0 = GPU, 1 = CPU
inline void SplitSolverBackend(SolverBackend b, int& outTypeIndex, int& outDeviceIndex) {
    if (b == SolverBackend::GPU) {
        outTypeIndex = 0;
        outDeviceIndex = 0;
    } else if (b == SolverBackend::CPU) {
        outTypeIndex = 0;
        outDeviceIndex = 1;
    } else if (b == SolverBackend::PanelGPU) {
        outTypeIndex = 1;
        outDeviceIndex = 0;
    } else if (b == SolverBackend::PanelCPU) {
        outTypeIndex = 1;
        outDeviceIndex = 1;
    } else {
        outTypeIndex = 2;
        outDeviceIndex = 1;
    }
}

inline SolverBackend CombineSolverBackend(int typeIndex, int deviceIndex) {
    if (typeIndex == 0) {
        return deviceIndex == 0 ? SolverBackend::GPU : SolverBackend::CPU;
    } else if (typeIndex == 1) {
        return deviceIndex == 0 ? SolverBackend::PanelGPU : SolverBackend::PanelCPU;
    }
    return SolverBackend::Naive;
}

// Three different ways to look at the same field: a single colored
// cross-section (the original), the whole volume as semi-transparent
// cubes (a rough "feel for the 3D shape" view — see
// FieldOverlay::BuildVoxelCloudMesh and OverlayRenderer::RenderTransparent),
// or a dye-injection "smoke trail" of particles advected by the flow (see
// SmokeTrailSystem). A sub-choice within Flow Overlay view mode, not a
// separate ViewMode — all three still just color the one field by
// Speed/Pressure using the one cached range (see AppState's comment on
// lastField/cachedRange*). Particles is simulated/meshed fresh every
// rendered frame rather than on demand like the other two — see
// UpdateSmokeTrailAndUpload's comment.
enum class OverlayStyle { Slice, VoxelCloud, Particles, StressMap, IsoSurface, Raymarch };

// Active stage tab in the Autodesk Fusion-style top ribbon bar.
enum class WorkspaceTab { Model, Voxelize, Simulate, Inspect };
constexpr float kFusionRibbonHeight = 92.0f;

const char* TaskKindLabel(TaskKind kind) {
    switch (kind) {
        case TaskKind::Voxelizing:
            return "Voxelizing mesh...";
        case TaskKind::InitializingSolver:
            return "Initializing solver...";
        case TaskKind::BuildingOverlay:
            return "Updating flow overlay...";
        case TaskKind::None:
        default:
            return "";
    }
}

struct AppState {
    Mesh mesh;
    bool meshLoaded = false;

    // UI fonts — populated once at startup onto ImGui::GetIO().Fonts so
    // switching between them is just changing io.FontDefault.
    ImFont* defaultFont = nullptr;
    ImFont* amigaTopazFont = nullptr;
    int uiFontIndex = 0;  // 0 = Default (ProggyClean), 1 = Amiga Topaz
    // The mesh's own raw source bytes — kept around (not just the parsed
    // Mesh) specifically so "Save Session" can embed them without needing
    // to re-read a file that may have moved, been renamed, or (if the
    // current mesh came from a *loaded* session rather than an original
    // STL file) never existed on disk at all. STL files are small next to
    // the multi-GB solver buffers this app already holds in memory, so
    // keeping a second copy here is not a meaningful cost.
    std::vector<uint8_t> meshStlBytes;
    std::string statusMessage = "Open an STL file to begin.";

    OrbitCamera camera;
    bool draggingCamera = false;

    // Smooth camera transition state (glides smoothly to Front, Top, Right, Left, Iso, etc.)
    bool cameraSmoothTransitions = true;
    bool cameraTransitioning = false;
    float cameraTransitionStartYaw = 0.0f;
    float cameraTransitionStartPitch = 0.0f;
    float cameraTransitionTargetYaw = 0.0f;
    float cameraTransitionTargetPitch = 0.0f;
    float cameraTransitionProgress = 0.0f;
    float cameraTransitionDuration = 0.35f; // 350 ms CAD-style smooth transition

    // Slice drag handles (3D view, Slice style only) — grabbing one of the
    // three little axis arrows at the slice plane's center both selects
    // that axis (if it wasn't already) and drags the slice position along
    // it. See DrawSliceDragHandles/ComputeSliceHandles and the mouse event
    // handling in main() for how these get set. sliceDragFractionalCells
    // accumulates sub-cell mouse motion between whole-cell steps (reset to
    // 0 at drag start) — see its use at the drag's own call site.
    bool draggingSliceHandle = false;
    SliceAxis draggingSliceHandleAxis = SliceAxis::X;
    float sliceDragFractionalCells = 0.0f;

    Mat4 modelOrientation = Mat4::Identity();
    Vec3 modelPivot{0.0f, 0.0f, 0.0f};
    // The orientation "Reset Orientation" resets *to*, and a freshly
    // loaded STL starts at — see the "Lock Orientation" button. Defaults
    // to identity, so until the user ever locks one, behavior is exactly
    // what it always was (reset/fresh-load both land at true identity).
    // Session-scoped by design: this lives only in memory (and, if saved,
    // in a .fcsession file) — there's no reason to reset it back to
    // identity just because a different STL got loaded, since the whole
    // point is "I already found the right orientation, stop making me
    // redo it."
    Mat4 lockedOrientation = Mat4::Identity();

    // STL files carry no unit. This is the user-supplied calibration:
    // "1 native model unit = scaleMM millimetres" (defaults to 1, i.e.
    // assume the file was already authored in mm). Once applied via
    // ModelMatrix, world space (what the camera and ruler measure in) is
    // directly in millimetres — see ModelMatrix()'s comment.
    float scaleMM = 1.0f;

    MeshRenderer renderer;

    // Voxelization is an explicit, one-shot action (not recomputed every
    // frame) since it's a non-trivial CPU pass — press "Voxelize" to
    // snapshot the current orientation/scale into a grid.
    VoxelGrid voxelGrid;
    // Guards voxelGrid itself against being replaced while a background
    // thread is still reading it. The UI thread is the only writer (the
    // main loop's "pending voxel result" block, and ResetVoxelization) and
    // takes it exclusively just for the assignment; the background
    // readers that can outlive a UI frame — BuildOverlayMeshFromField (on
    // the overlay auto-update thread, which is NOT gated by the task
    // "busy" flag and so can be mid-build when a re-voxelize lands) and
    // InitLbmSolver — hold it shared for as long as they use the grid.
    // The UI thread's own reads need no lock: it's the only writer.
    // Never taken while holding lbmMutex/lastFieldMutex/overlayMeshMutex
    // on the writer side, so there's no ordering to get wrong.
    std::shared_mutex voxelGridMutex;
    std::atomic<bool> voxelGridValid{false};
    int voxelResolution = 640;
    // Extra empty domain around the model's extent, per axis, as a
    // fraction of that extent — see Voxelizer::Params::paddingFactor
    // (x=lateral, y=flow axis, z=up). More padding on an axis means more
    // wind-tunnel room there (less wall-blockage effect on the physics)
    // at the same resolution/cell-count budget; less padding spends more
    // of that budget resolving the model itself.
    Vec3 voxelPaddingFactor{0.3f, 1.0f, 0.5f};
    MeshRenderer voxelRenderer;
    size_t voxelSolidCount = 0;  // counted once at voxelize time — see StartVoxelizeTask's comment

    // Voxelize() + BuildPreviewMesh() are pure CPU work (no GPU/renderer
    // touched), so the background task below computes a result here first;
    // the main loop then applies it (including the voxelRenderer upload,
    // which — like every other renderer mutation in this file — must stay
    // on the UI/render thread). See StartTask's comment and the main loop's
    // "pending voxel result" block.
    std::mutex voxelResultMutex;
    VoxelGrid pendingVoxelGrid;
    Mesh pendingVoxelPreview;
    size_t pendingVoxelSolidCount = 0;
    std::atomic<uint64_t> voxelResultVersion{0};
    uint64_t voxelConsumedResultVersion = 0;  // UI-thread-only

    ViewMode viewMode = ViewMode::Model;

    // One slow, blocking action (Voxelize / Initialize-or-Reset Solver /
    // Build Overlay) at a time, run on taskThread instead of the UI thread
    // so the window keeps rendering and responding to input while it runs —
    // see StartTask's comment. taskProgress/taskHasProgress feed
    // DrawBusyOverlay; only Voxelize currently reports real progress, the
    // other two show an indeterminate spinner plus elapsed time instead.
    std::thread taskThread;
    std::atomic<TaskKind> taskKind{TaskKind::None};
    std::atomic<float> taskProgress{0.0f};
    std::atomic<bool> taskHasProgress{false};
    Uint64 taskStartTicks = 0;  // UI-thread-only (set when launching, read by DrawBusyOverlay)
    // UI-thread-only (set by the Run button, read by DrawProcessingWindow)
    // — when Run was last pressed, for an "elapsed" readout while the
    // solver's running. Resets on each Run press, not cumulative across
    // pauses — this is a liveness reassurance ("still going"), not a
    // runtime odometer.
    Uint64 lbmRunStartTicks = 0;

    // Needed to (re)initialize the LBM solver from the UI (switching
    // lattice model requires new GPU buffers sized for the new Q).
    SDL_GPUDevice* gpuDevice = nullptr;

    // The solver runs continuously on its own background thread (see
    // LbmWorkerThreadFunc) so a slow/blocking GPU readback never stalls the
    // render/UI thread. app.lbmMutex serializes every access to lbmSolver
    // across the two threads — any direct call to it (Step/Reset/Init/
    // SetParams/ReadbackField/ReadbackStats) must hold it. The *Atomic
    // fields are the worker thread's read-only view of UI state that would
    // otherwise be a data race to read unsynchronized from the ImGui-bound
    // plain fields below; the UI thread refreshes them once per frame (see
    // the main loop, right after DrawUI).
    std::mutex lbmMutex;
    // Bumped (under lbmMutex) whenever something other than StepSolverPaced
    // replaces or advances lbmSolver's state — InitLbmSolver's Init/Reset,
    // or the manual Step button's StepSolverLocked. StepSolverPaced drops
    // lbmMutex between the Z-chunks of a substep, so either of those can
    // run in the middle of one; it compares this against the value it saw
    // when the substep began and abandons the substep (no AdvanceSubstep)
    // if it changed, rather than finishing a half-written substep on top
    // of state it didn't start from.
    uint64_t lbmSolverEpoch = 0;
    LbmSolver lbmSolver;
    LbmReference lbmReference;        // CPU LBM backend — see SolverBackend's comment
    PotentialFlowSolver potentialFlow;  // Naive backend — see SolverBackend's comment
    PanelSolver panelSolver;          // 3D Lifting Panel Method backend (GPU + CPU)
    SolverBackend lbmBackend = SolverBackend::GPU;
    std::atomic<bool> lbmValid{false};
    std::atomic<bool> lbmRunning{false};
    LatticeModel lbmModel = LatticeModel::D3Q19;
    LbmSolver::Params lbmParams;  // .inletVelocity is always kLatticeVelocityScale — see that constant's comment
    // Lock-free mirror of lbmParams.reverseFlow specifically for
    // UpdateSmokeTrailAndUpload, which (for the Particles style) reads it
    // on the render thread *every frame* — going through app.lbmMutex for
    // that, the way every other lbmParams access correctly does, meant
    // competing for that same lock against the stepper and readback
    // threads, which can be re-acquiring it continuously (back to back,
    // no idle gap) once an overlay update takes longer than its own
    // interval. A third thread trying to grab a mutex two others are
    // rapidly passing between themselves can be starved far longer than
    // the raw lock-hold time would suggest — measured, standalone, up to
    // several hundred ms of recurring stalls this way, which is exactly
    // what showed up as "periodic pauses" once resolution got large
    // enough for overlay updates to routinely run back to back. Written
    // immediately (and only) at its one write site, the Flow direction
    // combo's change handler — deliberately *not* also mirrored in the
    // once-per-frame atomic-sync block the way most other *Atomic fields
    // are, since that would need to read lbmParams.reverseFlow under
    // app.lbmMutex every frame to stay correct, which is exactly the
    // per-frame lock contention this field exists to avoid. Safe because
    // that combo handler is lbmParams.reverseFlow's only writer anywhere
    // in this file (InitLbmSolver never touches it).
    std::atomic<bool> lbmReverseFlowAtomic{false};  // mirrors lbmParams.reverseFlow's own new default
    // Same reasoning and same "only mirrored at its one write site" rule
    // as lbmReverseFlowAtomic just above — the Tau slider's own change
    // handler is lbmParams.tau's only writer anywhere in this file.
    // Added once resolution/cell-count grew to where auto-update's own
    // readback (see ReadbackFieldPacedGPU) started contending with
    // StepSolverPaced for app.lbmMutex *continuously*, not just
    // occasionally — enough to make even this display's once-a-frame
    // lock attempt a real chance to get starved for a long stretch
    // behind two threads both re-acquiring the same mutex in a tight
    // loop, which is what a "frozen window" with a running solver and a
    // working GPU turned out to actually be.
    std::atomic<float> lbmTauAtomic{0.8f};
    float lbmInletVelocityMmPerS = 10000.0f;  // 10 m/s: a reasonable default RC airspeed
    // When true, Reset()/Init() seed every cell at half the inlet speed
    // (direction per reverseFlow) and a Bernoulli-consistent density,
    // instead of quiescent rho=1,u=0 — see LbmReference::Reset's comment.
    // Read at Initialize/Reset Solver time (InitLbmSolver), not live.
    bool lbmSeedMidSpeed = true;
    // Warm start: before seeding the real grid, voxelize the same mesh
    // again at a much coarser resolution, run a quick solve on it, and
    // nearest-sample ITS field onto the real grid as the seed instead of
    // the uniform quiescent/mid-speed value above — see ApplyWarmStart's
    // comment for the full reasoning (and its physical caveat: tau fixed
    // + a smaller lattice extent means the coarse run isn't dynamically
    // similar to the fine one, just a rough approximation of roughly
    // where the wake/acceleration regions are). Only meaningful for
    // GPU/CPU (ApplyWarmStart is a no-op for Naive, which solves directly
    // via SOR rather than transient relaxation) — read at Initialize/
    // Reset Solver time, same as lbmSeedMidSpeed.
    bool warmStartEnabled = true;
    int warmStartSteps = 1000;
    int warmStartResolutionDivisor = 10;
    // Starts the solver running AND kicks off a Build Overlay the moment
    // "Run" is pressed (see DrawUI's Run/Pause button) — so the overlay
    // doesn't sit empty/stale until a separate manual click. On by
    // default: the whole point of this overlay is watching the flow
    // develop, and the first thing most people reach for after pressing
    // Run is the overlay anyway.
    bool buildOverlayOnRun = true;
    int lbmSubstepsPerFrame = 4;              // UI-bound; mirrored into lbmSubstepsAtomic for the worker thread
    std::atomic<int> lbmSubstepsAtomic{4};
    // Mirrors lbmBackend for LbmWorkerThreadFunc, which reads it every
    // tick (not just while a one-shot task is running, unlike lbmModel —
    // see SolverBackend's comment and the main loop's atomic-sync block).
    std::atomic<int> lbmBackendAtomic{static_cast<int>(SolverBackend::GPU)};
    std::atomic<int> lbmStepCount{0};
    // Progress (0..1) through the *current* call to StepReferenceLocked/
    // StepPotentialFlowLocked/StepSolverPaced — i.e. the current frame's
    // batch of app.lbmSubstepsPerFrame substeps, not progress of the run
    // as a whole (which has no natural end while Run is held down). Read
    // by DrawProcessingWindow, particularly useful on the CPU backend at
    // high resolution, where even one substep's own LbmReference::Step()
    // call can take several seconds to minutes — without this the window
    // could only say "running", with no sense of whether it's 10% or 90%
    // of the way through the step already in flight.
    std::atomic<float> lbmStepProgress{0.0f};
    LbmSolver::Stats lbmLastStats;  // only touched by the UI thread (Refresh Stats button), under lbmMutex
    // Mirror of lbmLastStats' individual fields, same "avoid a per-frame
    // lock" reasoning as lbmReverseFlowAtomic/lbmTauAtomic above — DrawUI
    // reads these every frame to keep showing the last-known stats, and
    // that read is what used to fall back to app.lbmMutex (skipped only
    // while a StartTask-based task is busy, which never covers ordinary
    // running+auto-update). lbmLastStats itself isn't a single scalar
    // (LbmSolver::Stats, 5 floats + 2 bools), too big to be one atomic
    // lock-free on any real platform, so this is one atomic per field
    // instead — written together (see SyncLbmStatsAtomics) everywhere
    // lbmLastStats itself is written (the Refresh Stats button, and
    // InitLbmSolver's reset-to-empty on a fresh Init).
    std::atomic<float> lbmStatsAvgDensityAtomic{0.0f};
    std::atomic<float> lbmStatsMinDensityAtomic{0.0f};
    std::atomic<float> lbmStatsMaxDensityAtomic{0.0f};
    std::atomic<float> lbmStatsAvgSpeedAtomic{0.0f};
    std::atomic<float> lbmStatsMaxSpeedAtomic{0.0f};
    std::atomic<bool> lbmStatsHasNaNAtomic{false};
    std::atomic<bool> lbmStatsValidAtomic{false};

    // Live flow divergence / NaN health monitor: tracks whether the solver
    // has blown up (NaN, Inf, negative density, or supersonic lattice speed),
    // which step it failed at, and whether to auto-pause immediately so high-
    // resolution runs don't waste minutes computing purple NaN cells.
    std::atomic<bool> lbmFlowDivergedAtomic{false};
    std::atomic<int> lbmDivergedAtStepAtomic{0};
    std::atomic<bool> divergenceBannerDismissedAtomic{false};
    bool autoPauseOnDivergence = true;
    std::atomic<bool> autoPauseOnDivergenceAtomic{true};

    // Aerodynamic forces (Wen et al. / Ladd Bouzidi momentum exchange),
    // peak surface stress in lattice units, and Centre of Pressure in
    // world coordinates (mm), updated both by Refresh Stats and
    // automatically whenever BuildOverlayMeshFromField reads back the field.
    std::atomic<float> lbmForceXAtomic{0.0f};
    std::atomic<float> lbmForceYAtomic{0.0f};
    std::atomic<float> lbmForceZAtomic{0.0f};
    std::atomic<float> lbmMaxStressAtomic{0.0f};
    std::atomic<float> lbmCpXAtomic{0.0f};
    std::atomic<float> lbmCpYAtomic{0.0f};
    std::atomic<float> lbmCpZAtomic{0.0f};
    std::atomic<float> lbmLiftBaseZAtomic{0.0f};
    std::atomic<bool> lbmForcesValidAtomic{false};
    bool showLiftArrow = true;
    std::string lbmInitError;  // set by InitLbmSolver on failure, read by DrawUI — both under lbmMutex

    std::atomic<bool> lbmThreadStop{false};

    // Flow overlay: a colored cross-section built on demand (explicit
    // "Build Overlay") or periodically (overlayAutoUpdate) from the
    // solver's current field.
    OverlayRenderer overlayRenderer;
    bool overlayValid = false;
    OverlayStyle overlayStyle = OverlayStyle::Particles;
    // One independent slice plane per world axis (X/Y/Z) — Slice style
    // shows however many of these are currently enabled at once, each at
    // its own position, rather than a single plane that switches between
    // axes. Indexed by static_cast<int>(SliceAxis). See SlicePlaneState's
    // own comment for the fields; enabledAtomic/indexAtomic are what
    // BuildOverlayMeshFromField reads (same reasoning as every other
    // *Atomic mirror here — it can run on a different thread), the plain
    // fields are UI-thread-only (TryFastReslice, DrawUI, the drag handles).
    struct SlicePlaneState {
        bool enabled = false;
        int index = -1;  // -1 = not yet defaulted to this axis's grid center
        std::atomic<bool> enabledAtomic{false};
        std::atomic<int> indexAtomic{-1};
        // Tracks what app.smokeTrails[this axis] was last confined to/
        // seeded at, for UpdateSmokeTrailAndUpload's own needsReset/
        // needsPlaneShift logic — see its comment. -1/false meaning
        // "never seeded yet", matching a freshly loaded/reset grid.
        bool smokeTrailWasEnabled = false;
        int smokeTrailLastIndex = -1;
    };
    std::array<SlicePlaneState, 3> slicePlanes;
    FieldColorMode overlayColorMode = FieldColorMode::Speed;
    float overlayMin = 0.0f, overlayMax = 0.0f;
    bool overlayHasNaN = false;  // UI-thread-only — set by the main loop's "pending overlay mesh" block
    bool overlayShowModel = true;  // draw the original (unvoxelized) mesh alongside the slice, for context
    // Slice style only: each enabled plane's colored cross-section is
    // replaced by its own Particles smoke trail confined to that same
    // plane (see SmokeTrailSystem::PlaneConfinement and
    // AppState::smokeTrails) instead of being drawn together with it —
    // not a third "Slice+Particles" enum value, so the per-plane enable/
    // position controls keep working unchanged to pick the planes, while
    // "Color by" is hidden (nothing left for it to affect — see the Slice
    // UI block in DrawUI). A single shared toggle, like particle count/
    // trail length/speed scale below — it affects every enabled plane's
    // particle field identically, not one plane at a time.
    bool sliceShowParticles = false;

    // 3D Iso-surface extraction via Marching Cubes (FieldOverlay::BuildIsoSurfaceMesh)
    float isoThreshold = 0.35f;       // Normalized threshold [0..1] within field range
    float isoAlpha = 0.40f;           // Surface opacity (0.1..1.0)
    bool isoEncloseGreater = false;   // false = enclose <= iso (wake bubble, vortex core), true = >= iso (jet, stagnation)
    int isoStride = 1;                // 1 = full resolution, 2 = 2x downsampled stride
    std::atomic<float> isoThresholdAtomic{0.35f};
    std::atomic<float> isoAlphaAtomic{0.40f};
    std::atomic<bool> isoEncloseGreaterAtomic{false};
    std::atomic<int> isoStrideAtomic{1};

    // 3D Volumetric Raymarching via view-aligned slicing (FieldOverlay::BuildRaymarchMesh)
    int raymarchSliceCount = 64;      // Number of view-aligned sample planes (32..128)
    float raymarchDensity = 1.2f;     // Optical absorption / emission density scale
    float raymarchCutoff = 0.08f;     // Low-end cutoff threshold to keep ambient fluid crystal clear
    float raymarchGamma = 1.4f;       // Contrast exponent for volume density mapping
    std::atomic<int> raymarchSliceCountAtomic{64};
    std::atomic<float> raymarchDensityAtomic{1.2f};
    std::atomic<float> raymarchCutoffAtomic{0.08f};
    std::atomic<float> raymarchGammaAtomic{1.4f};
    std::atomic<float> cameraPosXAtomic{0.0f}, cameraPosYAtomic{0.0f}, cameraPosZAtomic{0.0f};

    // Atomic mirrors of the UI-bound fields above that
    // BuildOverlayMeshFromField reads, refreshed once per frame (same
    // pattern as lbmSubstepsAtomic etc.) — needed because that function
    // can now run on the solver thread or a Build-Overlay task thread, not
    // just the UI thread, so it must never read the plain fields directly.
    std::atomic<int> overlayStyleAtomic{static_cast<int>(OverlayStyle::Particles)};
    std::atomic<int> overlayColorModeAtomic{static_cast<int>(FieldColorMode::Speed)};
    std::atomic<bool> sliceShowParticlesAtomic{false};
    // The camera's own forward vector, same reasoning as the atomics
    // above — Voxel Cloud's fuzzy billboards (FieldOverlay::
    // BuildVoxelCloudMesh) need to face the camera, but their mesh can be
    // built on a background thread (see BuildOverlayMeshFromField's own
    // comment), where reading app.camera directly would race the UI
    // thread's own mouse-drag updates to it. A frame or so of staleness
    // here is harmless — purely cosmetic billboard orientation, not
    // something that needs to be exactly current.
    std::atomic<float> cameraForwardXAtomic{1.0f}, cameraForwardYAtomic{0.0f}, cameraForwardZAtomic{0.0f};

    // Paces the periodic auto-update (mirrored into *Atomic fields below
    // for the solver thread to read); the UI thread only uploads the mesh
    // once new data shows up.
    bool overlayAutoUpdate = true;  // the whole point of this overlay is to watch the flow develop live
    int overlayUpdateIntervalMs = 300;
    std::atomic<bool> overlayAutoUpdateAtomic{false};
    std::atomic<int> overlayIntervalAtomic{300};
    // Suppresses auto-update's very first readback until app.lbmStepCount
    // (reset to 0 at every Initialize/Reset Solver) reaches this many
    // steps — so the first thing auto-update shows is a few seconds into
    // a developed flow rather than the barely-started transient right
    // after pressing Run. Purely a display-timing knob: the solver itself
    // runs identically either way, and once this threshold is crossed the
    // interval-based pacing above takes over exactly as before.
    int overlayAutoUpdateWarmupSteps = 100;
    std::atomic<int> overlayAutoUpdateWarmupStepsAtomic{100};

    // The solver thread only *starts* an auto-update (see
    // StartOverlayAutoUpdate) — the readback and mesh build themselves run
    // on this separate thread, so the solver's own stepping loop never
    // waits on them (a readback does still briefly take app.lbmMutex, the
    // same as a Step() call would, but the CPU-heavy mesh-building after
    // it needs no lock at all and no longer sits in the stepping loop's
    // way either). This was previously inline on the solver thread, which
    // meant auto-update directly ate into step throughput — measurably,
    // since a full field scan plus mesh build doesn't get cheaper just
    // because the solver is otherwise idle waiting on it. Overlapping
    // updates are prevented by overlayUpdateInFlight, not by blocking the
    // solver thread: if a readback+build takes longer than the configured
    // interval, the next one simply starts the moment this one finishes
    // rather than waiting out another full interval — the overlay lags
    // behind the simulation instead of slowing it down.
    std::atomic<bool> overlayUpdateInFlight{false};
    std::thread overlayUpdateThread;

    // BuildOverlayMeshFromField (run off the UI thread — by the solver
    // thread for overlayAutoUpdate, or a Build-Overlay task thread for the
    // manual button) stages its result here instead of uploading it
    // directly: the CPU-heavy part (a full-field scan for the color range,
    // plus the slice mesh itself — see FieldOverlay::BuildSliceMesh) used
    // to run on the UI thread and was a real, resolution-scaling source of
    // "freezes between solver updates" that none of the GPU-side pacing
    // work touched. The main loop's "pending overlay mesh" block does only
    // the actual GPU upload, which is cheap and must stay there anyway
    // (overlayRenderer is read every frame by the render loop).
    std::mutex overlayMeshMutex;
    OverlayMesh pendingOverlayMesh;
    float pendingOverlayMin = 0.0f, pendingOverlayMax = 0.0f;
    bool pendingOverlayHasNaN = false;
    std::atomic<uint64_t> overlayMeshVersion{0};
    uint64_t overlayConsumedMeshVersion = 0;  // UI-thread-only

    // The last field BuildOverlayMeshFromField was given, kept around so
    // moving the slice axis/position (or any Voxel Cloud alpha-window
    // control) can re-slice *without* a fresh solver readback — see
    // StartResliceTask/TryFastReslice. A shared_ptr, not a copied vector:
    // TryFastReslice runs on the UI thread and must not block it on
    // copying a potentially huge field (tens to hundreds of MB at high
    // resolution) — bumping a refcount is O(1) regardless of field size,
    // so holding lastFieldMutex here is always brief no matter how big
    // the grid is.
    //
    // cachedSpeed*/cachedPressure* are ComputeRange's result for this
    // exact field, one call per mode — the expensive O(whole field) part
    // (see FieldOverlay.h). Both are always computed for every new field
    // (not just whichever mode Color By is currently set to) because the
    // Voxel Cloud style needs both simultaneously (color from Speed,
    // alpha from Pressure — see BuildMeshForStyle), and caching both
    // unconditionally means switching Color By, or between Slice and
    // Voxel Cloud, never needs a fresh ComputeRange pass either — only
    // BuildMeshForStyle's bounded cost, cheap enough to run synchronously
    // on the UI thread, which is what makes all of those feel instant
    // instead of kicking off a background task.
    std::mutex lastFieldMutex;
    std::shared_ptr<const std::vector<float>> lastField;  // null until a field has ever been cached
    float cachedSpeedMin = 0.0f, cachedSpeedMax = 0.0f;
    bool cachedSpeedHasNaN = false;
    float cachedPressureMin = 0.0f, cachedPressureMax = 0.0f;
    bool cachedPressureHasNaN = false;
    float cachedStressMin = 0.0f, cachedStressMax = 0.0f;
    bool cachedStressHasNaN = false;
    bool cachedRangesValid = false;

    // The Voxel Cloud's pressure-to-alpha window (see
    // FieldOverlay::BuildVoxelCloudMesh) — pressure at/below
    // voxelAlphaPressureLo renders fully transparent, at/above
    // voxelAlphaPressureHi renders at the fixed alpha ceiling. Seeded
    // once from the field's actual pressure range the first time it
    // becomes known (see BuildOverlayMeshFromField), then left entirely
    // to the user — the sliders controlling these must never silently
    // reset to the latest field's range, since picking where "ambient"
    // falls is the whole point of this being adjustable.
    bool voxelAlphaRangeInitialized = false;
    float voxelAlphaPressureLo = 0.0f;
    float voxelAlphaPressureHi = 0.0f;
    std::atomic<float> voxelAlphaPressureLoAtomic{0.0f};
    std::atomic<float> voxelAlphaPressureHiAtomic{0.0f};

    // Subtracts each cell's own X-slice mean pressure wherever pressure is
    // used for color/alpha — removes the channel-length pressure gradient
    // that bounded (no-slip Y/Z wall) flow needs to overcome viscous
    // friction, which otherwise dwarfs the model's own local
    // stagnation/wake signature (see FieldOverlay::ComputeXSliceMeans). A
    // pure visualization transform, on by default — it never touches what
    // the solver computes.
    bool detrendPressure = true;
    std::atomic<bool> detrendPressureAtomic{true};

    // Overrides a below-zero-pressure Voxel Cloud cube/Particles trail
    // segment's color+alpha to a fixed dark grey at 30% instead of the
    // normal Speed colormap + alpha-window mapping — see
    // FieldOverlay::BuildVoxelCloudMesh's comment on the parameter of the
    // same name. On by default: a below-ambient cell can otherwise fade
    // to fully invisible under the alpha window, hiding exactly the
    // suction/low-pressure signal this exists to keep visible.
    bool voxelNegativePressureGrey = true;
    std::atomic<bool> voxelNegativePressureGreyAtomic{true};

    // Cached alongside cachedSpeed*/cachedPressure* (see that comment) so
    // toggling detrendPressure, or re-slicing while it's on, never needs a
    // fresh O(whole field) pass.
    std::vector<float> cachedPressureXMeans;
    float cachedPressureDetrendMin = 0.0f, cachedPressureDetrendMax = 0.0f;

    // The Particles overlay style's state — see SmokeTrailSystem and
    // UpdateSmokeTrailAndUpload. UI-thread-only (the sim/mesh step runs
    // once per rendered frame on that thread — see OverlayStyle's
    // comment), unlike everything else above that's shared with a
    // worker/task thread, so none of this needs an atomic mirror.
    SmokeTrailSystem smokeTrail;  // unconfined — used only by OverlayStyle::Particles
    // One independent confined particle field per slice plane (see
    // AppState::slicePlanes) — used only by Slice style's "Show particles"
    // option, each instance confined to its own plane via
    // SmokeTrailSystem::PlaneConfinement so every enabled plane gets its
    // own stream rather than sharing one. Indexed the same way
    // slicePlanes is (static_cast<int>(SliceAxis)); per-plane
    // reset/shift tracking lives on SlicePlaneState itself
    // (smokeTrailWasEnabled/smokeTrailLastIndex), not here, since that
    // part genuinely differs per plane — only grid/reverseFlow changes
    // (tracked below, shared) apply to all of them at once.
    std::array<SmokeTrailSystem, 3> smokeTrails;
    bool smokeTrailInitialized = false;
    bool smokeTrailReverseFlow = false;  // the reverseFlow smokeTrail(s) were last Reset with
    int smokeTrailGridNx = -1, smokeTrailGridNy = -1, smokeTrailGridNz = -1;  // the grid they were last Reset with
    // Whether app.smokeTrail (the unconfined instance) has itself ever
    // actually been Reset() under the current smokeTrailInitialized
    // "generation" — see UpdateSmokeTrailAndUpload's own comment. Needed
    // because smokeTrailInitialized above is shared with the 3 confined
    // smokeTrails[] instances: using *only* that to decide whether this
    // instance needs a Reset meant switching from Slice+particles (which
    // already set smokeTrailInitialized=true for its own instances) to
    // the plain Particles style for the first time since skipped this
    // instance's own Reset entirely — it had never been seeded, so it
    // just sat there with zero particles, reading as "the overlay didn't
    // update" until something else (e.g. changing particle count) forced
    // smokeTrailInitialized back to false and finally reset it too.
    bool smokeTrailUnconfinedEverReset = false;
    int smokeTrailParticleCount = 1500;
    int smokeTrailLength = 10;
    // "Scatter particles" button state: consumed (and cleared) by the
    // very next UpdateSmokeTrailAndUpload call, which calls
    // SmokeTrailSystem::ScatterRandom to instantly jump every particle to
    // a fresh random position across the whole domain (or whole slice
    // plane, confined) right now — unlike "Reset particles" below, which
    // only re-seeds at the inlet/plane edge and lets the stream refill
    // over the next smokeTrailLength or so frames.
    bool smokeTrailScatterRequested = false;
    // Multiplies the real-time advection displacement in
    // UpdateSmokeTrailAndUpload — a pure visualization control (like the
    // alpha window), not a physical input: it never touches the solver,
    // only how much of the flow's own velocity field a particle moves
    // through per real second. 1.0 = the physically accurate speed given
    // the current real inlet speed (mm/s) setting. Defaults to the
    // slider's lowest value (0.1x) rather than the physically-accurate
    // 1.0x — at full domain speed the trails read as a blur more than a
    // streakline, so the slow default reads better out of the box.
    float smokeTrailSpeedScale = 0.1f;

    // "Puff" button state: while smokeTrailPuffRemainingSeconds > 0,
    // UpdateSmokeTrailAndUpload passes suspendRespawn=true to
    // SmokeTrailSystem::Step (see its comment) — no new particles enter
    // at the inlet, so whatever's already in flight drains out and the
    // stream reads as a single puff rather than a continuous plume.
    // Ticks down in real (unscaled) seconds, independent of
    // smokeTrailSpeedScale — this is "how long to pause the source", not
    // part of the flow simulation itself.
    float smokeTrailPuffDurationSeconds = 3.0f;
    float smokeTrailPuffRemainingSeconds = 0.0f;

    // Whether the Particles style's trail opacity is modulated by the
    // same pressure alpha window ("Transparent at/below"/"Opaque
    // at/above") Voxel Cloud always uses — see
    // SmokeTrailSystem::BuildMesh's pressureAlphaEnabled parameter.
    // Defaults off: now that Reset() scatters the initial population
    // across the whole domain (see its own comment) rather than
    // bunching at the inlet, most particles read clearly without also
    // needing to fade in/out by pressure, and that fading was otherwise
    // making ambient/undisturbed-flow particles hard to see at all.
    // Voxel Cloud is unaffected either way — its own rendering always
    // uses the window, since there opacity *is* the visualization.
    bool smokeTrailPressureAlphaEnabled = false;

    // Autodesk Fusion-style top ribbon + context inspector UI state.
    WorkspaceTab activeTab = WorkspaceTab::Model;
    bool showInspectorPanel = true;
    bool showTelemetryHud = true;
    bool telemetryHudInitialized = false;
    bool telemetryHudUserMoved = false;
    ImVec2 telemetryHudLastPos{0.0f, 0.0f};
    ImVec2 telemetryHudLastViewportSize{0.0f, 0.0f};
    bool showProcessingWindow = true;
    // The Processing window's performance meters — see UpdatePerfMeters.
    // UI-thread-only. Rates are differences between two samples taken
    // kPerfSampleIntervalMs apart, so each field here is either "the
    // previous sample" (last*) or "the most recently computed rate".
    struct PerfMeters {
        Uint64 lastSampleTicks = 0;      // 0 = no previous sample to difference against yet
        double lastCpuSeconds = -1.0;
        int lastStepCount = 0;
        bool ratesValid = false;         // false until two samples exist
        float cpuCoresInUse = 0.0f;      // this process, all threads: 1.0 = one core fully busy
        float gpuUtilization = -1.0f;    // whole device, 0..1; negative = not available on this platform
        float stepsPerSecond = 0.0f;
    };
    PerfMeters perfMeters;
    bool processingWindowInitialized = false;
    bool processingWindowUserMoved = false;
    ImVec2 processingWindowLastPos{0.0f, 0.0f};
    // Gap between the window's right edge and the viewport's, as of the
    // user's last drag — see DrawProcessingWindow. Unused until then.
    float processingWindowRightInset = 0.0f;
    bool classicSinglePanelUI = false;

    // Aero Polars window state (XFLR5-style multi-point polar generation & graphing)
    bool showAeroPolarsWindow = false;
    // 0 = Whole Model (3D Assembly), 1 = Active Y-Slice (2D Section), 2 = Compare Both (Overlay)
    int polarScopeMode = 2;
    bool polarSliceInclude3DDownwash = false;
    int polarTableSource = 0;  // 0 = Whole Model, 1 = Active Slice
    float polarAlphaMin = -4.0f;
    float polarAlphaMax = 14.0f;
    float polarAlphaStep = 1.0f;
    bool polarAutoRefGeometry = true;
    float polarRefAreaM2 = 0.0f;
    float polarRefChordM = 0.0f;
    PanelSolver::PolarSweepResult polarResult;
    PanelSolver::SectionPolarResult sectionPolarResult;
    float lastSectionAlphaMin = 0.0f;
    float lastSectionAlphaMax = 0.0f;
    float lastSectionAlphaStep = 0.0f;
    float lastSectionInletVel = 0.0f;
    bool lastSectionDownwash = false;
    std::string polarExportStatus;

    // 0 = Normal (Fusion Ribbon + dark theme), 1 = Classic (single-window UI + dark theme),
    // 2 = Amiga (AmigaOS 1.3 window borders, gadgets, Topaz font, and 4-color palette).
    int uiThemeIndex = 0;
};

void ApplyUITheme(AppState& app, int themeIndex) {
    app.uiThemeIndex = themeIndex;
    app.classicSinglePanelUI = (themeIndex == 1);

    ImGuiStyle& style = ImGui::GetStyle();
    ImGuiIO& io = ImGui::GetIO();

    if (themeIndex == 2) {
        // AmigaOS 1.3 (Workbench 1.3) 4-color palette & window decorations
        style.AmigaOS13Mode = true;
        style.WindowRounding = 0.0f;
        style.ChildRounding = 0.0f;
        style.FrameRounding = 0.0f;
        style.PopupRounding = 0.0f;
        style.ScrollbarRounding = 0.0f;
        style.GrabRounding = 0.0f;
        style.TabRounding = 0.0f;
        style.WindowBorderSize = 2.0f;
        style.FrameBorderSize = 1.0f;
        style.PopupBorderSize = 2.0f;

        if (app.amigaTopazFont) {
            app.uiFontIndex = 1;
            io.FontDefault = app.amigaTopazFont;
        }

        // Authentic AmigaOS 1.3 colors:
        // Blue:      #0055AA -> (0.000f, 0.333f, 0.667f)
        // White:     #FFFFFF -> (1.000f, 1.000f, 1.000f)
        // Orange:    #FF8800 -> (1.000f, 0.533f, 0.000f)
        // Dark Blue: #000222 -> (0.000f, 0.008f, 0.133f)
        const ImVec4 amigaBlue(0.000f, 0.333f, 0.667f, 1.00f);
        const ImVec4 amigaWhite(1.000f, 1.000f, 1.000f, 1.00f);
        const ImVec4 amigaOrange(1.000f, 0.533f, 0.000f, 1.00f);
        const ImVec4 amigaDarkBlue(0.000f, 0.008f, 0.133f, 1.00f);

        ImVec4* colors = style.Colors;
        colors[ImGuiCol_Text]                  = amigaWhite;
        colors[ImGuiCol_TextDisabled]          = ImVec4(0.75f, 0.82f, 0.92f, 1.00f);
        colors[ImGuiCol_WindowBg]              = amigaBlue;
        colors[ImGuiCol_ChildBg]               = amigaBlue;
        colors[ImGuiCol_PopupBg]               = amigaBlue;
        colors[ImGuiCol_Border]                = amigaWhite;
        colors[ImGuiCol_BorderShadow]          = amigaDarkBlue;
        colors[ImGuiCol_FrameBg]               = amigaDarkBlue;
        colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.00f, 0.22f, 0.48f, 1.00f);
        colors[ImGuiCol_FrameBgActive]         = amigaOrange;
        colors[ImGuiCol_TitleBg]               = amigaWhite;
        colors[ImGuiCol_TitleBgActive]         = amigaWhite;
        colors[ImGuiCol_TitleBgCollapsed]      = amigaWhite;
        colors[ImGuiCol_MenuBarBg]             = amigaWhite;
        colors[ImGuiCol_ScrollbarBg]           = amigaDarkBlue;
        colors[ImGuiCol_ScrollbarGrab]         = amigaWhite;
        colors[ImGuiCol_ScrollbarGrabHovered]  = amigaOrange;
        colors[ImGuiCol_ScrollbarGrabActive]   = amigaOrange;
        colors[ImGuiCol_CheckMark]             = amigaOrange;
        colors[ImGuiCol_SliderGrab]            = amigaOrange;
        colors[ImGuiCol_SliderGrabActive]      = amigaWhite;
        colors[ImGuiCol_Button]                = amigaDarkBlue;
        colors[ImGuiCol_ButtonHovered]         = amigaOrange;
        colors[ImGuiCol_ButtonActive]          = amigaOrange;
        colors[ImGuiCol_Header]                = amigaDarkBlue;
        colors[ImGuiCol_HeaderHovered]         = amigaOrange;
        colors[ImGuiCol_HeaderActive]          = amigaOrange;
        colors[ImGuiCol_Separator]             = amigaWhite;
        colors[ImGuiCol_SeparatorHovered]      = amigaOrange;
        colors[ImGuiCol_SeparatorActive]       = amigaOrange;
        colors[ImGuiCol_ResizeGrip]            = amigaWhite;
        colors[ImGuiCol_ResizeGripHovered]     = amigaOrange;
        colors[ImGuiCol_ResizeGripActive]      = amigaOrange;
        colors[ImGuiCol_Tab]                   = amigaDarkBlue;
        colors[ImGuiCol_TabHovered]            = amigaOrange;
        colors[ImGuiCol_TabSelected]           = amigaOrange;
        colors[ImGuiCol_PlotLines]             = amigaWhite;
        colors[ImGuiCol_PlotLinesHovered]      = amigaOrange;
        colors[ImGuiCol_PlotHistogram]         = amigaOrange;
        colors[ImGuiCol_PlotHistogramHovered]  = amigaWhite;
        colors[ImGuiCol_TableHeaderBg]         = amigaDarkBlue;
        colors[ImGuiCol_TableBorderStrong]     = amigaWhite;
        colors[ImGuiCol_TableBorderLight]      = amigaWhite;
        colors[ImGuiCol_TextSelectedBg]        = amigaOrange;
    } else {
        ImGui::StyleColorsDark();
        style.AmigaOS13Mode = false;
        style.WindowBorderSize = 1.0f;
        style.FrameBorderSize = 0.0f;
        style.PopupBorderSize = 1.0f;
        app.uiFontIndex = 0;
        if (app.defaultFont) {
            io.FontDefault = app.defaultFont;
        }
    }
}

// Plain pass-through — no sign correction here. An earlier version of
// this negated `radians` unconditionally, on the theory that the
// camera's Y-right convention (see OrbitCamera::GetBasis()'s comment)
// reverses every rotation's apparent spin direction equally regardless
// of axis. That theory was wrong: verified by projecting test points
// through the real ViewMatrix, the axis about which a flat, in-plane
// spin happens already reads correctly with no sign change, while the
// other two need it — which a blanket negation here couldn't do. The
// correction instead lives in which axis each caller passes —
// kYawRotationAxis for Yaw, and for Pitch/Roll specifically, see
// kPitchRotationAxis's own comment (its name no longer matches which
// control uses it).
void RotateModel(AppState& app, const Vec3& axis, float radians) {
    if (radians == 0.0f) return;
    app.modelOrientation = Mat4::RotationAxisAngle(axis, radians) * app.modelOrientation;
}

bool TryFastReslice(AppState& app);

void SnapCameraView(AppState& app, float targetYaw, float targetPitch, bool animated = true) {
    if (!app.cameraSmoothTransitions || !animated) {
        app.cameraTransitioning = false;
        app.camera.yaw = targetYaw;
        app.camera.pitch = targetPitch;
        return;
    }

    constexpr float kPi = std::numbers::pi_v<float>;
    constexpr float kTwoPi = 2.0f * kPi;
    float deltaYaw = targetYaw - app.camera.yaw;
    while (deltaYaw > kPi) deltaYaw -= kTwoPi;
    while (deltaYaw < -kPi) deltaYaw += kTwoPi;

    if (std::abs(deltaYaw) < 1e-4f && std::abs(targetPitch - app.camera.pitch) < 1e-4f) {
        app.cameraTransitioning = false;
        app.camera.yaw = targetYaw;
        app.camera.pitch = targetPitch;
        return;
    }

    app.cameraTransitioning = true;
    app.cameraTransitionProgress = 0.0f;
    app.cameraTransitionDuration = 0.35f;
    app.cameraTransitionStartYaw = app.camera.yaw;
    app.cameraTransitionStartPitch = app.camera.pitch;
    app.cameraTransitionTargetYaw = app.camera.yaw + deltaYaw;
    app.cameraTransitionTargetPitch = targetPitch;
}

void UpdateCameraTransition(AppState& app, float dt) {
    if (!app.cameraTransitioning) return;

    float safeDt = std::clamp(dt, 0.0f, 0.1f);
    app.cameraTransitionProgress += safeDt / std::max(1e-3f, app.cameraTransitionDuration);

    constexpr float kPi = std::numbers::pi_v<float>;
    constexpr float kTwoPi = 2.0f * kPi;

    if (app.cameraTransitionProgress >= 1.0f) {
        app.cameraTransitionProgress = 1.0f;
        app.cameraTransitioning = false;
        app.camera.yaw = app.cameraTransitionTargetYaw;
        app.camera.pitch = app.cameraTransitionTargetPitch;
        while (app.camera.yaw > kPi) app.camera.yaw -= kTwoPi;
        while (app.camera.yaw < -kPi) app.camera.yaw += kTwoPi;
    } else {
        float t = app.cameraTransitionProgress;
        // Quintic smootherstep (6t^5 - 15t^4 + 10t^3) for zero jerk and silky ease-in / ease-out
        float ease = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
        app.camera.yaw = app.cameraTransitionStartYaw +
                         (app.cameraTransitionTargetYaw - app.cameraTransitionStartYaw) * ease;
        app.camera.pitch = app.cameraTransitionStartPitch +
                           (app.cameraTransitionTargetPitch - app.cameraTransitionStartPitch) * ease;
    }

    if (app.viewMode == ViewMode::FlowOverlay &&
        (app.overlayStyle == OverlayStyle::VoxelCloud || app.overlayStyle == OverlayStyle::Raymarch)) {
        TryFastReslice(app);
    }
}

// Keeps the camera's near/far clip margin sized to the model's *current*
// (scaled) extent rather than its native one, so cranking the scale up
// doesn't clip the now-larger geometry and cranking it down doesn't widen
// the depth range pointlessly. Deliberately does not touch camera.distance
// — the point of the scale control is to visibly see the model grow or
// shrink, which auto-reframing would hide.
void UpdateSceneRadiusForScale(AppState& app) {
    if (!app.meshLoaded) return;
    float nativeRadius = std::max(Length(app.mesh.bounds.Size()) * 0.5f, 0.001f);
    app.camera.sceneRadius = nativeRadius * app.scaleMM;
}

// Rotates and scales about the model's own bounding-box center rather than
// the world origin, so the model spins/grows in place regardless of where
// its STL coordinates happen to sit relative to that origin. Because the
// pivot is fixed by this construction (Translate(-pivot) sends it to the
// origin, where rotation and scale both leave the origin untouched, then
// Translate(pivot) sends it back), world space ends up scaled by exactly
// scaleMM relative to native mesh units — i.e. world-space distances are
// millimetres directly, which is what the scale ruler relies on.
Mat4 ModelMatrix(const AppState& app) {
    return Mat4::Translation(app.modelPivot) * app.modelOrientation * Mat4::UniformScale(app.scaleMM) *
           Mat4::Translation(app.modelPivot * -1.0f);
}

void TransformPoint(const Mat4& m, const Vec3& p, Vec3& out) {
    out.x = m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12];
    out.y = m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13];
    out.z = m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14];
}

// Estimates the voxel grid's cell count for the *current* mesh pose/scale
// at the given resolution/padding, without running the real (expensive,
// triangle-rasterizing) Voxelize() pass — cheap enough to call every
// frame while the user is still adjusting sliders, which is the whole
// point (a warning that only appears after already committing to a slow
// voxelization isn't very useful). Transforms the mesh's own bounding-box
// corners (not every vertex — Voxelize() itself unions all of them, but
// for an affine transform the AABB of a box's 8 corners already exactly
// bounds the transformed box, and the mesh's vertices are entirely
// contained in the original box by construction) through the current
// ModelMatrix and takes their AABB. That can come out a little larger
// than Voxelize()'s own tighter bound once the model is rotated off-axis
// — fine, even correct, for a pre-commit warning: better to over-warn
// than let a run silently exceed the limit this was asked to catch.
// Mirrors Voxelizer::Voxelize's own grid-sizing math exactly (see its
// comment) so the estimate and the real run always agree.
size_t EstimateVoxelCellCount(const AppState& app, int resolution, const Vec3& paddingFactor) {
    if (!app.meshLoaded) return 0;

    Mat4 model = ModelMatrix(app);
    Vec3 lo = app.mesh.bounds.min, hi = app.mesh.bounds.max;
    Vec3 corners[8] = {{lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z},
                        {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {lo.x, hi.y, hi.z}, {hi.x, hi.y, hi.z}};
    Bounds worldBounds;
    for (const Vec3& c : corners) {
        Vec3 transformed;
        TransformPoint(model, c, transformed);
        worldBounds.Expand(transformed);
    }

    Vec3 size = worldBounds.Size();
    Vec3 pad{size.x * paddingFactor.x, size.y * paddingFactor.y, size.z * paddingFactor.z};
    Vec3 domainSize = size + pad * 2.0f;
    float longestAxis = std::max({domainSize.x, domainSize.y, domainSize.z});
    int res = std::max(resolution, 4);
    float cellSize = std::max(longestAxis / static_cast<float>(res), 1e-6f);

    size_t nx = std::max(static_cast<size_t>(std::ceil(domainSize.x / cellSize)), size_t(1));
    size_t ny = std::max(static_cast<size_t>(std::ceil(domainSize.y / cellSize)), size_t(1));
    size_t nz = std::max(static_cast<size_t>(std::ceil(domainSize.z / cellSize)), size_t(1));
    return nx * ny * nz;
}

// Largest resolution (within the Resolution slider's own
// [kMinVoxelResolution, kMaxVoxelResolution] range) whose
// EstimateVoxelCellCount still fits under the GPU backend's 4GiB-per-
// buffer cap for the *current* lattice model and padding — backs the
// "Max for GPU Memory" button, regardless of which backend happens to be
// selected right now (same reasoning as the warning text next to the
// slider: the limit is a property of the GPU backend, worth knowing even
// if you're not currently on it).
//
// LbmSolver's GPU backend splits its population buffers across Z-slabs
// (see LbmSolver.h's class comment), so pop_'s own 4GiB cap no longer
// applies at any resolution — growing the domain just means more,
// equally-small slabs. The one buffer that decomposition deliberately
// leaves unsplit, solidMask_ (cellCount*sizeof(uint32_t) bytes), is what
// this now checks — a far higher ceiling (~1024 cubic) than pop_'s old
// one (~480-520), but still well short of kMaxVoxelResolution, so unlike
// when that constant was 1024 itself this now generally DOES return a
// real, resolution-dependent answer rather than just the slider's own
// ceiling outright. Binary search is valid because cell count only ever
// grows with resolution — smaller cells covering the same padded domain
// always means more of them, never fewer. Returns the slider's own floor
// if even that doesn't fit; there's nothing more this can do about that
// case, the warning text next to the slider already says so.
//
// Despite the button's name, this is still specifically SDL_GPU's
// per-buffer API cap, not an estimate of how much GPU memory is actually
// available on this machine (SDL_GPU exposes no such query) — see the
// button's own tooltip in DrawUI for the distinction.
int FindMaxGpuResolution(const AppState& app, const Vec3& paddingFactor) {
    auto fitsUnderCap = [&](int resolution) {
        size_t cells = EstimateVoxelCellCount(app, resolution, paddingFactor);
        size_t solidBytes = cells * sizeof(uint32_t);
        return solidBytes <= static_cast<size_t>(UINT32_MAX);
    };

    if (!fitsUnderCap(kMinVoxelResolution)) return kMinVoxelResolution;
    int lo = kMinVoxelResolution, hi = kMaxVoxelResolution;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;  // biased toward hi so lo==hi terminates, never loops forever
        if (fitsUnderCap(mid)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// Launches `work` on a background thread so the caller (always the UI
// thread, from a button handler) can return immediately and keep
// rendering/responding to input while it runs. Before this existed,
// Voxelize/Initialize-or-Reset-Solver/Build-Overlay all ran straight on
// the UI thread — fine at low resolution, but at high resolution each is
// slow enough (seconds, sometimes much more) that the window would just
// freeze with no feedback at all, which is what DrawBusyOverlay now fixes.
// Only one task runs at a time — relevant buttons are disabled meanwhile
// (see DrawUI) — so `app.taskThread` is always safe to join here: if one
// is still joinable, the task it ran already cleared app.taskKind back to
// None (its last action), so it's guaranteed finished, not just started.
void StartTask(AppState& app, TaskKind kind, bool hasProgress, std::function<void()> work) {
    if (app.taskKind.load(std::memory_order_acquire) != TaskKind::None) return;
    if (app.taskThread.joinable()) app.taskThread.join();
    app.taskProgress.store(0.0f, std::memory_order_relaxed);
    app.taskHasProgress.store(hasProgress, std::memory_order_relaxed);
    app.taskStartTicks = SDL_GetTicks();
    app.taskKind.store(kind, std::memory_order_release);
    app.taskThread = std::thread([&app, work = std::move(work)]() {
        work();
        app.taskKind.store(TaskKind::None, std::memory_order_release);
    });
}

// Snapshots the mesh's *current* world-space pose (orientation + scale)
// into a solid/fluid grid, on a background thread (see StartTask). The
// actual Voxelizer:: calls are pure CPU work — no GPU/renderer object is
// touched — so the result is staged into app.pendingVoxel* instead of
// being applied directly: the main loop's "pending voxel result" block
// does that on the UI thread, since the GPU upload (voxelRenderer) must
// stay there like every other renderer mutation in this file.
void StartVoxelizeTask(AppState& app) {
    if (!app.meshLoaded) return;

    Mat4 model = ModelMatrix(app);
    std::vector<Vec3> worldVerts(app.mesh.vertices.size());
    for (size_t i = 0; i < app.mesh.vertices.size(); ++i) TransformPoint(model, app.mesh.vertices[i], worldVerts[i]);
    std::vector<uint32_t> indices = app.mesh.indices;  // snapshot: safe even if a new mesh loads mid-task
    int resolution = app.voxelResolution;
    Vec3 paddingFactor = app.voxelPaddingFactor;

    StartTask(app, TaskKind::Voxelizing, /*hasProgress=*/true,
              [&app, worldVerts = std::move(worldVerts), indices = std::move(indices), resolution, paddingFactor]() {
                  Voxelizer::Params params;
                  params.targetResolution = resolution;
                  params.paddingFactor = paddingFactor;
                  VoxelGrid grid = Voxelizer::Voxelize(worldVerts, indices, params, &app.taskProgress);

                  size_t solidCount = 0;
                  for (uint8_t s : grid.solid) solidCount += s;

                  Mesh preview = Voxelizer::BuildPreviewMesh(grid, &app.taskProgress);

                  std::lock_guard<std::mutex> lock(app.voxelResultMutex);
                  app.pendingVoxelGrid = std::move(grid);
                  app.pendingVoxelPreview = std::move(preview);
                  app.pendingVoxelSolidCount = solidCount;
                  app.voxelResultVersion.fetch_add(1, std::memory_order_release);
              });
}

// Clears the current voxelization — the inverse of StartVoxelizeTask.
// Cheap and synchronous (unlike Voxelize/Init), so unlike every other
// button in this section this runs directly on the UI thread with no
// background task: it only clears flags and a few small cached fields,
// never touches GPU resources itself. app.voxelRenderer's existing
// preview mesh is simply left unrendered — the same way app.voxelGridValid
// already gates it off whenever it's false — until the next Voxelize's
// UploadMesh naturally replaces it; there's no cheap way to free it
// early without tearing down and re-Init'ing the whole renderer, and
// leaving unused GPU memory sit idle until it's overwritten is the same
// tradeoff every other "invalidate, don't eagerly destroy" reset in this
// file already makes (e.g. app.lbmValid=false elsewhere never frees
// lbmSolver's buffers either — only a fresh Init() does, by replacing
// them).
//
// Invalidates the solver too, same reasoning as the "pending voxel
// result" block in the main loop applies when a *fresh* Voxelize
// replaces the grid: whatever solver exists was sized/seeded for the
// grid that's going away, so it can't be left looking valid.
void ClearFlowDivergence(AppState& app) {
    app.lbmFlowDivergedAtomic.store(false, std::memory_order_relaxed);
    app.lbmDivergedAtStepAtomic.store(0, std::memory_order_relaxed);
    app.divergenceBannerDismissedAtomic.store(false, std::memory_order_relaxed);
    app.lbmStatsHasNaNAtomic.store(false, std::memory_order_relaxed);
    app.overlayHasNaN = false;
    app.pendingOverlayHasNaN = false;
}

void FlagFlowDivergence(AppState& app) {
    app.lbmStatsHasNaNAtomic.store(true, std::memory_order_relaxed);
    if (!app.lbmFlowDivergedAtomic.exchange(true, std::memory_order_acq_rel)) {
        app.lbmDivergedAtStepAtomic.store(app.lbmStepCount.load(std::memory_order_relaxed),
                                           std::memory_order_relaxed);
        app.divergenceBannerDismissedAtomic.store(false, std::memory_order_relaxed);
        if (app.autoPauseOnDivergenceAtomic.load(std::memory_order_relaxed)) {
            app.lbmRunning.store(false, std::memory_order_release);
            app.showProcessingWindow = true;
        }
    }
}

void ResetVoxelization(AppState& app) {
    app.voxelGridValid = false;
    // Stopped BEFORE the grid goes away, so the solver thread launches no
    // further overlay updates against it; the exclusive lock then waits
    // out any one still in flight — see voxelGridMutex's comment.
    app.lbmValid = false;
    app.lbmRunning = false;
    {
        std::unique_lock<std::shared_mutex> gridLock(app.voxelGridMutex);
        app.voxelGrid = VoxelGrid{};
    }
    app.voxelSolidCount = 0;

    app.lbmForcesValidAtomic.store(false, std::memory_order_relaxed);
    app.overlayValid = false;
    app.polarResult = {};
    app.sectionPolarResult = {};
    app.panelSolver.Shutdown();
    ClearFlowDivergence(app);
    // -1 so each plane (whichever are still enabled — that choice itself
    // persists across re-voxelization, same as before) re-defaults to the
    // *new* grid's own center rather than keeping a position sized for
    // the grid that's going away.
    for (AppState::SlicePlaneState& plane : app.slicePlanes) {
        plane.index = -1;
        plane.indexAtomic.store(-1, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lock(app.lbmMutex);
        app.lbmInitError.clear();
    }

    {
        std::lock_guard<std::mutex> lock(app.lastFieldMutex);
        app.lastField.reset();
        app.cachedRangesValid = false;
    }
    app.smokeTrailInitialized = false;

    if (app.viewMode == ViewMode::VoxelGrid || app.viewMode == ViewMode::FlowOverlay) {
        app.viewMode = ViewMode::Model;
    }
}

// How much real time one lattice step represents, given the grid's cell
// size (mm, fixed at voxelize time) and the user's chosen real inlet speed.
// See kLatticeVelocityScale's comment for why this is a free, instant
// reinterpretation rather than something that requires re-simulating.
float LatticeDtSeconds(const AppState& app) {
    if (!app.voxelGridValid || app.lbmInletVelocityMmPerS <= 0.0f) return 0.0f;
    return kLatticeVelocityScale * app.voxelGrid.cellSize / app.lbmInletVelocityMmPerS;
}

float LatticeSpeedToMmPerSecond(const AppState& app, float latticeSpeed) {
    float dt = LatticeDtSeconds(app);
    if (dt <= 0.0f) return 0.0f;
    return latticeSpeed * app.voxelGrid.cellSize / dt;
}

// Kinematic viscosity (mm²/s) that the given tau corresponds to, given the
// current dt — purely informational, to help calibrate tau against a real
// fluid (air is about 15 mm²/s at room temperature). Takes tau explicitly
// rather than reading app.lbmParams.tau itself, since InitLbmSolver can now
// run on a background task thread (see StartInitSolverTask) — the caller
// is expected to have already read tau under app.lbmMutex.
float LatticeTauToViscosityMm2PerSecond(const AppState& app, float tau) {
    float dt = LatticeDtSeconds(app);
    if (dt <= 0.0f) return 0.0f;
    float nuLattice = (tau - 0.5f) / 3.0f;
    float dx = app.voxelGrid.cellSize;
    return nuLattice * dx * dx / dt;
}

// Tau that makes the solver's kinematic viscosity equal that of sea-level
// air (~15 mm²/s) for the current inlet speed and cell size — the inverse of
// LatticeTauToViscosityMm2PerSecond. Clamped to the slider's range; very low
// results rely on the Smagorinsky sub-grid model in the collision step for
// stability.
constexpr float kAirViscosityMm2PerS = 15.0f;
constexpr float kMinTau = 0.50001f;
float LatticeTauMatchingAir(const AppState& app) {
    float dt = LatticeDtSeconds(app);
    if (dt <= 0.0f) return 0.8f;
    float dx = app.voxelGrid.cellSize;
    float nuLattice = kAirViscosityMm2PerS * dt / (dx * dx);
    return std::min(2.0f, std::max(kMinTau, 0.5f + 3.0f * nuLattice));
}

// Pushes a new tau to whichever backend is active (same logic the Tau slider uses).
void ApplyTau(AppState& app, float tau) {
    std::lock_guard<std::mutex> lock(app.lbmMutex);
    app.lbmParams.tau = tau;
    app.lbmTauAtomic.store(tau, std::memory_order_relaxed);
    if (app.lbmValid) {
        if (app.lbmBackend == SolverBackend::GPU) {
            app.lbmSolver.SetParams(app.lbmParams);
        } else {
            app.lbmReference.SetParams(LbmReference::Params{
                app.lbmParams.tau, app.lbmParams.inletVelocity, app.lbmParams.reverseFlow,
                app.lbmParams.freeSlipY, app.lbmParams.freeSlipZ, app.lbmParams.smagorinskyCs});
        }
    }
}

// Physical scaling from dimensionless lattice stress/force to SI units
// (Pascals and Newtons) for standard sea-level air (rho_air = 1.225 kg/m³):
//   C_u = (v_inlet [m/s]) / kLatticeVelocityScale   [m/s per lattice speed unit]
//   sigma_Pa = rho_air * C_u^2 * sigma_lattice      [Pa = N/m²]
//   F_N      = rho_air * C_u^2 * (dx [m])^2 * F_lat [N]
constexpr float kAirDensityKgPerM3 = 1.225f;

float LatticeStressToPascals(const AppState& app, float latticeStress) {
    if (!app.voxelGridValid || app.lbmInletVelocityMmPerS <= 0.0f) return 0.0f;
    float cu = (app.lbmInletVelocityMmPerS * 1e-3f) / kLatticeVelocityScale;
    return kAirDensityKgPerM3 * cu * cu * latticeStress;
}

float LatticeForceToNewtons(const AppState& app, float latticeForce) {
    if (!app.voxelGridValid || app.lbmInletVelocityMmPerS <= 0.0f) return 0.0f;
    float dxMeters = app.voxelGrid.cellSize * 1e-3f;
    return LatticeStressToPascals(app, latticeForce) * dxMeters * dxMeters;
}

// Same split-by-Z-slab-across-hardware_concurrency()-threads pattern as
// Voxelizer.cpp/LbmReference.cpp/PotentialFlowSolver.cpp's own
// ParallelForZ (duplicated rather than shared across files, same
// reasoning as theirs) — used by SampleCoarseFieldOntoFineGrid below,
// the one CPU-bound loop in main.cpp costly enough (O(fine cell count))
// to want threading.
template <typename Body>
void ParallelForZ(int nz, Body body) {
    unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
    numThreads = std::min(numThreads, static_cast<unsigned int>(std::max(nz, 1)));
    int rangePerThread = (nz + static_cast<int>(numThreads) - 1) / static_cast<int>(numThreads);

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (unsigned int t = 0; t < numThreads; ++t) {
        int zBegin = static_cast<int>(t) * rangePerThread;
        int zEnd = std::min(zBegin + rangePerThread, nz);
        if (zBegin >= zEnd) break;
        threads.emplace_back(body, zBegin, zEnd);
    }
    for (std::thread& th : threads) th.join();
}

// Builds a per-fine-cell (rho,ux,uy,uz) seed field for ApplyWarmStart
// below by nearest-sampling `coarseField` (GLOBAL-indexed, 4 floats/cell,
// same layout ReadbackField() returns) at each fine cell's own world
// position. Both grids come from the same mesh and padding factor (just
// different target resolutions — see ApplyWarmStart), so their origins
// are identical and only cellSize/nx/ny/nz differ; nearest-cell (not
// trilinear) sampling, since the coarse field is already a rough
// approximation (see ApplyWarmStart's comment) — the extra smoothness
// trilinear would buy isn't worth the code here.
std::vector<float> SampleCoarseFieldOntoFineGrid(const VoxelGrid& fineGrid, const VoxelGrid& coarseGrid,
                                                  const std::vector<float>& coarseField) {
    std::vector<float> fineSeed(fineGrid.CellCount() * 4);
    float invCoarseCellSize = 1.0f / coarseGrid.cellSize;
    ParallelForZ(fineGrid.nz, [&](int zBegin, int zEnd) {
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < fineGrid.ny; ++y) {
                for (int x = 0; x < fineGrid.nx; ++x) {
                    Vec3 worldCenter = fineGrid.CellCenter(x, y, z);
                    // Inverse of CellCenter's own mapping (VoxelGrid.h) —
                    // grid-x comes from world Y and grid-y from world X —
                    // so converting this world position back to the
                    // coarse grid's indices needs the same swap a plain
                    // per-component divide doesn't give. Missing this
                    // sampled each fine cell's seed (rho AND the ux/uy
                    // velocity components) from the wrong coarse cell
                    // entirely — the resulting velocity vectors were
                    // real values from *somewhere* in the coarse field,
                    // just not from anywhere near this fine cell, which
                    // is exactly "particles moving in strange directions"
                    // right after a warm-started Run: the seed field
                    // itself was scrambled, not the solver stepping it.
                    Vec3 local = (worldCenter - coarseGrid.origin) * invCoarseCellSize;
                    int cx = std::clamp(static_cast<int>(std::floor(local.y)), 0, coarseGrid.nx - 1);
                    int cy = std::clamp(static_cast<int>(std::floor(local.x)), 0, coarseGrid.ny - 1);
                    int cz = std::clamp(static_cast<int>(std::floor(local.z)), 0, coarseGrid.nz - 1);
                    size_t coarseCell = coarseGrid.Index(cx, cy, cz);
                    size_t fineCell = fineGrid.Index(x, y, z);
                    // Coarse solid cells store (stress, Fx, Fy, Fz) in
                    // macroOut rather than (rho, ux, uy, uz) — if a fine
                    // fluid cell nearest-samples a coarse solid cell near
                    // the boundary, seed it with quiescent fluid instead.
                    if (coarseGrid.solid[coarseCell]) {
                        fineSeed[fineCell * 4 + 0] = 1.0f;
                        fineSeed[fineCell * 4 + 1] = 0.0f;
                        fineSeed[fineCell * 4 + 2] = 0.0f;
                        fineSeed[fineCell * 4 + 3] = 0.0f;
                    } else {
                        fineSeed[fineCell * 4 + 0] = coarseField[coarseCell * 4 + 0];
                        fineSeed[fineCell * 4 + 1] = coarseField[coarseCell * 4 + 1];
                        fineSeed[fineCell * 4 + 2] = coarseField[coarseCell * 4 + 2];
                        fineSeed[fineCell * 4 + 3] = coarseField[coarseCell * 4 + 3];
                    }
                }
            }
        }
    });
    return fineSeed;
}

// "Warm start": rather than seed the real (fine) grid uniformly
// (quiescent or mid-speed — see AppState::lbmSeedMidSpeed), voxelize the
// SAME mesh again at a much coarser resolution
// (app.voxelResolution/app.warmStartResolutionDivisor), run a quick solve
// on that coarse grid for app.warmStartSteps steps, and nearest-sample
// ITS field onto the real grid as its seed instead — a cheap, rough
// approximation of the flow's steady state beats starting from nothing,
// even if it's not an exact one. Called from InitLbmSolver, under the
// same app.lbmMutex lock that function already holds for its entire
// body; the temporary coarse solver below is purely local to this call
// (never stored in AppState), so needs no locking of its own, and
// app.lbmRunning is already false by the time InitLbmSolver gets here —
// same assumption its own Init()/Reset() calls already rely on for safe
// GPU access.
//
// PHYSICAL CAVEAT: tau is fixed regardless of resolution, so lattice
// viscosity (nu = (tau-0.5)/3) doesn't change — but the model's size *in
// lattice cells* shrinks by warmStartResolutionDivisor. Since lattice
// Reynolds number is (lattice speed * lattice length)/nu, the coarse run
// ends up at roughly 1/warmStartResolutionDivisor the fine run's
// effective Reynolds number — not a scaled-down version of the same
// flow, just a rough approximation of roughly where the wake/
// acceleration regions are. Still a much better starting point than
// uniform quiescent/mid-speed in practice, just not literally the fine
// grid's own steady state.
//
// Only meaningful for the two backends with actual transient relaxation
// dynamics — returns false immediately for Naive (PotentialFlowSolver
// solves directly via SOR, not by relaxing toward equilibrium over many
// steps the way LBM does), and on any failure along the way (degenerate
// coarse grid, GPU allocation failure, etc.), so InitLbmSolver's callers
// can cleanly fall back to the normal uniform Reset() rather than being
// left with an unseeded solver.
bool ApplyWarmStart(AppState& app) {
    if (!IsLbmBackend(app.lbmBackend)) return false;

    Mat4 model = ModelMatrix(app);
    std::vector<Vec3> worldVerts(app.mesh.vertices.size());
    for (size_t i = 0; i < app.mesh.vertices.size(); ++i) TransformPoint(model, app.mesh.vertices[i], worldVerts[i]);

    Voxelizer::Params coarseParams;
    coarseParams.targetResolution = std::max(app.voxelResolution / std::max(app.warmStartResolutionDivisor, 1), 4);
    coarseParams.paddingFactor = app.voxelPaddingFactor;
    VoxelGrid coarseGrid = Voxelizer::Voxelize(worldVerts, app.mesh.indices, coarseParams, nullptr);
    if (coarseGrid.CellCount() == 0) return false;

    std::vector<float> coarseField;
    if (app.lbmBackend == SolverBackend::GPU) {
        if (!app.gpuDevice) return false;
        LbmSolver coarseSolver;
        if (!coarseSolver.Init(app.gpuDevice, coarseGrid, app.lbmModel)) return false;
        coarseSolver.Reset(app.lbmParams, app.lbmSeedMidSpeed);

        // Chunked the same way StepSolverPaced chunks the real solver —
        // not for frame-pacing (this already runs on a background task,
        // nothing is waiting on a frame), just to avoid queuing
        // warmStartSteps dispatches into a single command buffer with no
        // intermediate submission.
        constexpr int kChunkSubsteps = 100;
        for (int done = 0; done < app.warmStartSteps;) {
            int chunk = std::min(kChunkSubsteps, app.warmStartSteps - done);
            SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(app.gpuDevice);
            coarseSolver.Step(cmd, chunk);
            SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
            SDL_WaitForGPUFences(app.gpuDevice, true, &fence, 1);
            SDL_ReleaseGPUFence(app.gpuDevice, fence);
            done += chunk;
            // Free to report here despite being the GPU path — this loop
            // already blocks on a fence every chunk (for pacing, not for
            // this), so `done` is always an exact, already-synced count,
            // not an estimate. Capped at 0.9 so the readback/fine-grid
            // sampling/ResetFromField that follows (and InitLbmSolver's own
            // allocation before ApplyWarmStart is even called) still has
            // visible room left before the task reports 100%.
            app.taskProgress.store(0.9f * static_cast<float>(done) / static_cast<float>(app.warmStartSteps),
                                    std::memory_order_relaxed);
        }
        coarseField = coarseSolver.ReadbackField();
    } else {
        LbmReference coarseSolver;
        LbmReference::Params refParams{app.lbmParams.tau, app.lbmParams.inletVelocity, app.lbmParams.reverseFlow,
                                        app.lbmParams.freeSlipY, app.lbmParams.freeSlipZ, app.lbmParams.smagorinskyCs};
        coarseSolver.Init(coarseGrid, app.lbmModel, refParams, app.lbmSeedMidSpeed);
        for (int i = 0; i < app.warmStartSteps; ++i) {
            coarseSolver.Step();
            app.taskProgress.store(0.9f * static_cast<float>(i + 1) / static_cast<float>(app.warmStartSteps),
                                    std::memory_order_relaxed);
        }
        coarseField = coarseSolver.ReadbackField();
    }
    if (coarseField.empty()) return false;

    std::vector<float> fineSeed = SampleCoarseFieldOntoFineGrid(app.voxelGrid, coarseGrid, coarseField);

    if (app.lbmBackend == SolverBackend::GPU) {
        app.lbmSolver.SetParams(app.lbmParams);
        app.lbmSolver.ResetFromField(fineSeed);
    } else {
        app.lbmReference.ResetFromField(fineSeed);
    }
    return true;
}

// Mirrors app.lbmLastStats' fields into their own atomics — see those
// fields' own comment for why. Call right after any write to
// app.lbmLastStats, under the same app.lbmMutex lock that write is
// already under (no new locking here; these stores are just along for
// the ride).
void SyncLbmStatsAtomics(AppState& app) {
    app.lbmStatsAvgDensityAtomic.store(app.lbmLastStats.avgDensity, std::memory_order_relaxed);
    app.lbmStatsMinDensityAtomic.store(app.lbmLastStats.minDensity, std::memory_order_relaxed);
    app.lbmStatsMaxDensityAtomic.store(app.lbmLastStats.maxDensity, std::memory_order_relaxed);
    app.lbmStatsAvgSpeedAtomic.store(app.lbmLastStats.avgSpeed, std::memory_order_relaxed);
    app.lbmStatsMaxSpeedAtomic.store(app.lbmLastStats.maxSpeed, std::memory_order_relaxed);
    app.lbmStatsHasNaNAtomic.store(app.lbmLastStats.hasNaN, std::memory_order_relaxed);
    app.lbmStatsValidAtomic.store(app.lbmLastStats.valid, std::memory_order_relaxed);
    app.lbmForceXAtomic.store(app.lbmLastStats.forceX, std::memory_order_relaxed);
    app.lbmForceYAtomic.store(app.lbmLastStats.forceY, std::memory_order_relaxed);
    app.lbmForceZAtomic.store(app.lbmLastStats.forceZ, std::memory_order_relaxed);
    app.lbmMaxStressAtomic.store(app.lbmLastStats.maxStress, std::memory_order_relaxed);
    app.lbmCpXAtomic.store(app.lbmLastStats.centerOfPressure.x, std::memory_order_relaxed);
    app.lbmCpYAtomic.store(app.lbmLastStats.centerOfPressure.y, std::memory_order_relaxed);
    app.lbmCpZAtomic.store(app.lbmLastStats.centerOfPressure.z, std::memory_order_relaxed);
    app.lbmLiftBaseZAtomic.store(app.lbmLastStats.liftBaseZ, std::memory_order_relaxed);
    app.lbmForcesValidAtomic.store(app.lbmLastStats.valid, std::memory_order_relaxed);
}

// (Re)initializes the solver for the current voxel grid and lattice model.
// A structurally different action from SetParams()/tau changes — switching
// model or grid means different buffer sizes, so this always reallocates,
// while tweaking tau (or the real inlet speed, which the solver doesn't
// even see — see kLatticeVelocityScale) mid-run doesn't need it.
void InitLbmSolver(AppState& app) {
    // Held for the whole call (Init/warm start read app.voxelGrid
    // throughout) — see voxelGridMutex's comment. Taken before lbmMutex.
    std::shared_lock<std::shared_mutex> gridLock(app.voxelGridMutex);
    if (!app.voxelGridValid) return;
    if (IsGpuBackend(app.lbmBackend) && !app.gpuDevice) return;
    app.lbmParams.inletVelocity = kLatticeVelocityScale;
    app.lbmRunning = false;
    // Locked so this can't reallocate buffers out from under the worker
    // thread mid-Step/ReadbackField — see app.lbmMutex's comment. Also
    // guards app.lbmInitError, read by the UI under the same lock (see
    // DrawUI) since this can now run on a background task thread.
    std::lock_guard<std::mutex> lock(app.lbmMutex);
    ++app.lbmSolverEpoch;  // see its comment — tells a mid-substep StepSolverPaced to abandon that substep

    if (app.lbmBackend == SolverBackend::GPU) {
        app.lbmValid = app.lbmSolver.Init(app.gpuDevice, app.voxelGrid, app.lbmModel);
        if (app.lbmValid) {
            // ApplyWarmStart already calls SetParams()+ResetFromField()
            // itself on success; falls back to the normal uniform Reset()
            // on any failure (degenerate coarse grid, allocation failure,
            // etc.) — see its own comment.
            if (!app.warmStartEnabled || !ApplyWarmStart(app)) {
                app.lbmSolver.Reset(app.lbmParams, app.lbmSeedMidSpeed);
            }
            app.lbmInitError.clear();
        } else {
            app.lbmInitError = app.lbmSolver.LastError();
            if (app.lbmInitError.empty()) app.lbmInitError = "Initialization failed for an unknown reason.";
        }
    } else if (app.lbmBackend == SolverBackend::CPU) {
        // LbmReference::Init() is void (it just resizes a few vectors) and
        // has no 4GiB-per-buffer ceiling the way LbmSolver does — system
        // RAM is the only real limit — but a huge enough grid can still
        // throw std::bad_alloc, so this still needs to fail cleanly rather
        // than crash.
        try {
            LbmReference::Params refParams{app.lbmParams.tau, app.lbmParams.inletVelocity, app.lbmParams.reverseFlow,
                                            app.lbmParams.freeSlipY, app.lbmParams.freeSlipZ, app.lbmParams.smagorinskyCs};
            app.lbmReference.Init(app.voxelGrid, app.lbmModel, refParams, app.lbmSeedMidSpeed);
            // Init() above already seeded uniformly; if warm start
            // succeeds it just overwrites that with ApplyWarmStart's own
            // ResetFromField() call — a little redundant work on the
            // uniform seed when warm start is on, not worth restructuring
            // Init()'s long-standing always-resets contract to avoid.
            if (app.warmStartEnabled) ApplyWarmStart(app);
            app.lbmValid = true;
            app.lbmInitError.clear();
        } catch (const std::exception& e) {
            app.lbmValid = false;
            app.lbmInitError = std::string("CPU solver failed to allocate: ") + e.what();
        }
    } else if (IsPanelBackend(app.lbmBackend)) {
        try {
            bool useGpu = (app.lbmBackend == SolverBackend::PanelGPU);
            PanelSolver::Params panelParams{app.lbmParams.inletVelocity, app.lbmParams.reverseFlow};
            app.lbmValid = app.panelSolver.Init(app.gpuDevice, useGpu, app.voxelGrid, panelParams);
            if (app.lbmValid) {
                app.lbmInitError.clear();
            } else {
                app.lbmInitError = app.panelSolver.LastError();
                if (app.lbmInitError.empty()) app.lbmInitError = "Panel solver initialization failed.";
            }
        } catch (const std::exception& e) {
            app.lbmValid = false;
            app.lbmInitError = std::string("Panel solver failed to allocate: ") + e.what();
        }
    } else {
        // PotentialFlowSolver::Init() only allocates one scalar field (plus
        // the solid mask) — see its header comment — but the same
        // bad_alloc risk at extreme resolutions still applies.
        try {
            PotentialFlowSolver::Params naiveParams{app.lbmParams.inletVelocity, app.lbmParams.reverseFlow};
            app.potentialFlow.Init(app.voxelGrid, naiveParams);
            app.lbmValid = true;
            app.lbmInitError.clear();
        } catch (const std::exception& e) {
            app.lbmValid = false;
            app.lbmInitError = std::string("Naive solver failed to allocate: ") + e.what();
        }
    }

    if (app.lbmValid) {
        app.lbmStepCount = 0;
        app.lbmLastStats = LbmSolver::Stats{};
        SyncLbmStatsAtomics(app);
        ClearFlowDivergence(app);
    }

    // Unconditional final snap to 100%, regardless of which path above
    // ran: ApplyWarmStart (when it actually runs) only ever reports up to
    // 90% (see its own comment) to leave room for this; when warm start is
    // off, failed, or this is the Naive backend (which never calls it at
    // all), there's no cheap intermediate checkpoint to report instead, so
    // the bar just jumps straight to done once allocation/Reset() finishes
    // — still correct, just not gradual.
    app.taskProgress.store(1.0f, std::memory_order_relaxed);
}

// Runs InitLbmSolver on a background thread (see StartTask) — at high
// resolution its buffer allocation and (especially) Reset()'s population
// upload are slow enough to otherwise freeze the window with no feedback.
// Reports real fractional progress whenever warm start actually runs (see
// ApplyWarmStart's own progress-reporting comment, for both the GPU and
// CPU paths) — allocation/Reset()/the plain-uniform-seed fallback have no
// equivalent cheap checkpoint, so those stretches just hold at whatever
// fraction was last reported until InitLbmSolver's own final snap to 100%.
void StartInitSolverTask(AppState& app) {
    if (!app.voxelGridValid) return;
    StartTask(app, TaskKind::InitializingSolver, /*hasProgress=*/true, [&app]() { InitLbmSolver(app); });
}

void StabilizeAndResetSolver(AppState& app) {
    if (!app.voxelGridValid) return;
    {
        std::lock_guard<std::mutex> lock(app.lbmMutex);
        app.lbmParams.smagorinskyCs = std::max(app.lbmParams.smagorinskyCs, 0.18f);
        app.lbmParams.tau = std::max(app.lbmParams.tau, 0.53f);
        app.lbmTauAtomic.store(app.lbmParams.tau, std::memory_order_relaxed);
    }
    StartInitSolverTask(app);
}

int OverlaySliceDimension(const AppState& app, SliceAxis axis) {
    switch (axis) {
        case SliceAxis::X:
            return app.voxelGrid.nx;
        case SliceAxis::Y:
            return app.voxelGrid.ny;
        case SliceAxis::Z:
        default:
            return app.voxelGrid.nz;
    }
}

// Keeps the voxel cloud's cube count — and so its build/upload/render
// cost — bounded regardless of grid resolution. See
// FieldOverlay::ChooseVoxelCloudStride.
constexpr size_t kVoxelCloudTargetCount = 120000;

// A resolved (enabled, index) pair for one axis's slice plane — see
// AppState::SlicePlaneState. BuildOverlayMeshFromField and TryFastReslice
// each resolve app.slicePlanes their own way (atomic vs. plain reads, same
// reasoning as every other such pair in this file) into this common shape
// before handing it to BuildMeshForStyle/BuildCombinedSliceMesh, so
// neither of those needs to know which kind of read produced it.
struct ResolvedSlicePlane {
    bool enabled;
    int index;
};

constexpr std::array<SliceAxis, 3> kSliceAxes = {SliceAxis::X, SliceAxis::Y, SliceAxis::Z};

// Builds and merges the Slice style's mesh across however many of the 3
// planes are currently enabled (see ResolvedSlicePlane/AppState::
// slicePlanes) — a disabled plane contributes nothing, so this can return
// an empty mesh if none are enabled, same as "nothing to show yet".
OverlayMesh BuildCombinedSliceMesh(const VoxelGrid& grid, const std::vector<float>& field,
                                     const std::array<ResolvedSlicePlane, 3>& planes, FieldColorMode mode, float lo,
                                     float hi, const std::vector<float>* xMeans) {
    OverlayMesh combined;
    for (size_t i = 0; i < 3; ++i) {
        if (!planes[i].enabled) continue;
        OverlayMesh m = FieldOverlay::BuildSliceMeshFromRange(grid, field, kSliceAxes[i], planes[i].index, mode, lo,
                                                                 hi, xMeans);
        FieldOverlay::AppendMesh(combined, m);
    }
    return combined;
}

// Builds whichever mesh app.overlayStyle currently calls for, from
// already-known ranges — both are always computed together (see
// BuildOverlayMeshFromField) because Voxel Cloud needs both at once
// (color from Speed, alpha from Pressure — see its header comment);
// which shape to build, and which range(s) it actually uses, is the only
// thing that differs between styles. `pressureXMeans` is null when
// app.detrendPressure is off, otherwise the caller has already computed
// pressureLo/pressureHi against that same detrended basis (see
// BuildOverlayMeshFromField/TryFastReslice) — forwarded to
// FieldOverlay's Pressure-consuming builders, a no-op wherever Pressure
// isn't actually used.
OverlayMesh BuildMeshForStyle(const AppState& app, const std::vector<float>& field, OverlayStyle style,
                                const std::array<ResolvedSlicePlane, 3>& slicePlanes, FieldColorMode mode,
                                float speedLo, float speedHi, float pressureLo, float pressureHi, float stressLo,
                                float stressHi, float alphaLo, float alphaHi,
                                const std::vector<float>* pressureXMeans, bool highlightNegativePressure,
                                bool sliceShowParticles) {
    if (style == OverlayStyle::Particles || (style == OverlayStyle::Slice && sliceShowParticles)) {
        // Particles' own geometry (and Slice's, when "Also show
        // particles" replaces the colored cross-section with the smoke
        // trail confined to it — see SmokeTrailSystem::PlaneConfinement)
        // is built fresh every rendered frame by UpdateSmokeTrailAndUpload,
        // not on demand through this path — this call still exists (via
        // BuildOverlayMeshFromField/TryFastReslice) purely to seed
        // app.overlayValid and the cached Speed/Pressure ranges Particles
        // reads, same as every style.
        return OverlayMesh{};
    }
    if (style == OverlayStyle::StressMap) {
        return FieldOverlay::BuildStressMapMesh(app.voxelGrid, field, stressLo, stressHi);
    }
    if (style == OverlayStyle::IsoSurface) {
        float lo = mode == FieldColorMode::Speed ? speedLo : pressureLo;
        float hi = mode == FieldColorMode::Speed ? speedHi : pressureHi;
        float thresh = app.isoThresholdAtomic.load(std::memory_order_relaxed);
        float isoVal = lo + thresh * (hi - lo);
        float alpha = app.isoAlphaAtomic.load(std::memory_order_relaxed);
        bool encloseGreater = app.isoEncloseGreaterAtomic.load(std::memory_order_relaxed);
        int stride = app.isoStrideAtomic.load(std::memory_order_relaxed);
        const std::vector<float>* xMeans = mode == FieldColorMode::Pressure ? pressureXMeans : nullptr;
        return FieldOverlay::BuildIsoSurfaceMesh(app.voxelGrid, field, mode, isoVal, lo, hi,
                                                alpha, encloseGreater, stride, xMeans);
    }
    if (style == OverlayStyle::Raymarch) {
        float lo = mode == FieldColorMode::Speed ? speedLo : pressureLo;
        float hi = mode == FieldColorMode::Speed ? speedHi : pressureHi;
        Vec3 camForward{app.cameraForwardXAtomic.load(std::memory_order_relaxed),
                         app.cameraForwardYAtomic.load(std::memory_order_relaxed),
                         app.cameraForwardZAtomic.load(std::memory_order_relaxed)};
        Vec3 camRight = Normalize(Cross(camForward, Vec3{0.0f, 0.0f, 1.0f}));
        Vec3 camUp = Cross(camRight, camForward);
        Vec3 camPos{app.cameraPosXAtomic.load(std::memory_order_relaxed),
                     app.cameraPosYAtomic.load(std::memory_order_relaxed),
                     app.cameraPosZAtomic.load(std::memory_order_relaxed)};
        int sliceCount = app.raymarchSliceCountAtomic.load(std::memory_order_relaxed);
        float density = app.raymarchDensityAtomic.load(std::memory_order_relaxed);
        float cutoff = app.raymarchCutoffAtomic.load(std::memory_order_relaxed);
        float gamma = app.raymarchGammaAtomic.load(std::memory_order_relaxed);
        const std::vector<float>* xMeans = mode == FieldColorMode::Pressure ? pressureXMeans : nullptr;
        return FieldOverlay::BuildRaymarchMesh(app.voxelGrid, field, mode, lo, hi,
                                              camPos, camForward, camRight, camUp,
                                              sliceCount, density, cutoff, gamma, xMeans);
    }
    if (style == OverlayStyle::VoxelCloud) {
        int stride = FieldOverlay::ChooseVoxelCloudStride(app.voxelGrid, kVoxelCloudTargetCount);
        // Reconstructed from the cached atomic (not app.camera directly —
        // see cameraForwardXAtomic's own comment) via the same
        // right/up-from-forward formula OrbitCamera::GetBasis() uses, so
        // the fuzzy billboards face the camera the same way the
        // orientation gizmo/scale ruler already do.
        Vec3 camForward{app.cameraForwardXAtomic.load(std::memory_order_relaxed),
                         app.cameraForwardYAtomic.load(std::memory_order_relaxed),
                         app.cameraForwardZAtomic.load(std::memory_order_relaxed)};
        Vec3 camRight = Normalize(Cross(camForward, Vec3{0.0f, 0.0f, 1.0f}));
        Vec3 camUp = Cross(camRight, camForward);
        return FieldOverlay::BuildVoxelCloudMesh(app.voxelGrid, field, speedLo, speedHi, alphaLo, alphaHi, stride,
                                                   camRight, camUp, pressureXMeans, highlightNegativePressure);
    }
    float lo = mode == FieldColorMode::Speed ? speedLo : pressureLo;
    float hi = mode == FieldColorMode::Speed ? speedHi : pressureHi;
    const std::vector<float>* xMeans = mode == FieldColorMode::Pressure ? pressureXMeans : nullptr;
    return BuildCombinedSliceMesh(app.voxelGrid, field, slicePlanes, mode, lo, hi, xMeans);
}

// What the legend/"range [...]" readout should show for the current
// style+mode combination — Slice shows whichever range it's actually
// colored by; Voxel Cloud is always colored by Speed (see
// BuildMeshForStyle), so that's what its legend reflects too, even though
// its *opacity* is driven by Pressure through the separate alpha-window
// controls (not the legend at all).
void LegendRangeForStyle(OverlayStyle style, FieldColorMode mode, bool sliceShowParticles, float speedLo,
                          float speedHi, bool speedHasNaN, float pressureLo, float pressureHi, bool pressureHasNaN,
                          float stressLo, float stressHi, bool stressHasNaN, float& outLo, float& outHi,
                          bool& outHasNaN) {
    if (style == OverlayStyle::StressMap) {
        outLo = stressLo;
        outHi = stressHi;
        outHasNaN = stressHasNaN;
        return;
    }
    // Particles is colored by Speed exactly like Voxel Cloud (and also
    // has Pressure driving a second channel — trail opacity rather than
    // cube alpha, but the same idea) — see SmokeTrailSystem::BuildMesh.
    // Slice with "Also show particles" on reads the same way: it's the
    // smoke trail, not the colored cross-section, that's actually on
    // screen there — see BuildMeshForStyle's matching condition.
    bool particleLike = style == OverlayStyle::Particles || (style == OverlayStyle::Slice && sliceShowParticles);
    bool useSpeed = (style == OverlayStyle::VoxelCloud) || (!particleLike && mode == FieldColorMode::Speed);
    outLo = useSpeed ? speedLo : pressureLo;
    outHi = useSpeed ? speedHi : pressureHi;
    outHasNaN = (style == OverlayStyle::VoxelCloud || particleLike) ? (speedHasNaN || pressureHasNaN)
                                                                    : (useSpeed ? speedHasNaN : pressureHasNaN);
}

// Builds the current style's mesh from an already-read-back field and
// stages it for the UI thread to upload — never called from the UI
// thread itself (StartOverlayAutoUpdate's own thread calls it for
// overlayAutoUpdate, a Build-Overlay task thread for the manual button),
// which is exactly the point: this is the CPU-heavy part (a full-field
// scan for the color range, plus the mesh itself — see FieldOverlay.h),
// and running it on the UI thread was a real, resolution-scaling source
// of "freezes between solver updates" that none of the GPU-dispatch
// pacing work touched, because it was never a GPU cost to begin with. Nor
// does it run on the solver's own stepping thread any more — see
// StartOverlayAutoUpdate's comment for why that also used to eat directly
// into step throughput.
//
// Reads only the *Atomic mirrors of the UI-bound style/slice axis/index/
// color-mode/alpha-window fields (see AppState's comment on them), never
// those fields directly, since this can now run on a different thread
// than the one that owns them.
void BuildOverlayMeshFromField(AppState& app, const std::vector<float>& field) {
    if (field.empty()) return;
    // app.voxelGrid is read throughout, on a thread the UI's "busy" gating
    // doesn't cover — see voxelGridMutex's comment. A field read back for
    // a grid that has since been replaced (different size) is dropped
    // outright rather than cached as lastField / published as forces.
    std::shared_lock<std::shared_mutex> gridLock(app.voxelGridMutex);
    if (field.size() != app.voxelGrid.CellCount() * 4) return;

    OverlayStyle style = static_cast<OverlayStyle>(app.overlayStyleAtomic.load(std::memory_order_relaxed));
    FieldColorMode mode = static_cast<FieldColorMode>(app.overlayColorModeAtomic.load(std::memory_order_relaxed));
    // DrawUI defaults each plane's index to its axis's grid center before
    // it's ever negative here.
    std::array<ResolvedSlicePlane, 3> planes;
    for (size_t i = 0; i < 3; ++i) {
        planes[i].enabled = app.slicePlanes[i].enabledAtomic.load(std::memory_order_relaxed);
        planes[i].index = app.slicePlanes[i].indexAtomic.load(std::memory_order_relaxed);
    }
    float alphaLo = app.voxelAlphaPressureLoAtomic.load(std::memory_order_relaxed);
    float alphaHi = app.voxelAlphaPressureHiAtomic.load(std::memory_order_relaxed);
    bool sliceShowParticles = app.sliceShowParticlesAtomic.load(std::memory_order_relaxed);

    // Both ranges always, regardless of style/mode — see this function's
    // and BuildMeshForStyle's comments for why. Pressure's detrended
    // variant (see FieldOverlay::ComputeXSliceMeans) is also always
    // computed alongside it, so toggling app.detrendPressure later can
    // re-slice through TryFastReslice's cache instead of a fresh rebuild.
    float speedLo = 0.0f, speedHi = 0.0f, pressureLo = 0.0f, pressureHi = 0.0f;
    bool speedHasNaN = false, pressureHasNaN = false;
    FieldOverlay::ComputeRange(app.voxelGrid, field, FieldColorMode::Speed, speedLo, speedHi, speedHasNaN);
    FieldOverlay::ComputeRange(app.voxelGrid, field, FieldColorMode::Pressure, pressureLo, pressureHi,
                                pressureHasNaN);

    std::vector<float> pressureXMeans =
        FieldOverlay::ComputeXSliceMeans(app.voxelGrid, field, FieldColorMode::Pressure);
    float pressureDetrendLo = 0.0f, pressureDetrendHi = 0.0f;
    // Reuses pressureHasNaN: detrending subtracts a finite per-slice mean,
    // so it can never change which cells are finite — the same cells are
    // skipped either way.
    FieldOverlay::ComputeRange(app.voxelGrid, field, FieldColorMode::Pressure, pressureDetrendLo, pressureDetrendHi,
                                pressureHasNaN, &pressureXMeans);

    FieldOverlay::AerodynamicForces aero = FieldOverlay::ComputeAerodynamicForces(app.voxelGrid, field);
    app.lbmForceXAtomic.store(aero.forceX, std::memory_order_relaxed);
    app.lbmForceYAtomic.store(aero.forceY, std::memory_order_relaxed);
    app.lbmForceZAtomic.store(aero.forceZ, std::memory_order_relaxed);
    app.lbmMaxStressAtomic.store(aero.maxStress, std::memory_order_relaxed);
    app.lbmCpXAtomic.store(aero.centerOfPressure.x, std::memory_order_relaxed);
    app.lbmCpYAtomic.store(aero.centerOfPressure.y, std::memory_order_relaxed);
    app.lbmCpZAtomic.store(aero.centerOfPressure.z, std::memory_order_relaxed);
    app.lbmLiftBaseZAtomic.store(aero.liftArrowBase.z, std::memory_order_relaxed);
    app.lbmForcesValidAtomic.store(true, std::memory_order_relaxed);

    if (speedHasNaN || pressureHasNaN || aero.hasNaN) {
        FlagFlowDivergence(app);
    }

    bool detrend = app.detrendPressureAtomic.load(std::memory_order_relaxed);
    float pressureLoUsed = detrend ? pressureDetrendLo : pressureLo;
    float pressureHiUsed = detrend ? pressureDetrendHi : pressureHi;
    const std::vector<float>* xMeansUsed = detrend ? &pressureXMeans : nullptr;
    bool highlightNegative = app.voxelNegativePressureGreyAtomic.load(std::memory_order_relaxed);

    OverlayMesh mesh = BuildMeshForStyle(app, field, style, planes, mode, speedLo, speedHi, pressureLoUsed,
                                          pressureHiUsed, aero.minStress, aero.maxStress, alphaLo, alphaHi,
                                          xMeansUsed, highlightNegative, sliceShowParticles);

    float legendLo, legendHi;
    bool legendHasNaN;
    LegendRangeForStyle(style, mode, sliceShowParticles, speedLo, speedHi, speedHasNaN, pressureLoUsed,
                         pressureHiUsed, pressureHasNaN, aero.minStress, aero.maxStress, aero.hasNaN,
                         legendLo, legendHi, legendHasNaN);

    // Cached (field + both ranges) so StartResliceTask/TryFastReslice can
    // rebuild from this same field later — e.g. moving the slice position
    // while paused, adjusting the Voxel Cloud alpha window, or switching
    // Color By/style, while running but between auto-updates — without
    // needing a fresh solver readback or another ComputeRange pass. A
    // shared_ptr specifically so that cheap fast-path rebuild never has
    // to copy the field — see AppState's comment.
    //
    // The copy itself happens *before* taking the lock, deliberately: at
    // high resolution `field` can be close to a GiB, and copying that is
    // a genuinely slow, single-threaded memcpy — doing it while holding
    // lastFieldMutex meant every reader of app.lastField blocked for the
    // copy's entire duration, not just for the cheap pointer swap below.
    // That reader includes UpdateSmokeTrailAndUpload, which (for the
    // Particles style) runs on the render thread *every rendered frame*
    // — unlike Slice/Voxel Cloud, which only touch this lock on a
    // reslice. So this specific copy, done under the lock, showed up as
    // the render thread freezing for the copy's whole duration on every
    // single auto-update, regardless of the update interval (the copy
    // itself doesn't get any shorter) — exactly the symptom this fixes.
    auto fieldCopy = std::make_shared<std::vector<float>>(field);
    {
        std::lock_guard<std::mutex> lock(app.lastFieldMutex);
        app.lastField = std::move(fieldCopy);
        app.cachedSpeedMin = speedLo;
        app.cachedSpeedMax = speedHi;
        app.cachedSpeedHasNaN = speedHasNaN;
        app.cachedPressureMin = pressureLo;
        app.cachedPressureMax = pressureHi;
        app.cachedPressureHasNaN = pressureHasNaN;
        app.cachedPressureXMeans = std::move(pressureXMeans);
        app.cachedPressureDetrendMin = pressureDetrendLo;
        app.cachedPressureDetrendMax = pressureDetrendHi;
        app.cachedStressMin = aero.minStress;
        app.cachedStressMax = aero.maxStress;
        app.cachedStressHasNaN = aero.hasNaN;
        app.cachedRangesValid = true;
    }

    std::lock_guard<std::mutex> lock(app.overlayMeshMutex);
    app.pendingOverlayMesh = std::move(mesh);
    app.pendingOverlayMin = legendLo;
    app.pendingOverlayMax = legendHi;
    app.pendingOverlayHasNaN = legendHasNaN;
    app.overlayMeshVersion.fetch_add(1, std::memory_order_release);
}

// The fast path for StartResliceTask: both ranges are always cached
// together (see BuildOverlayMeshFromField), so moving the slice axis or
// position, switching Color By or style, or adjusting the Voxel Cloud
// alpha window never needs ComputeRange's O(whole field) scan again (see
// FieldOverlay.h's comment) — only BuildMeshForStyle's bounded cost
// (O(the visible slice) for Slice, O(target voxel count) for Voxel Cloud
// — see kVoxelCloudTargetCount), cheap enough to run right here on the
// UI thread. Returns false (nothing done) only if there's no cached
// field yet at all — the caller falls back to a full background rebuild
// in that case.
bool TryFastReslice(AppState& app) {
    std::shared_ptr<const std::vector<float>> field;
    float speedLo = 0.0f, speedHi = 0.0f, pressureLo = 0.0f, pressureHi = 0.0f;
    bool speedHasNaN = false, pressureHasNaN = false;
    float stressLo = 0.0f, stressHi = 0.0f;
    bool stressHasNaN = false;
    std::vector<float> pressureXMeans;  // small (one float per X) — a plain copy under the lock is fine
    float pressureDetrendLo = 0.0f, pressureDetrendHi = 0.0f;
    {
        std::lock_guard<std::mutex> lock(app.lastFieldMutex);
        if (!app.lastField || !app.cachedRangesValid) return false;
        field = app.lastField;  // shared_ptr copy: a refcount bump, not a field copy
        speedLo = app.cachedSpeedMin;
        speedHi = app.cachedSpeedMax;
        speedHasNaN = app.cachedSpeedHasNaN;
        pressureLo = app.cachedPressureMin;
        pressureHi = app.cachedPressureMax;
        pressureHasNaN = app.cachedPressureHasNaN;
        pressureXMeans = app.cachedPressureXMeans;
        pressureDetrendLo = app.cachedPressureDetrendMin;
        pressureDetrendHi = app.cachedPressureDetrendMax;
        stressLo = app.cachedStressMin;
        stressHi = app.cachedStressMax;
        stressHasNaN = app.cachedStressHasNaN;
    }

    // app.voxelAlphaPressureLo/Hi/detrendPressure read directly (not the
    // atomic mirrors): TryFastReslice only ever runs on the UI thread,
    // which is what owns them.
    float pressureLoUsed = app.detrendPressure ? pressureDetrendLo : pressureLo;
    float pressureHiUsed = app.detrendPressure ? pressureDetrendHi : pressureHi;
    const std::vector<float>* xMeansUsed = app.detrendPressure ? &pressureXMeans : nullptr;

    std::array<ResolvedSlicePlane, 3> planes;
    for (size_t i = 0; i < 3; ++i) {
        planes[i].enabled = app.slicePlanes[i].enabled;
        planes[i].index = app.slicePlanes[i].index;
    }
    OverlayMesh mesh = BuildMeshForStyle(app, *field, app.overlayStyle, planes,
                                          app.overlayColorMode, speedLo, speedHi, pressureLoUsed, pressureHiUsed,
                                          stressLo, stressHi, app.voxelAlphaPressureLo, app.voxelAlphaPressureHi,
                                          xMeansUsed, app.voxelNegativePressureGrey, app.sliceShowParticles);

    float lo, hi;
    bool hasNaN;
    LegendRangeForStyle(app.overlayStyle, app.overlayColorMode, app.sliceShowParticles, speedLo, speedHi,
                         speedHasNaN, pressureLoUsed, pressureHiUsed, pressureHasNaN, stressLo, stressHi,
                         stressHasNaN, lo, hi, hasNaN);

    app.overlayRenderer.UploadMesh(mesh);
    app.overlayMin = lo;
    app.overlayMax = hi;
    app.overlayHasNaN = hasNaN;
    app.overlayValid = true;
    return true;
}

// Re-slices the last field BuildOverlayMeshFromField was given — no fresh
// GPU/CPU readback — so the slice axis/position and color-mode controls
// work immediately even while the solver is paused (and has nothing new
// to read back), instead of only taking effect on the next auto-update or
// manual rebuild. Tries the instant path first (see TryFastReslice); only
// falls back to a background task — with the busy spinner, since it's an
// O(whole field) scan — when that's not available (no cached field yet,
// or the color mode just changed, which changes the range too).
void StartResliceTask(AppState& app) {
    if (!app.lbmValid) return;
    if (TryFastReslice(app)) return;

    std::shared_ptr<const std::vector<float>> field;
    {
        std::lock_guard<std::mutex> lock(app.lastFieldMutex);
        if (!app.lastField) return;
        field = app.lastField;
    }
    StartTask(app, TaskKind::BuildingOverlay, /*hasProgress=*/false,
              [&app, field]() { BuildOverlayMeshFromField(app, *field); });
}

// The Particles style's per-frame update: advances the smoke trail
// simulation by `dtSeconds` of real time and uploads its freshly-built
// mesh into `cmd` (the main loop's own command buffer for this frame, so
// the upload and the draw that follows share one submission — see
// OverlayRenderer::UploadDynamicMesh). Always UI-thread-only, unlike
// Slice/Voxel Cloud's mesh building — see OverlayStyle's comment for why
// this is cheap enough to afford there. A no-op if there's no cached
// field yet (nothing to advect through) or no valid solver.
void UpdateSmokeTrailAndUpload(AppState& app, SDL_GPUCommandBuffer* cmd, float dtSeconds) {
    if (!app.lbmValid) return;

    std::shared_ptr<const std::vector<float>> field;
    float pressureLo = 0.0f, pressureHi = 0.0f;
    std::vector<float> pressureXMeansCopy;  // small (one float per X) — a plain copy under the lock is fine
    {
        std::lock_guard<std::mutex> lock(app.lastFieldMutex);
        if (!app.lastField) return;
        field = app.lastField;
        pressureLo = app.detrendPressure ? app.cachedPressureDetrendMin : app.cachedPressureMin;
        pressureHi = app.detrendPressure ? app.cachedPressureDetrendMax : app.cachedPressureMax;
        if (app.detrendPressure) pressureXMeansCopy = app.cachedPressureXMeans;
    }

    // Lock-free — see AppState::lbmReverseFlowAtomic's comment for why
    // this doesn't go through app.lbmMutex like every other
    // lbmParams.reverseFlow read correctly does.
    bool reverseFlow = app.lbmReverseFlowAtomic.load(std::memory_order_relaxed);

    // Grid/reverseFlow changing needs a full Reset() for every smoke-trail
    // instance in play below (the unconfined Particles-style one, and
    // every per-plane confined one) — there's no sensible way to carry a
    // particle's position across to a differently-shaped or differently-
    // flowing domain. Tracked once, shared across all of them, since it
    // always happens to all of them at the same moment (a fresh
    // Initialize/Reset Solver) — unlike each confined instance's own
    // enable/position state below, which genuinely differs per plane.
    bool gridOrFlowChanged = !app.smokeTrailInitialized || app.smokeTrailReverseFlow != reverseFlow ||
                               app.smokeTrailGridNx != app.voxelGrid.nx || app.smokeTrailGridNy != app.voxelGrid.ny ||
                               app.smokeTrailGridNz != app.voxelGrid.nz;

    float mmPerSecPerLatticeUnit = LatticeSpeedToMmPerSecond(app, 1.0f);  // linear in lattice speed — see its comment
    bool suspendRespawn = app.smokeTrailPuffRemainingSeconds > 0.0f;
    // Scaling dtSeconds rather than mmPerSecPerLatticeUnit itself is
    // equivalent (Step only ever multiplies the two together) and keeps
    // smokeTrailSpeedScale's effect purely about how far a particle
    // travels per real second, not about reinterpreting the field.
    float scaledDt = dtSeconds * app.smokeTrailSpeedScale;
    const std::vector<float>* xMeans = app.detrendPressure ? &pressureXMeansCopy : nullptr;
    float quadHalfSize = 0.12f * app.voxelGrid.cellSize;

    OverlayMesh mesh;
    if (app.overlayStyle == OverlayStyle::Particles) {
        // Unconfined — the original single-instance behavior, except see
        // smokeTrailUnconfinedEverReset's own comment for why this checks
        // that too, not just the shared gridOrFlowChanged.
        if (gridOrFlowChanged || !app.smokeTrailUnconfinedEverReset) {
            app.smokeTrail.Reset(app.voxelGrid, reverseFlow, app.smokeTrailParticleCount, app.smokeTrailLength,
                                  nullptr);
            app.smokeTrailUnconfinedEverReset = true;
        }
        if (app.smokeTrailScatterRequested) app.smokeTrail.ScatterRandom(app.voxelGrid, nullptr);
        app.smokeTrail.Step(app.voxelGrid, *field, reverseFlow, mmPerSecPerLatticeUnit, scaledDt, suspendRespawn,
                              nullptr);
        mesh = app.smokeTrail.BuildMesh(app.voxelGrid, *field, app.camera.GetBasis().forward, quadHalfSize, pressureLo,
                                          pressureHi, app.voxelAlphaPressureLo, app.voxelAlphaPressureHi, xMeans,
                                          app.voxelNegativePressureGrey, app.smokeTrailPressureAlphaEnabled);
    } else if (app.overlayStyle == OverlayStyle::Slice && app.sliceShowParticles) {
        // Each enabled plane gets its own confined instance (see
        // AppState::smokeTrails) — moving one plane's position only
        // ShiftPlane()s *that* instance (see ShiftPlane's own comment for
        // why that matters — a full Reset reads as the stream snapping
        // back to the inlet), turning a plane off just stops stepping/
        // drawing its instance (smokeTrailWasEnabled=false so the next
        // time it's turned back on gets a fresh Reset, not a ShiftPlane
        // from a stale position), and the merged result is one mesh with
        // every enabled plane's own stream in it.
        for (size_t i = 0; i < 3; ++i) {
            AppState::SlicePlaneState& plane = app.slicePlanes[i];
            if (!plane.enabled) {
                plane.smokeTrailWasEnabled = false;
                continue;
            }
            SmokeTrailSystem::PlaneConfinement confine{kSliceAxes[i], plane.index};
            bool needsReset = gridOrFlowChanged || !plane.smokeTrailWasEnabled;
            bool needsPlaneShift = !needsReset && plane.smokeTrailLastIndex != plane.index;
            if (needsReset) {
                app.smokeTrails[i].Reset(app.voxelGrid, reverseFlow, app.smokeTrailParticleCount,
                                           app.smokeTrailLength, &confine);
            } else if (needsPlaneShift) {
                app.smokeTrails[i].ShiftPlane(app.voxelGrid, confine);
            }
            plane.smokeTrailWasEnabled = true;
            plane.smokeTrailLastIndex = plane.index;

            if (app.smokeTrailScatterRequested) app.smokeTrails[i].ScatterRandom(app.voxelGrid, &confine);
            app.smokeTrails[i].Step(app.voxelGrid, *field, reverseFlow, mmPerSecPerLatticeUnit, scaledDt,
                                      suspendRespawn, &confine);
            OverlayMesh planeMesh = app.smokeTrails[i].BuildMesh(
                app.voxelGrid, *field, app.camera.GetBasis().forward, quadHalfSize, pressureLo, pressureHi,
                app.voxelAlphaPressureLo, app.voxelAlphaPressureHi, xMeans, app.voxelNegativePressureGrey,
                app.smokeTrailPressureAlphaEnabled);
            FieldOverlay::AppendMesh(mesh, planeMesh);
        }
    }
    app.smokeTrailScatterRequested = false;

    if (gridOrFlowChanged) {
        app.smokeTrailInitialized = true;
        app.smokeTrailReverseFlow = reverseFlow;
        app.smokeTrailGridNx = app.voxelGrid.nx;
        app.smokeTrailGridNy = app.voxelGrid.ny;
        app.smokeTrailGridNz = app.voxelGrid.nz;
    }
    if (suspendRespawn) app.smokeTrailPuffRemainingSeconds = std::max(0.0f, app.smokeTrailPuffRemainingSeconds - dtSeconds);

    app.overlayRenderer.UploadDynamicMesh(cmd, mesh);
}

// Manual, on-demand readback (the "Build Overlay" button), on a background
// thread (see StartTask) — ReadbackField() is a blocking GPU download whose
// cost scales with grid resolution, same reasoning as StartVoxelizeTask.
// Switches to Flow Overlay view immediately (cheap, no need to wait); the
// slice mesh itself appears once BuildOverlayMeshFromField finishes and
// the main loop's "pending overlay mesh" block uploads it, same path
// overlayAutoUpdate already uses. Indeterminate, like StartInitSolverTask —
// a single blocking download has no natural midpoint to report.
// Defined further down, near StepSolverPaced (same pacing reasoning) —
// forward-declared here since StartBuildOverlayTask needs it and is
// defined first in the file.
void ReadbackFieldPacedGPU(AppState& app, std::vector<float>& outField);

void StartBuildOverlayTask(AppState& app) {
    if (!app.lbmValid) return;
    app.viewMode = ViewMode::FlowOverlay;
    StartTask(app, TaskKind::BuildingOverlay, /*hasProgress=*/false, [&app]() {
        // Safe to read app.lbmBackend directly (not the atomic mirror)
        // here: the Backend combo is disabled while any task is busy,
        // same reasoning as app.lbmModel just above it.
        std::vector<float> field;
        if (app.lbmBackend == SolverBackend::GPU) {
            // Chunked, with its own per-chunk locking — see
            // ReadbackFieldPacedGPU's comment. Without this, a manual
            // Build Overlay at high resolution would hold app.lbmMutex for
            // one long, unchunked download — stalling the solver's own
            // stepping (if running) for that whole duration, the exact
            // problem already fixed for auto-update's readback.
            ReadbackFieldPacedGPU(app, field);
        } else {
            std::lock_guard<std::mutex> lock(app.lbmMutex);
            if (app.lbmBackend == SolverBackend::CPU) {
                field = app.lbmReference.ReadbackField();
            } else if (IsPanelBackend(app.lbmBackend)) {
                field = app.panelSolver.ReadbackField();
            } else {
                field = app.potentialFlow.ReadbackField();
            }
        }
        BuildOverlayMeshFromField(app, field);
    });
}

// Steps the solver under app.lbmMutex — shared by the manual "Step" button
// (UI thread) and the background solver thread's continuous stepping, so
// the two can never touch lbmSolver at the same time.
void StepSolverLocked(AppState& app, int substeps) {
    bool diverged = false;
    {
        std::lock_guard<std::mutex> lock(app.lbmMutex);
        // These are whole substeps, each rewriting the full domain, so if
        // the solver thread is part-way through one of its own (the Step
        // button is enabled while running) that partial substep is now
        // stale — see lbmSolverEpoch's comment.
        ++app.lbmSolverEpoch;
        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(app.gpuDevice);
        app.lbmSolver.Step(cmd, substeps);
        SDL_SubmitGPUCommandBuffer(cmd);
        diverged = app.lbmSolver.ProbeSliceHealth(app.lbmSolver.Nz() / 2);
    }
    if (diverged) FlagFlowDivergence(app);
}

// CPU-backend equivalent of StepSolverLocked/StepSolverPaced. Much
// simpler: LbmReference::Step() is a plain (if internally multithreaded —
// see its comment) synchronous CPU call, with no command buffers, fences,
// or GPU-queue-contention concerns to pace around — those only exist
// because the GPU path shares a hardware queue with the render thread.
// Blocking this thread for however long `substeps` calls take is fine
// either way, since this only ever runs on the dedicated solver thread
// (or, for the manual "Step" button, briefly on the UI thread — same as
// StepSolverLocked).
void StepReferenceLocked(AppState& app, int substeps) {
    std::lock_guard<std::mutex> lock(app.lbmMutex);
    app.lbmStepProgress.store(0.0f, std::memory_order_relaxed);
    for (int i = 0; i < substeps; ++i) {
        app.lbmReference.Step(&app.lbmStepProgress, i, substeps);
        if (app.lbmReference.HasDiverged()) {
            FlagFlowDivergence(app);
            if (!app.lbmRunning.load(std::memory_order_acquire)) break;
        }
    }
}

// Naive-backend equivalent — PotentialFlowSolver::Step() is likewise a
// plain, already-internally-parallelized synchronous CPU call (see its
// comment), with no GPU-queue pacing concerns, same reasoning as
// StepReferenceLocked.
void StepPotentialFlowLocked(AppState& app, int sweeps) {
    std::lock_guard<std::mutex> lock(app.lbmMutex);
    app.lbmStepProgress.store(0.0f, std::memory_order_relaxed);
    app.potentialFlow.Step(sweeps, &app.lbmStepProgress);
}

void StepPanelLocked(AppState& app, int sweeps) {
    std::lock_guard<std::mutex> lock(app.lbmMutex);
    app.lbmStepProgress.store(0.0f, std::memory_order_relaxed);
    app.panelSolver.Step(sweeps, &app.lbmStepProgress);
}

// Same as StepSolverLocked, but submits one Z-chunk of one substep at a
// time (see StepRange), each its own command buffer, and waits for each
// one's GPU work to finish before moving to the next. Used only by the
// solver thread's own loop below.
//
// Three distinct problems, three reasons for this shape — the first two
// turned out not to be the whole story, which the third exists to fix:
//
// 1. Fence-waiting at all paces the solver to the GPU's real throughput
//    instead of piling up submissions faster than the GPU can retire them.
//    That backlog was starving the render thread's own submissions — not
//    lock contention, rendering never touches app.lbmMutex.
//
// 2. Chunking along Z, rather than dispatching the whole grid in one
//    command buffer, bounds how long any single *submitted* unit of GPU
//    work can run, since SDL_GPU/Metal execute command buffers on one
//    shared queue in submission order.
//
// 3. Neither of those bounds how long the GPU hardware itself spends
//    actually *executing* a chunk once it starts — and that turned out to
//    matter just as much as submission order. A standalone test (two
//    threads sharing one SDL_GPUDevice: one continuously dispatching a
//    deliberately heavy ~70ms compute workload and fence-waiting on it,
//    the other repeatedly submitting a *tiny* dispatch and timing its own
//    fence-wait) showed the tiny dispatch's latency jumping from ~0.01ms
//    uncontended to 60-80ms whenever it landed while the heavy one was
//    running — i.e. the GPU does not interleave a small job alongside a
//    large one already executing; it waits for the large one to finish
//    first, every time. A fixed chunk size (32 Z-cells, in an earlier
//    version of this function) only pushed the threshold where that
//    becomes visible up to a higher resolution, rather than removing it —
//    which is exactly why the freeze came back at resolution 512. So
//    chunk depth here is adaptive, targeting kTargetChunkMs of actual
//    measured GPU time per chunk regardless of resolution: shrink
//    (hurting solver throughput, never UI smoothness) when a chunk ran
//    over budget, grow back (so low resolutions aren't paying needless
//    per-chunk submission/fence overhead — ~0.17ms each, measured the
//    same way) when comfortably under it.
//
// The fence wait happens with the lock already released, since once a
// command buffer is submitted, SDL_GPU/Metal preserve submission order on
// their own — nothing else needs to block on this chunk's completion for
// correctness, only for pacing, and that's this thread's own problem to
// wait out, never the UI thread's.
//
// Returns how many substeps actually completed: fewer than `substeps` if
// the flow diverged and auto-paused, or if the solver was re-initialized
// or manually stepped from another thread part-way through a substep (see
// AppState::lbmSolverEpoch) — that substep is abandoned, not counted.
int StepSolverPaced(AppState& app, int substeps) {
    // static, not an AppState member: this function only ever runs on the
    // solver thread (see LbmWorkerThreadFunc), so there's no cross-thread
    // access to guard, and the adaptation should persist between calls.
    static int chunkZDepth = 32;
    constexpr int kMinChunkZDepth = 4;  // one Z-threadgroup (kThreadsZ, in LbmSolver.cpp)
    constexpr int kMaxChunkZDepth = 256;
    constexpr double kTargetChunkMs = 3.0;  // keeps a stalled render frame well under one 60fps budget

    Uint64 freq = SDL_GetPerformanceFrequency();

    app.lbmStepProgress.store(0.0f, std::memory_order_relaxed);
    for (int i = 0; i < substeps; ++i) {
        int nz;
        uint64_t epoch;
        {
            std::lock_guard<std::mutex> lock(app.lbmMutex);
            nz = app.lbmSolver.Nz();
            epoch = app.lbmSolverEpoch;
        }

        for (int zStart = 0; zStart < nz;) {
            int zCount = std::min(chunkZDepth, nz - zStart);
            Uint64 chunkStartTicks = SDL_GetPerformanceCounter();

            // Holds the lock through the fence wait itself — see
            // ReadbackFieldRange's own call site, which already does this.
            // The comment this replaced argued submission order alone made
            // that safe to skip, assuming every GPU operation here shares
            // one strictly-ordered queue; that assumption is the leading
            // suspect for an intermittent, timing-sensitive readback
            // corruption (disappears under added latency, e.g. logging) —
            // a copy/download dispatched from the readback thread may not
            // actually be guaranteed to execute after a compute dispatch
            // from this thread just because it was submitted later.
            // Serializing fully through app.lbmMutex removes that
            // assumption instead of relying on it.
            {
                std::lock_guard<std::mutex> lock(app.lbmMutex);
                // The chunks this substep already dispatched were computed
                // from (and written into) state that has since been
                // replaced — nothing here is worth finishing.
                if (app.lbmSolverEpoch != epoch) return i;
                SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(app.gpuDevice);
                app.lbmSolver.StepRange(cmd, zStart, zCount);
                SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
                if (fence) {
                    SDL_WaitForGPUFences(app.gpuDevice, true, &fence, 1);
                    SDL_ReleaseGPUFence(app.gpuDevice, fence);
                }
            }

            double chunkMs = 1000.0 * static_cast<double>(SDL_GetPerformanceCounter() - chunkStartTicks) /
                              static_cast<double>(freq);
            if (chunkMs > kTargetChunkMs && chunkZDepth > kMinChunkZDepth) {
                chunkZDepth = std::max(kMinChunkZDepth, chunkZDepth / 2);
            } else if (chunkMs < kTargetChunkMs * 0.25 && chunkZDepth < kMaxChunkZDepth) {
                chunkZDepth = std::min(kMaxChunkZDepth, chunkZDepth * 2);
            }

            // Advance by zCount — what this chunk actually covered — not
            // by chunkZDepth, which the adaptation just above may have
            // *already changed* for the next chunk. Using chunkZDepth
            // here (the loop's original increment clause) meant that
            // whenever it changed mid-substep, the next zStart either
            // overlapped the chunk just processed (shrink) or skipped a
            // range of Z-layers entirely (grow) — those skipped cells'
            // populations were never written this substep, so
            // AdvanceSubstep() below would hand stale/garbage data to the
            // next substep as "current", which then spreads into
            // neighbouring cells via streaming. This is what "the GPU
            // solver is broken, it worked before the performance changes"
            // was: static chunking (tested) was fine; the adaptive *size
            // change interacting with the tiling loop* (not independently
            // tested) was the actual bug.
            zStart += zCount;
            // Same (substepIndex + thisCallFraction) / substepCount shape
            // as LbmReference::Step()'s own progress reporting — zStart
            // is already the exact count of Z-layers this substep has
            // covered so far, no separate counter needed.
            app.lbmStepProgress.store(
                (static_cast<float>(i) + static_cast<float>(zStart) / static_cast<float>(nz)) /
                    static_cast<float>(substeps),
                std::memory_order_relaxed);
        }

        static Uint64 lastHealthProbeTicks = 0;
        static int healthProbePhase = 0;
        constexpr Uint64 kHealthProbeIntervalMs = 400;
        Uint64 nowTicks = SDL_GetTicks();
        bool doProbe = (nowTicks - lastHealthProbeTicks >= kHealthProbeIntervalMs);
        bool diverged = false;
        {
            std::lock_guard<std::mutex> lock(app.lbmMutex);
            // Same check as each chunk's: flipping the ping-pong index
            // here after an Init/Reset would make the freshly seeded
            // solver's never-written buffer "current".
            if (app.lbmSolverEpoch != epoch) return i;
            app.lbmSolver.AdvanceSubstep();
            if (doProbe && nz > 0) {
                lastHealthProbeTicks = nowTicks;
                int probeZ = (healthProbePhase == 0) ? (nz / 2)
                           : (healthProbePhase == 1) ? (nz / 3)
                                                     : ((2 * nz) / 3);
                healthProbePhase = (healthProbePhase + 1) % 3;
                diverged = app.lbmSolver.ProbeSliceHealth(probeZ);
            }
        }
        if (diverged) {
            FlagFlowDivergence(app);
            if (!app.lbmRunning.load(std::memory_order_acquire)) return i + 1;
        }
    }
    return substeps;
}

// Paced equivalent of app.lbmSolver.ReadbackField(), for the GPU backend's
// overlay auto-update. A plain ReadbackField() call holds app.lbmMutex for
// its *entire* duration — every chunk's download, all the way through —
// which blocks the solver thread's next Step() dispatch from even being
// submitted until the whole readback finishes. Worse, that block isn't
// only the C++ mutex: a GPU download and a GPU compute dispatch share one
// hardware queue, so even without the mutex a download big enough to take
// meaningfully long would still delay the next dispatch's actual
// execution, for the exact reason StepSolverPaced's comment measured and
// documents (the GPU doesn't interleave a small job alongside a large one
// already running on the same queue — it waits for the large one first).
// This applies that function's same fix to the download side: adaptively
// sized Z-chunks, each with its own short-lived lock, so neither the
// mutex nor the GPU queue is ever held/occupied by this for more than
// roughly kTargetChunkMs at a time — the periodic pauses this replaces
// were this exact mechanism, just on the read side instead of the step
// side, introduced when auto-update moved to its own thread but kept
// reading in one unchunked call.
void ReadbackFieldPacedGPU(AppState& app, std::vector<float>& outField) {
    // static, not an AppState member: this function only ever runs on
    // app.overlayUpdateThread, and overlayUpdateInFlight guarantees only
    // one of those runs at a time — same single-thread-owns-this
    // reasoning as StepSolverPaced's own chunkZDepth.
    static int chunkZDepth = 16;
    // 4, not 1: LbmSolver::ReadbackFieldRange now reuses one persistent
    // transfer buffer rather than creating/releasing one every call (see
    // its own comment), so the fixed per-chunk overhead that used to
    // make shrinking all the way to 1 genuinely costly is mostly gone —
    // but a command buffer acquire + submit + fence-wait + map/unmap
    // still happens once per chunk regardless, so this still keeps a
    // floor under how many of those a large readback can be split into,
    // the same reasoning StepSolverPaced's own floor already has.
    constexpr int kMinChunkZDepth = 4;
    constexpr int kMaxChunkZDepth = 256;
    constexpr double kTargetChunkMs = 3.0;  // same budget StepSolverPaced targets

    int nx, ny, nz;
    {
        std::lock_guard<std::mutex> lock(app.lbmMutex);
        nx = app.lbmSolver.Nx();
        ny = app.lbmSolver.Ny();
        nz = app.lbmSolver.Nz();
    }
    outField.assign(static_cast<size_t>(nx) * ny * nz * 4, 0.0f);

    Uint64 freq = SDL_GetPerformanceFrequency();
    for (int zStart = 0; zStart < nz;) {
        int zCount = std::min(chunkZDepth, nz - zStart);
        Uint64 chunkStartTicks = SDL_GetPerformanceCounter();
        {
            std::lock_guard<std::mutex> lock(app.lbmMutex);
            app.lbmSolver.ReadbackFieldRange(zStart, zCount, outField);
        }
        double chunkMs = 1000.0 * static_cast<double>(SDL_GetPerformanceCounter() - chunkStartTicks) /
                          static_cast<double>(freq);
        if (chunkMs > kTargetChunkMs && chunkZDepth > kMinChunkZDepth) {
            chunkZDepth = std::max(kMinChunkZDepth, chunkZDepth / 2);
        } else if (chunkMs < kTargetChunkMs * 0.25 && chunkZDepth < kMaxChunkZDepth) {
            chunkZDepth = std::min(kMaxChunkZDepth, chunkZDepth * 2);
        }
        // Advance by zCount, not chunkZDepth — same reasoning as
        // StepSolverPaced's own loop (see its comment): chunkZDepth may
        // have just changed for the *next* chunk, and using it here would
        // either skip or overlap Z-layers relative to what this chunk
        // actually covered.
        zStart += zCount;
    }
}

// Starts one overlay auto-update readback+rebuild on its own thread — see
// AppState::overlayUpdateThread's comment for why this exists separately
// from the solver's own stepping loop. Only ever called by
// LbmWorkerThreadFunc, and only when overlayUpdateInFlight is already
// false, so app.overlayUpdateThread is always safe to join here: if one
// is set, its thread function has already returned (it clears the flag
// as its last action) — join() on an already-finished thread is
// immediate, same reasoning as StartTask's join-before-reuse.
void StartOverlayAutoUpdate(AppState& app, SolverBackend backend) {
    if (app.overlayUpdateThread.joinable()) app.overlayUpdateThread.join();
    app.overlayUpdateInFlight.store(true, std::memory_order_relaxed);
    app.overlayUpdateThread = std::thread([&app, backend]() {
        std::vector<float> field;
        if (backend == SolverBackend::GPU) {
            // Chunked, with its own per-chunk locking — see
            // ReadbackFieldPacedGPU's comment for why a single
            // lbmMutex-for-the-whole-call readback (what this replaced)
            // caused periodic pauses in the solver's own stepping.
            ReadbackFieldPacedGPU(app, field);
        } else {
            // CPU/Naive readback is pure CPU work with no GPU-queue
            // contention to worry about, and is already parallelized
            // internally (ParallelForZ) — a single lock for the whole
            // call is the same cost either way, so there's no equivalent
            // benefit to chunking it the GPU path needed.
            std::lock_guard<std::mutex> lock(app.lbmMutex);
            if (backend == SolverBackend::CPU) {
                field = app.lbmReference.ReadbackField();
            } else if (IsPanelBackend(backend)) {
                field = app.panelSolver.ReadbackField();
            } else {
                field = app.potentialFlow.ReadbackField();
            }
        }
        // No lock needed from here on — BuildOverlayMeshFromField never
        // touches lbmSolver/lbmReference/potentialFlow, only `field`
        // (already downloaded) and app's own overlay-side state (which
        // has its own, separate mutexes) — see its own comment.
        BuildOverlayMeshFromField(app, field);
        app.overlayUpdateInFlight.store(false, std::memory_order_relaxed);
    });
}

// Runs on its own thread for the whole app lifetime, continuously stepping
// the solver whenever it's valid and running — decoupled from the render
// loop's frame rate, so a slow/blocking GPU readback (for the flow overlay)
// never stalls rendering or input handling. Reads only the *Atomic mirror
// fields (never the plain ImGui-bound ones) to avoid data races with the UI
// thread; see AppState's comment on lbmMutex.
void LbmWorkerThreadFunc(AppState* appPtr) {
    AppState& app = *appPtr;
    Uint64 lastFieldUpdateTicks = 0;
    while (!app.lbmThreadStop.load(std::memory_order_acquire)) {
        bool valid = app.lbmValid.load(std::memory_order_acquire);
        bool running = app.lbmRunning.load(std::memory_order_acquire);
        if (valid && running) {
            int substeps = app.lbmSubstepsAtomic.load(std::memory_order_relaxed);
            SolverBackend backend = static_cast<SolverBackend>(app.lbmBackendAtomic.load(std::memory_order_relaxed));
            int completed = substeps;
            if (backend == SolverBackend::GPU) {
                completed = StepSolverPaced(app, substeps);
            } else if (backend == SolverBackend::CPU) {
                StepReferenceLocked(app, substeps);
            } else if (IsPanelBackend(backend)) {
                StepPanelLocked(app, substeps);
            } else {
                StepPotentialFlowLocked(app, substeps);
            }
            // Only what actually ran — an Init/Reset that interrupted
            // StepSolverPaced has already zeroed this for the new run.
            app.lbmStepCount += completed;

            // warmedUp latches true forever once lbmStepCount (monotonic
            // while running, reset to 0 only by Initialize/Reset Solver)
            // crosses the threshold — so this only ever suppresses the
            // readback below for the first stretch after a fresh
            // start/reset, never after, with no extra state needed.
            bool warmedUp = app.lbmStepCount.load(std::memory_order_relaxed) >=
                             app.overlayAutoUpdateWarmupStepsAtomic.load(std::memory_order_relaxed);
            if (warmedUp && app.overlayAutoUpdateAtomic.load(std::memory_order_relaxed)) {
                Uint64 now = SDL_GetTicks();
                int interval = app.overlayIntervalAtomic.load(std::memory_order_relaxed);
                // Only the *decision* to start an update happens here, on
                // the stepping loop's own thread — the readback and mesh
                // build themselves run on a separate thread (see
                // StartOverlayAutoUpdate), so they never delay the next
                // Step() call the way doing this inline used to. Not
                // gating on overlayUpdateInFlight too would mean the
                // interval check stays true (lastFieldUpdateTicks is only
                // advanced on an actual start, below) the whole time a
                // slow update is still running, which is exactly what we
                // want: the moment it finishes, the next one starts
                // immediately rather than waiting out another full
                // interval — if updates are slower than the configured
                // interval, they simply run back-to-back as fast as they
                // can, falling behind the simulation instead of pacing it.
                if (!app.overlayUpdateInFlight.load(std::memory_order_relaxed) &&
                    now - lastFieldUpdateTicks >= static_cast<Uint64>(interval)) {
                    lastFieldUpdateTicks = now;
                    StartOverlayAutoUpdate(app, backend);
                }
            }
        } else {
            // Nothing to do (no solver, or paused) — avoid spinning a CPU
            // core at 100% polling those flags for no reason. When active,
            // StepSolverPaced's fence wait above already paces this loop to
            // the GPU's real throughput, so no extra delay is needed there.
            SDL_Delay(5);
        }
    }
}

// Reads an entire file into memory — used for stashing the mesh's own
// raw STL bytes (see AppState::meshStlBytes) alongside parsing it via
// STLLoader, which reads the file itself but doesn't hand the raw bytes
// back out. A second, independent read of a just-opened file is cheap
// next to everything else this app does (and happens once, on an
// explicit user action, not a hot path) — simpler than threading a
// "also return what you read" option through STLLoader for one caller.
std::vector<uint8_t> ReadWholeFile(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long sizeLong = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sizeLong < 0) {
        std::fclose(f);
        return {};
    }
    std::vector<uint8_t> data(static_cast<size_t>(sizeLong));
    size_t readCount = data.empty() ? 0 : std::fread(data.data(), 1, data.size(), f);
    std::fclose(f);
    if (readCount != data.size()) return {};
    return data;
}

// Common cleanup whenever the mesh is replaced wholesale — a fresh STL
// open, or loading a session — since every downstream stage (voxel grid,
// solver, overlay) was built from whatever mesh is going away. Does *not*
// touch app.mesh/meshStlBytes/meshLoaded themselves, or the model
// transform (orientation/pivot/scale) — callers set those to whatever's
// appropriate for their own case (identity + mesh-center pivot for a
// fresh STL; the saved values for a loaded session) before or after
// calling this, since the right values differ between the two.
void ResetStateForNewMesh(AppState& app) {
    app.voxelGridValid = false;
    app.viewMode = ViewMode::Model;
    app.lbmValid = false;
    app.lbmRunning = false;
    app.lbmForcesValidAtomic.store(false, std::memory_order_relaxed);
    app.overlayValid = false;
    app.polarResult = {};
    app.sectionPolarResult = {};
    app.panelSolver.Shutdown();
    ClearFlowDivergence(app);
    for (AppState::SlicePlaneState& plane : app.slicePlanes) {
        plane.index = -1;
        plane.indexAtomic.store(-1, std::memory_order_relaxed);
    }
    app.smokeTrailInitialized = false;
    {
        std::lock_guard<std::mutex> lock(app.lastFieldMutex);
        app.lastField.reset();
        app.cachedRangesValid = false;
    }
}

void LoadMeshFromDialog(AppState& app) {
    auto path = OpenSTLFileDialog();
    if (!path) return;

    STLLoader::Result result = STLLoader::Load(*path);
    if (!result.ok) {
        app.meshLoaded = false;
        app.statusMessage = "Failed to load: " + result.error;
        return;
    }

    app.mesh = std::move(result.mesh);
    app.meshStlBytes = ReadWholeFile(*path);
    app.meshLoaded = true;
    app.renderer.UploadMesh(app.mesh);
    app.camera.Frame(app.mesh.bounds);
    app.cameraTransitioning = false;
    app.modelOrientation = app.lockedOrientation;  // see its own comment — identity until ever locked
    app.modelPivot = app.mesh.bounds.Center();
    app.scaleMM = 1.0f;
    ResetStateForNewMesh(app);

    Vec3 size = app.mesh.bounds.Size();
    char buf[256];
    std::snprintf(buf, sizeof(buf), "Loaded %s: %zu triangles, bounds %.3f x %.3f x %.3f",
                  app.mesh.name.c_str(), app.mesh.TriangleCount(), size.x, size.y, size.z);
    app.statusMessage = buf;
}

// Snapshots the settings SessionFile.h cares about from the live
// AppState — everything needed to get back to this configuration, not a
// simulation snapshot (see SessionSettings' own header comment).
SessionSettings CaptureSessionSettings(const AppState& app) {
    SessionSettings s;
    s.modelOrientation = app.modelOrientation;
    s.scaleMM = app.scaleMM;
    s.modelPivot = app.modelPivot;
    s.lockedOrientation = app.lockedOrientation;

    s.voxelResolution = app.voxelResolution;
    s.voxelPaddingFactor = app.voxelPaddingFactor;

    s.lbmBackend = static_cast<int32_t>(app.lbmBackend);
    s.lbmModel = static_cast<int32_t>(app.lbmModel);
    s.lbmTau = app.lbmParams.tau;
    s.lbmReverseFlow = app.lbmParams.reverseFlow;
    s.lbmFreeSlipY = app.lbmParams.freeSlipY;
    s.lbmFreeSlipZ = app.lbmParams.freeSlipZ;
    s.lbmSmagorinskyCs = app.lbmParams.smagorinskyCs;
    s.lbmInletVelocityMmPerS = app.lbmInletVelocityMmPerS;
    s.lbmSubstepsPerFrame = app.lbmSubstepsPerFrame;
    s.lbmSeedMidSpeed = app.lbmSeedMidSpeed;
    s.warmStartEnabled = app.warmStartEnabled;
    s.warmStartSteps = app.warmStartSteps;
    s.warmStartResolutionDivisor = app.warmStartResolutionDivisor;
    s.buildOverlayOnRun = app.buildOverlayOnRun;

    s.overlayStyle = static_cast<int32_t>(app.overlayStyle);
    for (size_t i = 0; i < 3; ++i) {
        s.sliceEnabled[i] = app.slicePlanes[i].enabled;
        s.sliceIndex[i] = app.slicePlanes[i].index;
    }
    s.overlayColorMode = static_cast<int32_t>(app.overlayColorMode);
    s.overlayAutoUpdate = app.overlayAutoUpdate;
    s.overlayUpdateIntervalMs = app.overlayUpdateIntervalMs;
    s.overlayAutoUpdateWarmupSteps = app.overlayAutoUpdateWarmupSteps;
    s.overlayShowModel = app.overlayShowModel;
    s.sliceShowParticles = app.sliceShowParticles;
    s.voxelAlphaPressureLo = app.voxelAlphaPressureLo;
    s.voxelAlphaPressureHi = app.voxelAlphaPressureHi;
    s.voxelAlphaRangeInitialized = app.voxelAlphaRangeInitialized;
    s.detrendPressure = app.detrendPressure;
    s.voxelNegativePressureGrey = app.voxelNegativePressureGrey;

    s.smokeTrailParticleCount = app.smokeTrailParticleCount;
    s.smokeTrailLength = app.smokeTrailLength;
    s.smokeTrailSpeedScale = app.smokeTrailSpeedScale;
    s.smokeTrailPuffDurationSeconds = app.smokeTrailPuffDurationSeconds;
    s.smokeTrailPressureAlphaEnabled = app.smokeTrailPressureAlphaEnabled;

    s.activeTab = static_cast<int32_t>(app.activeTab);
    s.showInspectorPanel = app.showInspectorPanel;
    s.showTelemetryHud = app.showTelemetryHud;
    s.showLiftArrow = app.showLiftArrow;

    s.cameraYaw = app.camera.yaw;
    s.cameraPitch = app.camera.pitch;
    s.cameraDistance = app.camera.distance;
    return s;
}

// The inverse of CaptureSessionSettings — applies every session-scoped
// field onto `app`. Shared by LoadSessionDialog (with a loaded file's
// settings) and ResetSession (with a default-constructed SessionSettings{},
// i.e. every field's own in-class default — see SessionFile.h), so the
// two can never drift apart on which fields "the session" actually
// covers. Deliberately does NOT touch app.mesh/meshStlBytes/meshLoaded,
// the model transform's dependency on ResetStateForNewMesh having already
// run first (see its call sites' comments), or anything camera.Frame()
// itself derives (target/sceneRadius) — callers own those.
void ApplySessionSettings(AppState& app, const SessionSettings& s) {
    app.modelOrientation = s.modelOrientation;
    app.scaleMM = s.scaleMM;
    app.modelPivot = s.modelPivot;
    app.lockedOrientation = s.lockedOrientation;

    app.voxelResolution = s.voxelResolution;
    app.voxelPaddingFactor = s.voxelPaddingFactor;

    app.lbmBackend = static_cast<SolverBackend>(s.lbmBackend);
    app.lbmModel = static_cast<LatticeModel>(s.lbmModel);
    app.lbmParams.tau = s.lbmTau;
    app.lbmTauAtomic.store(s.lbmTau, std::memory_order_relaxed);
    app.lbmParams.reverseFlow = s.lbmReverseFlow;
    app.lbmReverseFlowAtomic.store(s.lbmReverseFlow, std::memory_order_relaxed);
    app.lbmParams.freeSlipY = s.lbmFreeSlipY;
    app.lbmParams.freeSlipZ = s.lbmFreeSlipZ;
    app.lbmParams.smagorinskyCs = s.lbmSmagorinskyCs;
    app.lbmInletVelocityMmPerS = s.lbmInletVelocityMmPerS;
    app.lbmSubstepsPerFrame = s.lbmSubstepsPerFrame;
    app.lbmSeedMidSpeed = s.lbmSeedMidSpeed;
    app.warmStartEnabled = s.warmStartEnabled;
    app.warmStartSteps = s.warmStartSteps;
    app.warmStartResolutionDivisor = s.warmStartResolutionDivisor;
    app.buildOverlayOnRun = s.buildOverlayOnRun;

    app.overlayStyle = static_cast<OverlayStyle>(s.overlayStyle);
    for (size_t i = 0; i < 3; ++i) {
        app.slicePlanes[i].enabled = s.sliceEnabled[i];
        app.slicePlanes[i].index = s.sliceIndex[i];
        app.slicePlanes[i].enabledAtomic.store(s.sliceEnabled[i], std::memory_order_relaxed);
        app.slicePlanes[i].indexAtomic.store(s.sliceIndex[i], std::memory_order_relaxed);
    }
    app.overlayColorMode = static_cast<FieldColorMode>(s.overlayColorMode);
    app.overlayAutoUpdate = s.overlayAutoUpdate;
    app.overlayUpdateIntervalMs = s.overlayUpdateIntervalMs;
    app.overlayAutoUpdateWarmupSteps = s.overlayAutoUpdateWarmupSteps;
    app.overlayShowModel = s.overlayShowModel;
    app.sliceShowParticles = s.sliceShowParticles;
    app.voxelAlphaPressureLo = s.voxelAlphaPressureLo;
    app.voxelAlphaPressureHi = s.voxelAlphaPressureHi;
    app.voxelAlphaRangeInitialized = s.voxelAlphaRangeInitialized;
    app.detrendPressure = s.detrendPressure;
    app.voxelNegativePressureGrey = s.voxelNegativePressureGrey;

    app.smokeTrailParticleCount = s.smokeTrailParticleCount;
    app.smokeTrailLength = s.smokeTrailLength;
    app.smokeTrailSpeedScale = s.smokeTrailSpeedScale;
    app.smokeTrailPuffDurationSeconds = s.smokeTrailPuffDurationSeconds;
    app.smokeTrailPressureAlphaEnabled = s.smokeTrailPressureAlphaEnabled;

    app.activeTab = static_cast<WorkspaceTab>(s.activeTab);
    app.showInspectorPanel = s.showInspectorPanel;
    app.showTelemetryHud = s.showTelemetryHud;
    app.showLiftArrow = s.showLiftArrow;

    app.camera.yaw = s.cameraYaw;
    app.camera.pitch = s.cameraPitch;
    app.camera.distance = s.cameraDistance;
    app.cameraTransitioning = false;
}

void SaveSessionDialog(AppState& app) {
    if (!app.meshLoaded) return;

    // Derive a sensible default filename from the mesh's own name (e.g.
    // "DroneType12.stl" -> "DroneType12.fcsession") rather than leaving
    // the save panel's field blank.
    std::string base = app.mesh.name;
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    auto path = SaveSessionFileDialog(base.empty() ? "session" : base);
    if (!path) return;

    SessionSettings settings = CaptureSessionSettings(app);
    std::string error;
    if (!SessionFile::Save(*path, settings, app.mesh.name, app.meshStlBytes, error)) {
        app.statusMessage = "Failed to save session: " + error;
        return;
    }
    app.statusMessage = "Saved session: " + *path;
}

void LoadSessionDialog(AppState& app) {
    auto path = OpenSessionFileDialog();
    if (!path) return;

    SessionFile::LoadResult result = SessionFile::Load(*path);
    if (!result.ok) {
        app.statusMessage = "Failed to load session: " + result.error;
        return;
    }

    STLLoader::Result stlResult =
        STLLoader::LoadFromMemory(result.stlBytes.data(), result.stlBytes.size(), result.stlFileName);
    if (!stlResult.ok) {
        app.statusMessage = "Session's embedded STL failed to parse: " + stlResult.error;
        return;
    }

    app.mesh = std::move(stlResult.mesh);
    app.meshStlBytes = std::move(result.stlBytes);
    app.meshLoaded = true;
    app.renderer.UploadMesh(app.mesh);
    app.camera.Frame(app.mesh.bounds);
    app.cameraTransitioning = false;  // seeds target/sceneRadius from the (new) mesh bounds first...

    // ResetStateForNewMesh resets each slice plane's index to -1 among
    // other things, so it must run BEFORE ApplySessionSettings, which
    // sets it (and everything else "the session" covers) to the saved
    // file's real values — see ApplySessionSettings' own comment.
    ResetStateForNewMesh(app);
    ApplySessionSettings(app, result.settings);  // ...including the saved view angle/zoom, overriding Frame()'s own

    app.statusMessage = "Loaded session: " + *path +
                         " (press Voxelize, then Initialize Solver, to resume simulating)";
}

// Discards the current mesh and every session-scoped setting (model
// transform, voxelization, solver, overlay, particles, camera angle —
// exactly what Save Session captures and Load Session restores, see
// CaptureSessionSettings/ApplySessionSettings) back to a blank slate, as
// if the app had just been launched with nothing loaded. Irreversible
// short of a previously saved .fcsession file — see DrawUI's "Reset
// Session" button for the confirmation modal in front of this.
void ResetSession(AppState& app) {
    app.lbmRunning = false;
    ResetStateForNewMesh(app);
    ApplySessionSettings(app, SessionSettings{});  // every field back to its own struct's default

    app.meshLoaded = false;
    app.telemetryHudInitialized = false;
    app.telemetryHudUserMoved = false;
    app.processingWindowInitialized = false;
    app.processingWindowUserMoved = false;
    app.mesh = Mesh{};
    app.meshStlBytes.clear();
    app.renderer.UploadMesh(Mesh{});  // clears what's actually drawn — HasMesh() depends on its own indexCount_,
                                        // not app.meshLoaded, see the render dispatch's ViewMode::Model branch
    app.statusMessage = "Session reset.";
}

// Polls held-down keys every frame (rather than reacting to discrete key
// events) so holding an arrow key rotates continuously, frame-rate
// independent via dt.
void PollKeyboardRotation(AppState& app, float dt) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    const bool* keys = SDL_GetKeyboardState(nullptr);
    constexpr float kRateRadiansPerSecond = 1.2f;
    float step = kRateRadiansPerSecond * dt;

    float yaw = 0.0f, pitch = 0.0f, roll = 0.0f;
    if (keys[SDL_SCANCODE_LEFT]) yaw += step;
    if (keys[SDL_SCANCODE_RIGHT]) yaw -= step;
    if (keys[SDL_SCANCODE_UP]) pitch -= step;
    if (keys[SDL_SCANCODE_DOWN]) pitch += step;
    if (keys[SDL_SCANCODE_Q]) roll += step;
    if (keys[SDL_SCANCODE_E]) roll -= step;

    RotateModel(app, kYawRotationAxis, yaw);
    RotateModel(app, kFlowAxis, pitch);
    RotateModel(app, kPitchRotationAxis, roll);
}

// Small inset widget (Blender/Maya-style) showing the fixed world axes as
// seen from the current camera angle, so it's clear how the model (which
// rotates in place, not the camera) is currently presented to the flow.
// The flow axis is labelled directly rather than drawn as a separate arrow.
//
// Its geometry is named (rather than left as literals in the function)
// because DrawProcessingWindow parks itself directly underneath by
// default and needs the same numbers: the widget is a dark backdrop disc
// of radius kGizmoRadius + kGizmoBackdropPad, centred kGizmoCenterInset in
// from the viewport's top-right corner (below `topInset`).
constexpr float kGizmoRadius = 42.0f;
constexpr float kGizmoBackdropPad = 16.0f;
constexpr float kGizmoCenterInset = 70.0f;
// Distance from the top of the widget's area (viewport top + topInset) to
// the backdrop's bottom edge, and from the viewport's right edge to the
// backdrop's right edge.
constexpr float kGizmoBottomInset = kGizmoCenterInset + kGizmoRadius + kGizmoBackdropPad;
constexpr float kGizmoRightMargin = kGizmoCenterInset - (kGizmoRadius + kGizmoBackdropPad);

void DrawOrientationGizmo(AppState& app, float topInset = 0.0f) {
    const OrbitCamera& camera = app.camera;
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float radius = kGizmoRadius;
    const ImVec2 center{viewport->Pos.x + viewport->Size.x - kGizmoCenterInset,
                        viewport->Pos.y + topInset + kGizmoCenterInset};

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddCircleFilled(center, radius + kGizmoBackdropPad, IM_COL32(18, 19, 23, 150));

    OrbitCamera::Basis basis = camera.GetBasis();
    struct AxisDef {
        Vec3 dir;
        ImU32 color;
        const char* label;
        int axisIndex;
    };
    // "Flow" is drawn along kPitchAxis (world Y) now, not kFlowAxis (world
    // X) — the solver's actual inlet/outlet axis moved to world Y (see
    // VoxelGrid::CellCenter's comment), and Roll already rotates around
    // that same Y axis (kPitchRotationAxis), so the gizmo's labeling
    // follows. kFlowAxis itself is unchanged (still world X, still what
    // Pitch rotates around) — just no longer the one worth calling
    // "Flow" here, so it gets the plain axis-letter label instead.
    const std::array<AxisDef, 3> axes = {{
        {kPitchAxis, IM_COL32(235, 90, 90, 255), "Flow", 0},
        {kYawAxis, IM_COL32(100, 150, 235, 255), "Z", 1},
        {kFlowAxis, IM_COL32(100, 210, 100, 255), "X", 2},
    }};

    struct AxisEnd {
        ImVec2 pos;
        float depth;
        ImU32 color;
        const char* label;
        bool positive;
        int axisIndex;
    };
    std::array<AxisEnd, 6> ends;
    size_t n = 0;
    for (const AxisDef& a : axes) {
        for (float sign : {1.0f, -1.0f}) {
            Vec3 e = a.dir * sign;
            Vec3 v{Dot(basis.right, e), Dot(basis.up, e), Dot(basis.forward, e)};
            ImVec2 p{center.x + v.x * radius, center.y - v.y * radius};
            ends[n++] = {p, v.z, a.color, a.label, sign > 0.0f, a.axisIndex};
        }
    }
    std::sort(ends.begin(), ends.end(), [](const AxisEnd& a, const AxisEnd& b) { return a.depth > b.depth; });

    ImVec2 mousePos = ImGui::GetIO().MousePos;
    bool mouseClicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    int hoveredIdx = -1;
    for (int i = 5; i >= 0; --i) {
        float dx = mousePos.x - ends[i].pos.x;
        float dy = mousePos.y - ends[i].pos.y;
        float hitR = ends[i].positive ? 12.0f : 9.0f;
        if (dx * dx + dy * dy <= hitR * hitR) {
            hoveredIdx = i;
            break;
        }
    }

    if (hoveredIdx >= 0 && mouseClicked && !ImGui::GetIO().WantCaptureMouseUnlessPopupClose) {
        const AxisEnd& clicked = ends[hoveredIdx];
        bool reverseFlow = app.lbmReverseFlowAtomic.load(std::memory_order_relaxed);
        if (clicked.axisIndex == 0) { // Flow
            if (clicked.positive) {
                SnapCameraView(app, reverseFlow ? kQuarterTurnRadians : -kQuarterTurnRadians, 0.0f);
            } else {
                SnapCameraView(app, reverseFlow ? -kQuarterTurnRadians : kQuarterTurnRadians, 0.0f);
            }
        } else if (clicked.axisIndex == 1) { // Z (Up)
            if (clicked.positive) {
                SnapCameraView(app, 0.0f, kTopViewPitch);
            } else {
                SnapCameraView(app, 0.0f, -kTopViewPitch);
            }
        } else if (clicked.axisIndex == 2) { // X (Lateral)
            if (clicked.positive) {
                SnapCameraView(app, 0.0f, 0.0f);
            } else {
                SnapCameraView(app, std::numbers::pi_v<float>, 0.0f);
            }
        }
    }

    for (size_t i = 0; i < ends.size(); ++i) {
        const AxisEnd& end = ends[i];
        bool isHovered = (static_cast<int>(i) == hoveredIdx);
        dl->AddLine(center, end.pos, end.color, isHovered ? 2.6f : 2.0f);
        if (end.positive) {
            float rDot = isHovered ? 8.0f : 6.0f;
            dl->AddCircleFilled(end.pos, rDot, end.color);
            if (isHovered) {
                dl->AddCircle(end.pos, rDot + 2.5f, IM_COL32(255, 255, 255, 220), 16, 1.5f);
            }
            dl->AddText(ImVec2(end.pos.x + 7.0f, end.pos.y - 7.0f), IM_COL32(235, 235, 235, 255), end.label);
        } else {
            float rDot = isHovered ? 6.0f : 4.0f;
            dl->AddCircle(end.pos, rDot, end.color, 12, isHovered ? 2.0f : 1.5f);
            if (isHovered) {
                dl->AddCircle(end.pos, rDot + 2.0f, IM_COL32(255, 255, 255, 220), 12, 1.0f);
            }
        }
    }
}

// Classic map/CAD-style scale bar: picks a "nice" round millimetre length
// and draws it at its true current on-screen pixel width, so the user gets
// direct visual + numeric feedback on the model's real-world size as they
// adjust Scale or zoom. Millimetres only for now; cm/m display units are a
// later addition (see header comment), not a unit *conversion* of the
// model — world space is already millimetres by construction.
void DrawScaleRuler(const AppState& app) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    float width = viewport->Size.x;
    float height = viewport->Size.y;
    if (width <= 1.0f || height <= 1.0f) return;

    Mat4 view = app.camera.ViewMatrix();
    OrbitCamera::ClipPlanes clip = app.camera.NearFar();
    Mat4 viewProj = Mat4::Perspective(0.9f, width / height, clip.nearZ, clip.farZ) * view;

    auto ProjectToScreen = [&](const Vec3& worldPos) -> ImVec2 {
        float x = viewProj.m[0] * worldPos.x + viewProj.m[4] * worldPos.y + viewProj.m[8] * worldPos.z + viewProj.m[12];
        float y = viewProj.m[1] * worldPos.x + viewProj.m[5] * worldPos.y + viewProj.m[9] * worldPos.z + viewProj.m[13];
        float w = viewProj.m[3] * worldPos.x + viewProj.m[7] * worldPos.y + viewProj.m[11] * worldPos.z + viewProj.m[15];
        w = std::max(w, 1e-5f);
        return ImVec2(viewport->Pos.x + (x / w * 0.5f + 0.5f) * width, viewport->Pos.y + (1.0f - (y / w * 0.5f + 0.5f)) * height);
    };

    // World space is millimetres already (see ModelMatrix()'s comment), so
    // 1 world unit along the camera's right vector, measured at the
    // target's depth, directly gives pixels-per-millimetre there.
    Vec3 right = app.camera.GetBasis().right;
    ImVec2 p0 = ProjectToScreen(app.camera.target);
    ImVec2 p1 = ProjectToScreen(app.camera.target + right);
    float pixelsPerMM = std::hypot(p1.x - p0.x, p1.y - p0.y);
    if (!(pixelsPerMM > 1e-6f) || !std::isfinite(pixelsPerMM)) return;

    constexpr float kNiceStepsMM[] = {1,    2,    5,    10,    20,    50,    100,   200,
                                       500,  1000, 2000, 5000,  10000, 20000, 50000, 100000};
    constexpr float kTargetPixels = 120.0f;
    float barMM = kNiceStepsMM[sizeof(kNiceStepsMM) / sizeof(kNiceStepsMM[0]) - 1];
    for (float mm : kNiceStepsMM) {
        if (mm * pixelsPerMM >= kTargetPixels) {
            barMM = mm;
            break;
        }
    }
    float barPixels = std::clamp(barMM * pixelsPerMM, 10.0f, width * 0.6f);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 origin{viewport->Pos.x + 24.0f, viewport->Pos.y + height - 34.0f};
    ImVec2 end{origin.x + barPixels, origin.y};
    ImU32 color = IM_COL32(230, 230, 230, 255);

    dl->AddLine(origin, end, color, 2.0f);
    dl->AddLine(ImVec2(origin.x, origin.y - 6.0f), ImVec2(origin.x, origin.y + 6.0f), color, 2.0f);
    dl->AddLine(ImVec2(end.x, end.y - 6.0f), ImVec2(end.x, end.y + 6.0f), color, 2.0f);

    char label[32];
    std::snprintf(label, sizeof(label), "%g mm", barMM);
    ImVec2 textSize = ImGui::CalcTextSize(label);
    dl->AddText(ImVec2(origin.x + (barPixels - textSize.x) * 0.5f, origin.y - textSize.y - 4.0f), color, label);
}

// Draws a 3D-projected red arrow with a thin yellow border pointing in the
// direction of the aerodynamic Lift force (+Z for positive upward lift, -Z
// for negative lift / downforce) anchored above the Centre of Pressure.
void DrawLiftArrow(const AppState& app) {
    if (!app.showLiftArrow || !app.meshLoaded || !app.voxelGridValid || !app.lbmValid) return;
    if (!app.lbmForcesValidAtomic.load(std::memory_order_relaxed)) return;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    float width = viewport->Size.x;
    float height = viewport->Size.y;
    if (width <= 1.0f || height <= 1.0f) return;

    Mat4 view = app.camera.ViewMatrix();
    OrbitCamera::ClipPlanes clip = app.camera.NearFar();
    Mat4 viewProj = Mat4::Perspective(0.9f, width / height, clip.nearZ, clip.farZ) * view;

    auto ProjectToScreen = [&](const Vec3& worldPos, float& outW) -> ImVec2 {
        float x = viewProj.m[0] * worldPos.x + viewProj.m[4] * worldPos.y + viewProj.m[8] * worldPos.z + viewProj.m[12];
        float y = viewProj.m[1] * worldPos.x + viewProj.m[5] * worldPos.y + viewProj.m[9] * worldPos.z + viewProj.m[13];
        float w = viewProj.m[3] * worldPos.x + viewProj.m[7] * worldPos.y + viewProj.m[11] * worldPos.z + viewProj.m[15];
        outW = w;
        w = std::max(w, 1e-5f);
        return ImVec2(viewport->Pos.x + (x / w * 0.5f + 0.5f) * width,
                      viewport->Pos.y + (1.0f - (y / w * 0.5f + 0.5f)) * height);
    };

    Vec3 cp{app.lbmCpXAtomic.load(std::memory_order_relaxed), app.lbmCpYAtomic.load(std::memory_order_relaxed),
            app.lbmCpZAtomic.load(std::memory_order_relaxed)};
    float baseZ = std::max(app.lbmLiftBaseZAtomic.load(std::memory_order_relaxed), cp.z);
    float liftLat = app.lbmForceZAtomic.load(std::memory_order_relaxed);
    float liftN = LatticeForceToNewtons(app, liftLat);
    constexpr float kNewtonsToGramsForce = 1000.0f / 9.80665f;
    float liftGf = liftN * kNewtonsToGramsForce;

    float clearance = std::max(app.voxelGrid.cellSize * 1.5f, app.camera.sceneRadius * 0.04f);
    Vec3 anchorAboveCp{cp.x, cp.y, baseZ + clearance};

    // Scale the arrow length smoothly between 24% and 48% of sceneRadius so
    // the Lift vector is always legible while still growing with lift magnitude.
    float magFactor = std::clamp(std::log10(1.0f + std::fabs(liftGf) * 0.25f) * 0.5f, 0.0f, 1.0f);
    float arrowWorldLen = app.camera.sceneRadius * (0.24f + 0.24f * magFactor);
    bool upward = (liftLat >= 0.0f);

    // Keep the entire arrow positioned above the Centre of Pressure (and above
    // the upper surface of the model) whether Lift points upward (+Z) or
    // downward (-Z, downforce).
    Vec3 tailWorld = upward ? anchorAboveCp : (anchorAboveCp + Vec3{0.0f, 0.0f, arrowWorldLen});
    Vec3 tipWorld = upward ? (anchorAboveCp + Vec3{0.0f, 0.0f, arrowWorldLen}) : anchorAboveCp;

    float wCp = 0.0f, wAnchor = 0.0f, wTail = 0.0f, wTip = 0.0f;
    ImVec2 screenCp = ProjectToScreen(cp, wCp);
    ImVec2 screenAnchor = ProjectToScreen(anchorAboveCp, wAnchor);
    ImVec2 screenTail = ProjectToScreen(tailWorld, wTail);
    ImVec2 screenTip = ProjectToScreen(tipWorld, wTip);
    if (wCp <= 0.0f || wTail <= 0.0f || wTip <= 0.0f) return;

    ImDrawList* dl = ImGui::GetForegroundDrawList();

    // Vertical dashed leader line from the 3D Centre of Pressure to the base
    // of the Lift arrow above the upper surface, plus a CAD-style CoP marker.
    float leaderDx = screenAnchor.x - screenCp.x;
    float leaderDy = screenAnchor.y - screenCp.y;
    float leaderLen = std::hypot(leaderDx, leaderDy);
    if (leaderLen > 6.0f) {
        int segments = std::clamp(static_cast<int>(leaderLen / 6.0f), 2, 24);
        for (int s = 0; s < segments; s += 2) {
            float t0 = static_cast<float>(s) / static_cast<float>(segments);
            float t1 = static_cast<float>(std::min(s + 1, segments)) / static_cast<float>(segments);
            dl->AddLine(ImVec2(screenCp.x + leaderDx * t0, screenCp.y + leaderDy * t0),
                        ImVec2(screenCp.x + leaderDx * t1, screenCp.y + leaderDy * t1),
                        IM_COL32(255, 230, 50, 175), 1.4f);
        }
    }
    dl->AddCircleFilled(screenCp, 4.5f, IM_COL32(225, 32, 32, 235));
    dl->AddCircle(screenCp, 4.5f, IM_COL32(255, 230, 50, 255), 12, 1.5f);

    // Ensure a readable minimum screen-space arrow length even when viewing
    // from nearly straight above/below (planform view).
    float dx = screenTip.x - screenTail.x;
    float dy = screenTip.y - screenTail.y;
    float lenPx = std::hypot(dx, dy);
    float ux = 0.0f;
    float uy = upward ? -1.0f : 1.0f;
    if (lenPx > 1.0f) {
        ux = dx / lenPx;
        uy = dy / lenPx;
    }
    float clampedLenPx = std::clamp(lenPx, 48.0f, 180.0f);
    screenTip = ImVec2(screenTail.x + ux * clampedLenPx, screenTail.y + uy * clampedLenPx);

    // Perpendicular screen normal
    float nx = -uy;
    float ny = ux;

    float headLen = std::clamp(clampedLenPx * 0.34f, 16.0f, 30.0f);
    float shaftHalfW = 5.0f;
    float headHalfW = 12.5f;
    ImVec2 neck{screenTip.x - ux * headLen, screenTip.y - uy * headLen};

    // 7-vertex perimeter of the arrow (shaft + head) so the red fill and thin
    // yellow border form one unified arrow shape with no internal seam line.
    ImVec2 p0{screenTail.x - nx * shaftHalfW, screenTail.y - ny * shaftHalfW};
    ImVec2 p1{screenTail.x + nx * shaftHalfW, screenTail.y + ny * shaftHalfW};
    ImVec2 p2{neck.x + nx * shaftHalfW, neck.y + ny * shaftHalfW};
    ImVec2 p3{neck.x + nx * headHalfW, neck.y + ny * headHalfW};
    ImVec2 p4 = screenTip;
    ImVec2 p5{neck.x - nx * headHalfW, neck.y - ny * headHalfW};
    ImVec2 p6{neck.x - nx * shaftHalfW, neck.y - ny * shaftHalfW};

    ImU32 redFill = IM_COL32(225, 32, 32, 240);
    ImU32 yellowBorder = IM_COL32(255, 230, 45, 255);

    dl->AddQuadFilled(p0, p1, p2, p6, redFill);
    dl->AddTriangleFilled(p5, p3, p4, redFill);

    const ImVec2 outline[7] = {p0, p1, p2, p3, p4, p5, p6};
    dl->AddPolyline(outline, 7, yellowBorder, ImDrawFlags_Closed, 1.8f);

    // Compact badge next to the arrow tip showing the Lift force
    char liftLabel[64];
    std::snprintf(liftLabel, sizeof(liftLabel), "Lift: %.3g N (%.1f gf)", liftN, liftGf);
    ImVec2 textSz = ImGui::CalcTextSize(liftLabel);
    ImVec2 labelPos{screenTip.x - textSz.x * 0.5f,
                    upward ? (screenTip.y - textSz.y - 10.0f) : (screenTip.y + 8.0f)};
    ImVec2 pillMin{labelPos.x - 5.0f, labelPos.y - 2.0f};
    ImVec2 pillMax{labelPos.x + textSz.x + 5.0f, labelPos.y + textSz.y + 2.0f};
    dl->AddRectFilled(pillMin, pillMax, IM_COL32(18, 20, 25, 210), 4.0f);
    dl->AddRect(pillMin, pillMax, IM_COL32(255, 230, 45, 190), 4.0f, 0, 1.0f);
    dl->AddText(labelPos, IM_COL32(255, 245, 220, 255), liftLabel);
}

// One slice-plane drag handle: a little axis arrow drawn at that plane's
// own current position, in both its world-space direction and its current
// on-screen projection. One per currently-*enabled* plane (see AppState::
// slicePlanes) — unlike the single-slice design this replaced, there's no
// "switch axis by grabbing a different handle": enabling a plane is the
// checkbox's job (see the Slice UI block in DrawUI), a handle only drags
// the position of a plane that's already on.
struct SliceHandle {
    SliceAxis axis;
    Vec3 worldDir;  // unit world-space direction this axis's slice index moves along
    ImVec2 screenCenter;
    ImVec2 screenTip;
    // Pixels corresponding to *one* world unit along worldDir, at the
    // slice's own depth — NOT pixels-per-handleLength (the tip is drawn
    // handleLength away, a separate, purely cosmetic distance). The drag
    // math needs this to convert raw pixel motion into world units.
    float pixelsPerWorldUnit;
};

// Shared by DrawSliceDragHandles (render) and the mouse handlers in main()
// (hit-test on button-down, axis-relative drag math on motion) so the two
// can never disagree about where a handle actually is or which way it
// points. Empty wherever there's nothing sensible to show/hit: no solved
// grid yet, a degenerate (zero-size) viewport, or simply no plane enabled.
std::vector<SliceHandle> ComputeSliceHandles(const AppState& app) {
    std::vector<SliceHandle> handles;
    if (!app.lbmValid || app.voxelGrid.CellCount() == 0) return handles;
    const VoxelGrid& grid = app.voxelGrid;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    float width = viewport->Size.x, height = viewport->Size.y;
    if (width <= 1.0f || height <= 1.0f) return handles;

    Mat4 view = app.camera.ViewMatrix();
    OrbitCamera::ClipPlanes clip = app.camera.NearFar();
    Mat4 viewProj = Mat4::Perspective(0.9f, width / height, clip.nearZ, clip.farZ) * view;
    auto ProjectToScreen = [&](const Vec3& worldPos) -> ImVec2 {
        float x = viewProj.m[0] * worldPos.x + viewProj.m[4] * worldPos.y + viewProj.m[8] * worldPos.z + viewProj.m[12];
        float y = viewProj.m[1] * worldPos.x + viewProj.m[5] * worldPos.y + viewProj.m[9] * worldPos.z + viewProj.m[13];
        float w = viewProj.m[3] * worldPos.x + viewProj.m[7] * worldPos.y + viewProj.m[11] * worldPos.z + viewProj.m[15];
        w = std::max(w, 1e-5f);
        return ImVec2(viewport->Pos.x + (x / w * 0.5f + 0.5f) * width,
                       viewport->Pos.y + (1.0f - (y / w * 0.5f + 0.5f)) * height);
    };

    // A fixed fraction of the scene's own radius, like DrawOrientationGizmo/
    // DrawScaleRuler's own sizing — reads as a sensible, consistent size
    // regardless of the model's actual scale or the current zoom level.
    float handleLength = std::max(app.camera.sceneRadius * 0.25f, grid.cellSize * 4.0f);

    // grid-x moves along world Y and grid-y along world X — see
    // VoxelGrid::CellCenter's own comment; grid-z/world Z agree already.
    constexpr std::array<Vec3, 3> kWorldDirs = {{{0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}};

    for (size_t i = 0; i < 3; ++i) {
        const AppState::SlicePlaneState& plane = app.slicePlanes[i];
        if (!plane.enabled) continue;

        // This plane's own center: grid's midpoint on the two axes it
        // doesn't fix, its own index on the one it does — same cell
        // FieldOverlay::BuildSliceMeshFromRange centers that plane's mesh
        // on.
        int gx = grid.nx / 2, gy = grid.ny / 2, gz = grid.nz / 2;
        switch (kSliceAxes[i]) {
            case SliceAxis::X:
                gx = std::clamp(plane.index, 0, grid.nx - 1);
                break;
            case SliceAxis::Y:
                gy = std::clamp(plane.index, 0, grid.ny - 1);
                break;
            case SliceAxis::Z:
                gz = std::clamp(plane.index, 0, grid.nz - 1);
                break;
        }
        Vec3 centerWorld = grid.CellCenter(gx, gy, gz);
        ImVec2 screenCenter = ProjectToScreen(centerWorld);
        ImVec2 screenTip = ProjectToScreen(centerWorld + kWorldDirs[i] * handleLength);
        float dx = screenTip.x - screenCenter.x, dy = screenTip.y - screenCenter.y;
        float pixelsPerWorldUnit = std::hypot(dx, dy) / handleLength;
        handles.push_back({kSliceAxes[i], kWorldDirs[i], screenCenter, screenTip, pixelsPerWorldUnit});
    }
    return handles;
}

// Draws every currently-enabled plane's drag handle computed above: a
// colored line from that plane's own center to its tip. Only called while
// Slice style is showing (see its call site) — Voxel Cloud/Particles have
// no slice plane to anchor these on.
void DrawSliceDragHandles(const AppState& app) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    for (const SliceHandle& h : ComputeSliceHandles(app)) {
        ImU32 color = h.axis == SliceAxis::X  ? IM_COL32(235, 90, 90, 255)
                      : h.axis == SliceAxis::Y ? IM_COL32(100, 210, 100, 255)
                                                 : IM_COL32(100, 150, 235, 255);
        dl->AddLine(h.screenCenter, h.screenTip, color, 3.5f);
        dl->AddCircleFilled(h.screenTip, 7.0f, color);
    }
}

// Hit-tests a mouse-down position (screen pixels) against every currently-
// enabled plane's handle tip, for starting a drag — see main()'s
// SDL_EVENT_MOUSE_BUTTON_DOWN handling. Picks the single closest tip
// within kPickRadiusPixels, nullopt if none are that close (so an ordinary
// click still falls through to orbiting the camera, same as before these
// handles existed) — including, now, every disabled plane's handle, since
// those simply don't exist in ComputeSliceHandles' result at all.
std::optional<SliceAxis> PickSliceHandle(const AppState& app, ImVec2 mousePos) {
    std::vector<SliceHandle> handles = ComputeSliceHandles(app);

    constexpr float kPickRadiusPixels = 10.0f;
    std::optional<SliceAxis> best;
    float bestDistSq = kPickRadiusPixels * kPickRadiusPixels;
    for (const SliceHandle& h : handles) {
        float dx = h.screenTip.x - mousePos.x, dy = h.screenTip.y - mousePos.y;
        float distSq = dx * dx + dy * dy;
        if (distSq <= bestDistSq) {
            bestDistSq = distSq;
            best = h.axis;
        }
    }
    return best;
}

// Applies one MOUSE_MOTION event's raw pixel delta to the currently-
// grabbed slice handle (app.draggingSliceHandle/draggingSliceHandleAxis) —
// projects the motion onto that handle's own on-screen direction, converts
// to world units via its pixelsPerWorldUnit, then to whole grid cells,
// accumulating the sub-cell remainder in sliceDragFractionalCells so a
// slow drag isn't lost to rounding every event. Updates that plane's own
// index (and its atomic mirror, and reslices) exactly the way the Slice
// Position slider's own change handler already does, once the accumulated
// motion actually crosses a whole cell.
void ApplySliceHandleDrag(AppState& app, float xrel, float yrel) {
    const SliceHandle* active = nullptr;
    std::vector<SliceHandle> handles = ComputeSliceHandles(app);
    for (const SliceHandle& h : handles) {
        if (h.axis == app.draggingSliceHandleAxis) {
            active = &h;
            break;
        }
    }
    if (!active || !(active->pixelsPerWorldUnit > 1e-6f) || !std::isfinite(active->pixelsPerWorldUnit)) return;

    size_t planeIndex = static_cast<size_t>(app.draggingSliceHandleAxis);

    float dirX = active->screenTip.x - active->screenCenter.x;
    float dirY = active->screenTip.y - active->screenCenter.y;
    float screenLen = std::hypot(dirX, dirY);
    if (!(screenLen > 1e-6f)) return;
    dirX /= screenLen;
    dirY /= screenLen;

    float motionAlongAxisPixels = xrel * dirX + yrel * dirY;  // signed
    float worldDelta = motionAlongAxisPixels / active->pixelsPerWorldUnit;
    float cellDelta = worldDelta / std::max(app.voxelGrid.cellSize, 1e-6f);

    app.sliceDragFractionalCells += cellDelta;
    int wholeCells = static_cast<int>(app.sliceDragFractionalCells);  // truncates toward 0
    if (wholeCells == 0) return;
    app.sliceDragFractionalCells -= static_cast<float>(wholeCells);

    AppState::SlicePlaneState& plane = app.slicePlanes[planeIndex];
    int sliceMax = std::max(OverlaySliceDimension(app, app.draggingSliceHandleAxis) - 1, 0);
    plane.index = std::clamp(plane.index + wholeCells, 0, sliceMax);
    plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
    StartResliceTask(app);
}

// Keeps a just-dragged (or resized) window fully within the main viewport.
// Must run right after Begin(), since it needs the window's actual
// position/size for this frame to clamp against — a window title bar
// dragged fully off screen would otherwise be hard to recover.
void ClampWindowToViewport(float topInset = 0.0f) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 pos = ImGui::GetWindowPos();
    ImVec2 size = ImGui::GetWindowSize();
    ImVec2 minPos{viewport->Pos.x, viewport->Pos.y + topInset};
    ImVec2 maxPos{viewport->Pos.x + viewport->Size.x - size.x, viewport->Pos.y + viewport->Size.y - size.y};
    // If the window is bigger than the viewport, pin it to the top-left
    // rather than leaving the clamp range inverted.
    maxPos.x = std::max(maxPos.x, minPos.x);
    maxPos.y = std::max(maxPos.y, minPos.y);

    ImVec2 clamped{std::clamp(pos.x, minPos.x, maxPos.x), std::clamp(pos.y, minPos.y, maxPos.y)};
    if (clamped.x != pos.x || clamped.y != pos.y) ImGui::SetWindowPos(clamped);
}

// Gradient bar + min/max readout for the flow overlay's color scale —
// without this there's no way to tell what a given color actually means.
void DrawOverlayLegend(const AppState& app) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float barWidth = 220.0f, barHeight = 16.0f;
    ImVec2 topLeft{viewport->Pos.x + viewport->Size.x - barWidth - 24.0f, viewport->Pos.y + viewport->Size.y - 56.0f};

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const int kSteps = 32;
    for (int i = 0; i < kSteps; ++i) {
        float t0 = static_cast<float>(i) / kSteps;
        float t1 = static_cast<float>(i + 1) / kSteps;
        Vec3 c = FieldOverlay::Colormap(t0);
        ImU32 col = IM_COL32(static_cast<int>(c.x * 255), static_cast<int>(c.y * 255), static_cast<int>(c.z * 255), 255);
        dl->AddRectFilled(ImVec2(topLeft.x + t0 * barWidth, topLeft.y), ImVec2(topLeft.x + t1 * barWidth, topLeft.y + barHeight), col);
    }
    dl->AddRect(topLeft, ImVec2(topLeft.x + barWidth, topLeft.y + barHeight), IM_COL32(20, 20, 24, 255));

    // Must mirror LegendRangeForStyle's own useSpeed logic exactly —
    // app.overlayMin/Max (which this legend displays) is whichever range
    // that function picked, so this has to agree on which one that was
    // (Voxel Cloud/Particles are always Speed, regardless of Color By).
    bool isStress = app.overlayStyle == OverlayStyle::StressMap;
    bool particleLike = app.overlayStyle == OverlayStyle::Particles ||
                        (app.overlayStyle == OverlayStyle::Slice && app.sliceShowParticles);
    bool isSpeed = !isStress && !particleLike &&
                   (app.overlayStyle == OverlayStyle::VoxelCloud || app.overlayColorMode == FieldColorMode::Speed);
    const char* title = isStress ? "Surface stress (Pa)"
                                 : (isSpeed ? "Speed (mm/s)" : "Pressure deviation (lattice units)");
    dl->AddText(ImVec2(topLeft.x, topLeft.y - 18.0f), IM_COL32(230, 230, 230, 255), title);

    char minLabel[32], maxLabel[32];
    if (isStress) {
        std::snprintf(minLabel, sizeof(minLabel), "%.3g Pa", LatticeStressToPascals(app, app.overlayMin));
        std::snprintf(maxLabel, sizeof(maxLabel), "%.3g Pa", LatticeStressToPascals(app, app.overlayMax));
    } else if (isSpeed) {
        std::snprintf(minLabel, sizeof(minLabel), "%.0f", LatticeSpeedToMmPerSecond(app, app.overlayMin));
        std::snprintf(maxLabel, sizeof(maxLabel), "%.0f", LatticeSpeedToMmPerSecond(app, app.overlayMax));
    } else {
        std::snprintf(minLabel, sizeof(minLabel), "%.4g", app.overlayMin);
        std::snprintf(maxLabel, sizeof(maxLabel), "%.4g", app.overlayMax);
    }
    dl->AddText(ImVec2(topLeft.x, topLeft.y + barHeight + 4.0f), IM_COL32(230, 230, 230, 255), minLabel);
    ImVec2 maxSize = ImGui::CalcTextSize(maxLabel);
    dl->AddText(ImVec2(topLeft.x + barWidth - maxSize.x, topLeft.y + barHeight + 4.0f), IM_COL32(230, 230, 230, 255),
                maxLabel);
}

// Centered "working" panel shown for the whole duration of a background
// task (see StartTask) — without this, Voxelize/Initialize-or-Reset-Solver
// /Build-Overlay at high resolution used to just look like the app had
// hung, with nothing on screen to say otherwise. Drawn with the foreground
// draw list, like DrawOrientationGizmo/DrawOverlayLegend above, so it
// doesn't need its own ImGui window. Elapsed time is shown either way —
// Voxelize gets a real fraction (see StartVoxelizeTask/Voxelizer.cpp), the
// other two an animated spinner, since neither has a cheap natural
// progress checkpoint to report.
void DrawBusyOverlay(const AppState& app) {
    TaskKind kind = app.taskKind.load(std::memory_order_acquire);
    if (kind == TaskKind::None) return;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float panelW = 320.0f, panelH = 92.0f;
    ImVec2 topLeft{viewport->Pos.x + (viewport->Size.x - panelW) * 0.5f,
                   viewport->Pos.y + (viewport->Size.y - panelH) * 0.5f};
    ImVec2 bottomRight{topLeft.x + panelW, topLeft.y + panelH};
    ImVec2 center{topLeft.x + panelW * 0.5f, topLeft.y + panelH * 0.5f};

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddRectFilled(topLeft, bottomRight, IM_COL32(18, 19, 23, 235), 8.0f);
    dl->AddRect(topLeft, bottomRight, IM_COL32(70, 72, 80, 255), 8.0f);

    const char* label = TaskKindLabel(kind);
    ImVec2 labelSize = ImGui::CalcTextSize(label);
    dl->AddText(ImVec2(center.x - labelSize.x * 0.5f, topLeft.y + 14.0f), IM_COL32(235, 235, 235, 255), label);

    float elapsedSeconds = static_cast<float>(SDL_GetTicks() - app.taskStartTicks) / 1000.0f;

    bool hasProgress = app.taskHasProgress.load(std::memory_order_relaxed);
    if (hasProgress) {
        float fraction = std::clamp(app.taskProgress.load(std::memory_order_relaxed), 0.0f, 1.0f);
        const float barW = panelW - 48.0f, barH = 14.0f;
        ImVec2 barTopLeft{center.x - barW * 0.5f, topLeft.y + 42.0f};
        ImVec2 barBottomRight{barTopLeft.x + barW, barTopLeft.y + barH};
        dl->AddRectFilled(barTopLeft, barBottomRight, IM_COL32(40, 41, 46, 255), 3.0f);
        dl->AddRectFilled(barTopLeft, ImVec2(barTopLeft.x + barW * fraction, barBottomRight.y),
                           IM_COL32(90, 160, 235, 255), 3.0f);
        dl->AddRect(barTopLeft, barBottomRight, IM_COL32(70, 72, 80, 255), 3.0f);

        char pctLabel[32];
        std::snprintf(pctLabel, sizeof(pctLabel), "%.0f%%  (%.1fs)", fraction * 100.0f, elapsedSeconds);
        ImVec2 pctSize = ImGui::CalcTextSize(pctLabel);
        dl->AddText(ImVec2(center.x - pctSize.x * 0.5f, barBottomRight.y + 8.0f), IM_COL32(190, 190, 195, 255),
                    pctLabel);
    } else {
        // No real progress fraction available (see this function's
        // comment) — an 8-dot ring with a traveling bright spot, driven by
        // wall-clock time, at least shows the app is alive and not frozen.
        const int kDots = 8;
        const float ringRadius = 12.0f;
        ImVec2 ringCenter{center.x, topLeft.y + 50.0f};
        float phase = std::fmod(static_cast<float>(ImGui::GetTime()) * 2.5f, static_cast<float>(kDots));
        for (int i = 0; i < kDots; ++i) {
            float angle = (static_cast<float>(i) / kDots) * 2.0f * std::numbers::pi_v<float>;
            ImVec2 pos{ringCenter.x + std::cos(angle) * ringRadius, ringCenter.y + std::sin(angle) * ringRadius};
            float dist = std::fmod(static_cast<float>(i) - phase + kDots, static_cast<float>(kDots));
            float brightness = std::max(0.0f, 1.0f - dist / 4.0f);
            int alpha = static_cast<int>(60 + 195 * brightness);
            dl->AddCircleFilled(pos, 3.0f, IM_COL32(150, 190, 235, alpha));
        }

        char elapsedLabel[32];
        std::snprintf(elapsedLabel, sizeof(elapsedLabel), "%.1fs elapsed", elapsedSeconds);
        ImVec2 elapsedSize = ImGui::CalcTextSize(elapsedLabel);
        dl->AddText(ImVec2(center.x - elapsedSize.x * 0.5f, topLeft.y + 68.0f), IM_COL32(190, 190, 195, 255),
                    elapsedLabel);
    }
}

// Pinned to the top-right corner (below DrawOrientationGizmo, which owns
// that corner's top ~128px — see this function's position math) instead
// of dimming/centering over the whole viewport like DrawBusyOverlay, and
// visible for as long as app.lbmRunning is true — not just while a
// background task happens to be in flight — so a slow machine still
// shows live reassurance that stepping is actually progressing rather
// than having frozen, for the whole time Run is held down, not just a
// brief flash during the next overlayAutoUpdate tick's BuildingOverlay
// task (that task is still folded in below, as a nested progress bar,
// when it happens to be running at the same time). Voxelizing and
// InitializingSolver are one-time setup steps before a run even starts,
// not "the solver running" itself, and DrawBusyOverlay's centered panel
// already covers those adequately on its own — this window has nothing
// to show for them.
// Solver progress & substep batch monitor — seeded in the top-right corner
// below DrawOrientationGizmo, keeping its position relative to the right edge
// when the main window is resized (while remaining freely draggable,
// collapsible, and closable).
// Refreshes app.perfMeters at most every kPerfSampleIntervalMs — called
// every frame the Processing window is showing a running solver. Half a
// second is long enough that the CPU-time difference isn't dominated by
// scheduler granularity (and that the numbers are readable rather than
// flickering), short enough to still feel live.
void UpdatePerfMeters(AppState& app) {
    constexpr Uint64 kPerfSampleIntervalMs = 500;
    // A gap this long means the window wasn't being drawn (paused, closed,
    // collapsed) — a rate averaged across it would be meaningless, so the
    // next sample just starts over.
    constexpr Uint64 kPerfStaleMs = 3000;

    AppState::PerfMeters& m = app.perfMeters;
    Uint64 now = SDL_GetTicks();
    int stepCount = app.lbmStepCount.load(std::memory_order_relaxed);
    bool restart = m.lastSampleTicks == 0 || now - m.lastSampleTicks > kPerfStaleMs ||
                   stepCount < m.lastStepCount;  // step count went backwards: the solver was re-initialized
    if (!restart && now - m.lastSampleTicks < kPerfSampleIntervalMs) return;

    double cpuSeconds = SystemMeters::ProcessCpuSeconds();
    if (restart) {
        m.ratesValid = false;
    } else {
        double wallSeconds = static_cast<double>(now - m.lastSampleTicks) / 1000.0;
        m.stepsPerSecond = static_cast<float>(static_cast<double>(stepCount - m.lastStepCount) / wallSeconds);
        m.cpuCoresInUse = (cpuSeconds >= 0.0 && m.lastCpuSeconds >= 0.0)
                              ? static_cast<float>(std::max(0.0, cpuSeconds - m.lastCpuSeconds) / wallSeconds)
                              : -1.0f;
        m.ratesValid = true;
    }
    m.gpuUtilization = SystemMeters::GpuUtilization();  // an instantaneous reading, not a difference
    m.lastSampleTicks = now;
    m.lastCpuSeconds = cpuSeconds;
    m.lastStepCount = stepCount;
}

void DrawProcessingWindow(AppState& app, float topInset = 0.0f) {
    bool running = app.lbmRunning.load(std::memory_order_relaxed);
    bool diverged = app.lbmFlowDivergedAtomic.load(std::memory_order_relaxed);
    if ((!running && !diverged) || !app.showProcessingWindow) return;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    // Default spot: directly under the orientation widget, right edges
    // lined up. Applied below with SetWindowPos every frame until the user
    // drags the window, rather than once via SetNextWindowPos — that
    // one-shot placement used to be silently cancelled by this function's
    // own SetWindowPos call (which only corrected x) while the
    // auto-resizing window was still hidden on its first frames, leaving
    // it at ImGui's stock y and so on top of the widget instead of
    // under it.
    constexpr float kRightMargin = kGizmoRightMargin;
    constexpr float kGapBelowGizmo = 8.0f;
    ImGui::SetNextWindowBgAlpha(0.85f);
    bool expanded = ImGui::Begin("Processing", &app.showProcessingWindow, ImGuiWindowFlags_AlwaysAutoResize);
    {
        ImVec2 currentPos = ImGui::GetWindowPos();
        ImVec2 currentSize = ImGui::GetWindowSize();
        float viewportRight = viewport->Pos.x + viewport->Size.x;

        // Every position change this function makes itself is recorded in
        // processingWindowLastPos at the bottom, so a position that
        // differs from it on the next frame, with the button held on the
        // focused window, can only be ImGui moving the window for a user
        // drag. (IsMouseDown, not IsMouseDragging: the latter stays false
        // for the first few pixels of a drag, during which the code below
        // would otherwise pull the window back.)
        if (!app.processingWindowInitialized) {
            app.processingWindowInitialized = true;
            app.processingWindowUserMoved = false;
        } else {
            bool posChanged =
                (currentPos.x != app.processingWindowLastPos.x || currentPos.y != app.processingWindowLastPos.y);
            if (posChanged && ImGui::IsWindowFocused() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                app.processingWindowUserMoved = true;
                app.processingWindowRightInset = viewportRight - (currentPos.x + currentSize.x);
            }
        }

        // The window is anchored by its RIGHT edge: x is re-derived every
        // frame from a distance to the viewport's right edge (the default
        // margin, or wherever the user last dropped it), so that distance
        // survives the main window being resized, the window's own
        // auto-resized width changing as rows come and go, and being
        // squeezed by ClampWindowToViewport in a too-narrow main window and
        // then given room again. (Shifting x by each frame's viewport-width
        // change, as this used to, kept the LEFT edge's offset instead and
        // lost it for good once clamped.) y is top-anchored: the default
        // slot under the orientation widget, or left wherever it was
        // dragged.
        float rightInset = app.processingWindowUserMoved ? app.processingWindowRightInset : kRightMargin;
        ImVec2 targetPos;
        targetPos.x = viewportRight - rightInset - currentSize.x;
        targetPos.y = app.processingWindowUserMoved
                          ? currentPos.y
                          : viewport->Pos.y + topInset + kGizmoBottomInset + kGapBelowGizmo;
        if (targetPos.x != currentPos.x || targetPos.y != currentPos.y) {
            ImGui::SetWindowPos(targetPos);
        }

        ClampWindowToViewport(topInset);
        app.processingWindowLastPos = ImGui::GetWindowPos();
    }

    if (expanded) {
        if (diverged) {
            ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.35f, 1.0f),
                               running ? "SOLVER DIVERGED (NaN)" : "SOLVER DIVERGED (NaN) — Auto-Paused");
            ImGui::Text("Failed around Step %d", app.lbmDivergedAtStepAtomic.load(std::memory_order_relaxed));
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.95f, 1.0f), "Purple/magenta cells indicate NaN / blowup.");
            bool busy = app.taskKind.load(std::memory_order_acquire) != TaskKind::None;
            ImGui::BeginDisabled(busy);
            if (ImGui::Button("Stabilize & Reset Solver", ImVec2(220.0f, 0.0f))) {
                StabilizeAndResetSolver(app);
            }
            ImGui::EndDisabled();
            if (running) ImGui::Separator();
        }
        if (running) {
            ImGui::TextUnformatted("Solver running...");
            ImGui::Text("Step %d", app.lbmStepCount.load(std::memory_order_relaxed));

            // The current frame's batch of substeps (see AppState::lbmStepProgress's
            // comment) — particularly worth having on the CPU backend, where one
            // substep alone can take seconds to minutes, so without this the
            // window could only say "running" with no sense of whether the step
            // already in flight is 10% or 90% done.
            float stepFraction = std::clamp(app.lbmStepProgress.load(std::memory_order_relaxed), 0.0f, 1.0f);
            char stepLabel[16];
            std::snprintf(stepLabel, sizeof(stepLabel), "%.0f%%", stepFraction * 100.0f);
            ImGui::ProgressBar(stepFraction, ImVec2(220.0f, 0.0f), stepLabel);

            float elapsedSeconds = static_cast<float>(SDL_GetTicks() - app.lbmRunStartTicks) / 1000.0f;
            ImGui::Text("%.1fs since Run", elapsedSeconds);

            // Performance meters. CPU is this process only (all threads),
            // as a fraction of every logical core; GPU is the whole
            // device as the OS reports it, so it includes other apps and
            // FlowCraft's own rendering — see SystemMeters.h.
            UpdatePerfMeters(app);
            const AppState::PerfMeters& perf = app.perfMeters;
            ImGui::Separator();
            if (!perf.ratesValid) {
                ImGui::TextDisabled("Measuring performance...");
            } else {
                // MLUPS (million lattice-cell updates per second) is the
                // standard, resolution-independent LBM throughput figure;
                // it means nothing for the panel/potential-flow backends,
                // whose "steps" are solver sweeps, so those show only the
                // plain rate.
                if (IsLbmBackend(app.lbmBackend) && app.voxelGridValid) {
                    double mlups = static_cast<double>(perf.stepsPerSecond) *
                                   static_cast<double>(app.voxelGrid.CellCount()) * 1e-6;
                    ImGui::Text("%.1f steps/s  (%.0f MLUPS)", perf.stepsPerSecond, mlups);
                } else {
                    ImGui::Text("%.1f steps/s", perf.stepsPerSecond);
                }

                if (perf.cpuCoresInUse >= 0.0f) {
                    unsigned int cores = std::max(1u, std::thread::hardware_concurrency());
                    float cpuFraction = std::clamp(perf.cpuCoresInUse / static_cast<float>(cores), 0.0f, 1.0f);
                    char cpuLabel[48];
                    std::snprintf(cpuLabel, sizeof(cpuLabel), "CPU %.0f%%  (%.1f of %u cores)", cpuFraction * 100.0f,
                                  perf.cpuCoresInUse, cores);
                    ImGui::ProgressBar(cpuFraction, ImVec2(220.0f, 0.0f), cpuLabel);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("CPU time used by FlowCraft (all threads),\nas a share of every logical core.");
                    }
                }
                if (perf.gpuUtilization >= 0.0f) {
                    char gpuLabel[32];
                    std::snprintf(gpuLabel, sizeof(gpuLabel), "GPU %.0f%%", perf.gpuUtilization * 100.0f);
                    ImGui::ProgressBar(perf.gpuUtilization, ImVec2(220.0f, 0.0f), gpuLabel);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("GPU utilisation for the whole device,\nincluding rendering and other apps.");
                    }
                }
            }
        }

        // Nested: the overlay-rebuild task specifically (the only TaskKind
        // that can be in flight *while* lbmRunning is true — Voxelizing/
        // InitializingSolver both require lbmRunning false to even start).
        TaskKind kind = app.taskKind.load(std::memory_order_acquire);
        if (kind == TaskKind::BuildingOverlay) {
            ImGui::Separator();
            ImGui::TextUnformatted(TaskKindLabel(kind));
            if (app.taskHasProgress.load(std::memory_order_relaxed)) {
                float fraction = std::clamp(app.taskProgress.load(std::memory_order_relaxed), 0.0f, 1.0f);
                char overlayLabel[16];
                std::snprintf(overlayLabel, sizeof(overlayLabel), "%.0f%%", fraction * 100.0f);
                ImGui::ProgressBar(fraction, ImVec2(220.0f, 0.0f), overlayLabel);
            } else {
                ImGui::TextDisabled("working...");
            }
        }
    }
    ImGui::End();
}

// Prominent top-center viewport alert banner shown whenever the solver
// diverges (NaN / Inf / negative density / supersonic lattice blowup,
// which renders as purple/magenta cells in the overlay).
void DrawFlowDivergenceBanner(AppState& app, float topInset, bool busy) {
    bool hasFailure = app.lbmValid.load(std::memory_order_relaxed) &&
                      (app.lbmFlowDivergedAtomic.load(std::memory_order_relaxed) || app.overlayHasNaN ||
                       app.lbmStatsHasNaNAtomic.load(std::memory_order_relaxed));
    if (!hasFailure || app.divergenceBannerDismissedAtomic.load(std::memory_order_relaxed)) return;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x + viewport->Size.x * 0.5f, viewport->Pos.y + topInset + 14.0f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.0f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(460.0f, 0.0f), ImVec2(580.0f, 300.0f));

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.24f, 0.05f, 0.14f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 0.2f, 0.65f, 0.95f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove;
    if (ImGui::Begin("##FlowDivergenceAlert", nullptr, flags)) {
        int failedStep = app.lbmDivergedAtStepAtomic.load(std::memory_order_relaxed);
        if (failedStep <= 0) failedStep = app.lbmStepCount.load(std::memory_order_relaxed);
        bool paused = !app.lbmRunning.load(std::memory_order_relaxed);

        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.55f, 1.0f),
                           paused ? "FLOW SOLVER DIVERGED (NaN / Instability) — Auto-Paused at Step %d"
                                  : "FLOW SOLVER DIVERGED (NaN / Instability) at Step %d",
                           failedStep);
        ImGui::Spacing();
        ImGui::TextWrapped(
            "Calculations encountered numerical blowup (NaN / Inf or supersonic lattice speeds), which renders as "
            "purple/magenta cells. At high resolutions, lower cell size reduces lattice viscosity (tau -> 0.5) and "
            "requires sub-grid turbulence damping (Smagorinsky Cs >= 0.18) or a slightly higher Tau.");
        ImGui::Spacing();

        ImGui::BeginDisabled(busy);
        if (ImGui::Button("Stabilize (Boost Cs & Tau) + Reset Solver")) {
            StabilizeAndResetSolver(app);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Solver Only")) {
            StartInitSolverTask(app);
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Dismiss")) {
            app.divergenceBannerDismissedAtomic.store(true, std::memory_order_relaxed);
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

// Projects the active Y-slice plane and the extracted 2D airfoil cross-section
// contour directly onto the 3D viewport when the Aero Polars window is open in
// Active Slice or Compare Both mode. Uses GetBackgroundDrawList() so it renders
// over the 3D scene and strictly underneath all floating ImGui windows.
void DrawPolarSliceHighlight(const AppState& app) {
    if (!app.showAeroPolarsWindow || !app.meshLoaded || !app.voxelGridValid) return;
    if (app.polarScopeMode != 1 && app.polarScopeMode != 2) return;

    const VoxelGrid& grid = app.voxelGrid;
    int sliceY = app.slicePlanes[static_cast<size_t>(SliceAxis::Y)].index;
    if (sliceY < 0) sliceY = grid.ny / 2;
    sliceY = std::clamp(sliceY, 0, std::max(0, grid.ny - 1));

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    float width = viewport->Size.x;
    float height = viewport->Size.y;
    if (width <= 1.0f || height <= 1.0f) return;

    Mat4 view = app.camera.ViewMatrix();
    OrbitCamera::ClipPlanes clip = app.camera.NearFar();
    Mat4 viewProj = Mat4::Perspective(0.9f, width / height, clip.nearZ, clip.farZ) * view;

    auto ProjectToScreen = [&](const Vec3& worldPos, bool& inFront) -> ImVec2 {
        float x = viewProj.m[0] * worldPos.x + viewProj.m[4] * worldPos.y + viewProj.m[8] * worldPos.z + viewProj.m[12];
        float y = viewProj.m[1] * worldPos.x + viewProj.m[5] * worldPos.y + viewProj.m[9] * worldPos.z + viewProj.m[13];
        float w = viewProj.m[3] * worldPos.x + viewProj.m[7] * worldPos.y + viewProj.m[11] * worldPos.z + viewProj.m[15];
        inFront = (w > 1e-4f);
        w = std::max(w, 1e-4f);
        return ImVec2(viewport->Pos.x + (x / w * 0.5f + 0.5f) * width,
                      viewport->Pos.y + (1.0f - (y / w * 0.5f + 0.5f)) * height);
    };

    ImDrawList* dl = ImGui::GetBackgroundDrawList();

    // 1. Translucent slice plane bounding frame at worldX
    float worldX = grid.origin.x + (static_cast<float>(sliceY) + 0.5f) * grid.cellSize;
    float y0 = grid.origin.y;
    float y1 = grid.origin.y + static_cast<float>(grid.nx) * grid.cellSize;
    float z0 = grid.origin.z;
    float z1 = grid.origin.z + static_cast<float>(grid.nz) * grid.cellSize;

    bool ok00 = false, ok10 = false, ok11 = false, ok01 = false;
    ImVec2 c00 = ProjectToScreen(Vec3{worldX, y0, z0}, ok00);
    ImVec2 c10 = ProjectToScreen(Vec3{worldX, y1, z0}, ok10);
    ImVec2 c11 = ProjectToScreen(Vec3{worldX, y1, z1}, ok11);
    ImVec2 c01 = ProjectToScreen(Vec3{worldX, y0, z1}, ok01);
    if (ok00 && ok10 && ok11 && ok01) {
        dl->AddQuadFilled(c00, c10, c11, c01, IM_COL32(0, 210, 255, 20));
        dl->AddQuad(c00, c10, c11, c01, IM_COL32(0, 210, 255, 130), 1.5f);
    }

    // 2. Highlight extracted 2D airfoil contour directly on the 3D model
    if (app.sectionPolarResult.valid && app.sectionPolarResult.profile.size() >= 2) {
        const auto& prof = app.sectionPolarResult.profile;
        for (size_t i = 0; i + 1 < prof.size(); ++i) {
            bool u0Ok = false, u1Ok = false, l0Ok = false, l1Ok = false;
            ImVec2 u0 = ProjectToScreen(prof[i].worldUpper, u0Ok);
            ImVec2 u1 = ProjectToScreen(prof[i + 1].worldUpper, u1Ok);
            ImVec2 l0 = ProjectToScreen(prof[i].worldLower, l0Ok);
            ImVec2 l1 = ProjectToScreen(prof[i + 1].worldLower, l1Ok);
            if (u0Ok && u1Ok) dl->AddLine(u0, u1, IM_COL32(255, 210, 40, 245), 2.6f);
            if (l0Ok && l1Ok) dl->AddLine(l0, l1, IM_COL32(40, 235, 255, 245), 2.6f);
        }
    }
}

// Interactive Aero Polars Window: runs multi-point angle-of-attack sweeps
// using the 3D Lifting Panel Method and/or the Active Y-Slice 2D Airfoil
// Section, plotting CL vs Alpha, CD vs CL, L/D vs Alpha, CM vs Alpha,
// 2D Airfoil Section Geometry, and Spanwise Lift Distribution via ImPlot.
void DrawAeroPolarsWindow(AppState& app, float topInset = 0.0f) {
    if (!app.showAeroPolarsWindow) return;

    DrawPolarSliceHighlight(app);

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowSize(ImVec2(920.0f, 660.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x + 80.0f, viewport->Pos.y + topInset + 20.0f), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Aero Polars & Flight Envelopes (XFLR5-Style)", &app.showAeroPolarsWindow)) {
        if (!app.meshLoaded || !app.voxelGridValid) {
            ImGui::TextDisabled("Load a 3D model and Voxelize it first to generate aerodynamic polars.");
            ImGui::End();
            return;
        }

        // Ensure Y-slice index is initialized
        AppState::SlicePlaneState& yPlane = app.slicePlanes[static_cast<size_t>(SliceAxis::Y)];
        int sliceMaxY = std::max(0, app.voxelGrid.ny - 1);
        if (yPlane.index < 0 || yPlane.index > sliceMaxY) {
            yPlane.index = sliceMaxY / 2;
            yPlane.indexAtomic.store(yPlane.index, std::memory_order_relaxed);
        }

        auto runFullPolarSweep = [&]() {
            if (!app.panelSolver.NumPanels() && app.voxelGridValid) {
                PanelSolver::Params pParams{app.lbmParams.inletVelocity, app.lbmParams.reverseFlow};
                app.panelSolver.Init(app.gpuDevice, false, app.voxelGrid, pParams);
            }
            float refArea = app.polarAutoRefGeometry ? 0.0f : app.polarRefAreaM2;
            float refChord = app.polarAutoRefGeometry ? 0.0f : app.polarRefChordM;
            app.polarResult = app.panelSolver.RunPolarSweep(
                app.polarAlphaMin, app.polarAlphaMax, app.polarAlphaStep,
                app.lbmInletVelocityMmPerS, app.voxelGrid.cellSize,
                refArea, refChord);
            if (app.polarAutoRefGeometry && app.polarResult.valid) {
                app.polarRefAreaM2 = app.polarResult.refAreaM2;
                app.polarRefChordM = app.polarResult.refChordM;
            }
        };

        auto refreshSectionPolar = [&]() {
            if (!app.panelSolver.NumPanels() && app.voxelGridValid) {
                PanelSolver::Params pParams{app.lbmParams.inletVelocity, app.lbmParams.reverseFlow};
                app.panelSolver.Init(app.gpuDevice, false, app.voxelGrid, pParams);
            }
            app.sectionPolarResult = app.panelSolver.RunSectionPolarSweep(
                app.voxelGrid, yPlane.index,
                app.polarAlphaMin, app.polarAlphaMax, app.polarAlphaStep,
                app.lbmInletVelocityMmPerS, app.polarSliceInclude3DDownwash);
            app.lastSectionAlphaMin = app.polarAlphaMin;
            app.lastSectionAlphaMax = app.polarAlphaMax;
            app.lastSectionAlphaStep = app.polarAlphaStep;
            app.lastSectionInletVel = app.lbmInletVelocityMmPerS;
            app.lastSectionDownwash = app.polarSliceInclude3DDownwash;
        };

        bool busy = app.taskKind.load(std::memory_order_acquire) != TaskKind::None;

        // Auto-populate initial 3D & section polars on first open
        if (!app.polarResult.valid && !busy) {
            runFullPolarSweep();
            refreshSectionPolar();
        }

        // Top controls: AoA Sweep Range & Run button
        ImGui::PushItemWidth(105.0f);
        ImGui::DragFloat("Min AoA", &app.polarAlphaMin, 0.5f, -20.0f, app.polarAlphaMax, "%.1f deg");
        ImGui::SameLine();
        ImGui::DragFloat("Max AoA", &app.polarAlphaMax, 0.5f, app.polarAlphaMin, 35.0f, "%.1f deg");
        ImGui::SameLine();
        ImGui::DragFloat("Step", &app.polarAlphaStep, 0.25f, 0.25f, 5.0f, "%.2f deg");
        ImGui::PopItemWidth();

        ImGui::SameLine(0.0f, 16.0f);
        ImGui::BeginDisabled(busy);
        if (ImGui::Button("Run Polar Sweep", ImVec2(145.0f, 0.0f))) {
            runFullPolarSweep();
            refreshSectionPolar();
        }
        ImGui::EndDisabled();

        ImGui::SameLine(0.0f, 12.0f);
        ImGui::Checkbox("Auto Ref Geometry", &app.polarAutoRefGeometry);
        if (!app.polarAutoRefGeometry) {
            ImGui::PushItemWidth(100.0f);
            ImGui::DragFloat("Ref Area S (m2)", &app.polarRefAreaM2, 0.001f, 0.0001f, 100.0f, "%.4f");
            ImGui::SameLine();
            ImGui::DragFloat("Ref Chord c (m)", &app.polarRefChordM, 0.001f, 0.0001f, 10.0f, "%.4f");
            ImGui::PopItemWidth();
        }

        // Scope Selection & Active Y-Slice Bar
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.35f, 0.82f, 1.0f, 1.0f), "Scope:");
        ImGui::SameLine();
        ImGui::RadioButton("Whole Model (3D)", &app.polarScopeMode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Active Y-Slice (2D Section)", &app.polarScopeMode, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Compare Both (Overlay)", &app.polarScopeMode, 2);

        if (app.polarScopeMode == 1 || app.polarScopeMode == 2) {
            ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.25f, 1.0f), "Active Y-Slice:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(180.0f);
            if (ImGui::SliderInt("##polar_slice_y", &yPlane.index, 0, sliceMaxY, "Y Index %d")) {
                yPlane.indexAtomic.store(yPlane.index, std::memory_order_relaxed);
                if (app.viewMode == ViewMode::FlowOverlay && app.overlayStyle == OverlayStyle::Slice && yPlane.enabled) {
                    StartResliceTask(app);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Center##polar_slice_center")) {
                yPlane.index = sliceMaxY / 2;
                yPlane.indexAtomic.store(yPlane.index, std::memory_order_relaxed);
                if (app.viewMode == ViewMode::FlowOverlay && app.overlayStyle == OverlayStyle::Slice && yPlane.enabled) {
                    StartResliceTask(app);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Show Flow Slice")) {
                yPlane.enabled = true;
                yPlane.enabledAtomic.store(true, std::memory_order_relaxed);
                app.overlayStyle = OverlayStyle::Slice;
                app.overlayStyleAtomic.store(static_cast<int>(app.overlayStyle), std::memory_order_relaxed);
                if (app.overlayValid) {
                    app.viewMode = ViewMode::FlowOverlay;
                    StartResliceTask(app);
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Enable the Y (side-on) slice plane in the 3D visualizer.\n"
                                  "Even in Solid Mesh view, the active slice contour is highlighted directly on the 3D model.");
            }
            ImGui::SameLine(0.0f, 12.0f);
            ImGui::Checkbox("Include 3D Downwash at Slice", &app.polarSliceInclude3DDownwash);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Unchecked: Pure 2D isolated airfoil polar (XFoil-style, no tip induced drag).\n"
                                  "Checked: Includes local 3D finite-wing downwash and induced drag at this span station.");
            }
        }

        // Live refresh of the active Y-slice section polar whenever the slice moves or parameters change
        if (app.sectionPolarResult.sliceY != yPlane.index ||
            app.lastSectionAlphaMin != app.polarAlphaMin ||
            app.lastSectionAlphaMax != app.polarAlphaMax ||
            app.lastSectionAlphaStep != app.polarAlphaStep ||
            app.lastSectionInletVel != app.lbmInletVelocityMmPerS ||
            app.lastSectionDownwash != app.polarSliceInclude3DDownwash) {
            refreshSectionPolar();
        }

        const bool show3D = (app.polarScopeMode == 0 || app.polarScopeMode == 2) && app.polarResult.valid;
        const bool showSec = (app.polarScopeMode == 1 || app.polarScopeMode == 2) && app.sectionPolarResult.valid;

        if (app.polarResult.valid || app.sectionPolarResult.valid) {
            ImGui::Separator();
            if (show3D) {
                const PanelSolver::PolarSweepResult& res = app.polarResult;
                ImGui::TextColored(ImVec4(0.4f, 0.95f, 0.5f, 1.0f),
                                   "[3D Model]  CL_alpha: %.3f /deg (%.2f /rad)  |  alpha_0: %.2f deg  |  (L/D)max: %.1f @ %.1f deg  |  CD0: %.4f  |  e: %.2f  |  AR: %.2f",
                                   res.clAlphaPerDeg, res.clAlphaPerRad, res.alphaZeroLiftDeg,
                                   res.maxLD, res.alphaAtMaxLD, res.minCD0, res.oswaldEfficiency, res.aspectRatio);
            }
            if (app.polarScopeMode == 1 || app.polarScopeMode == 2) {
                if (app.sectionPolarResult.valid) {
                    const PanelSolver::SectionPolarResult& sec = app.sectionPolarResult;
                    ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.28f, 1.0f),
                                       "[Y-Slice @ %+.1f mm]  Chord: %.1f mm  |  t/c: %.1f%%  |  Camber: %+.1f%%  |  Twist: %+.1f deg  |  Re: %.0f  |  (l/d)max: %.1f @ %.1f deg",
                                       sec.spanPosMm, sec.localChordMm, sec.maxThicknessRatio * 100.0f,
                                       sec.maxCamberRatio * 100.0f, sec.localTwistDeg, sec.reynoldsNumber,
                                       sec.polar.maxLD, sec.polar.alphaAtMaxLD);
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f),
                                       "[Y-Slice Index %d]  Slice is in empty air outside the model — drag the Active Y-Slice slider onto the airframe.",
                                       yPlane.index);
                }
            }
            ImGui::Spacing();

            // Prepare 3D Whole-Model vectors
            const PanelSolver::PolarSweepResult& res3D = app.polarResult;
            size_t n3D = res3D.points.size();
            std::vector<float> alphas3D(n3D), cls3D(n3D), cds3D(n3D), cdis3D(n3D), lds3D(n3D), cms3D(n3D);
            std::vector<float> clFitAlphas3D(n3D), clFitCls3D(n3D);
            for (size_t i = 0; i < n3D; ++i) {
                alphas3D[i] = res3D.points[i].alphaDeg;
                cls3D[i] = res3D.points[i].cl;
                cds3D[i] = res3D.points[i].cd;
                cdis3D[i] = res3D.points[i].cdi;
                lds3D[i] = res3D.points[i].ld;
                cms3D[i] = res3D.points[i].cm;
                clFitAlphas3D[i] = alphas3D[i];
                clFitCls3D[i] = res3D.clAlphaPerDeg * (alphas3D[i] - res3D.alphaZeroLiftDeg);
            }

            // Prepare 2D Active-Slice vectors
            const PanelSolver::PolarSweepResult& resSec = app.sectionPolarResult.polar;
            size_t nSec = resSec.points.size();
            std::vector<float> alphasSec(nSec), clsSec(nSec), cdsSec(nSec), cdisSec(nSec), ldsSec(nSec), cmsSec(nSec);
            for (size_t i = 0; i < nSec; ++i) {
                alphasSec[i] = resSec.points[i].alphaDeg;
                clsSec[i] = resSec.points[i].cl;
                cdsSec[i] = resSec.points[i].cd;
                cdisSec[i] = resSec.points[i].cdi;
                ldsSec[i] = resSec.points[i].ld;
                cmsSec[i] = resSec.points[i].cm;
            }

            if (ImGui::BeginTabBar("##AeroPolarTabs")) {
                if (ImGui::BeginTabItem("Lift & Drag Polar")) {
                    float availW = ImGui::GetContentRegionAvail().x;
                    float plotW = std::max(280.0f, (availW - 16.0f) * 0.5f);
                    float plotH = std::max(260.0f, ImGui::GetContentRegionAvail().y - 10.0f);

                    if (ImPlot::BeginPlot("Lift Curve (CL vs Alpha)", ImVec2(plotW, plotH))) {
                        ImPlot::SetupAxes("Angle of Attack Alpha (deg)", "Lift Coefficient CL");
                        ImPlot::SetupLegend(ImPlotLocation_NorthWest);
                        if (show3D && n3D > 0) {
                            ImPlot::PlotLine("3D Whole Model CL", alphas3D.data(), cls3D.data(), static_cast<int>(n3D));
                            ImPlot::PlotScatter("##3d_cl_pts", alphas3D.data(), cls3D.data(), static_cast<int>(n3D));
                            if (app.polarScopeMode == 0) {
                                ImPlot::PlotLine("Linear Slope dCL/dAlpha", clFitAlphas3D.data(), clFitCls3D.data(), static_cast<int>(n3D));
                            }
                        }
                        if (showSec && nSec > 0) {
                            ImPlot::PlotLine("Active Y-Slice Section cl", alphasSec.data(), clsSec.data(), static_cast<int>(nSec));
                            ImPlot::PlotScatter("##sec_cl_pts", alphasSec.data(), clsSec.data(), static_cast<int>(nSec));
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::SameLine();
                    if (ImPlot::BeginPlot("Drag Polar (CD vs CL)", ImVec2(plotW, plotH))) {
                        ImPlot::SetupAxes("Total Drag CD", "Lift Coefficient CL");
                        ImPlot::SetupLegend(ImPlotLocation_SouthEast);
                        if (show3D && n3D > 0) {
                            ImPlot::PlotLine("3D Whole Model Polar", cds3D.data(), cls3D.data(), static_cast<int>(n3D));
                            ImPlot::PlotScatter("##3d_cd_pts", cds3D.data(), cls3D.data(), static_cast<int>(n3D));
                            if (app.polarScopeMode == 0) {
                                ImPlot::PlotLine("3D Induced Drag CDi", cdis3D.data(), cls3D.data(), static_cast<int>(n3D));
                            }
                        }
                        if (showSec && nSec > 0) {
                            ImPlot::PlotLine("Active Y-Slice Section Polar", cdsSec.data(), clsSec.data(), static_cast<int>(nSec));
                            ImPlot::PlotScatter("##sec_cd_pts", cdsSec.data(), clsSec.data(), static_cast<int>(nSec));
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Glide Ratio & Moment")) {
                    float availW = ImGui::GetContentRegionAvail().x;
                    float plotW = std::max(280.0f, (availW - 16.0f) * 0.5f);
                    float plotH = std::max(260.0f, ImGui::GetContentRegionAvail().y - 10.0f);

                    if (ImPlot::BeginPlot("Glide Ratio (L/D vs Alpha)", ImVec2(plotW, plotH))) {
                        ImPlot::SetupAxes("Angle of Attack Alpha (deg)", "Lift-to-Drag Ratio L/D");
                        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
                        if (show3D && n3D > 0) {
                            ImPlot::PlotLine("3D Whole Model L/D", alphas3D.data(), lds3D.data(), static_cast<int>(n3D));
                            ImPlot::PlotScatter("##3d_ld_pts", alphas3D.data(), lds3D.data(), static_cast<int>(n3D));
                        }
                        if (showSec && nSec > 0) {
                            ImPlot::PlotLine("Active Y-Slice Section l/d", alphasSec.data(), ldsSec.data(), static_cast<int>(nSec));
                            ImPlot::PlotScatter("##sec_ld_pts", alphasSec.data(), ldsSec.data(), static_cast<int>(nSec));
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::SameLine();
                    if (ImPlot::BeginPlot("Pitching Moment (CM vs Alpha)", ImVec2(plotW, plotH))) {
                        ImPlot::SetupAxes("Angle of Attack Alpha (deg)", "Pitching Moment CM (c/4)");
                        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
                        if (show3D && n3D > 0) {
                            ImPlot::PlotLine("3D Whole Model CM", alphas3D.data(), cms3D.data(), static_cast<int>(n3D));
                            ImPlot::PlotScatter("##3d_cm_pts", alphas3D.data(), cms3D.data(), static_cast<int>(n3D));
                        }
                        if (showSec && nSec > 0) {
                            ImPlot::PlotLine("Active Y-Slice Section cm", alphasSec.data(), cmsSec.data(), static_cast<int>(nSec));
                            ImPlot::PlotScatter("##sec_cm_pts", alphasSec.data(), cmsSec.data(), static_cast<int>(nSec));
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Slice Airfoil & Span Load")) {
                    float availW = ImGui::GetContentRegionAvail().x;
                    float plotW = std::max(280.0f, (availW - 16.0f) * 0.5f);
                    float plotH = std::max(260.0f, ImGui::GetContentRegionAvail().y - 10.0f);

                    // Left Plot: 1:1 Aspect-Locked 2D Airfoil Cross-Section at Active Y-Slice
                    if (ImPlot::BeginPlot("Active Y-Slice Airfoil Profile (mm)", ImVec2(plotW, plotH), ImPlotFlags_Equal)) {
                        ImPlot::SetupAxes("Chord X from Leading Edge (mm)", "Thickness / Height Z (mm)");
                        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
                        if (app.sectionPolarResult.valid && !app.sectionPolarResult.profile.empty()) {
                            const auto& prof = app.sectionPolarResult.profile;
                            size_t mCount = prof.size();
                            std::vector<float> px(mCount), zUp(mCount), zLo(mCount), zCam(mCount), zChd(mCount);
                            for (size_t i = 0; i < mCount; ++i) {
                                px[i] = prof[i].xMm;
                                zUp[i] = prof[i].zUpperMm;
                                zLo[i] = prof[i].zLowerMm;
                                zCam[i] = prof[i].zCamberMm;
                                zChd[i] = prof[i].zChordMm;
                            }
                            ImPlot::PlotLine("Upper Surface", px.data(), zUp.data(), static_cast<int>(mCount));
                            ImPlot::PlotLine("Lower Surface", px.data(), zLo.data(), static_cast<int>(mCount));
                            ImPlot::PlotLine("Mean Camber Line", px.data(), zCam.data(), static_cast<int>(mCount));
                            ImPlot::PlotLine("Chord Line", px.data(), zChd.data(), static_cast<int>(mCount));
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::SameLine();

                    // Right Plot: Spanwise Lift Distribution with Interactive Active Slice Drag Line
                    if (ImPlot::BeginPlot("Spanwise Lift Distribution (cl vs Span Y)", ImVec2(plotW, plotH))) {
                        ImPlot::SetupAxes("Spanwise Station Y (mm)", "Local Lift Coefficient cl(y)");
                        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
                        if (!res3D.spanStations.empty()) {
                            size_t sCount = res3D.spanStations.size();
                            std::vector<float> sy(sCount), scl(sCount), sellip(sCount);
                            float yGridSum = 0.0f, yMmSum = 0.0f;
                            for (size_t i = 0; i < sCount; ++i) {
                                sy[i] = res3D.spanStations[i].yMm;
                                scl[i] = res3D.spanStations[i].clLocal;
                                sellip[i] = res3D.spanStations[i].clElliptic;
                                yGridSum += res3D.spanStations[i].yGrid;
                                yMmSum += res3D.spanStations[i].yMm;
                            }
                            ImPlot::PlotLine("3D Spanwise cl(y)", sy.data(), scl.data(), static_cast<int>(sCount));
                            ImPlot::PlotLine("Ideal Elliptical Load", sy.data(), sellip.data(), static_cast<int>(sCount));

                            // Interactive vertical line showing and controlling the Active Y-Slice station
                            float yCenterGrid = (yGridSum / static_cast<float>(sCount)) -
                                                (yMmSum / static_cast<float>(sCount)) / std::max(1e-4f, app.voxelGrid.cellSize);
                            double sliceYMm = (static_cast<double>(yPlane.index) - static_cast<double>(yCenterGrid)) *
                                              static_cast<double>(app.voxelGrid.cellSize);
                            if (ImPlot::DragLineX(0, &sliceYMm, ImVec4(1.0f, 0.85f, 0.2f, 1.0f), 2.0f)) {
                                int newIdx = static_cast<int>(std::round(sliceYMm / std::max(1e-4f, app.voxelGrid.cellSize) + yCenterGrid));
                                newIdx = std::clamp(newIdx, 0, sliceMaxY);
                                if (newIdx != yPlane.index) {
                                    yPlane.index = newIdx;
                                    yPlane.indexAtomic.store(yPlane.index, std::memory_order_relaxed);
                                    if (app.viewMode == ViewMode::FlowOverlay && app.overlayStyle == OverlayStyle::Slice && yPlane.enabled) {
                                        StartResliceTask(app);
                                    }
                                }
                            }
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Data Table & Export")) {
                    ImGui::TextUnformatted("Table Source:");
                    ImGui::SameLine();
                    ImGui::RadioButton("3D Whole Model##tbl", &app.polarTableSource, 0);
                    ImGui::SameLine();
                    ImGui::RadioButton("Active Y-Slice Section##tbl", &app.polarTableSource, 1);
                    ImGui::SameLine(0.0f, 20.0f);

                    const PanelSolver::PolarSweepResult& activeRes =
                        (app.polarTableSource == 1 && app.sectionPolarResult.valid) ? app.sectionPolarResult.polar : app.polarResult;

                    if (ImGui::Button("Export Polar to CSV (polar.csv)")) {
                        FILE* f = std::fopen("polar.csv", "w");
                        if (f) {
                            if (app.polarTableSource == 1 && app.sectionPolarResult.valid) {
                                const auto& sec = app.sectionPolarResult;
                                std::fprintf(f, "# FlowCraft 2D Active Slice Section Polar (Y = %.2f mm)\n", sec.spanPosMm);
                                std::fprintf(f, "# Local Chord = %.2f mm, t/c = %.2f%%, Camber = %.2f%%, Twist = %.2f deg, Re = %.0f\n",
                                             sec.localChordMm, sec.maxThicknessRatio * 100.0f, sec.maxCamberRatio * 100.0f,
                                             sec.localTwistDeg, sec.reynoldsNumber);
                            } else {
                                std::fprintf(f, "# FlowCraft 3D Whole-Model Aero Polar Sweep (XFLR5-Style)\n");
                                std::fprintf(f, "# S = %.6f m2, Mean Chord = %.6f m, Span = %.6f m, AR = %.3f\n",
                                             activeRes.refAreaM2, activeRes.refChordM, activeRes.refSpanM, activeRes.aspectRatio);
                            }
                            std::fprintf(f, "# CL_alpha = %.4f /deg, alpha_0 = %.3f deg, max(L/D) = %.2f @ %.1f deg\n",
                                         activeRes.clAlphaPerDeg, activeRes.alphaZeroLiftDeg, activeRes.maxLD, activeRes.alphaAtMaxLD);
                            std::fprintf(f, "Alpha_deg,CL,CD,CDi,CD0,L_over_D,CM_c4,Lift_N,Drag_N,CoP_X_mm\n");
                            for (const auto& pt : activeRes.points) {
                                std::fprintf(f, "%.2f,%.5f,%.5f,%.5f,%.5f,%.3f,%.5f,%.4f,%.4f,%.2f\n",
                                             pt.alphaDeg, pt.cl, pt.cd, pt.cdi, pt.cd0, pt.ld, pt.cm,
                                             pt.liftN, pt.dragN, pt.cpX_mm);
                            }
                            std::fclose(f);
                            app.polarExportStatus = "Exported polar.csv successfully!";
                        } else {
                            app.polarExportStatus = "Failed to write polar.csv";
                        }
                    }
                    if (!app.polarExportStatus.empty()) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f), "%s", app.polarExportStatus.c_str());
                    }

                    if (ImGui::BeginTable("##PolarDataTable", 10,
                                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                                          ImVec2(0.0f, 320.0f))) {
                        ImGui::TableSetupColumn("AoA (deg)");
                        ImGui::TableSetupColumn("CL");
                        ImGui::TableSetupColumn("CD");
                        ImGui::TableSetupColumn("CDi");
                        ImGui::TableSetupColumn("CD0");
                        ImGui::TableSetupColumn("L/D");
                        ImGui::TableSetupColumn("CM");
                        ImGui::TableSetupColumn("Lift (N)");
                        ImGui::TableSetupColumn("Drag (N)");
                        ImGui::TableSetupColumn("CoP X (mm)");
                        ImGui::TableHeadersRow();

                        for (const auto& pt : activeRes.points) {
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0); ImGui::Text("%.1f deg", pt.alphaDeg);
                            ImGui::TableSetColumnIndex(1); ImGui::Text("%.4f", pt.cl);
                            ImGui::TableSetColumnIndex(2); ImGui::Text("%.4f", pt.cd);
                            ImGui::TableSetColumnIndex(3); ImGui::Text("%.4f", pt.cdi);
                            ImGui::TableSetColumnIndex(4); ImGui::Text("%.4f", pt.cd0);
                            ImGui::TableSetColumnIndex(5); ImGui::Text("%.2f", pt.ld);
                            ImGui::TableSetColumnIndex(6); ImGui::Text("%.4f", pt.cm);
                            ImGui::TableSetColumnIndex(7); ImGui::Text("%.3f", pt.liftN);
                            ImGui::TableSetColumnIndex(8); ImGui::Text("%.3f", pt.dragN);
                            ImGui::TableSetColumnIndex(9); ImGui::Text("%.1f", pt.cpX_mm);
                        }
                        ImGui::EndTable();
                    }
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Spacing();
            ImGui::TextDisabled("Click \"Run Polar Sweep\" above to calculate lift, drag, efficiency curves and stability derivatives.");
        }
    }
    ImGui::End();
}

// Vector icons drawn procedurally via ImDrawList for the Autodesk Fusion–style
// top ribbon toolbar — crisp at any DPI with zero external image assets.
enum class RibbonIcon {
    OpenSTL,
    LoadSession,
    SaveSession,
    ResetSession,
    ViewModel,
    ViewVoxels,
    ViewOverlay,
    CamFront,
    CamTop,
    CamRight,
    CamLeft,
    CamIso,
    Voxelize,
    InitSolver,
    Play,
    Pause,
    Step,
    RefreshStats,
    BuildOverlay,
    StyleSlice,
    StyleCloud,
    StyleParticles,
    StyleStress,
    StyleIso,
    StyleRaymarch,
    AeroPolars
};

void DrawRibbonVectorIcon(ImDrawList* dl, ImVec2 center, float radius, RibbonIcon icon, ImU32 primaryCol,
                          ImU32 accentCol) {
    float r = radius;
    switch (icon) {
        case RibbonIcon::OpenSTL: {
            // Folder outline + front flap
            ImVec2 p0{center.x - r, center.y - r * 0.65f};
            ImVec2 p1{center.x - r * 0.2f, center.y - r * 0.65f};
            ImVec2 p2{center.x + r * 0.05f, center.y - r * 0.35f};
            ImVec2 p3{center.x + r, center.y - r * 0.35f};
            ImVec2 p4{center.x + r, center.y + r * 0.75f};
            ImVec2 p5{center.x - r, center.y + r * 0.75f};
            dl->AddLine(p0, p1, primaryCol, 1.8f);
            dl->AddLine(p1, p2, primaryCol, 1.8f);
            dl->AddLine(p2, p3, primaryCol, 1.8f);
            dl->AddLine(p3, p4, primaryCol, 1.8f);
            dl->AddLine(p4, p5, primaryCol, 1.8f);
            dl->AddLine(p5, p0, primaryCol, 1.8f);
            dl->AddLine(ImVec2(center.x - r * 0.9f, center.y - r * 0.05f),
                        ImVec2(center.x + r * 0.9f, center.y - r * 0.05f), accentCol, 1.5f);
            break;
        }
        case RibbonIcon::LoadSession: {
            // Document outline + inward arrow
            ImVec2 tl{center.x - r * 0.75f, center.y - r * 0.85f};
            ImVec2 br{center.x + r * 0.75f, center.y + r * 0.85f};
            dl->AddRect(tl, br, primaryCol, 2.0f, 0, 1.7f);
            ImVec2 a0{center.x, center.y + r * 0.45f};
            ImVec2 a1{center.x, center.y - r * 0.35f};
            dl->AddLine(a0, a1, accentCol, 2.0f);
            dl->AddLine(a1, ImVec2(center.x - r * 0.35f, center.y), accentCol, 2.0f);
            dl->AddLine(a1, ImVec2(center.x + r * 0.35f, center.y), accentCol, 2.0f);
            break;
        }
        case RibbonIcon::SaveSession: {
            // Floppy / save badge
            ImVec2 tl{center.x - r * 0.8f, center.y - r * 0.8f};
            ImVec2 br{center.x + r * 0.8f, center.y + r * 0.8f};
            dl->AddRect(tl, br, primaryCol, 2.0f, 0, 1.7f);
            dl->AddRectFilled(ImVec2(center.x - r * 0.45f, tl.y + 2.0f),
                              ImVec2(center.x + r * 0.45f, center.y - r * 0.15f), accentCol, 1.0f);
            dl->AddRect(ImVec2(center.x - r * 0.5f, center.y + r * 0.15f),
                        ImVec2(center.x + r * 0.5f, br.y - 1.0f), primaryCol, 1.0f, 0, 1.4f);
            break;
        }
        case RibbonIcon::ResetSession: {
            // Circular counter-clockwise reset arc + arrowhead
            dl->PathArcTo(center, r * 0.75f, -0.4f, 4.4f, 18);
            dl->PathStroke(primaryCol, 0, 1.9f);
            ImVec2 tip{center.x + std::cos(-0.4f) * r * 0.75f, center.y + std::sin(-0.4f) * r * 0.75f};
            dl->AddTriangleFilled(tip, ImVec2(tip.x - 6.0f, tip.y - 3.0f), ImVec2(tip.x - 1.5f, tip.y - 7.0f),
                                  accentCol);
            break;
        }
        case RibbonIcon::ViewModel: {
            // Isometric 3D CAD prism / crystal
            ImVec2 top{center.x, center.y - r * 0.9f};
            ImVec2 midL{center.x - r * 0.85f, center.y - r * 0.25f};
            ImVec2 midR{center.x + r * 0.85f, center.y - r * 0.25f};
            ImVec2 midC{center.x, center.y + r * 0.15f};
            ImVec2 bot{center.x, center.y + r * 0.9f};
            ImVec2 botL{center.x - r * 0.85f, center.y + r * 0.45f};
            ImVec2 botR{center.x + r * 0.85f, center.y + r * 0.45f};
            dl->AddQuadFilled(top, midR, midC, midL, IM_COL32(56, 171, 223, 85));
            dl->AddLine(top, midL, primaryCol, 1.7f);
            dl->AddLine(top, midR, primaryCol, 1.7f);
            dl->AddLine(midL, midC, accentCol, 1.6f);
            dl->AddLine(midR, midC, accentCol, 1.6f);
            dl->AddLine(midL, botL, primaryCol, 1.7f);
            dl->AddLine(midR, botR, primaryCol, 1.7f);
            dl->AddLine(midC, bot, accentCol, 1.6f);
            dl->AddLine(botL, bot, primaryCol, 1.7f);
            dl->AddLine(botR, bot, primaryCol, 1.7f);
            break;
        }
        case RibbonIcon::ViewVoxels:
        case RibbonIcon::Voxelize: {
            // 3x3 voxel lattice grid with highlighted active cells
            float half = r * 0.82f;
            float step = (half * 2.0f) / 3.0f;
            ImVec2 start{center.x - half, center.y - half};
            dl->AddRectFilled(ImVec2(start.x + step, start.y), ImVec2(start.x + step * 2.0f, start.y + step),
                              accentCol);
            dl->AddRectFilled(ImVec2(start.x, start.y + step), ImVec2(start.x + step * 2.0f, start.y + step * 2.0f),
                              accentCol);
            for (int i = 0; i <= 3; ++i) {
                float off = static_cast<float>(i) * step;
                dl->AddLine(ImVec2(start.x + off, start.y), ImVec2(start.x + off, start.y + half * 2.0f), primaryCol,
                            1.4f);
                dl->AddLine(ImVec2(start.x, start.y + off), ImVec2(start.x + half * 2.0f, start.y + off), primaryCol,
                            1.4f);
            }
            break;
        }
        case RibbonIcon::ViewOverlay:
        case RibbonIcon::BuildOverlay: {
            // Three aerodynamic streamlines
            for (int row = -1; row <= 1; ++row) {
                float yBase = center.y + static_cast<float>(row) * r * 0.52f;
                ImU32 col = (row == 0) ? accentCol : primaryCol;
                ImVec2 p0{center.x - r * 0.9f, yBase};
                ImVec2 p1{center.x - r * 0.3f, yBase - r * 0.25f};
                ImVec2 p2{center.x + r * 0.3f, yBase + r * 0.25f};
                ImVec2 p3{center.x + r * 0.9f, yBase};
                dl->AddBezierCubic(p0, p1, p2, p3, col, 1.9f);
            }
            break;
        }
        case RibbonIcon::CamFront:
        case RibbonIcon::CamTop:
        case RibbonIcon::CamRight:
        case RibbonIcon::CamLeft: {
            // ViewCube with the active face highlighted
            float s = r * 0.78f;
            ImVec2 tl{center.x - s, center.y - s};
            ImVec2 br{center.x + s, center.y + s};
            dl->AddRect(tl, br, primaryCol, 2.0f, 0, 1.6f);
            if (icon == RibbonIcon::CamFront) {
                dl->AddRectFilled(ImVec2(center.x - s * 0.55f, center.y - s * 0.55f),
                                  ImVec2(center.x + s * 0.55f, center.y + s * 0.55f), accentCol, 1.5f);
            } else if (icon == RibbonIcon::CamTop) {
                dl->AddRectFilled(ImVec2(tl.x + 2.0f, tl.y + 2.0f), ImVec2(br.x - 2.0f, center.y - s * 0.15f),
                                  accentCol, 1.0f);
            } else if (icon == RibbonIcon::CamRight) {
                dl->AddRectFilled(ImVec2(center.x + s * 0.15f, tl.y + 2.0f), ImVec2(br.x - 2.0f, br.y - 2.0f),
                                  accentCol, 1.0f);
            } else {
                dl->AddRectFilled(ImVec2(tl.x + 2.0f, tl.y + 2.0f), ImVec2(center.x - s * 0.15f, br.y - 2.0f),
                                  accentCol, 1.0f);
            }
            break;
        }
        case RibbonIcon::CamIso: {
            // Isometric 3D box outline with top face shaded
            float s = r * 0.72f;
            float dx = s * 0.866f;
            float dy = s * 0.5f;
            ImVec2 c = center;
            ImVec2 pTop{c.x, c.y - s};
            ImVec2 pTopR{c.x + dx, c.y - dy * 0.5f};
            ImVec2 pBotR{c.x + dx, c.y + dy * 0.5f + s * 0.35f};
            ImVec2 pBot{c.x, c.y + s * 0.85f};
            ImVec2 pBotL{c.x - dx, c.y + dy * 0.5f + s * 0.35f};
            ImVec2 pTopL{c.x - dx, c.y - dy * 0.5f};
            ImVec2 pMid{c.x, c.y};

            dl->AddQuad(pTop, pTopR, pMid, pTopL, primaryCol, 1.5f);
            dl->AddQuadFilled(pTop, pTopR, pMid, pTopL, accentCol);
            dl->AddQuad(pMid, pTopR, pBotR, pBot, primaryCol, 1.5f);
            dl->AddQuad(pMid, pBot, pBotL, pTopL, primaryCol, 1.5f);
            break;
        }
        case RibbonIcon::InitSolver: {
            // D3Q19-style star lattice node with center hub
            for (int i = 0; i < 4; ++i) {
                float a = static_cast<float>(i) * (std::numbers::pi_v<float> * 0.25f);
                ImVec2 d{std::cos(a) * r * 0.85f, std::sin(a) * r * 0.85f};
                dl->AddLine(ImVec2(center.x - d.x, center.y - d.y), ImVec2(center.x + d.x, center.y + d.y),
                            primaryCol, 1.6f);
            }
            dl->AddCircleFilled(center, r * 0.34f, accentCol);
            break;
        }
        case RibbonIcon::Play: {
            ImVec2 p0{center.x - r * 0.55f, center.y - r * 0.78f};
            ImVec2 p1{center.x - r * 0.55f, center.y + r * 0.78f};
            ImVec2 p2{center.x + r * 0.78f, center.y};
            dl->AddTriangleFilled(p0, p1, p2, IM_COL32(72, 215, 120, 255));
            break;
        }
        case RibbonIcon::Pause: {
            dl->AddRectFilled(ImVec2(center.x - r * 0.6f, center.y - r * 0.72f),
                              ImVec2(center.x - r * 0.15f, center.y + r * 0.72f), IM_COL32(245, 175, 55, 255), 1.5f);
            dl->AddRectFilled(ImVec2(center.x + r * 0.15f, center.y - r * 0.72f),
                              ImVec2(center.x + r * 0.6f, center.y + r * 0.72f), IM_COL32(245, 175, 55, 255), 1.5f);
            break;
        }
        case RibbonIcon::Step: {
            ImVec2 p0{center.x - r * 0.7f, center.y - r * 0.7f};
            ImVec2 p1{center.x - r * 0.7f, center.y + r * 0.7f};
            ImVec2 p2{center.x + r * 0.25f, center.y};
            dl->AddTriangleFilled(p0, p1, p2, accentCol);
            dl->AddRectFilled(ImVec2(center.x + r * 0.4f, center.y - r * 0.7f),
                              ImVec2(center.x + r * 0.72f, center.y + r * 0.7f), primaryCol, 1.0f);
            break;
        }
        case RibbonIcon::RefreshStats: {
            // Telemetry pulse waveform
            ImVec2 p0{center.x - r * 0.9f, center.y};
            ImVec2 p1{center.x - r * 0.4f, center.y};
            ImVec2 p2{center.x - r * 0.15f, center.y - r * 0.75f};
            ImVec2 p3{center.x + r * 0.2f, center.y + r * 0.75f};
            ImVec2 p4{center.x + r * 0.45f, center.y};
            ImVec2 p5{center.x + r * 0.9f, center.y};
            dl->AddLine(p0, p1, primaryCol, 1.8f);
            dl->AddLine(p1, p2, accentCol, 2.0f);
            dl->AddLine(p2, p3, accentCol, 2.0f);
            dl->AddLine(p3, p4, accentCol, 2.0f);
            dl->AddLine(p4, p5, primaryCol, 1.8f);
            break;
        }
        case RibbonIcon::StyleSlice: {
            // Slanted cross-section plane
            ImVec2 p0{center.x - r * 0.8f, center.y + r * 0.55f};
            ImVec2 p1{center.x - r * 0.25f, center.y - r * 0.65f};
            ImVec2 p2{center.x + r * 0.8f, center.y - r * 0.55f};
            ImVec2 p3{center.x + r * 0.25f, center.y + r * 0.65f};
            dl->AddQuadFilled(p0, p1, p2, p3, IM_COL32(56, 171, 223, 95));
            dl->AddQuad(p0, p1, p2, p3, accentCol, 1.8f);
            break;
        }
        case RibbonIcon::StyleCloud: {
            // Volumetric cloud cluster
            dl->AddCircleFilled(ImVec2(center.x - r * 0.35f, center.y + r * 0.15f), r * 0.42f,
                                IM_COL32(80, 175, 235, 160));
            dl->AddCircleFilled(ImVec2(center.x + r * 0.32f, center.y + r * 0.18f), r * 0.38f,
                                IM_COL32(110, 215, 165, 160));
            dl->AddCircleFilled(ImVec2(center.x, center.y - r * 0.22f), r * 0.46f, accentCol);
            break;
        }
        case RibbonIcon::StyleParticles: {
            // Smoke streaklines with glowing particle heads
            for (int i = -1; i <= 1; ++i) {
                float y = center.y + static_cast<float>(i) * r * 0.5f;
                dl->AddLine(ImVec2(center.x - r * 0.85f, y), ImVec2(center.x + r * 0.45f, y), primaryCol, 1.6f);
                dl->AddCircleFilled(ImVec2(center.x + r * 0.65f, y), 2.8f, accentCol);
            }
            break;
        }
        case RibbonIcon::StyleStress: {
            // Swept delta wing / airframe skin with thermal stress gradient
            ImVec2 nose{center.x, center.y - r * 0.85f};
            ImVec2 tipL{center.x - r * 0.9f, center.y + r * 0.65f};
            ImVec2 tail{center.x, center.y + r * 0.3f};
            ImVec2 tipR{center.x + r * 0.9f, center.y + r * 0.65f};
            dl->AddTriangleFilled(nose, tipL, tail, IM_COL32(245, 185, 45, 220));
            dl->AddTriangleFilled(nose, tail, tipR, IM_COL32(60, 195, 135, 220));
            dl->AddLine(nose, tipL, primaryCol, 1.7f);
            dl->AddLine(tipL, tail, primaryCol, 1.7f);
            dl->AddLine(tail, tipR, primaryCol, 1.7f);
            dl->AddLine(tipR, nose, primaryCol, 1.7f);
            break;
        }
        case RibbonIcon::StyleIso: {
            // Contoured iso-surface 3D droplet / bulb shape
            dl->AddEllipseFilled(ImVec2(center.x, center.y), ImVec2(r * 0.75f, r * 0.55f), IM_COL32(70, 190, 240, 180));
            dl->AddEllipse(ImVec2(center.x, center.y), ImVec2(r * 0.75f, r * 0.55f), accentCol, 0.0f, 0, 1.8f);
            dl->AddLine(ImVec2(center.x - r * 0.65f, center.y), ImVec2(center.x + r * 0.65f, center.y),
                        IM_COL32(255, 255, 255, 130), 1.2f);
            break;
        }
        case RibbonIcon::StyleRaymarch: {
            // Stack of view-aligned volumetric ray slices with depth gradient
            for (int i = -2; i <= 2; ++i) {
                float y = center.y + static_cast<float>(i) * r * 0.35f;
                float halfW = r * (0.85f - std::fabs(static_cast<float>(i)) * 0.12f);
                ImU32 col = (i == 0) ? accentCol : IM_COL32(80, 185, 235, 140 + i * 25);
                dl->AddLine(ImVec2(center.x - halfW, y), ImVec2(center.x + halfW, y), col, 2.0f);
            }
            break;
        }
        case RibbonIcon::AeroPolars: {
            // Coordinate axes + lifting curve
            ImVec2 o{center.x - r * 0.7f, center.y + r * 0.7f};
            ImVec2 x{center.x + r * 0.8f, center.y + r * 0.7f};
            ImVec2 y{center.x - r * 0.7f, center.y - r * 0.8f};
            dl->AddLine(o, x, primaryCol, 1.6f);
            dl->AddLine(o, y, primaryCol, 1.6f);
            // Curved aero lift line (C_L vs alpha curve)
            ImVec2 p0{center.x - r * 0.5f, center.y + r * 0.55f};
            ImVec2 p1{center.x - r * 0.1f, center.y + r * 0.1f};
            ImVec2 p2{center.x + r * 0.35f, center.y - r * 0.45f};
            ImVec2 p3{center.x + r * 0.75f, center.y - r * 0.40f};
            dl->AddBezierCubic(p0, p1, p2, p3, accentCol, 2.0f);
            break;
        }
    }
}

// Renders a single Autodesk Fusion–style vertical icon+caption button.
bool RibbonIconButton(const char* id, const char* label, RibbonIcon icon, bool active = false, bool enabled = true,
                      const char* tooltip = nullptr, float width = 58.0f, float height = 52.0f) {
    ImGui::BeginDisabled(!enabled);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton(id, ImVec2(width, height));
    bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
    bool held = ImGui::IsItemActive();

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 br{pos.x + width, pos.y + height};
    const bool amiga = ImGui::GetStyle().AmigaOS13Mode;

    if (amiga) {
        const ImU32 amigaBlue = IM_COL32(0, 85, 170, 255);
        const ImU32 amigaDarkBlue = IM_COL32(0, 2, 34, 255);
        const ImU32 amigaWhite = IM_COL32(255, 255, 255, 255);
        const ImU32 amigaOrange = IM_COL32(255, 136, 0, 255);

        if (active || (held && enabled)) {
            dl->AddRectFilled(pos, br, amigaDarkBlue, 0.0f);
            dl->AddRect(pos, br, amigaOrange, 0.0f, 0, 2.0f);
        } else if (hovered && enabled) {
            dl->AddRectFilled(pos, br, amigaDarkBlue, 0.0f);
            dl->AddRect(pos, br, amigaWhite, 0.0f, 0, 1.5f);
        } else {
            dl->AddRectFilled(pos, br, amigaBlue, 0.0f);
            dl->AddRect(pos, br, amigaWhite, 0.0f, 0, 1.0f);
        }

        ImU32 primaryCol = !enabled ? IM_COL32(150, 170, 200, 140) : amigaWhite;
        ImU32 accentCol = !enabled ? IM_COL32(120, 140, 170, 140) : amigaOrange;
        ImVec2 iconCenter{pos.x + width * 0.5f, pos.y + 18.0f};
        DrawRibbonVectorIcon(dl, iconCenter, 11.0f, icon, primaryCol, accentCol);

        ImVec2 textSize = ImGui::CalcTextSize(label);
        ImU32 textCol = !enabled ? IM_COL32(150, 170, 200, 150) : (active ? amigaOrange : amigaWhite);
        dl->AddText(ImVec2(pos.x + (width - textSize.x) * 0.5f, pos.y + height - textSize.y - 4.0f), textCol, label);
    } else {
        if (active) {
            dl->AddRectFilled(pos, br, IM_COL32(48, 92, 128, 225), 5.0f);
            dl->AddRect(pos, br, IM_COL32(72, 182, 242, 255), 5.0f, 0, 1.5f);
        } else if (held && enabled) {
            dl->AddRectFilled(pos, br, IM_COL32(62, 68, 78, 230), 5.0f);
        } else if (hovered && enabled) {
            dl->AddRectFilled(pos, br, IM_COL32(50, 55, 64, 210), 5.0f);
            dl->AddRect(pos, br, IM_COL32(95, 102, 116, 180), 5.0f);
        }

        ImU32 primaryCol = !enabled ? IM_COL32(105, 110, 118, 150)
                                    : (active ? IM_COL32(245, 250, 255, 255) : IM_COL32(215, 220, 228, 255));
        ImU32 accentCol = !enabled ? IM_COL32(95, 115, 130, 140)
                                   : (active ? IM_COL32(105, 215, 255, 255) : IM_COL32(56, 171, 223, 255));

        ImVec2 iconCenter{pos.x + width * 0.5f, pos.y + 18.0f};
        DrawRibbonVectorIcon(dl, iconCenter, 11.0f, icon, primaryCol, accentCol);

        ImVec2 textSize = ImGui::CalcTextSize(label);
        ImU32 textCol = !enabled ? IM_COL32(115, 120, 128, 160)
                                 : (active ? IM_COL32(255, 255, 255, 255) : IM_COL32(210, 214, 222, 255));
        dl->AddText(ImVec2(pos.x + (width - textSize.x) * 0.5f, pos.y + height - textSize.y - 4.0f), textCol, label);
    }

    ImGui::EndDisabled();
    if (hovered && tooltip && tooltip[0] != '\0') {
        ImGui::SetTooltip("%s", tooltip);
    }
    return clicked && enabled;
}

void DrawRibbonDivider() {
    ImGui::SameLine(0.0f, 6.0f);
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddLine(ImVec2(p.x, p.y + 6.0f), ImVec2(p.x, p.y + 46.0f), IM_COL32(68, 73, 82, 200), 1.0f);
    ImGui::Dummy(ImVec2(7.0f, 52.0f));
    ImGui::SameLine(0.0f, 2.0f);
}

// Helper shared by the Ribbon and the Simulate panel to refresh solver stats.
void TriggerRefreshStats(AppState& app) {
    std::lock_guard<std::mutex> lock(app.lbmMutex);
    std::vector<float> field;
    if (app.lbmBackend == SolverBackend::GPU) {
        app.lbmLastStats = app.lbmSolver.ReadbackStats();
        field = app.lbmSolver.ReadbackField();
    } else if (app.lbmBackend == SolverBackend::CPU) {
        LbmReference::Stats refStats = app.lbmReference.ComputeStats();
        app.lbmLastStats.avgDensity = refStats.avgDensity;
        app.lbmLastStats.minDensity = refStats.minDensity;
        app.lbmLastStats.maxDensity = refStats.maxDensity;
        app.lbmLastStats.avgSpeed = refStats.avgSpeed;
        app.lbmLastStats.maxSpeed = refStats.maxSpeed;
        app.lbmLastStats.hasNaN = refStats.hasNaN;
        app.lbmLastStats.valid = true;
        field = app.lbmReference.ReadbackField();
    } else if (IsPanelBackend(app.lbmBackend)) {
        field = app.panelSolver.ReadbackField();
        PanelSolver::Stats pStats = app.panelSolver.ComputeStatsFromField(field);
        app.lbmLastStats.avgDensity = pStats.avgDensity;
        app.lbmLastStats.minDensity = pStats.minDensity;
        app.lbmLastStats.maxDensity = pStats.maxDensity;
        app.lbmLastStats.avgSpeed = pStats.avgSpeed;
        app.lbmLastStats.maxSpeed = pStats.maxSpeed;
        app.lbmLastStats.hasNaN = pStats.hasNaN;
        app.lbmLastStats.valid = pStats.valid;
    } else {
        PotentialFlowSolver::Stats naiveStats = app.potentialFlow.ComputeStats();
        app.lbmLastStats.avgDensity = naiveStats.avgDensity;
        app.lbmLastStats.minDensity = naiveStats.minDensity;
        app.lbmLastStats.maxDensity = naiveStats.maxDensity;
        app.lbmLastStats.avgSpeed = naiveStats.avgSpeed;
        app.lbmLastStats.maxSpeed = naiveStats.maxSpeed;
        app.lbmLastStats.hasNaN = naiveStats.hasNaN;
        app.lbmLastStats.valid = true;
        field = app.potentialFlow.ReadbackField();
    }
    if (!field.empty() && app.voxelGridValid) {
        FieldOverlay::AerodynamicForces aero = FieldOverlay::ComputeAerodynamicForces(app.voxelGrid, field);
        app.lbmLastStats.forceX = aero.forceX;
        app.lbmLastStats.forceY = aero.forceY;
        app.lbmLastStats.forceZ = aero.forceZ;
        app.lbmLastStats.maxStress = aero.maxStress;
        app.lbmLastStats.centerOfPressure = aero.centerOfPressure;
        app.lbmLastStats.liftBaseZ = aero.liftArrowBase.z;
    }
    SyncLbmStatsAtomics(app);
    if (app.lbmLastStats.hasNaN) {
        FlagFlowDivergence(app);
    }
}

// Autodesk Fusion–style top bar: workspace stage tabs along the top strip
// + grouped vector-icon toolbar across the second strip.
void DrawFusionRibbon(AppState& app, bool busy) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, kFusionRibbonHeight), ImGuiCond_Always);

    const bool amiga = (app.uiThemeIndex == 2);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 5.0f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg,
                          amiga ? ImVec4(0.00f, 0.333f, 0.667f, 1.00f) : ImVec4(0.11f, 0.12f, 0.14f, 0.98f));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                             ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoBringToFrontOnFocus;

    if (ImGui::Begin("##FusionRibbon", nullptr, flags)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 winPos = ImGui::GetWindowPos();
        ImVec2 winSize = ImGui::GetWindowSize();

        // Top tab strip backdrop + bottom separator border
        dl->AddRectFilled(winPos, ImVec2(winPos.x + winSize.x, winPos.y + 30.0f),
                          amiga ? IM_COL32(0, 2, 34, 255) : IM_COL32(20, 22, 26, 255));
        dl->AddLine(ImVec2(winPos.x, winPos.y + 30.0f), ImVec2(winPos.x + winSize.x, winPos.y + 30.0f),
                    amiga ? IM_COL32(255, 255, 255, 255) : IM_COL32(52, 56, 64, 255), amiga ? 2.0f : 1.0f);
        dl->AddLine(ImVec2(winPos.x, winPos.y + winSize.y - 1.0f),
                    ImVec2(winPos.x + winSize.x, winPos.y + winSize.y - 1.0f),
                    amiga ? IM_COL32(255, 255, 255, 255) : IM_COL32(58, 64, 74, 255), amiga ? 2.0f : 1.0f);

        // --- Row 1: Brand badge + Workspace Stage Tabs + Right-aligned utility controls ---
        ImGui::SetCursorPos(ImVec2(12.0f, 5.0f));
        ImGui::TextColored(amiga ? ImVec4(1.0f, 0.533f, 0.0f, 1.0f) : ImVec4(0.28f, 0.73f, 0.95f, 1.0f),
                           "FLOWCRAFT");
        ImGui::SameLine(0.0f, 18.0f);

        struct TabDef {
            WorkspaceTab tab;
            const char* label;
            bool ready;
        };
        const TabDef tabs[] = {
            {WorkspaceTab::Model, "1. MODEL & ORIENT", app.meshLoaded},
            {WorkspaceTab::Voxelize, "2. VOXELIZE", app.voxelGridValid},
            {WorkspaceTab::Simulate, "3. SIMULATION SETTINGS", app.lbmValid},
            {WorkspaceTab::Inspect, "4. FLOW VISUALISATION", app.overlayValid},
        };

        for (const TabDef& t : tabs) {
            bool isSelected = (app.activeTab == t.tab);
            ImVec2 tabPos = ImGui::GetCursorScreenPos();
            ImVec2 textSz = ImGui::CalcTextSize(t.label);
            float tabW = textSz.x + 28.0f;
            float tabH = 22.0f;

            if (ImGui::InvisibleButton(t.label, ImVec2(tabW, tabH))) {
                app.activeTab = t.tab;
                app.showInspectorPanel = true;
            }
            bool tabHovered = ImGui::IsItemHovered();
            if (isSelected || tabHovered) {
                ImU32 tabFill = amiga ? (isSelected ? IM_COL32(0, 85, 170, 255) : IM_COL32(0, 55, 120, 255))
                                      : (isSelected ? IM_COL32(38, 44, 54, 255) : IM_COL32(30, 34, 42, 200));
                dl->AddRectFilled(tabPos, ImVec2(tabPos.x + tabW, tabPos.y + tabH), tabFill, amiga ? 0.0f : 3.0f);
            }
            if (isSelected) {
                dl->AddLine(ImVec2(tabPos.x + 4.0f, tabPos.y + tabH - 1.0f),
                            ImVec2(tabPos.x + tabW - 4.0f, tabPos.y + tabH - 1.0f),
                            amiga ? IM_COL32(255, 136, 0, 255) : IM_COL32(56, 171, 223, 255), 2.5f);
            }
            // Status dot indicating whether this stage is populated
            ImU32 dotCol = amiga ? (t.ready ? IM_COL32(255, 136, 0, 255) : IM_COL32(255, 255, 255, 160))
                                 : (t.ready ? IM_COL32(72, 215, 120, 255) : IM_COL32(95, 100, 110, 180));
            dl->AddCircleFilled(ImVec2(tabPos.x + 10.0f, tabPos.y + tabH * 0.5f), 3.2f, dotCol);
            dl->AddText(ImVec2(tabPos.x + 18.0f, tabPos.y + (tabH - textSz.y) * 0.5f),
                        isSelected ? IM_COL32(255, 255, 255, 255)
                                   : (amiga ? IM_COL32(220, 235, 255, 255) : IM_COL32(185, 192, 202, 255)),
                        t.label);
            ImGui::SameLine(0.0f, 6.0f);
        }

        // Right-aligned utilities on Row 1: Inspector toggle, Telemetry HUD toggle, Font combo, UI style dropdown
        float rightControlsWidth = 455.0f;
        if (winSize.x > rightControlsWidth + 520.0f) {
            ImGui::SameLine(winSize.x - rightControlsWidth);
            if (ImGui::SmallButton(app.showInspectorPanel ? "Inspector: ON" : "Inspector: OFF")) {
                app.showInspectorPanel = !app.showInspectorPanel;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton(app.showTelemetryHud ? "HUD: ON" : "HUD: OFF")) {
                app.showTelemetryHud = !app.showTelemetryHud;
            }
            ImGui::SameLine();
            const char* fontNames[] = {"Font: Default", "Font: Amiga Topaz"};
            ImGui::SetNextItemWidth(135.0f);
            if (ImGui::Combo("##RibbonFont", &app.uiFontIndex, fontNames, 2)) {
                ImGuiIO& io = ImGui::GetIO();
                io.FontDefault = (app.uiFontIndex == 1 && app.amigaTopazFont) ? app.amigaTopazFont : app.defaultFont;
            }
            ImGui::SameLine();
            const char* uiStyleNames[] = {"UI: Normal", "UI: Classic", "UI: Amiga"};
            ImGui::SetNextItemWidth(115.0f);
            if (ImGui::Combo("##UIStyle", &app.uiThemeIndex, uiStyleNames, 3)) {
                ApplyUITheme(app, app.uiThemeIndex);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Switch UI style between Normal (Fusion Ribbon), Classic single-window, and AmigaOS 1.3.");
            }
        }

        // --- Row 2: Autodesk Fusion–style Vector Icon Bar ---
        ImGui::SetCursorPos(ImVec2(10.0f, 34.0f));

        // 1. FILE group
        if (RibbonIconButton("##rib_open", "Open STL", RibbonIcon::OpenSTL, false, !busy,
                             "Open an STL mesh file for simulation.")) {
            LoadMeshFromDialog(app);
            if (app.meshLoaded) app.activeTab = WorkspaceTab::Model;
        }
        ImGui::SameLine(0.0f, 3.0f);
        if (RibbonIconButton("##rib_load", "Load", RibbonIcon::LoadSession, false, !busy,
                             "Load a saved .fcsession file (includes embedded STL and all settings).", 50.0f)) {
            LoadSessionDialog(app);
        }
        ImGui::SameLine(0.0f, 3.0f);
        if (RibbonIconButton("##rib_save", "Save", RibbonIcon::SaveSession, false, !busy && app.meshLoaded,
                             "Save model and all settings to a .fcsession file.", 50.0f)) {
            SaveSessionDialog(app);
        }
        ImGui::SameLine(0.0f, 3.0f);
        if (RibbonIconButton("##rib_reset", "Reset", RibbonIcon::ResetSession, false, !busy && app.meshLoaded,
                             "Discard the current mesh and reset all settings.", 50.0f)) {
            ImGui::OpenPopup("Reset Session?");
        }

        DrawRibbonDivider();

        // 2. DISPLAY VIEW MODE group
        if (RibbonIconButton("##rib_vmodel", "Model", RibbonIcon::ViewModel,
                             app.meshLoaded && app.viewMode == ViewMode::Model, app.meshLoaded,
                             "Display the CAD triangle mesh (1. MODEL & ORIENT).", 54.0f)) {
            app.viewMode = ViewMode::Model;
            app.activeTab = WorkspaceTab::Model;
        }
        ImGui::SameLine(0.0f, 3.0f);
        if (RibbonIconButton("##rib_vvox", "Voxels", RibbonIcon::ViewVoxels,
                             app.voxelGridValid && app.viewMode == ViewMode::VoxelGrid, app.voxelGridValid,
                             "Display the voxelized LBM solid grid.", 54.0f)) {
            app.viewMode = ViewMode::VoxelGrid;
            app.activeTab = WorkspaceTab::Voxelize;
        }
        ImGui::SameLine(0.0f, 3.0f);
        if (RibbonIconButton("##rib_vovl", "Overlay", RibbonIcon::ViewOverlay,
                             app.overlayValid && app.viewMode == ViewMode::FlowOverlay, app.overlayValid,
                             "Display the 3D Flow Overlay visualization.", 56.0f)) {
            app.viewMode = ViewMode::FlowOverlay;
            app.activeTab = WorkspaceTab::Inspect;
        }

        DrawRibbonDivider();

        // 3. CAMERA SNAP group
        if (RibbonIconButton("##rib_cfront", "Front", RibbonIcon::CamFront, false, app.meshLoaded,
                             "Snap camera to Front (nose-on to flow).", 48.0f)) {
            bool reverseFlow = app.lbmReverseFlowAtomic.load(std::memory_order_relaxed);
            SnapCameraView(app, reverseFlow ? kQuarterTurnRadians : -kQuarterTurnRadians, 0.0f);
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (RibbonIconButton("##rib_ctop", "Top", RibbonIcon::CamTop, false, app.meshLoaded,
                             "Snap camera to Top (planform view).", 46.0f)) {
            SnapCameraView(app, 0.0f, kTopViewPitch);
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (RibbonIconButton("##rib_cright", "Right", RibbonIcon::CamRight, false, app.meshLoaded,
                             "Snap camera to Right profile view.", 48.0f)) {
            SnapCameraView(app, 0.0f, 0.0f);
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (RibbonIconButton("##rib_cleft", "Left", RibbonIcon::CamLeft, false, app.meshLoaded,
                             "Snap camera to Left profile view.", 46.0f)) {
            SnapCameraView(app, std::numbers::pi_v<float>, 0.0f);
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (RibbonIconButton("##rib_ciso", "Iso", RibbonIcon::CamIso, false, app.meshLoaded,
                             "Snap camera to 3D Isometric perspective view.", 46.0f)) {
            bool reverseFlow = app.lbmReverseFlowAtomic.load(std::memory_order_relaxed);
            SnapCameraView(app, reverseFlow ? (kQuarterTurnRadians * 0.5f) : (-kQuarterTurnRadians * 0.5f), 0.55f);
        }

        DrawRibbonDivider();

        // 4. SOLVER & PIPELINE ACTIONS group
        if (RibbonIconButton("##rib_vox", "Voxelize", RibbonIcon::Voxelize, false, !busy && app.meshLoaded,
                             "Voxelize the mesh at the current resolution & padding.", 58.0f)) {
            app.activeTab = WorkspaceTab::Voxelize;
            StartVoxelizeTask(app);
        }
        ImGui::SameLine(0.0f, 3.0f);
        if (RibbonIconButton("##rib_init", "Init Solver", RibbonIcon::InitSolver, false, !busy && app.voxelGridValid,
                             "Initialize or Reset the CFD solver on the current voxel grid.", 66.0f)) {
            app.activeTab = WorkspaceTab::Simulate;
            StartInitSolverTask(app);
        }
        ImGui::SameLine(0.0f, 3.0f);
        bool isRunning = app.lbmRunning.load(std::memory_order_relaxed);
        if (RibbonIconButton("##rib_run", isRunning ? "Pause" : "Run", isRunning ? RibbonIcon::Pause : RibbonIcon::Play,
                             isRunning, !busy && app.lbmValid, "Start or pause continuous background solver stepping.",
                             52.0f)) {
            bool startingRun = !app.lbmRunning;
            app.lbmRunning = !app.lbmRunning;
            if (startingRun) {
                app.lbmRunStartTicks = SDL_GetTicks();
                app.showProcessingWindow = true;
                if (app.buildOverlayOnRun) StartBuildOverlayTask(app);
            }
        }
        ImGui::SameLine(0.0f, 3.0f);
        if (RibbonIconButton("##rib_step", "Step", RibbonIcon::Step, false, !busy && app.lbmValid,
                             "Advance the solver by one tick of substeps.", 48.0f)) {
            if (app.lbmBackend == SolverBackend::GPU) {
                StepSolverLocked(app, app.lbmSubstepsPerFrame);
            } else if (app.lbmBackend == SolverBackend::CPU) {
                StepReferenceLocked(app, app.lbmSubstepsPerFrame);
            } else if (IsPanelBackend(app.lbmBackend)) {
                StepPanelLocked(app, app.lbmSubstepsPerFrame);
            } else {
                StepPotentialFlowLocked(app, app.lbmSubstepsPerFrame);
            }
            app.lbmStepCount += app.lbmSubstepsPerFrame;
        }
        ImGui::SameLine(0.0f, 3.0f);
        if (RibbonIconButton("##rib_stats", "Stats", RibbonIcon::RefreshStats, false, !busy && app.lbmValid,
                             "Read back live density, speed, and Lift/Drag aerodynamic forces.", 48.0f)) {
            TriggerRefreshStats(app);
        }
        ImGui::SameLine(0.0f, 3.0f);
        if (RibbonIconButton("##rib_polars", "Polars", RibbonIcon::AeroPolars, app.showAeroPolarsWindow,
                             app.voxelGridValid, "Aero Polars & Flight Envelopes (XFLR5-style polar sweep).", 52.0f)) {
            app.showAeroPolarsWindow = !app.showAeroPolarsWindow;
        }

        DrawRibbonDivider();

        // 5. OVERLAY STYLE & BUILD group
        auto selectOverlayStyle = [&](OverlayStyle style) {
            app.overlayStyle = style;
            app.overlayStyleAtomic.store(static_cast<int>(app.overlayStyle), std::memory_order_relaxed);
            app.activeTab = WorkspaceTab::Inspect;
            if (app.lbmValid) {
                if (app.overlayValid) {
                    app.viewMode = ViewMode::FlowOverlay;
                    StartResliceTask(app);
                } else if (!busy) {
                    StartBuildOverlayTask(app);
                }
            }
        };

        if (RibbonIconButton("##rib_sslice", "Slice", RibbonIcon::StyleSlice,
                             app.overlayStyle == OverlayStyle::Slice && app.viewMode == ViewMode::FlowOverlay,
                             app.lbmValid, "2D colored cross-section slice planes.", 48.0f)) {
            selectOverlayStyle(OverlayStyle::Slice);
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (RibbonIconButton("##rib_scloud", "Cloud", RibbonIcon::StyleCloud,
                             app.overlayStyle == OverlayStyle::VoxelCloud && app.viewMode == ViewMode::FlowOverlay,
                             app.lbmValid, "3D volumetric pressure-faded speed cloud.", 48.0f)) {
            selectOverlayStyle(OverlayStyle::VoxelCloud);
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (RibbonIconButton("##rib_spart", "Smoke", RibbonIcon::StyleParticles,
                             app.overlayStyle == OverlayStyle::Particles && app.viewMode == ViewMode::FlowOverlay,
                             app.lbmValid, "Advected dye-injection smoke trail particles.", 50.0f)) {
            selectOverlayStyle(OverlayStyle::Particles);
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (RibbonIconButton("##rib_sstress", "Stress Map", RibbonIcon::StyleStress,
                             app.overlayStyle == OverlayStyle::StressMap && app.viewMode == ViewMode::FlowOverlay,
                             app.lbmValid, "3D SDF-projected airframe surface stress map (Pa).", 68.0f)) {
            selectOverlayStyle(OverlayStyle::StressMap);
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (RibbonIconButton("##rib_siso", "3D Iso", RibbonIcon::StyleIso,
                             app.overlayStyle == OverlayStyle::IsoSurface && app.viewMode == ViewMode::FlowOverlay,
                             app.lbmValid, "3D Iso-surface extraction (wake bubble / vortex core).", 50.0f)) {
            selectOverlayStyle(OverlayStyle::IsoSurface);
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (RibbonIconButton("##rib_sray", "Raymarch", RibbonIcon::StyleRaymarch,
                             app.overlayStyle == OverlayStyle::Raymarch && app.viewMode == ViewMode::FlowOverlay,
                             app.lbmValid, "3D Volumetric raymarching continuous field view.", 60.0f)) {
            selectOverlayStyle(OverlayStyle::Raymarch);
        }
        ImGui::SameLine(0.0f, 3.0f);
        if (RibbonIconButton("##rib_buildovl", "Update", RibbonIcon::BuildOverlay, false, !busy && app.lbmValid,
                             "Read back the solver field and rebuild the Flow Overlay.", 54.0f)) {
            StartBuildOverlayTask(app);
        }

        // Modal confirmation popup for Reset Session triggered from the ribbon
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal("Reset Session?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped(
                "This discards the current mesh, voxelization, and solver\n"
                "state, and resets every setting (model orientation,\n"
                "voxelization, solver, overlay, particles, camera angle) to\n"
                "its default. There's no way back afterward short of a\n"
                "previously saved .fcsession file.");
            ImGui::Spacing();
            if (ImGui::Button("Reset Session", ImVec2(160, 0))) {
                ResetSession(app);
                ImGui::CloseCurrentPopup();
            }
            ImGui::SetItemDefaultFocus();
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }
    ImGui::End();

    ImGui::PopStyleColor(1);
    ImGui::PopStyleVar(3);
}

// Live Telemetry & Aerodynamics HUD — initially centered horizontally near the
// bottom of the main window, staying centered and moving relative to the bottom
// edge during main-window resizing (and remaining freely draggable, collapsible,
// and closable).
void DrawTelemetryHud(AppState& app, float topInset) {
    if (!app.showTelemetryHud || !app.meshLoaded) return;

    constexpr float kBottomMargin = 16.0f;
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (!app.telemetryHudInitialized) {
        ImGui::SetNextWindowPos(
            ImVec2(viewport->Pos.x + viewport->Size.x * 0.5f, viewport->Pos.y + viewport->Size.y - kBottomMargin),
            ImGuiCond_Appearing, ImVec2(0.5f, 1.0f));
    }
    ImGui::SetNextWindowBgAlpha(app.uiThemeIndex == 2 ? 0.98f : 0.84f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, app.uiThemeIndex == 2 ? 0.0f : 6.0f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing;

    bool expanded = ImGui::Begin("Telemetry & Aerodynamics", &app.showTelemetryHud, flags);
    {
        ImVec2 currentPos = ImGui::GetWindowPos();
        ImVec2 currentSize = ImGui::GetWindowSize();
        bool collapsed = ImGui::IsWindowCollapsed();

        if (!app.telemetryHudInitialized) {
            app.telemetryHudInitialized = true;
            app.telemetryHudUserMoved = false;
            app.telemetryHudLastViewportSize = viewport->Size;
        } else {
            bool posChanged =
                (currentPos.x != app.telemetryHudLastPos.x || currentPos.y != app.telemetryHudLastPos.y);
            if (posChanged && ImGui::IsWindowFocused() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                app.telemetryHudUserMoved = true;
            }
        }

        float deltaViewportY = viewport->Size.y - app.telemetryHudLastViewportSize.y;
        if (!app.telemetryHudUserMoved) {
            ImVec2 targetPos;
            targetPos.x = std::floor(viewport->Pos.x + (viewport->Size.x - currentSize.x) * 0.5f);
            targetPos.y = collapsed ? (currentPos.y + deltaViewportY)
                                    : (viewport->Pos.y + viewport->Size.y - currentSize.y - kBottomMargin);
            if (targetPos.x != currentPos.x || targetPos.y != currentPos.y) {
                ImGui::SetWindowPos(targetPos);
            }
        } else if (deltaViewportY != 0.0f) {
            ImGui::SetWindowPos(ImVec2(currentPos.x, currentPos.y + deltaViewportY));
        }

        ClampWindowToViewport(topInset);
        app.telemetryHudLastPos = ImGui::GetWindowPos();
        app.telemetryHudLastViewportSize = viewport->Size;
    }

    if (expanded) {
        ImGui::TextDisabled("%s", app.statusMessage.c_str());

        if (app.voxelGridValid) {
            const VoxelGrid& grid = app.voxelGrid;
            double mb = static_cast<double>(grid.CellCount()) / (1024.0 * 1024.0);
            ImGui::Text("Grid: %d x %d x %d (%.1fM cells, dx=%.2f mm)", grid.ny, grid.nx, grid.nz, mb, grid.cellSize);
        }

        if (app.lbmValid) {
            ImGui::Separator();
            if (app.lbmBackend == SolverBackend::Naive) {
                ImGui::Text("Solver: Potential Flow (CPU)   Sweeps: %d", app.lbmStepCount.load());
            } else if (IsPanelBackend(app.lbmBackend)) {
                ImGui::Text("Solver: 3D Panel + Wake (%s)   Sweeps: %d",
                            app.lbmBackend == SolverBackend::PanelGPU ? "GPU" : "CPU", app.lbmStepCount.load());
            } else {
                ImGui::Text("Solver: %s (%s)   Steps: %d", LatticeModelName(app.lbmModel),
                            app.lbmBackend == SolverBackend::GPU ? "GPU" : "CPU", app.lbmStepCount.load());
            }

            LbmSolver::Stats statsCopy;
            statsCopy.avgDensity = app.lbmStatsAvgDensityAtomic.load(std::memory_order_relaxed);
            statsCopy.minDensity = app.lbmStatsMinDensityAtomic.load(std::memory_order_relaxed);
            statsCopy.maxDensity = app.lbmStatsMaxDensityAtomic.load(std::memory_order_relaxed);
            statsCopy.avgSpeed = app.lbmStatsAvgSpeedAtomic.load(std::memory_order_relaxed);
            statsCopy.maxSpeed = app.lbmStatsMaxSpeedAtomic.load(std::memory_order_relaxed);
            statsCopy.hasNaN = app.lbmStatsHasNaNAtomic.load(std::memory_order_relaxed);
            statsCopy.valid = app.lbmStatsValidAtomic.load(std::memory_order_relaxed);

            if (statsCopy.valid) {
                ImGui::Text("Speed: avg %.0f mm/s   max %.0f mm/s",
                            LatticeSpeedToMmPerSecond(app, statsCopy.avgSpeed),
                            LatticeSpeedToMmPerSecond(app, statsCopy.maxSpeed));
                ImGui::Text("Density: avg %.4f  [%.4f, %.4f]", statsCopy.avgDensity, statsCopy.minDensity,
                            statsCopy.maxDensity);
            }
            if (statsCopy.hasNaN || app.lbmFlowDivergedAtomic.load(std::memory_order_relaxed) || app.overlayHasNaN) {
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.45f, 1.0f),
                                    "FLOW DIVERGED (NaN / purple cells) — raise Smagorinsky Cs or Tau.");
            }

            if (app.lbmForcesValidAtomic.load(std::memory_order_relaxed)) {
                float rawFx = app.lbmForceXAtomic.load(std::memory_order_relaxed);
                float rawFy = app.lbmForceYAtomic.load(std::memory_order_relaxed);
                float rawFz = app.lbmForceZAtomic.load(std::memory_order_relaxed);
                float maxStressLat = app.lbmMaxStressAtomic.load(std::memory_order_relaxed);

                float dragLat = app.lbmParams.reverseFlow ? -rawFx : rawFx;
                float liftLat = rawFz;
                float sideLat = rawFy;

                float dragN = LatticeForceToNewtons(app, dragLat);
                float liftN = LatticeForceToNewtons(app, liftLat);
                float sideN = LatticeForceToNewtons(app, sideLat);
                constexpr float kNewtonsToGramsForce = 1000.0f / 9.80665f;
                float peakStressPa = LatticeStressToPascals(app, maxStressLat);

                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.35f, 0.82f, 1.0f, 1.0f), "Aerodynamic Forces (Sea-Level Air)");
                ImGui::Text("Drag:  %.4g N  (%.2f gf)", dragN, dragN * kNewtonsToGramsForce);
                ImGui::Text("Lift:  %.4g N  (%.2f gf)", liftN, liftN * kNewtonsToGramsForce);
                if (std::fabs(dragLat) > 1e-6f) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.55f, 0.92f, 0.65f, 1.0f), "  L/D: %.2f", liftLat / dragLat);
                }
                ImGui::Text("Side:  %.4g N  (%.2f gf)", sideN, sideN * kNewtonsToGramsForce);
                ImGui::Text("Peak Skin Stress:  %.3g Pa", peakStressPa);
                ImGui::Text("Centre of Pressure: (%.1f, %.1f, %.1f) mm",
                            app.lbmCpXAtomic.load(std::memory_order_relaxed),
                            app.lbmCpYAtomic.load(std::memory_order_relaxed),
                            app.lbmCpZAtomic.load(std::memory_order_relaxed));
                ImGui::Checkbox("Show Lift arrow above CoP", &app.showLiftArrow);
            } else if (!statsCopy.valid) {
                ImGui::TextDisabled("Click \"Stats\" or \"Update\" in the ribbon for live telemetry.");
            }
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

// Docked left-hand context inspector showing only the active workspace tab's controls.
void DrawFusionInspector(AppState& app, bool busy) {
    if (!app.showInspectorPanel) return;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x + 12.0f, viewport->Pos.y + kFusionRibbonHeight + 12.0f),
                             ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(340.0f, 120.0f),
                                         ImVec2(440.0f, std::max(200.0f, viewport->Size.y - kFusionRibbonHeight - 68.0f)));
    ImGui::SetNextWindowBgAlpha(app.uiThemeIndex == 2 ? 0.98f : 0.92f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, app.uiThemeIndex == 2 ? 0.0f : 6.0f);

    const char* panelTitle = "Inspector — Model & Orientation###FusionInspector";
    switch (app.activeTab) {
        case WorkspaceTab::Model:
            panelTitle = "Inspector — 1. Model & Orientation###FusionInspector";
            break;
        case WorkspaceTab::Voxelize:
            panelTitle = "Inspector — 2. Voxelize Domain###FusionInspector";
            break;
        case WorkspaceTab::Simulate:
            panelTitle = "Inspector — 3. Simulation Settings###FusionInspector";
            break;
        case WorkspaceTab::Inspect:
            panelTitle = "Inspector — 4. Flow Visualisation###FusionInspector";
            break;
    }

    if (ImGui::Begin(panelTitle, &app.showInspectorPanel, ImGuiWindowFlags_AlwaysAutoResize)) {
        ClampWindowToViewport(kFusionRibbonHeight);

        if (!app.meshLoaded) {
            ImGui::TextWrapped("%s", app.statusMessage.c_str());
            ImGui::Spacing();
            ImGui::BeginDisabled(busy);
            if (ImGui::Button("Open STL...", ImVec2(140.0f, 0.0f))) {
                LoadMeshFromDialog(app);
            }
            ImGui::SameLine();
            if (ImGui::Button("Load Session...", ImVec2(140.0f, 0.0f))) {
                LoadSessionDialog(app);
            }
            ImGui::EndDisabled();
        } else if (app.activeTab == WorkspaceTab::Model) {
            Vec3 size = app.mesh.bounds.Size();
            ImGui::Text("Triangles: %zu   Vertices: %zu", app.mesh.TriangleCount(), app.mesh.vertices.size());
            ImGui::Text("Native bounds: %.3f x %.3f x %.3f", size.x, size.y, size.z);

            ImGui::Separator();
            ImGui::TextUnformatted("Scale");
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("mm per model unit", &app.scaleMM, 0.0f, 0.0001f, 100000.0f, "%.4g",
                              ImGuiSliderFlags_Logarithmic);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "STL files carry no unit. This multiplies the model's\n"
                    "native coordinates; the result is what the ruler (and\n"
                    "later, the CFD solver) treats as millimetres.");
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset##scale_reset")) app.scaleMM = 1.0f;
            Vec3 scaledSize = size * app.scaleMM;
            ImGui::Text("Scaled size: %.2f x %.2f x %.2f mm", scaledSize.x, scaledSize.y, scaledSize.z);

            ImGui::Separator();
            ImGui::Text("Orientation (relative to flow, +Y)");
            ImGui::TextDisabled("Arrow keys: yaw/pitch   Q/E: roll");

            ImGui::PushItemFlag(ImGuiItemFlags_ButtonRepeat, true);
            float step = kRotationStepRadians;

            ImGui::TextUnformatted("Yaw");
            ImGui::SameLine(90);
            if (ImGui::ArrowButton("##yaw_left", ImGuiDir_Left)) RotateModel(app, kYawRotationAxis, step);
            ImGui::SameLine();
            if (ImGui::ArrowButton("##yaw_right", ImGuiDir_Right)) RotateModel(app, kYawRotationAxis, -step);
            ImGui::SameLine(170);
            if (ImGui::Button("-90##yaw90_left", ImVec2(40, 0))) RotateModel(app, kYawRotationAxis, kQuarterTurnRadians);
            ImGui::SameLine();
            if (ImGui::Button("+90##yaw90_right", ImVec2(40, 0)))
                RotateModel(app, kYawRotationAxis, -kQuarterTurnRadians);

            ImGui::TextUnformatted("Pitch");
            ImGui::SameLine(90);
            if (ImGui::ArrowButton("##pitch_up", ImGuiDir_Up)) RotateModel(app, kFlowAxis, -step);
            ImGui::SameLine();
            if (ImGui::ArrowButton("##pitch_down", ImGuiDir_Down)) RotateModel(app, kFlowAxis, step);
            ImGui::SameLine(170);
            if (ImGui::Button("-90##pitch90_up", ImVec2(40, 0))) RotateModel(app, kFlowAxis, -kQuarterTurnRadians);
            ImGui::SameLine();
            if (ImGui::Button("+90##pitch90_down", ImVec2(40, 0))) RotateModel(app, kFlowAxis, kQuarterTurnRadians);

            ImGui::TextUnformatted("Roll");
            ImGui::SameLine(90);
            if (ImGui::Button("-##roll_left", ImVec2(21, 0))) RotateModel(app, kPitchRotationAxis, step);
            ImGui::SameLine();
            if (ImGui::Button("+##roll_right", ImVec2(21, 0))) RotateModel(app, kPitchRotationAxis, -step);
            ImGui::SameLine(170);
            if (ImGui::Button("-90##roll90_left", ImVec2(40, 0)))
                RotateModel(app, kPitchRotationAxis, kQuarterTurnRadians);
            ImGui::SameLine();
            if (ImGui::Button("+90##roll90_right", ImVec2(40, 0)))
                RotateModel(app, kPitchRotationAxis, -kQuarterTurnRadians);

            ImGui::PopItemFlag();

            ImGui::Spacing();
            if (ImGui::Button("Reset Orientation")) {
                app.modelOrientation = app.lockedOrientation;
            }
            ImGui::SameLine();
            if (ImGui::Button("Lock Orientation")) {
                app.lockedOrientation = app.modelOrientation;
            }

            ImGui::Separator();
            ImGui::TextDisabled("Camera View (Snap & Smooth Glide)");
            bool reverseFlow = app.lbmReverseFlowAtomic.load(std::memory_order_relaxed);
            if (ImGui::Button("Front##insp")) {
                SnapCameraView(app, reverseFlow ? kQuarterTurnRadians : -kQuarterTurnRadians, 0.0f);
            }
            ImGui::SameLine();
            if (ImGui::Button("Back##insp")) {
                SnapCameraView(app, reverseFlow ? -kQuarterTurnRadians : kQuarterTurnRadians, 0.0f);
            }
            ImGui::SameLine();
            if (ImGui::Button("Top##insp")) SnapCameraView(app, 0.0f, kTopViewPitch);
            ImGui::SameLine();
            if (ImGui::Button("Bottom##insp")) SnapCameraView(app, 0.0f, -kTopViewPitch);
            ImGui::SameLine();
            if (ImGui::Button("Right##insp")) SnapCameraView(app, 0.0f, 0.0f);
            ImGui::SameLine();
            if (ImGui::Button("Left##insp")) SnapCameraView(app, std::numbers::pi_v<float>, 0.0f);
            ImGui::SameLine();
            if (ImGui::Button("Iso##insp")) {
                SnapCameraView(app, reverseFlow ? (kQuarterTurnRadians * 0.5f) : (-kQuarterTurnRadians * 0.5f), 0.55f);
            }

            ImGui::Checkbox("Smooth Transitions", &app.cameraSmoothTransitions);
            ImGui::SameLine();
            ImGui::Checkbox("Invert X", &app.camera.invertX);
            ImGui::SameLine();
            ImGui::Checkbox("Invert Y", &app.camera.invertY);
            ImGui::SameLine();
            ImGui::Checkbox("Invert Scroll Zoom", &app.camera.invertScrollZoom);
        } else if (app.activeTab == WorkspaceTab::Voxelize) {
            ImGui::BeginDisabled(busy);
            ImGui::SetNextItemWidth(140);
            ImGui::SliderInt("Resolution", &app.voxelResolution, kMinVoxelResolution, kMaxVoxelResolution);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Voxels along the longest axis of the padded domain. Total\n"
                    "cell count (and solver memory/compute cost) scales with\n"
                    "the CUBE of this, not linearly with it.");
            }
            ImGui::SameLine();
            if (ImGui::Button("Max for GPU Memory")) {
                app.voxelResolution = FindMaxGpuResolution(app, app.voxelPaddingFactor);
            }

            ImGui::SetNextItemWidth(200);
            ImGui::SliderFloat3("Padding (lat, flow, up)", &app.voxelPaddingFactor.x, 0.0f, 2.0f, "%.2f");

            size_t estimatedCells = EstimateVoxelCellCount(app, app.voxelResolution, app.voxelPaddingFactor);
            int q = GetLatticeSet(app.lbmModel).count;
            size_t popBytes = static_cast<size_t>(q) * estimatedCells * sizeof(uint16_t);
            size_t solidBytes = estimatedCells * sizeof(uint32_t);
            double popGiB = static_cast<double>(popBytes) / (1024.0 * 1024.0 * 1024.0);
            ImGui::TextDisabled("Estimated: %.1fM cells (~%.2f GiB population storage)",
                                 static_cast<double>(estimatedCells) / 1e6, popGiB);
            if (solidBytes > static_cast<size_t>(UINT32_MAX)) {
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                                    "Exceeds GPU 4GiB solid-mask buffer limit.\n"
                                    "Lower resolution/padding or use CPU/Naive.");
            }

            if (ImGui::Button("Voxelize", ImVec2(130.0f, 0.0f))) {
                StartVoxelizeTask(app);
            }
            if (app.voxelGridValid) {
                ImGui::SameLine();
                if (ImGui::Button("Reset Voxelization")) {
                    ResetVoxelization(app);
                }
            }
            ImGui::EndDisabled();

            if (app.voxelGridValid) {
                ImGui::Separator();
                const VoxelGrid& grid = app.voxelGrid;
                double mb = static_cast<double>(grid.CellCount()) / (1024.0 * 1024.0);
                ImGui::Text("Grid: %d x %d x %d  (%.1f%% solid)", grid.ny, grid.nx, grid.nz,
                            100.0 * static_cast<double>(app.voxelSolidCount) / static_cast<double>(grid.CellCount()));
                ImGui::Text("Cell size: %.3f mm  (%.1fM cells)", grid.cellSize, mb);
                ImGui::Text("Domain: %.1f x %.1f x %.1f mm", grid.ny * grid.cellSize, grid.nx * grid.cellSize,
                            grid.nz * grid.cellSize);
            }
        } else if (app.activeTab == WorkspaceTab::Simulate) {
            if (!app.voxelGridValid) {
                ImGui::TextDisabled("Voxelize the mesh first to configure and run the solver.");
                ImGui::BeginDisabled(busy);
                if (ImGui::Button("Voxelize Now")) {
                    StartVoxelizeTask(app);
                }
                ImGui::EndDisabled();
            } else {
                bool isLbm = IsLbmBackend(app.lbmBackend);
                bool isPanel = IsPanelBackend(app.lbmBackend);
                bool isNaive = app.lbmBackend == SolverBackend::Naive;

                ImGui::BeginDisabled(busy);
                int solverTypeIdx = 0, deviceIdx = 0;
                SplitSolverBackend(app.lbmBackend, solverTypeIdx, deviceIdx);
                const char* solverTypeNames[] = {"LBM (Navier-Stokes)", "Panel Method (3D Lifting)", "Potential Flow (Naive)"};
                ImGui::SetNextItemWidth(195);
                if (ImGui::Combo("Solver Type", &solverTypeIdx, solverTypeNames, 3)) {
                    app.lbmBackend = CombineSolverBackend(solverTypeIdx, deviceIdx);
                    app.lbmValid = false;
                    app.lbmRunning = false;
                }
                if (solverTypeIdx != 2) {
                    const char* deviceNames[] = {"GPU (Compute Shader)", "CPU (Multithreaded)"};
                    ImGui::SetNextItemWidth(195);
                    if (ImGui::Combo("Device", &deviceIdx, deviceNames, 2)) {
                        app.lbmBackend = CombineSolverBackend(solverTypeIdx, deviceIdx);
                        app.lbmValid = false;
                        app.lbmRunning = false;
                    }
                } else {
                    ImGui::TextDisabled("Device: CPU (multithreaded red-black SOR)");
                }
                ImGui::EndDisabled();

                if (isPanel) {
                    ImGui::TextDisabled("3D Source Panels + Trailing-Edge Kutta Horseshoe Wake");
                }

                if (isLbm) {
                    ImGui::BeginDisabled(busy);
                    int modelIndex = app.lbmModel == LatticeModel::D3Q19 ? 0 : 1;
                    const char* modelNames[] = {"D3Q19 (accurate)", "D3Q15 (fast)"};
                    ImGui::SetNextItemWidth(160);
                    if (ImGui::Combo("Lattice", &modelIndex, modelNames, 2)) {
                        app.lbmModel = modelIndex == 0 ? LatticeModel::D3Q19 : LatticeModel::D3Q15;
                        app.lbmValid = false;
                        app.lbmRunning = false;
                    }
                    ImGui::EndDisabled();
                }

                bool reverseFlow = app.lbmReverseFlowAtomic.load(std::memory_order_relaxed);
                int flowDirIndex = reverseFlow ? 1 : 0;
                const char* flowDirNames[] = {"Left -> Right", "Right -> Left"};
                ImGui::SetNextItemWidth(160);
                if (ImGui::Combo("Flow direction", &flowDirIndex, flowDirNames, 2)) {
                    std::lock_guard<std::mutex> lock(app.lbmMutex);
                    app.lbmParams.reverseFlow = flowDirIndex == 1;
                    app.lbmReverseFlowAtomic.store(app.lbmParams.reverseFlow, std::memory_order_relaxed);
                    if (app.lbmValid) {
                        if (app.lbmBackend == SolverBackend::GPU) {
                            app.lbmSolver.SetParams(app.lbmParams);
                        } else if (app.lbmBackend == SolverBackend::CPU) {
                            app.lbmReference.SetParams(LbmReference::Params{
                                app.lbmParams.tau, app.lbmParams.inletVelocity, app.lbmParams.reverseFlow,
                                app.lbmParams.freeSlipY, app.lbmParams.freeSlipZ, app.lbmParams.smagorinskyCs});
                        } else if (IsPanelBackend(app.lbmBackend)) {
                            app.panelSolver.SetParams(
                                PanelSolver::Params{app.lbmParams.inletVelocity, app.lbmParams.reverseFlow});
                        } else {
                            app.potentialFlow.SetParams(
                                PotentialFlowSolver::Params{app.lbmParams.inletVelocity, app.lbmParams.reverseFlow});
                        }
                    }
                }

                ImGui::SetNextItemWidth(140);
                ImGui::DragFloat("Inlet velocity (mm/s)", &app.lbmInletVelocityMmPerS, 50.0f, 1.0f, 200000.0f, "%.0f");

                if (isLbm) {
                    float tau = app.lbmTauAtomic.load(std::memory_order_relaxed);
                    ImGui::SetNextItemWidth(140);
                    if (ImGui::SliderFloat("Tau", &tau, kMinTau, 2.0f, "%.5f")) {
                        std::lock_guard<std::mutex> lock(app.lbmMutex);
                        app.lbmParams.tau = tau;
                        app.lbmTauAtomic.store(tau, std::memory_order_relaxed);
                        if (app.lbmValid) {
                            if (app.lbmBackend == SolverBackend::GPU) {
                                app.lbmSolver.SetParams(app.lbmParams);
                            } else {
                                app.lbmReference.SetParams(LbmReference::Params{
                                    app.lbmParams.tau, app.lbmParams.inletVelocity, app.lbmParams.reverseFlow,
                                    app.lbmParams.freeSlipY, app.lbmParams.freeSlipZ, app.lbmParams.smagorinskyCs});
                            }
                        }
                    }
                    float viscosity = LatticeTauToViscosityMm2PerSecond(app, tau);
                    ImGui::TextDisabled("≈ kinematic viscosity: %.3g mm²/s  (air ≈ 15 mm²/s)", viscosity);
                    if (ImGui::SmallButton("Match air viscosity")) {
                        ApplyTau(app, LatticeTauMatchingAir(app));
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Sets Tau so the viscosity equals sea-level air (15 mm²/s)\nat the current inlet speed and voxel size. Very low Tau is\nstabilised by a Smagorinsky sub-grid turbulence model.");
                    }
                    ImGui::SetNextItemWidth(140);
                    if (ImGui::SliderFloat("Smagorinsky Cs", &app.lbmParams.smagorinskyCs, 0.0f, 0.3f, "%.3f")) {
                        ApplyTau(app, app.lbmParams.tau);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Sub-grid turbulence strength. 0 = off (plain regularised LBM).\nRaise (0.15-0.2) if low-Tau runs go unstable; lower for less\nadded numerical viscosity.");
                    }

                    auto applyFreeSlip = [&]() {
                        std::lock_guard<std::mutex> lock(app.lbmMutex);
                        if (app.lbmValid) {
                            if (app.lbmBackend == SolverBackend::GPU) {
                                app.lbmSolver.SetParams(app.lbmParams);
                            } else {
                                app.lbmReference.SetParams(LbmReference::Params{
                                    app.lbmParams.tau, app.lbmParams.inletVelocity, app.lbmParams.reverseFlow,
                                    app.lbmParams.freeSlipY, app.lbmParams.freeSlipZ, app.lbmParams.smagorinskyCs});
                            }
                        }
                    };
                    if (ImGui::Checkbox("Free-slip Y walls", &app.lbmParams.freeSlipY)) applyFreeSlip();
                    ImGui::SameLine();
                    if (ImGui::Checkbox("Free-slip Z walls", &app.lbmParams.freeSlipZ)) applyFreeSlip();
                }

                ImGui::SetNextItemWidth(140);
                ImGui::SliderInt(!isLbm ? "Sweeps/tick" : "Substeps/tick", &app.lbmSubstepsPerFrame, 1,
                                  isNaive ? 500 : 20);

                if (isLbm) {
                    ImGui::Separator();
                    ImGui::Checkbox("Start flow at mid-speed/pressure", &app.lbmSeedMidSpeed);
                    ImGui::Checkbox("Warm start from coarse grid", &app.warmStartEnabled);
                    if (app.warmStartEnabled) {
                        ImGui::SetNextItemWidth(140);
                        ImGui::SliderInt("Warm start steps", &app.warmStartSteps, 10, 10000, "%d",
                                          ImGuiSliderFlags_Logarithmic);
                        ImGui::SetNextItemWidth(140);
                        ImGui::SliderInt("Resolution divisor", &app.warmStartResolutionDivisor, 2, 50);
                    }
                }

                ImGui::Separator();
                ImGui::BeginDisabled(busy);
                if (ImGui::Button("Initialize / Reset Solver")) {
                    StartInitSolverTask(app);
                }
                ImGui::EndDisabled();

                if (!app.lbmValid) {
                    static std::string initErrorDisplay;
                    if (!busy) {
                        std::lock_guard<std::mutex> lock(app.lbmMutex);
                        initErrorDisplay = app.lbmInitError;
                    }
                    if (!initErrorDisplay.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Failed to initialize:");
                        ImGui::TextWrapped("%s", initErrorDisplay.c_str());
                    }
                } else {
                    ImGui::Checkbox("Build overlay on Run", &app.buildOverlayOnRun);
                    if (ImGui::Checkbox("Auto-pause on flow divergence (NaN)", &app.autoPauseOnDivergence)) {
                        app.autoPauseOnDivergenceAtomic.store(app.autoPauseOnDivergence, std::memory_order_relaxed);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Periodically probes the solver field for NaN, Inf, negative\n"
                            "density, or supersonic lattice blowup (purple/magenta cells)\n"
                            "and immediately pauses the run so high-resolution simulations\n"
                            "don't waste time stepping a failed calculation.");
                    }
                    ImGui::BeginDisabled(busy);
                    if (ImGui::Button(app.lbmRunning ? "Pause" : "Run", ImVec2(80.0f, 0.0f))) {
                        bool startingRun = !app.lbmRunning;
                        app.lbmRunning = !app.lbmRunning;
                        if (startingRun) {
                            app.lbmRunStartTicks = SDL_GetTicks();
                            app.showProcessingWindow = true;
                            if (app.buildOverlayOnRun) StartBuildOverlayTask(app);
                        }
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Step", ImVec2(70.0f, 0.0f))) {
                        if (app.lbmBackend == SolverBackend::GPU) {
                            StepSolverLocked(app, app.lbmSubstepsPerFrame);
                        } else if (app.lbmBackend == SolverBackend::CPU) {
                            StepReferenceLocked(app, app.lbmSubstepsPerFrame);
                        } else if (IsPanelBackend(app.lbmBackend)) {
                            StepPanelLocked(app, app.lbmSubstepsPerFrame);
                        } else {
                            StepPotentialFlowLocked(app, app.lbmSubstepsPerFrame);
                        }
                        app.lbmStepCount += app.lbmSubstepsPerFrame;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Refresh Stats")) {
                        TriggerRefreshStats(app);
                    }
                    ImGui::EndDisabled();

                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.35f, 0.82f, 1.0f, 1.0f), "Aero Polars (XFLR5-Style)");
                    if (ImGui::Button("Open Aero Polars Window...", ImVec2(220.0f, 0.0f))) {
                        app.showAeroPolarsWindow = true;
                    }
                }
            }
        } else if (app.activeTab == WorkspaceTab::Inspect) {
            if (!app.lbmValid) {
                ImGui::TextDisabled("Initialize the CFD solver first to build and inspect flow overlays.");
                if (app.voxelGridValid) {
                    ImGui::BeginDisabled(busy);
                    if (ImGui::Button("Initialize Solver Now")) {
                        StartInitSolverTask(app);
                    }
                    ImGui::EndDisabled();
                }
            } else {
                int styleIndex = static_cast<int>(app.overlayStyle);
                const char* styleNames[] = {"Slice", "Voxel Cloud", "Particles (smoke trail)", "Airframe Stress Map", "3D Iso-Surface", "Volumetric Raymarching"};
                ImGui::SetNextItemWidth(200);
                if (ImGui::Combo("Style", &styleIndex, styleNames, 6)) {
                    app.overlayStyle = static_cast<OverlayStyle>(styleIndex);
                    app.overlayStyleAtomic.store(static_cast<int>(app.overlayStyle), std::memory_order_relaxed);
                    StartResliceTask(app);
                }

                if (app.overlayStyle == OverlayStyle::StressMap) {
                    ImGui::TextDisabled("Colored by local surface stress (Pa) on the SDF-projected skin.");
                } else {
                    if (ImGui::Checkbox("Remove channel pressure gradient", &app.detrendPressure)) {
                        app.detrendPressureAtomic.store(app.detrendPressure, std::memory_order_relaxed);
                        app.voxelAlphaRangeInitialized = false;
                        StartResliceTask(app);
                    }
                }

                if (app.overlayStyle == OverlayStyle::Raymarch) {
                    int colorIndex = app.overlayColorMode == FieldColorMode::Speed ? 0 : 1;
                    const char* colorNames[] = {"Speed", "Pressure"};
                    ImGui::SetNextItemWidth(160);
                    if (ImGui::Combo("Volume field", &colorIndex, colorNames, 2)) {
                        app.overlayColorMode = colorIndex == 0 ? FieldColorMode::Speed : FieldColorMode::Pressure;
                        app.overlayColorModeAtomic.store(static_cast<int>(app.overlayColorMode), std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    ImGui::SetNextItemWidth(160);
                    if (ImGui::SliderFloat("Absorption Density", &app.raymarchDensity, 0.10f, 5.0f, "%.2fx")) {
                        app.raymarchDensityAtomic.store(app.raymarchDensity, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    ImGui::SetNextItemWidth(160);
                    if (ImGui::SliderFloat("Ambient Cutoff", &app.raymarchCutoff, 0.0f, 0.50f, "%.2f")) {
                        app.raymarchCutoffAtomic.store(app.raymarchCutoff, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    ImGui::SetNextItemWidth(160);
                    if (ImGui::SliderFloat("Contrast Gamma", &app.raymarchGamma, 0.5f, 3.0f, "%.2f")) {
                        app.raymarchGammaAtomic.store(app.raymarchGamma, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    int sliceChoice = (app.raymarchSliceCount <= 32) ? 0 : (app.raymarchSliceCount <= 64 ? 1 : (app.raymarchSliceCount <= 96 ? 2 : 3));
                    const char* sliceNames[] = {"32 Slices (Fast)", "64 Slices (Balanced)", "96 Slices (Fine)", "128 Slices (Ultra)"};
                    ImGui::SetNextItemWidth(170);
                    if (ImGui::Combo("Sampling Quality", &sliceChoice, sliceNames, 4)) {
                        const int counts[] = {32, 64, 96, 128};
                        app.raymarchSliceCount = counts[sliceChoice];
                        app.raymarchSliceCountAtomic.store(app.raymarchSliceCount, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                }

                if (app.overlayStyle == OverlayStyle::IsoSurface) {
                    int colorIndex = app.overlayColorMode == FieldColorMode::Speed ? 0 : 1;
                    const char* colorNames[] = {"Speed", "Pressure"};
                    ImGui::SetNextItemWidth(160);
                    if (ImGui::Combo("Contour field", &colorIndex, colorNames, 2)) {
                        app.overlayColorMode = colorIndex == 0 ? FieldColorMode::Speed : FieldColorMode::Pressure;
                        app.overlayColorModeAtomic.store(static_cast<int>(app.overlayColorMode), std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    float lo = app.overlayColorMode == FieldColorMode::Speed ? app.cachedSpeedMin :
                               (app.detrendPressure ? app.cachedPressureDetrendMin : app.cachedPressureMin);
                    float hi = app.overlayColorMode == FieldColorMode::Speed ? app.cachedSpeedMax :
                               (app.detrendPressure ? app.cachedPressureDetrendMax : app.cachedPressureMax);
                    float currentVal = lo + app.isoThreshold * (hi - lo);

                    ImGui::SetNextItemWidth(160);
                    if (ImGui::SliderFloat("Iso Threshold", &app.isoThreshold, 0.01f, 0.99f, "%.2f")) {
                        app.isoThresholdAtomic.store(app.isoThreshold, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                    if (app.overlayColorMode == FieldColorMode::Speed) {
                        float speedMmS = LatticeSpeedToMmPerSecond(app, currentVal);
                        ImGui::TextDisabled("Contour: %.1f mm/s", speedMmS);
                    } else {
                        ImGui::TextDisabled("Contour: %.4g lat. p", currentVal);
                    }

                    int encloseMode = app.isoEncloseGreater ? 1 : 0;
                    const char* encloseNames[] = {"<= Iso (Wake / Suction)", ">= Iso (Jet / Stagnation)"};
                    ImGui::SetNextItemWidth(190);
                    if (ImGui::Combo("Enclose", &encloseMode, encloseNames, 2)) {
                        app.isoEncloseGreater = (encloseMode == 1);
                        app.isoEncloseGreaterAtomic.store(app.isoEncloseGreater, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    ImGui::SetNextItemWidth(160);
                    if (ImGui::SliderFloat("Surface Opacity", &app.isoAlpha, 0.10f, 1.0f, "%.2f")) {
                        app.isoAlphaAtomic.store(app.isoAlpha, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    int strideChoice = app.isoStride == 1 ? 0 : 1;
                    const char* strideNames[] = {"Full (1x)", "Fast (2x Stride)"};
                    ImGui::SetNextItemWidth(160);
                    if (ImGui::Combo("Quality", &strideChoice, strideNames, 2)) {
                        app.isoStride = strideChoice == 0 ? 1 : 2;
                        app.isoStrideAtomic.store(app.isoStride, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                }

                if (app.overlayStyle == OverlayStyle::Slice) {
                    const char* axisLabels[3] = {"X (front-on)", "Y (side-on)", "Z (plan/top)"};
                    for (size_t i = 0; i < 3; ++i) {
                        AppState::SlicePlaneState& plane = app.slicePlanes[i];
                        ImGui::PushID(static_cast<int>(i));
                        if (ImGui::Checkbox(axisLabels[i], &plane.enabled)) {
                            if (plane.enabled && plane.index < 0) {
                                plane.index = OverlaySliceDimension(app, kSliceAxes[i]) / 2;
                            }
                            plane.enabledAtomic.store(plane.enabled, std::memory_order_relaxed);
                            plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
                            StartResliceTask(app);
                        }
                        if (plane.enabled) {
                            int sliceMax = std::max(OverlaySliceDimension(app, kSliceAxes[i]) - 1, 0);
                            if (plane.index < 0) {
                                plane.index = sliceMax / 2;
                                plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
                            }
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(125);
                            if (ImGui::SliderInt("##pos", &plane.index, 0, sliceMax)) {
                                plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
                                StartResliceTask(app);
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("Center")) {
                                plane.index = sliceMax / 2;
                                plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
                                StartResliceTask(app);
                            }
                        }
                        ImGui::PopID();
                    }

                    if (!app.sliceShowParticles) {
                        int colorIndex = app.overlayColorMode == FieldColorMode::Speed ? 0 : 1;
                        const char* colorNames[] = {"Speed", "Pressure"};
                        ImGui::SetNextItemWidth(160);
                        if (ImGui::Combo("Color by", &colorIndex, colorNames, 2)) {
                            app.overlayColorMode = colorIndex == 0 ? FieldColorMode::Speed : FieldColorMode::Pressure;
                            app.overlayColorModeAtomic.store(static_cast<int>(app.overlayColorMode),
                                                               std::memory_order_relaxed);
                            StartResliceTask(app);
                        }
                    }

                    if (ImGui::Checkbox("Show particles", &app.sliceShowParticles)) {
                        app.sliceShowParticlesAtomic.store(app.sliceShowParticles, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                }

                bool showingParticles = app.overlayStyle == OverlayStyle::Particles ||
                                         (app.overlayStyle == OverlayStyle::Slice && app.sliceShowParticles);
                if (showingParticles) {
                    ImGui::Checkbox("Fade trail opacity by pressure", &app.smokeTrailPressureAlphaEnabled);
                    bool countChanged =
                        ImGui::SliderInt("Particle count", &app.smokeTrailParticleCount, 100, 8000);
                    bool lengthChanged = ImGui::SliderInt("Trail length", &app.smokeTrailLength, 2, 30);
                    if (countChanged || lengthChanged) app.smokeTrailInitialized = false;
                    if (ImGui::Button("Reset particles")) app.smokeTrailInitialized = false;
                    ImGui::SameLine();
                    if (ImGui::Button("Scatter particles")) app.smokeTrailScatterRequested = true;
                    ImGui::SetNextItemWidth(160);
                    ImGui::SliderFloat("Particle speed", &app.smokeTrailSpeedScale, 0.1f, 5.0f, "%.2fx");

                    bool puffing = app.smokeTrailPuffRemainingSeconds > 0.0f;
                    ImGui::BeginDisabled(puffing);
                    if (ImGui::Button("Puff")) {
                        app.smokeTrailPuffRemainingSeconds = app.smokeTrailPuffDurationSeconds;
                    }
                    ImGui::EndDisabled();
                    ImGui::SameLine();
                    if (puffing) {
                        ImGui::TextDisabled("puffing... %.1fs left", app.smokeTrailPuffRemainingSeconds);
                    } else {
                        ImGui::SetNextItemWidth(110);
                        ImGui::SliderFloat("Puff duration", &app.smokeTrailPuffDurationSeconds, 0.5f, 10.0f, "%.1fs");
                    }
                }

                if (showingParticles || app.overlayStyle == OverlayStyle::VoxelCloud) {
                    if (!app.voxelAlphaRangeInitialized) {
                        std::lock_guard<std::mutex> lock(app.lastFieldMutex);
                        if (app.cachedRangesValid) {
                            app.voxelAlphaPressureLo =
                                app.detrendPressure ? app.cachedPressureDetrendMin : app.cachedPressureMin;
                            app.voxelAlphaPressureHi =
                                app.detrendPressure ? app.cachedPressureDetrendMax : app.cachedPressureMax;
                            app.voxelAlphaRangeInitialized = true;
                        }
                    }
                    ImGui::SetNextItemWidth(160);
                    bool alphaChanged = ImGui::DragFloat("Transparent at/below", &app.voxelAlphaPressureLo, 0.001f,
                                                           -10.0f, app.voxelAlphaPressureHi, "%.4f");
                    ImGui::SetNextItemWidth(160);
                    alphaChanged |= ImGui::DragFloat("Opaque at/above", &app.voxelAlphaPressureHi, 0.001f,
                                                       app.voxelAlphaPressureLo, 10.0f, "%.4f");
                    if (ImGui::Checkbox("Highlight negative pressure (dark grey)", &app.voxelNegativePressureGrey)) {
                        app.voxelNegativePressureGreyAtomic.store(app.voxelNegativePressureGrey,
                                                                    std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                    if (alphaChanged) {
                        app.voxelAlphaPressureLoAtomic.store(app.voxelAlphaPressureLo, std::memory_order_relaxed);
                        app.voxelAlphaPressureHiAtomic.store(app.voxelAlphaPressureHi, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                }

                ImGui::Separator();
                ImGui::BeginDisabled(busy);
                if (ImGui::Button("Build Overlay")) {
                    StartBuildOverlayTask(app);
                }
                ImGui::EndDisabled();

                if (app.overlayStyle != OverlayStyle::StressMap) {
                    ImGui::Checkbox("Show model", &app.overlayShowModel);
                }
                ImGui::Checkbox("Auto-update while running", &app.overlayAutoUpdate);
                ImGui::SetNextItemWidth(140);
                ImGui::SliderInt("Update interval (ms)", &app.overlayUpdateIntervalMs, 50, 2000);
                ImGui::SetNextItemWidth(140);
                ImGui::SliderInt("Warmup steps", &app.overlayAutoUpdateWarmupSteps, 0, 2000);
            }
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void DrawUI(AppState& app) {
    bool busy = app.taskKind.load(std::memory_order_acquire) != TaskKind::None;

    if (!app.classicSinglePanelUI) {
        DrawFusionRibbon(app, busy);
        DrawFusionInspector(app, busy);
        DrawTelemetryHud(app, kFusionRibbonHeight);

        if (app.meshLoaded) {
            DrawOrientationGizmo(app, kFusionRibbonHeight);
            DrawScaleRuler(app);
            DrawLiftArrow(app);
            if (app.viewMode == ViewMode::FlowOverlay && app.overlayStyle == OverlayStyle::Slice) {
                DrawSliceDragHandles(app);
            }
            if (app.viewMode == ViewMode::FlowOverlay && app.overlayValid) DrawOverlayLegend(app);
        }
        DrawBusyOverlay(app);
        DrawProcessingWindow(app, kFusionRibbonHeight);
        DrawFlowDivergenceBanner(app, kFusionRibbonHeight, busy);
        DrawAeroPolarsWindow(app, kFusionRibbonHeight);
        return;
    }

    bool classicOpen = true;
    ImGui::Begin("FlowCraft", &classicOpen);
    if (!classicOpen) {
        ApplyUITheme(app, 0);
    }
    ClampWindowToViewport();
    const char* uiStyleNames[] = {"Normal", "Classic", "Amiga"};
    ImGui::SetNextItemWidth(140.0f);
    if (ImGui::Combo("UI Style", &app.uiThemeIndex, uiStyleNames, 3)) {
        ApplyUITheme(app, app.uiThemeIndex);
    }
    ImGui::Separator();

    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Open STL...")) {
        LoadMeshFromDialog(app);
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Session...")) {
        LoadSessionDialog(app);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Restores the model, its orientation/scale, and every\n"
            "solver/voxelize/overlay setting from a saved .fcsession\n"
            "file — including the STL itself, embedded in the file, so\n"
            "the original isn't needed. Does NOT re-run Voxelize or\n"
            "Initialize Solver automatically, and does not include a\n"
            "live simulation snapshot (the solver's own state can be\n"
            "many gigabytes) — press those yourself afterward to resume\n"
            "simulating with the restored settings.");
    }
    if (app.meshLoaded) {
        ImGui::SameLine();
        if (ImGui::Button("Save Session...")) {
            SaveSessionDialog(app);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Saves the model (with the STL itself embedded — the\n"
                "original file isn't needed to reload it), its\n"
                "orientation/scale, and every solver/voxelize/overlay\n"
                "setting to a single .fcsession file. Does not include a\n"
                "live simulation snapshot — see Load Session's tooltip.");
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Session")) {
            ImGui::OpenPopup("Reset Session?");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Discards the current mesh and every setting (model\n"
                "orientation, voxelization, solver, overlay, particles,\n"
                "camera angle) back to a blank session, as if the app\n"
                "had just been launched. Asks for confirmation first —\n"
                "there's no way back afterward short of a previously\n"
                "saved .fcsession file.");
        }
    }
    ImGui::EndDisabled();

    // Centered, same as ImGui's own demo convention for a confirmation
    // dialog — placed outside the BeginDisabled(busy) block above (a
    // modal popup shouldn't itself be disabled; the button that opens it
    // already was, while busy, so OpenPopup can't fire then anyway).
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Reset Session?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped(
            "This discards the current mesh, voxelization, and solver\n"
            "state, and resets every setting (model orientation,\n"
            "voxelization, solver, overlay, particles, camera angle) to\n"
            "its default. There's no way back afterward short of a\n"
            "previously saved .fcsession file.");
        ImGui::Spacing();
        if (ImGui::Button("Reset Session", ImVec2(160, 0))) {
            ResetSession(app);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::Separator();
    {
        const char* fontNames[] = {"Default (ProggyClean)", "Amiga Topaz"};
        ImGui::SetNextItemWidth(180);
        if (ImGui::Combo("UI Font", &app.uiFontIndex, fontNames, 2)) {
            ImGuiIO& io = ImGui::GetIO();
            if (app.uiFontIndex == 1 && app.amigaTopazFont) {
                io.FontDefault = app.amigaTopazFont;
            } else {
                io.FontDefault = app.defaultFont;
            }
        }
    }
    ImGui::Separator();
    ImGui::TextWrapped("%s", app.statusMessage.c_str());
    if (app.meshLoaded) {
        ImGui::Spacing();
        ImGui::Text("Triangles: %zu", app.mesh.TriangleCount());
        ImGui::Text("Vertices: %zu", app.mesh.vertices.size());
        Vec3 size = app.mesh.bounds.Size();
        ImGui::Text("Bounds: %.3f x %.3f x %.3f", size.x, size.y, size.z);
        ImGui::Spacing();
        ImGui::TextDisabled("Drag to orbit, scroll to zoom.");
        ImGui::Checkbox("Invert X", &app.camera.invertX);
        ImGui::SameLine();
        ImGui::Checkbox("Invert Y", &app.camera.invertY);
        ImGui::Checkbox("Invert Scroll Zoom", &app.camera.invertScrollZoom);

        ImGui::Separator();
        ImGui::TextUnformatted("Display");
        if (ImGui::RadioButton("Model", app.viewMode == ViewMode::Model)) app.viewMode = ViewMode::Model;
        ImGui::SameLine();
        ImGui::BeginDisabled(!app.voxelGridValid);
        if (ImGui::RadioButton("Voxel Grid", app.viewMode == ViewMode::VoxelGrid)) app.viewMode = ViewMode::VoxelGrid;
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!app.overlayValid);
        if (ImGui::RadioButton("Flow Overlay", app.viewMode == ViewMode::FlowOverlay))
            app.viewMode = ViewMode::FlowOverlay;
        ImGui::EndDisabled();

        ImGui::Separator();
        ImGui::TextUnformatted("Scale");
        ImGui::SetNextItemWidth(140);
        ImGui::DragFloat("mm per model unit", &app.scaleMM, 0.0f, 0.0001f, 100000.0f, "%.4g",
                          ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "STL files carry no unit. This multiplies the model's\n"
                "native coordinates; the result is what the ruler (and\n"
                "later, the CFD solver) treats as millimetres.");
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset##scale_reset")) app.scaleMM = 1.0f;
        Vec3 scaledSize = size * app.scaleMM;
        ImGui::Text("Scaled size: %.3f x %.3f x %.3f mm", scaledSize.x, scaledSize.y, scaledSize.z);

        ImGui::Separator();
        ImGui::TextUnformatted("View (Snap & Smooth Glide)");
        bool reverseFlow = app.lbmReverseFlowAtomic.load(std::memory_order_relaxed);
        if (ImGui::Button("Front")) {
            SnapCameraView(app, reverseFlow ? kQuarterTurnRadians : -kQuarterTurnRadians, 0.0f);
        }
        ImGui::SameLine();
        if (ImGui::Button("Back")) {
            SnapCameraView(app, reverseFlow ? -kQuarterTurnRadians : kQuarterTurnRadians, 0.0f);
        }
        ImGui::SameLine();
        if (ImGui::Button("Top")) SnapCameraView(app, 0.0f, kTopViewPitch);
        ImGui::SameLine();
        if (ImGui::Button("Bottom")) SnapCameraView(app, 0.0f, -kTopViewPitch);
        ImGui::SameLine();
        if (ImGui::Button("Right")) SnapCameraView(app, 0.0f, 0.0f);
        ImGui::SameLine();
        if (ImGui::Button("Left")) SnapCameraView(app, std::numbers::pi_v<float>, 0.0f);
        ImGui::SameLine();
        if (ImGui::Button("Iso")) {
            SnapCameraView(app, reverseFlow ? (kQuarterTurnRadians * 0.5f) : (-kQuarterTurnRadians * 0.5f), 0.55f);
        }
        ImGui::Checkbox("Smooth Transitions", &app.cameraSmoothTransitions);

        ImGui::Separator();
        ImGui::Text("Orientation (relative to flow, +Y)");
        ImGui::TextDisabled("Arrow keys: yaw/pitch   Q/E: roll");

        ImGui::PushItemFlag(ImGuiItemFlags_ButtonRepeat, true);
        float step = kRotationStepRadians;

        ImGui::TextUnformatted("Yaw");
        ImGui::SameLine(100);
        if (ImGui::ArrowButton("##yaw_left", ImGuiDir_Left)) RotateModel(app, kYawRotationAxis, step);
        ImGui::SameLine();
        if (ImGui::ArrowButton("##yaw_right", ImGuiDir_Right)) RotateModel(app, kYawRotationAxis, -step);

        ImGui::TextUnformatted("Pitch");
        ImGui::SameLine(100);
        if (ImGui::ArrowButton("##pitch_up", ImGuiDir_Up)) RotateModel(app, kFlowAxis, -step);
        ImGui::SameLine();
        if (ImGui::ArrowButton("##pitch_down", ImGuiDir_Down)) RotateModel(app, kFlowAxis, step);

        ImGui::TextUnformatted("Roll");
        ImGui::SameLine(100);
        if (ImGui::Button("-##roll_left", ImVec2(21, 0))) RotateModel(app, kPitchRotationAxis, step);
        ImGui::SameLine();
        if (ImGui::Button("+##roll_right", ImVec2(21, 0))) RotateModel(app, kPitchRotationAxis, -step);

        ImGui::PopItemFlag();

        ImGui::Spacing();
        ImGui::TextUnformatted("Snap 90°");
        float quarter = kQuarterTurnRadians;

        ImGui::TextUnformatted("Yaw");
        ImGui::SameLine(100);
        if (ImGui::Button("-90##yaw90_left", ImVec2(40, 0))) RotateModel(app, kYawRotationAxis, quarter);
        ImGui::SameLine();
        if (ImGui::Button("+90##yaw90_right", ImVec2(40, 0))) RotateModel(app, kYawRotationAxis, -quarter);

        ImGui::TextUnformatted("Pitch");
        ImGui::SameLine(100);
        if (ImGui::Button("-90##pitch90_up", ImVec2(40, 0))) RotateModel(app, kFlowAxis, -quarter);
        ImGui::SameLine();
        if (ImGui::Button("+90##pitch90_down", ImVec2(40, 0))) RotateModel(app, kFlowAxis, quarter);

        ImGui::TextUnformatted("Roll");
        ImGui::SameLine(100);
        if (ImGui::Button("-90##roll90_left", ImVec2(40, 0))) RotateModel(app, kPitchRotationAxis, quarter);
        ImGui::SameLine();
        if (ImGui::Button("+90##roll90_right", ImVec2(40, 0))) RotateModel(app, kPitchRotationAxis, -quarter);

        ImGui::Spacing();
        if (ImGui::Button("Reset Orientation")) {
            app.modelOrientation = app.lockedOrientation;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Resets to the locked orientation below (plain identity\n"
                "until you've ever pressed Lock Orientation).");
        }
        ImGui::SameLine();
        if (ImGui::Button("Lock Orientation")) {
            app.lockedOrientation = app.modelOrientation;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Makes the model's *current* orientation the new default\n"
                "for Reset Orientation above, and for the next STL you\n"
                "open — so once you've rotated a model into the pose you\n"
                "actually want, you don't have to redo that rotation\n"
                "every time you reload it or start a new one this\n"
                "session. Saved in .fcsession files too.");
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Voxelize (LBM grid)");
        ImGui::BeginDisabled(busy);
        ImGui::SetNextItemWidth(140);
        ImGui::SliderInt("Resolution", &app.voxelResolution, kMinVoxelResolution, kMaxVoxelResolution);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Voxels along the longest axis of the padded domain. Total\n"
                "cell count (and solver memory/compute cost) scales with\n"
                "the CUBE of this, not linearly with it — e.g. 512 is ~8x\n"
                "the cells of 256. Watch the \"M cells\" readout below after\n"
                "Voxelizing, not this slider's value, when judging scaling.\n"
                "Very high values can exhaust GPU or system memory on Init.");
        }
        if (app.meshLoaded) {
            ImGui::SameLine();
            if (ImGui::Button("Max for GPU Memory")) {
                app.voxelResolution = FindMaxGpuResolution(app, app.voxelPaddingFactor);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Sets Resolution to the highest value whose estimated\n"
                    "solid-mask buffer — the one buffer the GPU backend's\n"
                    "Z-slab decomposition doesn't split, see the estimate/\n"
                    "warning below — still fits under SDL_GPU's 4GiB-per-\n"
                    "buffer cap, for the current lattice model, padding,\n"
                    "and the model's current pose. That cap is a fixed\n"
                    "SDL_GPU API limit, NOT a measurement of how much GPU\n"
                    "memory this machine actually has (SDL_GPU exposes no\n"
                    "such query) — a resolution under this button's result\n"
                    "can still exhaust real GPU memory on Init, same as\n"
                    "any other setting here. CPU and Naive have no 4GiB\n"
                    "ceiling at all (only real system memory), so this is\n"
                    "specifically about what the GPU backend can handle,\n"
                    "regardless of which backend is currently selected.");
            }
        }
        ImGui::SetNextItemWidth(220);
        ImGui::SliderFloat3("Padding factor (lateral, flow, up)", &app.voxelPaddingFactor.x, 0.0f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Extra empty domain added around the model's extent, per\n"
                "axis, as a fraction of that extent — e.g. 0.3 adds 30%%\n"
                "margin on that axis. First is lateral, second is the flow\n"
                "axis (upstream/downstream — more room here costs the\n"
                "most cells, since it's usually the longest), third is\n"
                "up. More padding on an axis means a bigger wind\n"
                "tunnel there (less wall-blockage on the physics) at the\n"
                "same resolution/cell-count budget; less padding spends\n"
                "more of that budget resolving the model itself. Takes\n"
                "effect on the next Voxelize.");
        }

        // Estimated *before* committing to a (potentially slow, at high
        // resolution) Voxelize — see EstimateVoxelCellCount's comment.
        // LbmSolver's GPU backend splits its population buffers across
        // Z-slabs (see LbmSolver.h's class comment), so the 4GiB figure
        // below is no longer about pop_ at all (that cap is gone,
        // regardless of resolution) — it's SDL_GPU's hard per-buffer cap
        // applied to the one buffer decomposition deliberately leaves
        // unsplit, solidMask_ (cellCount*sizeof(uint32_t) bytes — see
        // LbmSolver::Init's own, identical check), which is now the real
        // ceiling. The population-storage total is still shown since
        // that's real GPU memory the lattice needs regardless of how
        // many buffers it's spread across — just no longer a hard
        // failure on its own. CPU and Naive have no such limit either
        // way, so this warns regardless of which backend is currently
        // selected rather than hiding it behind that choice.
        if (app.meshLoaded) {
            size_t estimatedCells = EstimateVoxelCellCount(app, app.voxelResolution, app.voxelPaddingFactor);
            int q = GetLatticeSet(app.lbmModel).count;
            size_t popBytes = static_cast<size_t>(q) * estimatedCells * sizeof(uint16_t);  // total, across every slab
            size_t solidBytes = estimatedCells * sizeof(uint32_t);
            double popGiB = static_cast<double>(popBytes) / (1024.0 * 1024.0 * 1024.0);
            ImGui::TextDisabled("estimated: %.1fM cells (~%.2f GiB total population storage)",
                                 static_cast<double>(estimatedCells) / 1e6, popGiB);
            if (solidBytes > static_cast<size_t>(UINT32_MAX)) {
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                                    "Exceeds the GPU backend's 4GiB-per-buffer limit at this\n"
                                    "resolution/padding — the solid-mask buffer, which Z-slab\n"
                                    "decomposition doesn't split, would need to be larger than\n"
                                    "that. GPU Initialize will fail. Lower the resolution or\n"
                                    "padding, or use the CPU/Naive backend instead (no such\n"
                                    "ceiling).");
            }
        }

        if (ImGui::Button("Voxelize")) {
            StartVoxelizeTask(app);
        }
        if (app.voxelGridValid) {
            ImGui::SameLine();
            if (ImGui::Button("Reset Voxelization")) {
                ResetVoxelization(app);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Clears the current grid (and the solver/overlay built\n"
                    "from it, which can't outlive it) so you can change\n"
                    "resolution, padding, or the model's pose before\n"
                    "voxelizing again — without needing to Voxelize first\n"
                    "just to get back to a clean slate.");
            }
        }
        ImGui::EndDisabled();
        if (app.voxelGridValid) {
            const VoxelGrid& grid = app.voxelGrid;
            double mb = static_cast<double>(grid.CellCount()) / (1024.0 * 1024.0);
            // Printed in WORLD (lateral, flow, up) order — grid.ny, then
            // grid.nx, then grid.nz — not the grid's own (nx,ny,nz)
            // order, since grid-x/grid-y are world Y/X respectively
            // (see VoxelGrid::CellCenter's comment) and showing them
            // nx-first here would silently print the flow-axis count
            // where a reader expects the lateral one.
            ImGui::Text("Grid: %d x %d x %d  (%.1f%% solid)", grid.ny, grid.nx, grid.nz,
                        100.0 * static_cast<double>(app.voxelSolidCount) / static_cast<double>(grid.CellCount()));
            ImGui::Text("Cell size: %.3f mm  (%.1fM cells)", grid.cellSize, mb);
            ImGui::Text("Domain: %.1f x %.1f x %.1f mm", grid.ny * grid.cellSize, grid.nx * grid.cellSize,
                        grid.nz * grid.cellSize);

            ImGui::Separator();
            ImGui::TextUnformatted("Solver");

            bool isLbm = IsLbmBackend(app.lbmBackend);
            bool isPanel = IsPanelBackend(app.lbmBackend);
            bool isNaive = app.lbmBackend == SolverBackend::Naive;

            ImGui::BeginDisabled(busy);
            int solverTypeIdx = 0, deviceIdx = 0;
            SplitSolverBackend(app.lbmBackend, solverTypeIdx, deviceIdx);
            const char* solverTypeNames[] = {"LBM (Navier-Stokes)", "Panel Method (3D Lifting)", "Potential Flow (Naive)"};
            ImGui::SetNextItemWidth(220);
            if (ImGui::Combo("Solver Type", &solverTypeIdx, solverTypeNames, 3)) {
                app.lbmBackend = CombineSolverBackend(solverTypeIdx, deviceIdx);
                app.lbmValid = false;
                app.lbmRunning = false;
            }
            if (solverTypeIdx != 2) {
                const char* deviceNames[] = {"GPU (Compute Shader)", "CPU (Multithreaded)"};
                ImGui::SetNextItemWidth(220);
                if (ImGui::Combo("Device", &deviceIdx, deviceNames, 2)) {
                    app.lbmBackend = CombineSolverBackend(solverTypeIdx, deviceIdx);
                    app.lbmValid = false;
                    app.lbmRunning = false;
                }
            } else {
                ImGui::TextDisabled("Device: CPU (multithreaded red-black SOR)");
            }
            ImGui::EndDisabled();

            if (isLbm) {
                ImGui::BeginDisabled(busy);
                int modelIndex = app.lbmModel == LatticeModel::D3Q19 ? 0 : 1;
                const char* modelNames[] = {"D3Q19 (accurate)", "D3Q15 (fast)"};
                ImGui::SetNextItemWidth(160);
                if (ImGui::Combo("Lattice", &modelIndex, modelNames, 2)) {
                    app.lbmModel = modelIndex == 0 ? LatticeModel::D3Q19 : LatticeModel::D3Q15;
                    app.lbmValid = false;  // different Q -> different buffer sizes, needs re-init
                    app.lbmRunning = false;
                }
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "D3Q19: 19 directions, the common choice for aerodynamics.\n"
                        "D3Q15: 15 directions — a genuinely different, faster lattice\n"
                        "(fewer directions to stream/collide per cell), not just a\n"
                        "lower-quality setting; still a real momentum-conserving model.");
                }
            }

            if (isPanel) {
                ImGui::TextDisabled(
                    "3D Lifting Panel Method: desingularized surface source panels\n"
                    "+ spanwise trailing-edge Kutta horseshoe vortex wake.\n"
                    "Generates circulation, Lift, wingtip vortices, and Induced Drag.");
            } else if (isNaive) {
                ImGui::TextDisabled(
                    "Steady, inviscid potential flow — fast, resolution-unbounded\n"
                    "(system RAM only), but no viscosity: no separation, no\n"
                    "low-pressure wake behind the model (pressure recovers to\n"
                    "ambient at the rear instead). Good for a quick look at flow\n"
                    "ahead of and alongside the model at resolutions LBM can't\n"
                    "reach; not a substitute for GPU/CPU when the wake matters.");
            }

            // app.lbmReverseFlowAtomic already exists (see its own
            // comment) for exactly this: reading it here lock-free avoids
            // what used to be an unconditional-every-frame app.lbmMutex
            // acquisition — fine on its own against an occasional
            // Initialize/Reset Solver (covered by the old `busy` guard),
            // but at large enough resolution StepSolverPaced and
            // ReadbackFieldPacedGPU both re-acquire that same mutex in a
            // tight loop continuously (not just occasionally) while
            // running+auto-updating, and this read — along with the Tau
            // display and the stats display below, now fixed the same
            // way — isn't gated by `busy` at all (that only covers
            // StartTask-based one-shot tasks), so it was trying to
            // acquire that same hammered mutex every single frame. A
            // mutex gives no fairness guarantee between a periodic
            // once-a-frame acquirer and two threads re-acquiring it
            // hundreds of times a second each — which is what a
            // genuinely frozen window, not just a stale display, turned
            // out to actually be.
            bool reverseFlow = app.lbmReverseFlowAtomic.load(std::memory_order_relaxed);
            int flowDirIndex = reverseFlow ? 1 : 0;
            // Describes flow direction as seen from the "Right" camera
            // view. Swapped from the previous guess per the user's own
            // direct report — flowDirIndex itself is untouched (it only
            // controls which text corresponds to which actual
            // reverseFlow value, not which text says which way, so no
            // risk of this being out of sync with the Combo's write-back
            // below).
            // ASCII "->" rather than a Unicode arrow glyph (U+2192) —
            // ImGui's default bundled font doesn't include the Arrows
            // block, so that rendered as a tofu/"?" box instead.
            const char* flowDirNames[] = {"Left -> Right", "Right -> Left"};
            ImGui::SetNextItemWidth(160);
            if (ImGui::Combo("Flow direction", &flowDirIndex, flowDirNames, 2)) {
                std::lock_guard<std::mutex> lock(app.lbmMutex);
                app.lbmParams.reverseFlow = flowDirIndex == 1;
                app.lbmReverseFlowAtomic.store(app.lbmParams.reverseFlow, std::memory_order_relaxed);
                if (app.lbmValid) {
                    if (app.lbmBackend == SolverBackend::GPU) {
                        app.lbmSolver.SetParams(app.lbmParams);
                    } else if (app.lbmBackend == SolverBackend::CPU) {
                        app.lbmReference.SetParams(
                            LbmReference::Params{app.lbmParams.tau, app.lbmParams.inletVelocity,
                                                  app.lbmParams.reverseFlow, app.lbmParams.freeSlipY,
                                                  app.lbmParams.freeSlipZ, app.lbmParams.smagorinskyCs});
                    } else if (IsPanelBackend(app.lbmBackend)) {
                        app.panelSolver.SetParams(
                            PanelSolver::Params{app.lbmParams.inletVelocity, app.lbmParams.reverseFlow});
                    } else {
                        app.potentialFlow.SetParams(
                            PotentialFlowSolver::Params{app.lbmParams.inletVelocity, app.lbmParams.reverseFlow});
                    }
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Which face of the domain is the inlet — i.e. which way\n"
                    "the flow goes, as seen from the Right view (View ->\n"
                    "Right). Swaps live, no re-initialize needed, same as\n"
                    "Tau or Inlet velocity.");
            }

            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Inlet velocity (mm/s)", &app.lbmInletVelocityMmPerS, 50.0f, 1.0f, 200000.0f, "%.0f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Real-world inflow speed. The solver itself always runs\n"
                    "at a fixed, numerically-safe lattice velocity — this\n"
                    "just relabels how much real time one lattice step\n"
                    "represents, so it's free to change even while running\n"
                    "(10000 mm/s = 10 m/s ≈ 36 km/h, a typical RC airspeed).");
            }

            if (isLbm) {
                // app.lbmTauAtomic — see its own comment — is read here
                // lock-free instead of locking app.lbmMutex every frame
                // (ImGui's SliderFloat needs a plain float* to mutate
                // directly, so it operates on this local copy either
                // way, atomic-sourced or not).
                float tau = app.lbmTauAtomic.load(std::memory_order_relaxed);
                ImGui::SetNextItemWidth(140);
                if (ImGui::SliderFloat("Tau", &tau, kMinTau, 2.0f, "%.5f")) {
                    std::lock_guard<std::mutex> lock(app.lbmMutex);
                    app.lbmParams.tau = tau;
                    app.lbmTauAtomic.store(tau, std::memory_order_relaxed);
                    if (app.lbmValid) {
                        if (app.lbmBackend == SolverBackend::GPU) {
                            app.lbmSolver.SetParams(app.lbmParams);
                        } else {
                            app.lbmReference.SetParams(LbmReference::Params{
                                app.lbmParams.tau, app.lbmParams.inletVelocity, app.lbmParams.reverseFlow,
                                app.lbmParams.freeSlipY, app.lbmParams.freeSlipZ, app.lbmParams.smagorinskyCs});
                        }
                    }
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Relaxation time — a dimensionless numerical-stability\n"
                        "knob, lower = less viscous but less stable, must stay\n"
                        "above 0.5. See the equivalent viscosity below to help\n"
                        "calibrate it against a real fluid.");
                }
                float viscosity = LatticeTauToViscosityMm2PerSecond(app, tau);
                ImGui::TextDisabled("≈ kinematic viscosity: %.3g mm²/s  (air ≈ 15 mm²/s)", viscosity);
                if (ImGui::SmallButton("Match air viscosity")) {
                    ApplyTau(app, LatticeTauMatchingAir(app));
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Sets Tau so the viscosity equals sea-level air (15 mm²/s)\nat the current inlet speed and voxel size. Very low Tau is\nstabilised by a Smagorinsky sub-grid turbulence model.");
                }
                ImGui::SetNextItemWidth(140);
                if (ImGui::SliderFloat("Smagorinsky Cs", &app.lbmParams.smagorinskyCs, 0.0f, 0.3f, "%.3f")) {
                    ApplyTau(app, app.lbmParams.tau);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Sub-grid turbulence strength. 0 = off (plain regularised LBM).\nRaise (0.15-0.2) if low-Tau runs go unstable; lower for less\nadded numerical viscosity.");
                }
            }

            // Both LBM backends support this now (GPU and CPU reference) —
            // not Naive (PotentialFlowSolver), which has no concept of a
            // domain-edge boundary condition to begin with.
            if (isLbm) {
                auto applyFreeSlip = [&]() {
                    std::lock_guard<std::mutex> lock(app.lbmMutex);
                    if (app.lbmValid) {
                        if (app.lbmBackend == SolverBackend::GPU) {
                            app.lbmSolver.SetParams(app.lbmParams);
                        } else {
                            app.lbmReference.SetParams(LbmReference::Params{
                                app.lbmParams.tau, app.lbmParams.inletVelocity, app.lbmParams.reverseFlow,
                                app.lbmParams.freeSlipY, app.lbmParams.freeSlipZ, app.lbmParams.smagorinskyCs});
                        }
                    }
                };
                if (ImGui::Checkbox("Free-slip Y walls", &app.lbmParams.freeSlipY)) applyFreeSlip();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Off (default): the domain's lateral (grid-y) side\n"
                        "walls use no-slip bounce-back, same as the model's\n"
                        "own surface — a real boundary layer forms against\n"
                        "them, like a physical wind tunnel's walls. On: those\n"
                        "walls become free-slip (specular/mirror reflection)\n"
                        "instead — only the velocity component straight into\n"
                        "the wall is reflected, flow along it passes through\n"
                        "unslowed, reading as an open domain edge rather than\n"
                        "a real wall. Applies live, no re-initialize needed.");
                }
                ImGui::SameLine();
                if (ImGui::Checkbox("Free-slip Z walls", &app.lbmParams.freeSlipZ)) applyFreeSlip();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Same as Free-slip Y walls, for the top/bottom (grid-z) side walls instead.");
                }
            }

            ImGui::SetNextItemWidth(140);
            // Naive's sweeps are far cheaper per-call than an LBM substep
            // (a 6-neighbour average vs. a full collide+stream over 15-19
            // directions — see PotentialFlowSolver::Step's comment), so a
            // useful range for it goes well past 20.
            ImGui::SliderInt(!isLbm ? "Sweeps/tick" : "Substeps/tick", &app.lbmSubstepsPerFrame, 1,
                              isNaive ? 500 : 20);
            if (ImGui::IsItemHovered()) {
                if (!isLbm) {
                    ImGui::SetTooltip(
                        "The solver runs continuously on its own background\n"
                        "thread, independent of the render/UI frame rate — this\n"
                        "is how many relaxation sweeps it does per iteration\n"
                        "of that loop. More sweeps per tick converges faster in\n"
                        "wall-clock time but makes each tick take longer.");
                } else {
                    ImGui::SetTooltip(
                        "The solver runs continuously on its own background\n"
                        "thread, independent of the render/UI frame rate — this\n"
                        "is how many fused collision+streaming substeps it does\n"
                        "per iteration of that loop.");
                }
            }

            if (isLbm) {
                ImGui::Checkbox("Start flow at mid-speed/pressure", &app.lbmSeedMidSpeed);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Seeds every cell at half the inlet speed (and a\n"
                        "Bernoulli-consistent pressure) instead of quiescent\n"
                        "rho=1, u=0 — reaches a representative-looking state\n"
                        "faster, since the whole domain isn't starting from\n"
                        "rest. Not a physically real initial condition (nothing\n"
                        "produced it via the actual dynamics), so early steps\n"
                        "after it are relaxing away an artificial start, same\n"
                        "caveat a quiescent start's own early transient has,\n"
                        "just a different one. Takes effect on the next\n"
                        "Initialize/Reset Solver below.");
                }

                ImGui::Checkbox("Warm start from coarse grid", &app.warmStartEnabled);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Before seeding the real grid, voxelizes the same\n"
                        "mesh again at a much coarser resolution, runs a\n"
                        "quick solve on it, and nearest-samples ITS field\n"
                        "onto the real grid as the seed instead of the\n"
                        "uniform value above — a rough approximation of\n"
                        "where the wake/acceleration regions roughly are\n"
                        "beats starting from nothing, even if it's not an\n"
                        "exact match. Not dynamically identical to the real\n"
                        "grid though: tau is fixed regardless of\n"
                        "resolution, so the coarse run's effective Reynolds\n"
                        "number ends up far lower (the model is far fewer\n"
                        "cells across there) — a rough head start, not a\n"
                        "shortcut to the fine grid's actual steady state.\n"
                        "Takes effect on the next Initialize/Reset Solver\n"
                        "below; no effect on Naive, which solves directly\n"
                        "rather than relaxing toward equilibrium over many\n"
                        "steps the way LBM does.");
                }
                if (app.warmStartEnabled) {
                    ImGui::SetNextItemWidth(140);
                    ImGui::SliderInt("Warm start steps", &app.warmStartSteps, 10, 10000, "%d",
                                      ImGuiSliderFlags_Logarithmic);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("How many steps the coarse grid runs before being sampled onto the real one.");
                    }
                    ImGui::SetNextItemWidth(140);
                    ImGui::SliderInt("Resolution divisor", &app.warmStartResolutionDivisor, 2, 50);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "The coarse grid's Resolution is the real\n"
                            "grid's own divided by this — e.g. 10 means a\n"
                            "coarse grid 1/10th the resolution (so ~1/1000th\n"
                            "the cells) of the real one.");
                    }
                }
            }

            ImGui::BeginDisabled(busy);
            if (ImGui::Button("Initialize / Reset Solver")) {
                StartInitSolverTask(app);
            }
            ImGui::EndDisabled();

            if (!app.lbmValid) {
                // Skipped while busy: this is exactly the read that would
                // otherwise block the render loop for Initialize/Reset
                // Solver's entire first-run duration, since app.lbmValid
                // starts false and only flips true once Init finishes
                // (still holding the lock the whole time) — the busy
                // overlay (drawn later this same frame) is what actually
                // tells the user something's happening in that case, not
                // this text. Unlike the reverseFlow/tau/stats displays
                // above and below (now read from lock-free atomics
                // instead — see app.lbmTauAtomic's comment), this one's
                // only ever reached while !app.lbmValid, which is false
                // throughout ordinary running — so it was never the
                // per-frame-during-normal-operation contention source
                // those were; the plain `busy` guard here is enough.
                static std::string initErrorDisplay;
                if (!busy) {
                    std::lock_guard<std::mutex> lock(app.lbmMutex);
                    initErrorDisplay = app.lbmInitError;
                }
                const std::string& initError = initErrorDisplay;
                if (!initError.empty()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Failed to initialize:");
                    ImGui::TextWrapped("%s", initError.c_str());
                    if (app.lbmBackend == SolverBackend::GPU && initError.find("too large") != std::string::npos) {
                        ImGui::TextDisabled(
                            "Try the CPU backend above instead — it has no 4GiB-per-buffer\n"
                            "ceiling, only system RAM, though it's far slower. Or Naive,\n"
                            "for an even higher resolution at the cost of no wake/viscosity.");
                    }
                }
            }

            if (app.lbmValid) {
                ImGui::Checkbox("Build overlay on Run", &app.buildOverlayOnRun);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "The moment Run is pressed, also kick off a Build\n"
                        "Overlay (same as the button below) instead of leaving\n"
                        "the overlay empty/stale until a separate manual click.");
                }
                if (ImGui::Checkbox("Auto-pause on flow divergence (NaN)", &app.autoPauseOnDivergence)) {
                    app.autoPauseOnDivergenceAtomic.store(app.autoPauseOnDivergence, std::memory_order_relaxed);
                }

                ImGui::BeginDisabled(busy);
                if (ImGui::Button(app.lbmRunning ? "Pause" : "Run")) {
                    bool startingRun = !app.lbmRunning;
                    app.lbmRunning = !app.lbmRunning;
                    if (startingRun) {
                        // Feeds DrawProcessingWindow's "elapsed" readout —
                        // UI-thread-only, same as app.taskStartTicks.
                        app.lbmRunStartTicks = SDL_GetTicks();
                        app.showProcessingWindow = true;
                        if (app.buildOverlayOnRun) StartBuildOverlayTask(app);
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Step")) {
                    if (app.lbmBackend == SolverBackend::GPU) {
                        StepSolverLocked(app, app.lbmSubstepsPerFrame);
                    } else if (app.lbmBackend == SolverBackend::CPU) {
                        StepReferenceLocked(app, app.lbmSubstepsPerFrame);
                    } else if (IsPanelBackend(app.lbmBackend)) {
                        StepPanelLocked(app, app.lbmSubstepsPerFrame);
                    } else {
                        StepPotentialFlowLocked(app, app.lbmSubstepsPerFrame);
                    }
                    app.lbmStepCount += app.lbmSubstepsPerFrame;
                }
                ImGui::SameLine();
                if (ImGui::Button("Refresh Stats")) {
                    TriggerRefreshStats(app);
                }
                ImGui::EndDisabled();

                // Reconstructed from the lock-free *Atomic mirrors (see
                // their own comment on AppState::lbmLastStats) instead of
                // locking app.lbmMutex every frame to copy app.lbmLastStats
                // directly — this read is unconditional every frame (not
                // gated by app.lbmValid the way initErrorDisplay above
                // is), so during ordinary running+auto-update at high
                // resolution it was contending with StepSolverPaced and
                // ReadbackFieldPacedGPU both re-acquiring that same mutex
                // continuously — see app.lbmTauAtomic's comment for the
                // full reasoning.
                LbmSolver::Stats statsCopy;
                statsCopy.avgDensity = app.lbmStatsAvgDensityAtomic.load(std::memory_order_relaxed);
                statsCopy.minDensity = app.lbmStatsMinDensityAtomic.load(std::memory_order_relaxed);
                statsCopy.maxDensity = app.lbmStatsMaxDensityAtomic.load(std::memory_order_relaxed);
                statsCopy.avgSpeed = app.lbmStatsAvgSpeedAtomic.load(std::memory_order_relaxed);
                statsCopy.maxSpeed = app.lbmStatsMaxSpeedAtomic.load(std::memory_order_relaxed);
                statsCopy.hasNaN = app.lbmStatsHasNaNAtomic.load(std::memory_order_relaxed);
                statsCopy.valid = app.lbmStatsValidAtomic.load(std::memory_order_relaxed);
                if (isNaive) {
                    ImGui::Text("Model: potential flow (CPU)   Sweeps: %d", app.lbmStepCount.load());
                } else if (isPanel) {
                    ImGui::Text("Model: 3D Panel + Wake (%s)   Sweeps: %d",
                                app.lbmBackend == SolverBackend::PanelGPU ? "GPU" : "CPU", app.lbmStepCount.load());
                } else {
                    ImGui::Text("Model: %s (%s)   Steps: %d", LatticeModelName(app.lbmModel),
                                app.lbmBackend == SolverBackend::GPU ? "GPU" : "CPU", app.lbmStepCount.load());
                }
                if (statsCopy.valid) {
                    ImGui::Text("Density: avg %.4f  [%.4f, %.4f]", statsCopy.avgDensity, statsCopy.minDensity,
                                statsCopy.maxDensity);
                    ImGui::Text("Speed: avg %.0f mm/s   max %.0f mm/s",
                                LatticeSpeedToMmPerSecond(app, statsCopy.avgSpeed),
                                LatticeSpeedToMmPerSecond(app, statsCopy.maxSpeed));
                } else {
                    ImGui::TextDisabled("Press \"Refresh Stats\" to read back diagnostics.");
                }
                if (statsCopy.hasNaN || app.lbmFlowDivergedAtomic.load(std::memory_order_relaxed) || app.overlayHasNaN) {
                    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.45f, 1.0f),
                                        "FLOW DIVERGED (NaN / purple cells) — raise Smagorinsky Cs or Tau.");
                }

                // Real-time Lift, Drag & Peak Surface Stress readout
                // (computed from Wen et al. / Ladd Bouzidi momentum
                // exchange stored in macroOut's solid cells; updated by
                // both "Refresh Stats" and overlay auto-update / "Build
                // Overlay").
                if (app.lbmForcesValidAtomic.load(std::memory_order_relaxed)) {
                    float rawFx = app.lbmForceXAtomic.load(std::memory_order_relaxed);
                    float rawFy = app.lbmForceYAtomic.load(std::memory_order_relaxed);
                    float rawFz = app.lbmForceZAtomic.load(std::memory_order_relaxed);
                    float maxStressLat = app.lbmMaxStressAtomic.load(std::memory_order_relaxed);

                    // Grid-x is the flow axis (world Y); when flow is
                    // reversed (-Y), positive Drag points along -x.
                    float dragLat = app.lbmParams.reverseFlow ? -rawFx : rawFx;
                    float liftLat = rawFz;
                    float sideLat = rawFy;

                    float dragN = LatticeForceToNewtons(app, dragLat);
                    float liftN = LatticeForceToNewtons(app, liftLat);
                    float sideN = LatticeForceToNewtons(app, sideLat);
                    constexpr float kNewtonsToGramsForce = 1000.0f / 9.80665f;
                    float peakStressPa = LatticeStressToPascals(app, maxStressLat);

                    ImGui::Spacing();
                    ImGui::TextUnformatted("Aerodynamic Forces (sea-level air):");
                    ImGui::Text("  Drag: %.4g N  (%.2f gf)", dragN, dragN * kNewtonsToGramsForce);
                    ImGui::Text("  Lift: %.4g N  (%.2f gf)", liftN, liftN * kNewtonsToGramsForce);
                    if (std::fabs(dragLat) > 1e-6f) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("  L/D: %.2f", liftLat / dragLat);
                    }
                    ImGui::Text("  Side: %.4g N  (%.2f gf)   Peak stress: %.3g Pa",
                                sideN, sideN * kNewtonsToGramsForce, peakStressPa);
                    ImGui::Text("  Centre of Pressure: (%.1f, %.1f, %.1f) mm",
                                app.lbmCpXAtomic.load(std::memory_order_relaxed),
                                app.lbmCpYAtomic.load(std::memory_order_relaxed),
                                app.lbmCpZAtomic.load(std::memory_order_relaxed));
                    ImGui::Checkbox("Show Lift arrow above CoP", &app.showLiftArrow);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Integrated over all solid boundary links via\n"
                            "Wen et al. / Ladd momentum exchange using the\n"
                            "Bouzidi SDF-interpolated bounce-back populations,\n"
                            "scaled to sea-level air (1.225 kg/m³) at the\n"
                            "current inlet speed and voxel scale.");
                    }
                }

                if (ImGui::Button("Aero Polars (XFLR5-Style)...", ImVec2(220.0f, 0.0f))) {
                    app.showAeroPolarsWindow = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Open interactive XFLR5-style multi-point polar sweeps\n"
                                      "and flight envelope curves (CL, CD, L/D, CM vs alpha).");
                }

                ImGui::Separator();
                ImGui::TextUnformatted("Flow Overlay");

                int styleIndex = static_cast<int>(app.overlayStyle);
                const char* styleNames[] = {"Slice", "Voxel Cloud", "Particles (smoke trail)", "Airframe Stress Map", "3D Iso-Surface", "Volumetric Raymarching"};
                ImGui::SetNextItemWidth(220);
                if (ImGui::Combo("Style", &styleIndex, styleNames, 6)) {
                    app.overlayStyle = static_cast<OverlayStyle>(styleIndex);
                    app.overlayStyleAtomic.store(static_cast<int>(app.overlayStyle), std::memory_order_relaxed);
                    StartResliceTask(app);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Slice: one colored cross-section, exact per cell —\n"
                        "colored by whatever Color By is set to below.\n"
                        "Voxel Cloud: the whole volume as semi-transparent\n"
                        "cubes, always colored by Speed with opacity driven\n"
                        "by Pressure (see the window below) — a rough feel\n"
                        "for the field's 3D shape, not a precise reading.\n"
                        "Downsampled to a fixed budget regardless of\n"
                        "resolution, and no back-to-front sorting, so\n"
                        "blending order among the cubes isn't exact either.\n"
                        "Particles: a dye-injection style smoke trail —\n"
                        "particles seeded at the inlet, carried downstream\n"
                        "by the flow itself, colored by Speed with trail\n"
                        "opacity driven by Pressure (the same window).\n"
                        "Airframe Stress Map: projects exposed solid boundary\n"
                        "faces onto the sub-cell SDF surface and colors the\n"
                        "skin by local aerodynamic stress (Pa).\n"
                        "3D Iso-Surface: parallel Marching Cubes extraction of\n"
                        "smooth 3D boundary surfaces at chosen speed or pressure\n"
                        "thresholds (e.g. wake bubbles or vortex cores).\n"
                        "Volumetric Raymarching: view-aligned continuous volume\n"
                        "integration through the velocity or pressure field.");
                }

                if (app.overlayStyle == OverlayStyle::StressMap) {
                    ImGui::TextDisabled("Colored by local surface stress (Pa) on the SDF-projected skin.");
                } else {
                    // Affects any pressure-driven color/alpha below, regardless
                    // of style — see AppState's comment on detrendPressure.
                    // Toggling it re-seeds the Voxel Cloud alpha window (its
                    // old values are in the wrong basis otherwise) and
                    // re-slices immediately.
                    if (ImGui::Checkbox("Remove channel pressure gradient", &app.detrendPressure)) {
                        app.detrendPressureAtomic.store(app.detrendPressure, std::memory_order_relaxed);
                        app.voxelAlphaRangeInitialized = false;
                        StartResliceTask(app);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "The no-slip side walls mean flow needs a pressure\n"
                            "gradient along the channel just to overcome\n"
                            "viscous friction, like a pipe — often larger than\n"
                            "the model's own local stagnation/wake effect. This\n"
                            "subtracts each cross-section's own mean pressure\n"
                            "before coloring/fading, removing that gradient so\n"
                            "the local effect is what shows.");
                    }
                }

                if (app.overlayStyle == OverlayStyle::Raymarch) {
                    int colorIndex = app.overlayColorMode == FieldColorMode::Speed ? 0 : 1;
                    const char* colorNames[] = {"Speed", "Pressure"};
                    ImGui::SetNextItemWidth(160);
                    if (ImGui::Combo("Volume field", &colorIndex, colorNames, 2)) {
                        app.overlayColorMode = colorIndex == 0 ? FieldColorMode::Speed : FieldColorMode::Pressure;
                        app.overlayColorModeAtomic.store(static_cast<int>(app.overlayColorMode), std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    ImGui::SetNextItemWidth(160);
                    if (ImGui::SliderFloat("Absorption Density", &app.raymarchDensity, 0.10f, 5.0f, "%.2fx")) {
                        app.raymarchDensityAtomic.store(app.raymarchDensity, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    ImGui::SetNextItemWidth(160);
                    if (ImGui::SliderFloat("Ambient Cutoff", &app.raymarchCutoff, 0.0f, 0.50f, "%.2f")) {
                        app.raymarchCutoffAtomic.store(app.raymarchCutoff, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    ImGui::SetNextItemWidth(160);
                    if (ImGui::SliderFloat("Contrast Gamma", &app.raymarchGamma, 0.5f, 3.0f, "%.2f")) {
                        app.raymarchGammaAtomic.store(app.raymarchGamma, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    int sliceChoice = (app.raymarchSliceCount <= 32) ? 0 : (app.raymarchSliceCount <= 64 ? 1 : (app.raymarchSliceCount <= 96 ? 2 : 3));
                    const char* sliceNames[] = {"32 Slices (Fast)", "64 Slices (Balanced)", "96 Slices (Fine)", "128 Slices (Ultra)"};
                    ImGui::SetNextItemWidth(170);
                    if (ImGui::Combo("Sampling Quality", &sliceChoice, sliceNames, 4)) {
                        const int counts[] = {32, 64, 96, 128};
                        app.raymarchSliceCount = counts[sliceChoice];
                        app.raymarchSliceCountAtomic.store(app.raymarchSliceCount, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                }

                if (app.overlayStyle == OverlayStyle::IsoSurface) {
                    int colorIndex = app.overlayColorMode == FieldColorMode::Speed ? 0 : 1;
                    const char* colorNames[] = {"Speed", "Pressure"};
                    ImGui::SetNextItemWidth(160);
                    if (ImGui::Combo("Contour field", &colorIndex, colorNames, 2)) {
                        app.overlayColorMode = colorIndex == 0 ? FieldColorMode::Speed : FieldColorMode::Pressure;
                        app.overlayColorModeAtomic.store(static_cast<int>(app.overlayColorMode), std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    float lo = app.overlayColorMode == FieldColorMode::Speed ? app.cachedSpeedMin :
                               (app.detrendPressure ? app.cachedPressureDetrendMin : app.cachedPressureMin);
                    float hi = app.overlayColorMode == FieldColorMode::Speed ? app.cachedSpeedMax :
                               (app.detrendPressure ? app.cachedPressureDetrendMax : app.cachedPressureMax);
                    float currentVal = lo + app.isoThreshold * (hi - lo);

                    ImGui::SetNextItemWidth(160);
                    if (ImGui::SliderFloat("Iso Threshold", &app.isoThreshold, 0.01f, 0.99f, "%.2f")) {
                        app.isoThresholdAtomic.store(app.isoThreshold, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                    if (app.overlayColorMode == FieldColorMode::Speed) {
                        float speedMmS = LatticeSpeedToMmPerSecond(app, currentVal);
                        ImGui::TextDisabled("Contour: %.1f mm/s", speedMmS);
                    } else {
                        ImGui::TextDisabled("Contour: %.4g lat. p", currentVal);
                    }

                    int encloseMode = app.isoEncloseGreater ? 1 : 0;
                    const char* encloseNames[] = {"<= Iso (Wake / Suction)", ">= Iso (Jet / Stagnation)"};
                    ImGui::SetNextItemWidth(190);
                    if (ImGui::Combo("Enclose", &encloseMode, encloseNames, 2)) {
                        app.isoEncloseGreater = (encloseMode == 1);
                        app.isoEncloseGreaterAtomic.store(app.isoEncloseGreater, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    ImGui::SetNextItemWidth(160);
                    if (ImGui::SliderFloat("Surface Opacity", &app.isoAlpha, 0.10f, 1.0f, "%.2f")) {
                        app.isoAlphaAtomic.store(app.isoAlpha, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }

                    int strideChoice = app.isoStride == 1 ? 0 : 1;
                    const char* strideNames[] = {"Full (1x)", "Fast (2x Stride)"};
                    ImGui::SetNextItemWidth(160);
                    if (ImGui::Combo("Quality", &strideChoice, strideNames, 2)) {
                        app.isoStride = strideChoice == 0 ? 1 : 2;
                        app.isoStrideAtomic.store(app.isoStride, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                }

                if (app.overlayStyle == OverlayStyle::Slice) {
                    // SliceAxis::X/Y are grid axes, not world axes — grid-x
                    // maps to world Y (the solver's actual flow axis) and
                    // grid-y to world X (lateral) — see VoxelGrid::
                    // CellCenter's comment — so a slice fixing grid-x is
                    // perpendicular to flow (front-on) and one fixing
                    // grid-y is perpendicular to the lateral axis
                    // (side-on), the opposite pairing from when the grid
                    // was still world-identity-mapped. Z (grid and world
                    // agree there) is still the plan/top cut.
                    //
                    // Up to all 3 can be enabled at once — each row's own
                    // checkbox turns that plane on/off independently (see
                    // AppState::slicePlanes), and its slider only appears,
                    // and only then moves *that* plane, while it's on. The
                    // matching drag handle in the 3D view (see
                    // DrawSliceDragHandles) does the same thing by mouse.
                    const char* axisLabels[3] = {"X (front-on)", "Y (side-on)", "Z (plan/top)"};
                    for (size_t i = 0; i < 3; ++i) {
                        AppState::SlicePlaneState& plane = app.slicePlanes[i];
                        ImGui::PushID(static_cast<int>(i));
                        if (ImGui::Checkbox(axisLabels[i], &plane.enabled)) {
                            // Freshly enabling defaults to this axis's own
                            // grid center — same reasoning the lazy -1
                            // correction below exists for, just applied
                            // immediately instead of waiting a frame.
                            if (plane.enabled && plane.index < 0) {
                                plane.index = OverlaySliceDimension(app, kSliceAxes[i]) / 2;
                            }
                            // Stored immediately (not just in the once-per-
                            // frame sync below) so StartResliceTask's
                            // background task — which may start before that
                            // sync runs — sees the new value right away
                            // rather than one frame stale.
                            plane.enabledAtomic.store(plane.enabled, std::memory_order_relaxed);
                            plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
                            StartResliceTask(app);
                        }
                        if (plane.enabled) {
                            int sliceMax = std::max(OverlaySliceDimension(app, kSliceAxes[i]) - 1, 0);
                            if (plane.index < 0) {
                                plane.index = sliceMax / 2;
                                // Also sync the atomic mirror here, not just
                                // the plain field — BuildOverlayMeshFromField
                                // (which can run on a background thread
                                // before the user ever touches the Slider
                                // below) trusts this defaulting to have
                                // already happened by the time it reads
                                // indexAtomic; a background rebuild that ran
                                // first and saw -1 would have had
                                // BuildSliceMeshFromRange's own clamp(index,
                                // 0, dim-1) turn it into 0 instead — the
                                // slice appearing at one end rather than the
                                // center the UI shows.
                                plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
                            }
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(140);
                            // Re-slices immediately, even while paused — see
                            // StartResliceTask's comment; without it, this
                            // slider did nothing until the next auto-update
                            // or manual rebuild.
                            if (ImGui::SliderInt("##pos", &plane.index, 0, sliceMax)) {
                                plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
                                StartResliceTask(app);
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("Center")) {
                                plane.index = sliceMax / 2;
                                plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
                                StartResliceTask(app);
                            }
                        }
                        ImGui::PopID();
                    }

                    // "Color by" only matters when the colored cross-
                    // section itself is actually drawn — hidden once
                    // "Also show particles" replaces it with the smoke
                    // trail below, which is always colored by Speed
                    // (same as the Particles style).
                    if (!app.sliceShowParticles) {
                        int colorIndex = app.overlayColorMode == FieldColorMode::Speed ? 0 : 1;
                        const char* colorNames[] = {"Speed", "Pressure"};
                        ImGui::SetNextItemWidth(180);
                        if (ImGui::Combo("Color by", &colorIndex, colorNames, 2)) {
                            app.overlayColorMode = colorIndex == 0 ? FieldColorMode::Speed : FieldColorMode::Pressure;
                            app.overlayColorModeAtomic.store(static_cast<int>(app.overlayColorMode),
                                                               std::memory_order_relaxed);
                            StartResliceTask(app);
                        }
                    }

                    if (ImGui::Checkbox("Show particles", &app.sliceShowParticles)) {
                        // Stored immediately — see each plane row's own
                        // comment above for why. StartResliceTask rebuilds
                        // (or, once this is on, skips building — see
                        // BuildMeshForStyle) the colored mesh right away;
                        // UpdateSmokeTrailAndUpload separately notices
                        // app.overlayStyle/sliceShowParticles now call for
                        // the confined path and Resets each enabled plane's
                        // own smokeTrails[] instance on its very next call,
                        // so no explicit smokeTrail call is needed here.
                        app.sliceShowParticlesAtomic.store(app.sliceShowParticles, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Replaces each enabled plane's colored cross-\n"
                            "section with its own Particles smoke trail\n"
                            "confined to that plane: particles move within\n"
                            "the plane (the flow component through it is\n"
                            "ignored), colored by Speed, using the particle\n"
                            "and opacity controls below — same as the\n"
                            "Particles style, just restricted to each\n"
                            "plane, with its own independent stream.");
                    }
                }

                // Particle controls and the pressure-driven alpha window
                // below apply whenever particles are actually drawn this
                // frame — the Particles style itself, or Slice with "Also
                // show particles" on. Mirrors the render dispatch's own
                // showParticles condition further down.
                bool showingParticles = app.overlayStyle == OverlayStyle::Particles ||
                                         (app.overlayStyle == OverlayStyle::Slice && app.sliceShowParticles);
                if (showingParticles) {
                    ImGui::TextDisabled(app.smokeTrailPressureAlphaEnabled
                                             ? "Colored by Speed; trail opacity below is driven by Pressure."
                                             : "Colored by Speed; trail opacity is fixed (see checkbox below).");

                    ImGui::Checkbox("Fade trail opacity by pressure", &app.smokeTrailPressureAlphaEnabled);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Off (default): every trail segment renders at\n"
                            "a fixed opacity (still faded along the trail\n"
                            "itself, head to tail) — most particles stay\n"
                            "clearly visible rather than fading out across\n"
                            "ambient/undisturbed flow. On: also modulates\n"
                            "opacity by Pressure, using the same window as\n"
                            "Voxel Cloud below — the model's wake/\n"
                            "stagnation disturbance stays visible while\n"
                            "undisturbed flow fades out.");
                    }

                    bool countChanged =
                        ImGui::SliderInt("Particle count", &app.smokeTrailParticleCount, 100, 8000);
                    bool lengthChanged = ImGui::SliderInt("Trail length", &app.smokeTrailLength, 2, 30);
                    if (countChanged || lengthChanged) app.smokeTrailInitialized = false;  // forces a Reset
                    if (ImGui::Button("Reset particles")) app.smokeTrailInitialized = false;
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Re-seeds every particle at the inlet (or, with\n"
                            "\"Show particles\" on, the edge of the\n"
                            "slice plane) and lets the stream refill from\n"
                            "there over the next several frames, same as\n"
                            "when the particle count/trail length changes.");
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Scatter particles")) app.smokeTrailScatterRequested = true;
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Instantly jumps every particle to a fresh\n"
                            "random position across the WHOLE domain (or\n"
                            "whole slice plane, with \"Show particles\"\n"
                            "on) right now, instead of only at the inlet/\n"
                            "plane edge and waiting for the stream to\n"
                            "refill — a one-shot shuffle of the\n"
                            "visualization.");
                    }

                    ImGui::SetNextItemWidth(180);
                    ImGui::SliderFloat("Particle speed", &app.smokeTrailSpeedScale, 0.1f, 5.0f, "%.2fx");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Scales how far a particle travels per real\n"
                            "second through the flow's velocity field.\n"
                            "1.0x matches the real inlet speed (mm/s)\n"
                            "setting; purely a visualization control —\n"
                            "doesn't affect the solver or any other\n"
                            "style's reading of speed/pressure.");
                    }

                    bool puffing = app.smokeTrailPuffRemainingSeconds > 0.0f;
                    ImGui::BeginDisabled(puffing);
                    if (ImGui::Button("Puff")) {
                        app.smokeTrailPuffRemainingSeconds = app.smokeTrailPuffDurationSeconds;
                    }
                    ImGui::EndDisabled();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Briefly stops new particles entering at the\n"
                            "inlet. Whatever's already in flight keeps\n"
                            "moving and drains out the outlet, so the\n"
                            "stream thins to nothing and then refills —\n"
                            "reads as one puff of smoke rather than a\n"
                            "continuous plume.");
                    }
                    ImGui::SameLine();
                    if (puffing) {
                        ImGui::TextDisabled("puffing... %.1fs left", app.smokeTrailPuffRemainingSeconds);
                    } else {
                        ImGui::SetNextItemWidth(120);
                        ImGui::SliderFloat("Puff duration", &app.smokeTrailPuffDurationSeconds, 0.5f, 10.0f,
                                            "%.1fs");
                    }
                } else if (app.overlayStyle == OverlayStyle::VoxelCloud) {
                    ImGui::TextDisabled("Colored by Speed; opacity below is driven by Pressure.");
                }

                if (showingParticles || app.overlayStyle == OverlayStyle::VoxelCloud) {
                    // Lazily seeded from the field's actual pressure range
                    // the first time it's known, then left to the user —
                    // see AppState's comment on voxelAlphaRangeInitialized.
                    if (!app.voxelAlphaRangeInitialized) {
                        std::lock_guard<std::mutex> lock(app.lastFieldMutex);
                        if (app.cachedRangesValid) {
                            app.voxelAlphaPressureLo =
                                app.detrendPressure ? app.cachedPressureDetrendMin : app.cachedPressureMin;
                            app.voxelAlphaPressureHi =
                                app.detrendPressure ? app.cachedPressureDetrendMax : app.cachedPressureMax;
                            app.voxelAlphaRangeInitialized = true;
                        }
                    }

                    ImGui::SetNextItemWidth(180);
                    bool alphaChanged = ImGui::DragFloat("Transparent at/below", &app.voxelAlphaPressureLo, 0.001f,
                                                           -10.0f, app.voxelAlphaPressureHi, "%.4f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Pressure deviation (lattice units) at or below\n"
                            "which a cube/trail segment renders fully\n"
                            "invisible. Set this at or above the ambient/\n"
                            "free-flowing value to make undisturbed fluid\n"
                            "disappear, leaving only the regions the model\n"
                            "actually disturbs.");
                    }
                    ImGui::SetNextItemWidth(180);
                    alphaChanged |= ImGui::DragFloat("Opaque at/above", &app.voxelAlphaPressureHi, 0.001f,
                                                      app.voxelAlphaPressureLo, 10.0f, "%.4f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Pressure deviation at or above which a cube/\n"
                            "trail segment renders at the (still translucent)\n"
                            "alpha ceiling. Narrowing the gap to \"Transparent\n"
                            "at/below\" makes the fade sharper.");
                    }
                    if (app.cachedRangesValid) {
                        std::lock_guard<std::mutex> lock(app.lastFieldMutex);
                        float rangeLo = app.detrendPressure ? app.cachedPressureDetrendMin : app.cachedPressureMin;
                        float rangeHi = app.detrendPressure ? app.cachedPressureDetrendMax : app.cachedPressureMax;
                        ImGui::TextDisabled("actual pressure range: [%.4g, %.4g]", rangeLo, rangeHi);
                    }

                    if (ImGui::Checkbox("Highlight negative pressure (dark grey)",
                                         &app.voxelNegativePressureGrey)) {
                        app.voxelNegativePressureGreyAtomic.store(app.voxelNegativePressureGrey,
                                                                    std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "A below-ambient (negative deviation) cube/trail\n"
                            "segment renders a fixed dark grey at 30%% opacity\n"
                            "instead of the usual Speed color + alpha-window\n"
                            "fade — otherwise a mildly below-ambient cell can\n"
                            "fade all the way to invisible under the window\n"
                            "above, hiding exactly the suction/low-pressure\n"
                            "path this makes visible instead.");
                    }

                    if (alphaChanged) {
                        app.voxelAlphaPressureLoAtomic.store(app.voxelAlphaPressureLo, std::memory_order_relaxed);
                        app.voxelAlphaPressureHiAtomic.store(app.voxelAlphaPressureHi, std::memory_order_relaxed);
                        StartResliceTask(app);
                    }
                }

                ImGui::BeginDisabled(busy);
                if (ImGui::Button("Build Overlay")) {
                    StartBuildOverlayTask(app);
                }
                ImGui::EndDisabled();
                if (app.overlayValid) {
                    ImGui::SameLine();
                    // See DrawOverlayLegend's isSpeed comment — must agree
                    // with LegendRangeForStyle on which range app.overlayMin/
                    // Max actually is.
                    bool legendIsStress = app.overlayStyle == OverlayStyle::StressMap;
                    bool legendParticleLike = app.overlayStyle == OverlayStyle::Particles ||
                                              (app.overlayStyle == OverlayStyle::Slice && app.sliceShowParticles);
                    bool legendIsSpeed = !legendIsStress && !legendParticleLike &&
                                         (app.overlayStyle == OverlayStyle::VoxelCloud ||
                                          app.overlayColorMode == FieldColorMode::Speed);
                    if (legendIsStress) {
                        ImGui::TextDisabled("range [%.3g, %.3g] Pa", LatticeStressToPascals(app, app.overlayMin),
                                             LatticeStressToPascals(app, app.overlayMax));
                    } else if (legendIsSpeed) {
                        ImGui::TextDisabled("range [%.0f, %.0f] mm/s", LatticeSpeedToMmPerSecond(app, app.overlayMin),
                                             LatticeSpeedToMmPerSecond(app, app.overlayMax));
                    } else {
                        ImGui::TextDisabled("range [%.4g, %.4g]", app.overlayMin, app.overlayMax);
                    }
                    if (app.overlayHasNaN) {
                        ImGui::TextColored(ImVec4(1.0f, 0.1f, 1.0f, 1.0f),
                                            "NaN detected — solver diverged (magenta cells). Try raising tau.");
                    }
                }

                if (app.overlayStyle != OverlayStyle::StressMap) {
                    ImGui::Checkbox("Show model", &app.overlayShowModel);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Draws the original (unvoxelized) mesh alongside the\n"
                            "slice for context. The slice already skips solid\n"
                            "cells, and is properly depth-tested against the\n"
                            "model, so parts of it hidden behind the model from\n"
                            "this viewpoint won't show through.");
                    }
                }

                ImGui::Checkbox("Auto-update while running", &app.overlayAutoUpdate);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Periodically rebuilds the overlay while the solver\n"
                        "runs, so it animates instead of staying frozen at\n"
                        "the last Build. The GPU readback happens on the\n"
                        "solver's own background thread, not here, so it\n"
                        "never stalls the UI — tune the interval below if\n"
                        "the animation feels too choppy or too stale.");
                }
                ImGui::SetNextItemWidth(140);
                ImGui::SliderInt("Update interval (ms)", &app.overlayUpdateIntervalMs, 50, 2000);

                ImGui::SetNextItemWidth(140);
                ImGui::SliderInt("Warmup steps before first auto-update", &app.overlayAutoUpdateWarmupSteps, 0, 2000);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Holds off auto-update's very first readback until\n"
                        "this many steps have run since the last Initialize/\n"
                        "Reset Solver — so the first thing it shows is a few\n"
                        "seconds into a developed flow rather than the\n"
                        "barely-started transient right after pressing Run.\n"
                        "Purely a display-timing knob; the solver itself runs\n"
                        "identically either way.");
                }
            }
        }
    }
    ImGui::End();

    if (app.meshLoaded) {
        DrawOrientationGizmo(app);
        DrawScaleRuler(app);
        DrawLiftArrow(app);
        if (app.viewMode == ViewMode::FlowOverlay && app.overlayStyle == OverlayStyle::Slice) {
            DrawSliceDragHandles(app);
        }
        if (app.viewMode == ViewMode::FlowOverlay && app.overlayValid) DrawOverlayLegend(app);
    }
    DrawBusyOverlay(app);
    DrawProcessingWindow(app);
    DrawFlowDivergenceBanner(app, 0.0f, busy);
    DrawAeroPolarsWindow(app);
}

}  // namespace

int main(int, char**) {
    SDL_SetAppMetadata("FlowCraft", "0.1", "com.mattparsons.flowcraft");
    SDL_SetAppMetadataProperty(SDL_PROP_APP_METADATA_CREATOR_STRING, "Matt Parsons");
    SDL_SetAppMetadataProperty(SDL_PROP_APP_METADATA_COPYRIGHT_STRING, "FlowCraft v0.1 by Matt Parsons");

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return 1;
    }
    ConfigureMacOSAboutMenu();
    ConfigureMacOSAppIcon();

    int initialWidth = 1920;
    int initialHeight = 1080;
    SDL_DisplayID primaryDisplay = SDL_GetPrimaryDisplay();
    SDL_Rect usableBounds{};
    if (primaryDisplay != 0 &&
        (SDL_GetDisplayUsableBounds(primaryDisplay, &usableBounds) ||
         SDL_GetDisplayBounds(primaryDisplay, &usableBounds)) &&
        usableBounds.w > 0 && usableBounds.h > 0) {
        initialWidth = std::min(initialWidth, usableBounds.w);
        initialHeight = std::min(initialHeight, usableBounds.h);
    }

    SDL_WindowFlags windowFlags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    SDL_Window* window = SDL_CreateWindow("FlowCraft", initialWidth, initialHeight, windowFlags);
    if (!window) {
        SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
        return 1;
    }
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);

    SDL_Surface* iconSurface = SDL_CreateSurfaceFrom(kFlowCraftIconWidth, kFlowCraftIconHeight,
                                                     SDL_PIXELFORMAT_RGBA32,
                                                     const_cast<uint8_t*>(GetFlowCraftIconRgba()),
                                                     kFlowCraftIconWidth * 4);
    if (iconSurface) {
        SDL_SetWindowIcon(window, iconSurface);
        SDL_DestroySurface(iconSurface);
    }

    SDL_GPUDevice* gpuDevice = SDL_CreateGPUDevice(kEmbeddedShaderFormats | SDL_GPU_SHADERFORMAT_METALLIB, true, nullptr);
    if (!gpuDevice) {
        SDL_Log("SDL_CreateGPUDevice failed: %s", SDL_GetError());
        return 1;
    }
    if (!SDL_ClaimWindowForGPUDevice(gpuDevice, window)) {
        SDL_Log("SDL_ClaimWindowForGPUDevice failed: %s", SDL_GetError());
        return 1;
    }
    SDL_SetGPUSwapchainParameters(gpuDevice, window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_VSYNC);

    SDL_ShowWindow(window);

    AppState app;
    // Y on by default — matches the single-slice design's own default
    // (SliceAxis::Y) before it became 3 independently-enabled planes.
    app.slicePlanes[static_cast<size_t>(SliceAxis::Y)].enabled = true;
    app.slicePlanes[static_cast<size_t>(SliceAxis::Y)].enabledAtomic.store(true, std::memory_order_relaxed);
    app.gpuDevice = gpuDevice;
    if (!app.renderer.Init(gpuDevice, window)) {
        SDL_Log("MeshRenderer::Init failed");
        return 1;
    }
    if (!app.voxelRenderer.Init(gpuDevice, window)) {
        SDL_Log("MeshRenderer::Init (voxel) failed");
        return 1;
    }
    if (!app.overlayRenderer.Init(gpuDevice, window)) {
        SDL_Log("OverlayRenderer::Init failed");
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();

    {
        ImGuiIO& io = ImGui::GetIO();
        app.defaultFont = io.Fonts->AddFontDefault();

        ImFontConfig topazCfg;
        topazCfg.FontDataOwnedByAtlas = false;
        topazCfg.PixelSnapH = true;
        std::snprintf(topazCfg.Name, sizeof(topazCfg.Name), "Amiga Topaz");
        app.amigaTopazFont = io.Fonts->AddFontFromMemoryTTF(
            const_cast<uint8_t*>(kAmigaTopazFontData),
            static_cast<int>(kAmigaTopazFontDataSize),
            16.0f,
            &topazCfg);
    }

    ImGui_ImplSDL3_InitForSDLGPU(window);
    ImGui_ImplSDLGPU3_InitInfo initInfo = {};
    initInfo.Device = gpuDevice;
    initInfo.ColorTargetFormat = SDL_GetGPUSwapchainTextureFormat(gpuDevice, window);
    initInfo.MSAASamples = SDL_GPU_SAMPLECOUNT_1;
    ImGui_ImplSDLGPU3_Init(&initInfo);

    // Runs the solver on its own thread for the rest of the app's life —
    // see LbmWorkerThreadFunc's comment for why.
    std::thread lbmThread(LbmWorkerThreadFunc, &app);

    Uint64 lastTicks = SDL_GetTicks();
    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            ImGuiIO& io = ImGui::GetIO();

            if (event.type == SDL_EVENT_QUIT) running = false;
            if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window))
                running = false;

            if (!io.WantCaptureMouse) {
                if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT) {
                    // Slice drag handles only exist to pick while they're
                    // actually shown — same condition as DrawSliceDragHandles'
                    // own call site below.
                    std::optional<SliceAxis> picked;
                    if (app.viewMode == ViewMode::FlowOverlay && app.overlayStyle == OverlayStyle::Slice) {
                        picked = PickSliceHandle(app, ImVec2(event.button.x, event.button.y));
                    }
                    if (picked) {
                        // Only an already-enabled plane's handle can ever
                        // be picked (see ComputeSliceHandles/
                        // PickSliceHandle) — enabling one is the panel
                        // checkbox's job, not a side effect of grabbing a
                        // handle, so there's nothing to switch here.
                        app.draggingSliceHandle = true;
                        app.draggingSliceHandleAxis = *picked;
                        app.sliceDragFractionalCells = 0.0f;
                    } else {
                        app.draggingCamera = true;
                        app.cameraTransitioning = false;
                    }
                }
                if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_LEFT) {
                    app.draggingCamera = false;
                    app.draggingSliceHandle = false;
                }
                if (event.type == SDL_EVENT_MOUSE_MOTION && app.draggingSliceHandle) {
                    ApplySliceHandleDrag(app, event.motion.xrel, event.motion.yrel);
                } else if (event.type == SDL_EVENT_MOUSE_MOTION && app.draggingCamera) {
                    app.cameraTransitioning = false;
                    app.camera.OnDrag(event.motion.xrel, event.motion.yrel);
                    Vec3 forward = app.camera.GetBasis().forward;
                    app.cameraForwardXAtomic.store(forward.x, std::memory_order_relaxed);
                    app.cameraForwardYAtomic.store(forward.y, std::memory_order_relaxed);
                    app.cameraForwardZAtomic.store(forward.z, std::memory_order_relaxed);
                    if (app.viewMode == ViewMode::FlowOverlay &&
                        (app.overlayStyle == OverlayStyle::VoxelCloud || app.overlayStyle == OverlayStyle::Raymarch)) {
                        TryFastReslice(app);
                    }
                }
                if (event.type == SDL_EVENT_MOUSE_WHEEL) {
                    app.camera.OnScroll(event.wheel.y);
                    if (app.viewMode == ViewMode::FlowOverlay &&
                        (app.overlayStyle == OverlayStyle::VoxelCloud || app.overlayStyle == OverlayStyle::Raymarch)) {
                        TryFastReslice(app);
                    }
                }
            } else {
                app.draggingCamera = false;
                app.draggingSliceHandle = false;
            }
        }

        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
            SDL_Delay(10);
            continue;
        }

        Uint64 nowTicks = SDL_GetTicks();
        float dt = static_cast<float>(nowTicks - lastTicks) / 1000.0f;
        lastTicks = nowTicks;
        PollKeyboardRotation(app, dt);
        UpdateCameraTransition(app, dt);
        UpdateSceneRadiusForScale(app);

        // The solver thread (or a Build-Overlay task thread) already did
        // the readback *and* the CPU-heavy mesh build (see
        // BuildOverlayMeshFromField); all that's left here is the actual
        // GPU upload, which is cheap and must stay on this thread anyway
        // (overlayRenderer is read every frame by the render loop below).
        uint64_t meshVersion = app.overlayMeshVersion.load(std::memory_order_acquire);
        if (meshVersion != app.overlayConsumedMeshVersion) {
            OverlayMesh mesh;
            float lo, hi;
            bool hasNaN;
            {
                std::lock_guard<std::mutex> lock(app.overlayMeshMutex);
                mesh = std::move(app.pendingOverlayMesh);
                lo = app.pendingOverlayMin;
                hi = app.pendingOverlayMax;
                hasNaN = app.pendingOverlayHasNaN;
            }
            app.overlayRenderer.UploadMesh(mesh);
            app.overlayMin = lo;
            app.overlayMax = hi;
            app.overlayHasNaN = hasNaN;
            app.overlayValid = true;
            app.overlayConsumedMeshVersion = meshVersion;
        }

        // Same pattern for StartVoxelizeTask's result: the heavy CPU work
        // already happened on the background thread, so applying it here
        // is just a handful of field copies plus the (cheap, async)
        // voxelRenderer upload — never the slow part.
        uint64_t voxelVersion = app.voxelResultVersion.load(std::memory_order_acquire);
        // The solver is stopped as soon as a new grid is waiting (it's
        // about to be invalidated just below anyway), so the solver thread
        // stops launching overlay updates; the swap itself then only
        // happens on a frame where no background reader still holds the
        // old grid — try_lock, so a long overlay build in flight delays
        // the swap by a few frames instead of freezing the window. See
        // voxelGridMutex's comment.
        std::unique_lock<std::shared_mutex> voxelGridLock(app.voxelGridMutex, std::defer_lock);
        if (voxelVersion != app.voxelConsumedResultVersion) {
            app.lbmValid = false;
            app.lbmRunning = false;
        }
        if (voxelVersion != app.voxelConsumedResultVersion && voxelGridLock.try_lock()) {
            VoxelGrid grid;
            Mesh preview;
            size_t solidCount;
            {
                std::lock_guard<std::mutex> lock(app.voxelResultMutex);
                grid = std::move(app.pendingVoxelGrid);
                preview = std::move(app.pendingVoxelPreview);
                solidCount = app.pendingVoxelSolidCount;
            }
            app.voxelGrid = std::move(grid);
            voxelGridLock.unlock();
            app.voxelSolidCount = solidCount;
            app.voxelRenderer.UploadMesh(preview);
            app.voxelGridValid = true;
            app.viewMode = ViewMode::VoxelGrid;

            // The old solver (if any) was sized for the previous grid; it
            // must be explicitly re-initialized before running again, and
            // any overlay built from it is now stale too.
            app.lbmValid = false;
            app.lbmRunning = false;
            app.overlayValid = false;
            app.polarResult = {};
            app.sectionPolarResult = {};
            app.panelSolver.Shutdown();
            // Every plane's index defaulted to the *new* grid's own center
            // right here, immediately — both the plain field and its
            // atomic mirror — rather than left at -1 for DrawUI's own
            // lazy per-plane correction to notice later (see the Slice UI
            // block). That lazy correction (still kept, as a harmless
            // no-op once this has already run) only ever touched the
            // plain field, and only once the user's frame reached that
            // part of DrawUI; a background overlay rebuild triggered by
            // "Build overlay on Run" could start — and read the still-
            // stale -1 from the atomic — before DrawUI's own code got
            // that far in the very same frame, or even in the same frame
            // the Run press happened. Reading -1 isn't a crash, just
            // wrong: BuildSliceMeshFromRange's own clamp(index, 0, dim-1)
            // turns it into 0, a slice pinned to one end of the domain
            // instead of the center the UI meanwhile showed correctly.
            for (size_t i = 0; i < 3; ++i) {
                AppState::SlicePlaneState& plane = app.slicePlanes[i];
                plane.index = OverlaySliceDimension(app, kSliceAxes[i]) / 2;
                plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
            }

            app.voxelConsumedResultVersion = voxelVersion;
        }

        ImGui_ImplSDLGPU3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        DrawUI(app);
        ImGui::Render();
        ImDrawData* drawData = ImGui::GetDrawData();
        bool minimized = drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f;

        // Refresh the worker thread's view of UI-controlled knobs (see
        // AppState's comment on the *Atomic fields — it must never read the
        // plain ones directly, since those are only safe from this thread).
        app.lbmSubstepsAtomic.store(app.lbmSubstepsPerFrame, std::memory_order_relaxed);
        app.lbmBackendAtomic.store(static_cast<int>(app.lbmBackend), std::memory_order_relaxed);
        app.overlayAutoUpdateAtomic.store(app.overlayAutoUpdate && app.viewMode == ViewMode::FlowOverlay,
                                          std::memory_order_relaxed);
        app.overlayIntervalAtomic.store(app.overlayUpdateIntervalMs, std::memory_order_relaxed);
        app.overlayAutoUpdateWarmupStepsAtomic.store(app.overlayAutoUpdateWarmupSteps, std::memory_order_relaxed);
        app.overlayStyleAtomic.store(static_cast<int>(app.overlayStyle), std::memory_order_relaxed);
        for (AppState::SlicePlaneState& plane : app.slicePlanes) {
            plane.enabledAtomic.store(plane.enabled, std::memory_order_relaxed);
            plane.indexAtomic.store(plane.index, std::memory_order_relaxed);
        }
        app.overlayColorModeAtomic.store(static_cast<int>(app.overlayColorMode), std::memory_order_relaxed);
        app.sliceShowParticlesAtomic.store(app.sliceShowParticles, std::memory_order_relaxed);
        app.voxelAlphaPressureLoAtomic.store(app.voxelAlphaPressureLo, std::memory_order_relaxed);
        app.voxelAlphaPressureHiAtomic.store(app.voxelAlphaPressureHi, std::memory_order_relaxed);
        app.detrendPressureAtomic.store(app.detrendPressure, std::memory_order_relaxed);
        app.voxelNegativePressureGreyAtomic.store(app.voxelNegativePressureGrey, std::memory_order_relaxed);
        app.isoThresholdAtomic.store(app.isoThreshold, std::memory_order_relaxed);
        app.isoAlphaAtomic.store(app.isoAlpha, std::memory_order_relaxed);
        app.isoEncloseGreaterAtomic.store(app.isoEncloseGreater, std::memory_order_relaxed);
        app.isoStrideAtomic.store(app.isoStride, std::memory_order_relaxed);
        app.raymarchSliceCountAtomic.store(app.raymarchSliceCount, std::memory_order_relaxed);
        app.raymarchDensityAtomic.store(app.raymarchDensity, std::memory_order_relaxed);
        app.raymarchCutoffAtomic.store(app.raymarchCutoff, std::memory_order_relaxed);
        app.raymarchGammaAtomic.store(app.raymarchGamma, std::memory_order_relaxed);
        {
            Vec3 forward = app.camera.GetBasis().forward;
            app.cameraForwardXAtomic.store(forward.x, std::memory_order_relaxed);
            app.cameraForwardYAtomic.store(forward.y, std::memory_order_relaxed);
            app.cameraForwardZAtomic.store(forward.z, std::memory_order_relaxed);
            Vec3 eye = app.camera.Eye();
            app.cameraPosXAtomic.store(eye.x, std::memory_order_relaxed);
            app.cameraPosYAtomic.store(eye.y, std::memory_order_relaxed);
            app.cameraPosZAtomic.store(eye.z, std::memory_order_relaxed);
        }

        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(gpuDevice);

        SDL_GPUTexture* swapchainTexture = nullptr;
        uint32_t width = 0, height = 0;
        SDL_WaitAndAcquireGPUSwapchainTexture(cmd, window, &swapchainTexture, &width, &height);

        if (swapchainTexture && !minimized) {
            ImGui_ImplSDLGPU3_PrepareDrawData(drawData, cmd);

            Mat4 view = app.camera.ViewMatrix();
            float aspect = height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
            OrbitCamera::ClipPlanes clip = app.camera.NearFar();
            Mat4 proj = Mat4::Perspective(0.9f, aspect, clip.nearZ, clip.farZ);
            SDL_FColor clearColor{0.08f, 0.09f, 0.11f, 1.0f};
            if (app.viewMode == ViewMode::VoxelGrid && app.voxelGridValid) {
                // The voxel preview mesh is already in world space (built
                // from vertices transformed by ModelMatrix at voxelize
                // time), so it must not be transformed again here.
                app.voxelRenderer.Render(cmd, swapchainTexture, width, height, view, proj, Mat4::Identity(),
                                          app.camera.Eye(), clearColor);
            } else if (app.viewMode == ViewMode::FlowOverlay && app.overlayValid) {
                // The slice/voxel-cloud mesh is built from a GPU readback
                // of the solver's field, already in world space. Particles
                // is the odd one out: its mesh is rebuilt and uploaded
                // fresh this frame (into `cmd`, so the upload and the draw
                // below share one submission) rather than already sitting
                // in overlayRenderer from an earlier on-demand rebuild —
                // see UpdateSmokeTrailAndUpload's comment.
                //
                // All three styles draw through the alpha-blended
                // pipeline now, not just Voxel Cloud/Particles — Slice
                // carries its own fixed near-opaque alpha per vertex (see
                // FieldOverlay::BuildSliceMeshFromRange's kSliceAlpha) so
                // it reads as a very slightly translucent plane rather
                // than a flat, harsh wall, especially composited over the
                // unvoxelized model below. OverlayRenderer::Render() (the
                // fully-opaque pipeline) is unused by any style as of
                // this change, but deliberately left in OverlayRenderer
                // itself rather than removed — a cheap capability to keep
                // around, not one this change has an opinion on deleting.
                if (app.overlayStyle == OverlayStyle::StressMap) {
                    // Airframe Stress Map is an opaque 3D surface skin of
                    // the model itself — render with depth-test and
                    // depth-write enabled (OverlayRenderer::Render) so
                    // front-facing surfaces properly occlude back-facing
                    // ones without z-fighting against the CAD mesh.
                    app.overlayRenderer.Render(cmd, swapchainTexture, width, height, view, proj, clearColor, nullptr);
                } else if (app.overlayStyle == OverlayStyle::IsoSurface && app.isoAlpha >= 0.98f) {
                    // Fully opaque 3D Iso-surface renders with depth write enabled
                    if (app.overlayShowModel && app.renderer.HasMesh()) {
                        Mat4 model = ModelMatrix(app);
                        app.renderer.Render(cmd, swapchainTexture, width, height, view, proj, model, app.camera.Eye(),
                                            clearColor);
                        app.overlayRenderer.Render(cmd, swapchainTexture, width, height, view, proj, clearColor,
                                                   app.renderer.DepthTexture());
                    } else {
                        app.overlayRenderer.Render(cmd, swapchainTexture, width, height, view, proj, clearColor, nullptr);
                    }
                } else {
                    // Particles are drawn either for the Particles style
                    // itself, or confined to the plane in place of the
                    // colored cross-section when Slice's "Also show
                    // particles" is on (see the Slice-section checkbox
                    // above and SmokeTrailSystem::PlaneConfinement) — the two
                    // never both draw at once, unlike an earlier version of
                    // this feature that layered them.
                    bool sliceWithParticles = app.overlayStyle == OverlayStyle::Slice && app.sliceShowParticles;
                    bool showParticles = app.overlayStyle == OverlayStyle::Particles || sliceWithParticles;
                    if (showParticles) UpdateSmokeTrailAndUpload(app, cmd, ImGui::GetIO().DeltaTime);

                    auto drawTransparent = [&](SDL_GPUTexture* backdropDepth) {
                        // Draw the static mesh (Slice/Voxel Cloud geometry)
                        // except for Slice+particles, where BuildMeshForStyle
                        // leaves it empty anyway (nothing to draw) and the
                        // smoke trail replaces it entirely. For the pure
                        // Particles style this static pass still runs — its
                        // own mesh is empty too, but running it anyway keeps
                        // the dynamic draw below as the always-second pass,
                        // so forceLoadColor's meaning (see OverlayRenderer.h)
                        // stays simple: true exactly when something was
                        // already drawn into color this frame.
                        bool drawStatic = !sliceWithParticles;
                        if (drawStatic) {
                            app.overlayRenderer.RenderTransparent(cmd, swapchainTexture, width, height, view, proj,
                                                                    clearColor, backdropDepth);
                        }
                        if (showParticles) {
                            app.overlayRenderer.RenderTransparentDynamic(cmd, swapchainTexture, width, height, view,
                                                                            proj, clearColor, backdropDepth,
                                                                            /*forceLoadColor=*/drawStatic);
                        }
                    };

                    if (app.overlayShowModel && app.renderer.HasMesh()) {
                        // Draw the real mesh first so it populates the depth
                        // buffer, then composite the overlay on top of it
                        // using that same depth buffer — this hides the parts
                        // of it that are behind the model from this
                        // viewpoint, on top of RenderTransparent['Dynamic]'s
                        // own depth-test-but-don't-write behavior (see its
                        // header comment) — the slice/cubes/trails still
                        // correctly hide behind the opaque model, they just
                        // don't occlude *each other*.
                        Mat4 model = ModelMatrix(app);
                        app.renderer.Render(cmd, swapchainTexture, width, height, view, proj, model, app.camera.Eye(),
                                            clearColor);
                        drawTransparent(app.renderer.DepthTexture());
                    } else {
                        drawTransparent(nullptr);
                    }
                }
            } else {
                Mat4 model = ModelMatrix(app);
                app.renderer.Render(cmd, swapchainTexture, width, height, view, proj, model, app.camera.Eye(),
                                     clearColor);
            }

            SDL_GPUColorTargetInfo uiTarget = {};
            uiTarget.texture = swapchainTexture;
            uiTarget.load_op = SDL_GPU_LOADOP_LOAD;
            uiTarget.store_op = SDL_GPU_STOREOP_STORE;
            SDL_GPURenderPass* uiPass = SDL_BeginGPURenderPass(cmd, &uiTarget, 1, nullptr);
            ImGui_ImplSDLGPU3_RenderDrawData(drawData, cmd, uiPass);
            SDL_EndGPURenderPass(uiPass);
        }

        SDL_SubmitGPUCommandBuffer(cmd);
    }

    // Stop and join the solver thread before touching lbmSolver again —
    // it must not still be mid-Step/ReadbackField when Shutdown() frees
    // the GPU buffers out from under it.
    app.lbmThreadStop.store(true, std::memory_order_release);
    lbmThread.join();

    // Join any in-flight overlay auto-update too — see
    // AppState::overlayUpdateThread's comment; it also touches lbmSolver.
    if (app.overlayUpdateThread.joinable()) app.overlayUpdateThread.join();

    // Join any in-flight background task (Voxelize/Init/Build Overlay) too
    // — it may still be running if the window was closed mid-operation,
    // and it touches lbmSolver/renderers that are about to be torn down.
    if (app.taskThread.joinable()) app.taskThread.join();

    SDL_WaitForGPUIdle(gpuDevice);
    app.renderer.Shutdown();
    app.voxelRenderer.Shutdown();
    app.overlayRenderer.Shutdown();
    app.lbmSolver.Shutdown();
    app.panelSolver.Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui_ImplSDLGPU3_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    SDL_ReleaseWindowFromGPUDevice(gpuDevice, window);
    SDL_DestroyGPUDevice(gpuDevice);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
