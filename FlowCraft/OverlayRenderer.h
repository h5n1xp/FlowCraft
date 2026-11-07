#pragma once

#include <SDL3/SDL.h>

#include "FieldOverlay.h"
#include "FcMath.h"

// Minimal SDL_GPU renderer for vertex-colored, unlit geometry — the flow
// overlay's color values are data (speed/pressure), not something a
// directional light should modulate, so this deliberately doesn't share
// MeshRenderer's lit pipeline.
class OverlayRenderer {
public:
    bool Init(SDL_GPUDevice* device, SDL_Window* window);
    void Shutdown();

    // Used for the Slice/Voxel Cloud styles, rebuilt on demand (a manual
    // reslice, or an overlayAutoUpdate tick) rather than every frame.
    // Originally destroyed and recreated its GPU buffers, plus a
    // one-off transfer buffer, on every single call — harmless for an
    // occasional manual reslice, but a real cost with auto-update
    // running every overlayUpdateIntervalMs on a large domain, where
    // creating multi-hundred-MB buffers under the GPU memory pressure a
    // large LBM grid already puts on the device can itself stall (a
    // real, measured contributor to the GPU-backend-only main-loop
    // freeze investigated via main.cpp's *MsDebug fields — this was
    // happening inside what that instrumentation calls "poll"). Now
    // grows its own persistent vertex/index/transfer buffers only when a
    // mesh doesn't fit their current capacity, the same pattern
    // UploadDynamicMesh already used below.
    void UploadMesh(const OverlayMesh& mesh);
    bool HasMesh() const { return indexCount_ > 0; }

    // For geometry that changes every rendered frame (the Particles
    // overlay style's smoke trail) rather than occasionally on demand
    // (Slice/Voxel Cloud, via UploadMesh above). Grows its own persistent
    // vertex/index/transfer buffers only when a mesh doesn't fit their
    // current capacity (same pattern UploadMesh now uses — see its own
    // comment for why), and records its copy into `cmd` — the caller's
    // own command buffer for this frame — instead of submitting one of
    // its own, so the upload and the draw that follows
    // (RenderTransparentDynamic) share one submission.
    void UploadDynamicMesh(SDL_GPUCommandBuffer* cmd, const OverlayMesh& mesh);

    // `backdropDepth` is nullptr for a normal standalone draw (clears color
    // and depth, owns its own depth buffer). Pass a renderer's DepthTexture()
    // to composite the slice on top of what that renderer already drew in
    // this frame: color is loaded (not cleared) and the slice depth-tests
    // against the backdrop's depth buffer, so geometry hides the parts of
    // the slice that are behind it instead of the slice always winning.
    // `forceLoadColor`, when true, loads the color target even if
    // `backdropDepth` is null — for layering a second overlay draw on top
    // of a first one *within the same frame* (e.g. the Particles style's
    // own empty static-mesh pass through RenderTransparent, immediately
    // followed by its real geometry through RenderTransparentDynamic —
    // see main.cpp's render dispatch), where there's no model backdrop to
    // composite against but the color target still needs to be preserved
    // from whatever the first draw just put there, not cleared out from
    // under it. Normally compositing-against-a-model (backdropDepth
    // non-null) already implies this and this can be left false; it only
    // needs to be set explicitly for the "no model, but a second overlay
    // this frame" case that backdropDepth alone can't distinguish from
    // "no model, first/only overlay this frame".
    void Render(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchainTexture, uint32_t width, uint32_t height,
                const Mat4& view, const Mat4& proj, const SDL_FColor& clearColor,
                SDL_GPUTexture* backdropDepth = nullptr, bool forceLoadColor = false);

    // Same mesh/buffers, a different pipeline: alpha blending on (a fixed
    // 20% opacity baked into the shader — see OverlayRenderer.cpp), depth
    // *test* on but depth *write* off. The write being off is deliberate —
    // it means the voxel cubes never occlude each other via depth no
    // matter how many overlap along a sightline (so looking through
    // several reads as increasing opacity, the whole point of a
    // semi-transparent voxel cloud), while depth test staying on still
    // lets `backdropDepth` correctly hide whatever's behind real opaque
    // geometry (e.g. the model, if shown). `backdropDepth` means the same
    // thing as in Render() above. No back-to-front sorting is done among
    // the cubes themselves — acceptable for "get a feel for the shape",
    // not perfect blending order.
    // `forceLoadColor` — see Render()'s comment.
    void RenderTransparent(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchainTexture, uint32_t width,
                            uint32_t height, const Mat4& view, const Mat4& proj, const SDL_FColor& clearColor,
                            SDL_GPUTexture* backdropDepth = nullptr, bool forceLoadColor = false);

    // Same pipeline/blend behavior as RenderTransparent, but draws the
    // dynamic buffers UploadDynamicMesh fills instead of the static ones
    // UploadMesh fills — see the Particles overlay style in main.cpp.
    // `forceLoadColor` — see Render()'s comment.
    void RenderTransparentDynamic(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchainTexture, uint32_t width,
                                   uint32_t height, const Mat4& view, const Mat4& proj, const SDL_FColor& clearColor,
                                   SDL_GPUTexture* backdropDepth = nullptr, bool forceLoadColor = false);

private:
    void EnsureDepthTexture(uint32_t width, uint32_t height);
    void RenderWithPipeline(SDL_GPUGraphicsPipeline* pipeline, SDL_GPUCommandBuffer* cmd,
                             SDL_GPUTexture* swapchainTexture, uint32_t width, uint32_t height, const Mat4& view,
                             const Mat4& proj, const SDL_FColor& clearColor, SDL_GPUTexture* backdropDepth,
                             bool forceLoadColor, SDL_GPUBuffer* vertexBuffer, SDL_GPUBuffer* indexBuffer,
                             uint32_t indexCount);

    SDL_GPUDevice* device_ = nullptr;
    SDL_GPUGraphicsPipeline* pipeline_ = nullptr;             // opaque, for the slice
    SDL_GPUGraphicsPipeline* pipelineTransparent_ = nullptr;  // alpha-blended, for the voxel cloud and particles
    // Capacities track actual allocated size (bytes), which can exceed
    // what the latest mesh needed — see UploadMesh's comment. `transfer_`
    // is kept and re-mapped (with the cycle flag) across calls rather
    // than recreated each time, the same reasoning as `dynamicTransfer_`
    // below, even though this one's rewritten far less often.
    SDL_GPUBuffer* vertexBuffer_ = nullptr;
    SDL_GPUBuffer* indexBuffer_ = nullptr;
    uint32_t vertexCapacityBytes_ = 0;
    uint32_t indexCapacityBytes_ = 0;
    uint32_t indexCount_ = 0;
    SDL_GPUTransferBuffer* transfer_ = nullptr;
    uint32_t transferCapacityBytes_ = 0;

    // The Particles style's per-frame geometry — see UploadDynamicMesh.
    // Capacities track the buffers' actual allocated size (bytes), which
    // can exceed what the latest mesh needed; `dynamicTransfer_` is kept
    // and re-mapped (with the cycle flag) across frames rather than
    // recreated each time, since its lifetime would otherwise have to be
    // tied to a command buffer submission this class doesn't do itself.
    SDL_GPUBuffer* dynamicVertexBuffer_ = nullptr;
    SDL_GPUBuffer* dynamicIndexBuffer_ = nullptr;
    uint32_t dynamicVertexCapacityBytes_ = 0;
    uint32_t dynamicIndexCapacityBytes_ = 0;
    uint32_t dynamicIndexCount_ = 0;
    SDL_GPUTransferBuffer* dynamicTransfer_ = nullptr;
    uint32_t dynamicTransferCapacityBytes_ = 0;

    SDL_GPUTexture* depthTexture_ = nullptr;
    uint32_t depthWidth_ = 0;
    uint32_t depthHeight_ = 0;
};
