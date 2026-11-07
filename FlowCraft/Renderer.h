#pragma once

#include <SDL3/SDL.h>

#include "FcMath.h"
#include "Mesh.h"

// Minimal SDL_GPU mesh renderer: one pipeline, one vertex+index buffer
// (recreated whenever a new mesh is loaded), flat-shaded with a single
// directional light. Enough to visually confirm the STL parser produced a
// sane mesh; the CFD overlay will extend this later.
class MeshRenderer {
public:
    bool Init(SDL_GPUDevice* device, SDL_Window* window);
    void Shutdown();

    void UploadMesh(const Mesh& mesh);
    bool HasMesh() const { return indexCount_ > 0; }

    // Renders into swapchainTexture (color, loaded/cleared per clearColor)
    // with its own depth buffer sized to (width, height). Must be called
    // before any ImGui render pass on the same command buffer/texture.
    void Render(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchainTexture, uint32_t width, uint32_t height,
                const Mat4& view, const Mat4& proj, const Mat4& model, const Vec3& cameraPos,
                const SDL_FColor& clearColor);

    // The depth buffer this renderer last wrote (sized to the last Render()
    // call's width/height). Lets a renderer drawn afterward in the same
    // frame — e.g. OverlayRenderer compositing the flow slice on top —
    // depth-test against the mesh instead of a blank buffer, so the slice
    // is correctly hidden behind solid geometry.
    SDL_GPUTexture* DepthTexture() const { return depthTexture_; }

private:
    void EnsureDepthTexture(uint32_t width, uint32_t height);

    SDL_GPUDevice* device_ = nullptr;
    SDL_GPUGraphicsPipeline* pipeline_ = nullptr;
    SDL_GPUBuffer* vertexBuffer_ = nullptr;
    SDL_GPUBuffer* indexBuffer_ = nullptr;
    uint32_t indexCount_ = 0;

    SDL_GPUTexture* depthTexture_ = nullptr;
    uint32_t depthWidth_ = 0;
    uint32_t depthHeight_ = 0;
};
