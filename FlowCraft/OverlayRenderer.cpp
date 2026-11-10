#include "OverlayRenderer.h"
#include "GpuShaders.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace {

constexpr SDL_GPUTextureFormat kDepthFormat = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;

struct GpuVertex {
    float px, py, pz;
    float cr, cg, cb, ca;  // ca only read by the transparent pipeline — see OverlayVertex::alpha
};

struct VSUniforms {
    float mvp[16];
};

SDL_GPUShader* CompileShader(SDL_GPUDevice* device, const ShaderBlob& blob, SDL_GPUShaderStage stage,
                             uint32_t numUniformBuffers) {
    return CreateShaderFromBlob(device, blob, stage, numUniformBuffers);
}

}  // namespace

bool OverlayRenderer::Init(SDL_GPUDevice* device, SDL_Window* window) {
    device_ = device;

    SDL_GPUShader* vs = CompileShader(device_, FC_SHADER_BLOB(OverlayVert), SDL_GPU_SHADERSTAGE_VERTEX, 1);
    SDL_GPUShader* fs = CompileShader(device_, FC_SHADER_BLOB(OverlayFrag), SDL_GPU_SHADERSTAGE_FRAGMENT, 0);
    SDL_GPUShader* fsTransparent = CompileShader(device_, FC_SHADER_BLOB(OverlayTransparentFrag), SDL_GPU_SHADERSTAGE_FRAGMENT, 0);
    if (!vs || !fs || !fsTransparent) {
        SDL_Log("OverlayRenderer: failed to compile shaders: %s", SDL_GetError());
        if (vs) SDL_ReleaseGPUShader(device_, vs);
        if (fs) SDL_ReleaseGPUShader(device_, fs);
        if (fsTransparent) SDL_ReleaseGPUShader(device_, fsTransparent);
        return false;
    }

    SDL_GPUVertexBufferDescription vbDesc = {};
    vbDesc.slot = 0;
    vbDesc.pitch = sizeof(GpuVertex);
    vbDesc.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;

    SDL_GPUVertexAttribute attrs[2] = {};
    attrs[0].location = 0;
    attrs[0].buffer_slot = 0;
    attrs[0].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
    attrs[0].offset = offsetof(GpuVertex, px);
    attrs[1].location = 1;
    attrs[1].buffer_slot = 0;
    attrs[1].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4;
    attrs[1].offset = offsetof(GpuVertex, cr);

    SDL_GPUColorTargetDescription colorTarget = {};
    colorTarget.format = SDL_GetGPUSwapchainTextureFormat(device_, window);

    SDL_GPUGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.vertex_shader = vs;
    pipelineInfo.fragment_shader = fs;
    pipelineInfo.vertex_input_state.vertex_buffer_descriptions = &vbDesc;
    pipelineInfo.vertex_input_state.num_vertex_buffers = 1;
    pipelineInfo.vertex_input_state.vertex_attributes = attrs;
    pipelineInfo.vertex_input_state.num_vertex_attributes = 2;
    pipelineInfo.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    pipelineInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;  // slice/cubes must be visible from either side
    pipelineInfo.depth_stencil_state.enable_depth_test = true;
    pipelineInfo.depth_stencil_state.enable_depth_write = true;
    pipelineInfo.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS;
    pipelineInfo.target_info.num_color_targets = 1;
    pipelineInfo.target_info.color_target_descriptions = &colorTarget;
    pipelineInfo.target_info.has_depth_stencil_target = true;
    pipelineInfo.target_info.depth_stencil_format = kDepthFormat;

    pipeline_ = SDL_CreateGPUGraphicsPipeline(device_, &pipelineInfo);

    // Same as above except: alpha blending on, and depth *write* off (test
    // stays on) — see RenderTransparent's header comment for why.
    SDL_GPUColorTargetBlendState blendState = {};
    blendState.enable_blend = true;
    blendState.src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA;
    blendState.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    blendState.color_blend_op = SDL_GPU_BLENDOP_ADD;
    blendState.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
    blendState.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    blendState.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
    SDL_GPUColorTargetDescription colorTargetTransparent = colorTarget;
    colorTargetTransparent.blend_state = blendState;

    SDL_GPUGraphicsPipelineCreateInfo pipelineInfoTransparent = pipelineInfo;
    pipelineInfoTransparent.fragment_shader = fsTransparent;
    pipelineInfoTransparent.depth_stencil_state.enable_depth_write = false;
    pipelineInfoTransparent.target_info.color_target_descriptions = &colorTargetTransparent;

    pipelineTransparent_ = SDL_CreateGPUGraphicsPipeline(device_, &pipelineInfoTransparent);

    SDL_ReleaseGPUShader(device_, vs);
    SDL_ReleaseGPUShader(device_, fs);
    SDL_ReleaseGPUShader(device_, fsTransparent);

    if (!pipeline_ || !pipelineTransparent_) {
        SDL_Log("OverlayRenderer: failed to create pipeline: %s", SDL_GetError());
        return false;
    }
    return true;
}

void OverlayRenderer::Shutdown() {
    if (vertexBuffer_) SDL_ReleaseGPUBuffer(device_, vertexBuffer_);
    if (indexBuffer_) SDL_ReleaseGPUBuffer(device_, indexBuffer_);
    if (transfer_) SDL_ReleaseGPUTransferBuffer(device_, transfer_);
    if (dynamicVertexBuffer_) SDL_ReleaseGPUBuffer(device_, dynamicVertexBuffer_);
    if (dynamicIndexBuffer_) SDL_ReleaseGPUBuffer(device_, dynamicIndexBuffer_);
    if (dynamicTransfer_) SDL_ReleaseGPUTransferBuffer(device_, dynamicTransfer_);
    if (depthTexture_) SDL_ReleaseGPUTexture(device_, depthTexture_);
    if (pipeline_) SDL_ReleaseGPUGraphicsPipeline(device_, pipeline_);
    if (pipelineTransparent_) SDL_ReleaseGPUGraphicsPipeline(device_, pipelineTransparent_);
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    transfer_ = nullptr;
    vertexCapacityBytes_ = 0;
    indexCapacityBytes_ = 0;
    transferCapacityBytes_ = 0;
    indexCount_ = 0;
    dynamicVertexBuffer_ = nullptr;
    dynamicIndexBuffer_ = nullptr;
    dynamicTransfer_ = nullptr;
    dynamicVertexCapacityBytes_ = 0;
    dynamicIndexCapacityBytes_ = 0;
    dynamicTransferCapacityBytes_ = 0;
    dynamicIndexCount_ = 0;
    depthTexture_ = nullptr;
    pipeline_ = nullptr;
    pipelineTransparent_ = nullptr;
}

namespace {

// SDL_GPU buffer sizes are uint32, so no single vertex/index/transfer
// buffer can exceed 4GiB. Both Upload* functions do their size arithmetic
// in 64 bits and go through these, because doing it in uint32 wrapped
// silently: a 3.85GiB vertex buffer's "+50% headroom" capacity overflowed
// to 1.49GiB, the buffer was created at that size, and the full-size copy
// into it then tripped Metal's validation (and would corrupt memory
// without it).
constexpr uint64_t kMaxGpuBufferBytes = UINT32_MAX;

// True if a mesh with these byte sizes can be uploaded at all (vertex,
// index, and the one transfer buffer holding both).
bool MeshFitsGpuLimits(uint64_t vbBytes, uint64_t ibBytes) {
    return vbBytes <= kMaxGpuBufferBytes && ibBytes <= kMaxGpuBufferBytes && vbBytes + ibBytes <= kMaxGpuBufferBytes;
}

// `bytes` plus 50% growth headroom, clamped to the per-buffer limit.
uint32_t CapacityWithHeadroom(uint64_t bytes) {
    return static_cast<uint32_t>(std::min(bytes + bytes / 2, kMaxGpuBufferBytes));
}

}  // namespace

void OverlayRenderer::UploadMesh(const OverlayMesh& mesh) {
    indexCount_ = 0;
    if (mesh.vertices.empty()) return;

    // Checked before the CPU-side copy below, which for a mesh this large
    // would itself be several GiB.
    if (!MeshFitsGpuLimits(static_cast<uint64_t>(mesh.vertices.size()) * sizeof(GpuVertex),
                           static_cast<uint64_t>(mesh.indices.size()) * sizeof(uint32_t))) {
        SDL_Log("OverlayRenderer: mesh too large to upload (%zu vertices, %zu indices) - not drawn.",
                mesh.vertices.size(), mesh.indices.size());
        return;
    }

    std::vector<GpuVertex> gpuVerts(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        const OverlayVertex& v = mesh.vertices[i];
        gpuVerts[i] = {v.pos.x, v.pos.y, v.pos.z, v.color.x, v.color.y, v.color.z, v.alpha};
    }

    uint32_t vbBytes = static_cast<uint32_t>(gpuVerts.size() * sizeof(GpuVertex));
    uint32_t ibBytes = static_cast<uint32_t>(mesh.indices.size() * sizeof(uint32_t));

    // Grows with 50% headroom, same reasoning/pattern as
    // UploadDynamicMesh — once a reslice's mesh size settles, this never
    // reallocates (and so never re-triggers the GPU memory pressure
    // stall this replaced) again.
    if (vbBytes > vertexCapacityBytes_) {
        if (vertexBuffer_) SDL_ReleaseGPUBuffer(device_, vertexBuffer_);
        vertexCapacityBytes_ = CapacityWithHeadroom(vbBytes);
        SDL_GPUBufferCreateInfo vbInfo = {};
        vbInfo.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
        vbInfo.size = vertexCapacityBytes_;
        vertexBuffer_ = SDL_CreateGPUBuffer(device_, &vbInfo);
        if (!vertexBuffer_) vertexCapacityBytes_ = 0;
    }
    if (ibBytes > indexCapacityBytes_) {
        if (indexBuffer_) SDL_ReleaseGPUBuffer(device_, indexBuffer_);
        indexCapacityBytes_ = CapacityWithHeadroom(ibBytes);
        SDL_GPUBufferCreateInfo ibInfo = {};
        ibInfo.usage = SDL_GPU_BUFFERUSAGE_INDEX;
        ibInfo.size = indexCapacityBytes_;
        indexBuffer_ = SDL_CreateGPUBuffer(device_, &ibInfo);
        if (!indexBuffer_) indexCapacityBytes_ = 0;
    }

    uint32_t transferBytes = vbBytes + ibBytes;
    if (transferBytes > transferCapacityBytes_) {
        if (transfer_) SDL_ReleaseGPUTransferBuffer(device_, transfer_);
        transferCapacityBytes_ = CapacityWithHeadroom(transferBytes);
        SDL_GPUTransferBufferCreateInfo tbInfo = {};
        tbInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        tbInfo.size = transferCapacityBytes_;
        transfer_ = SDL_CreateGPUTransferBuffer(device_, &tbInfo);
        if (!transfer_) transferCapacityBytes_ = 0;
    }
    if (!vertexBuffer_ || !indexBuffer_ || !transfer_) {
        SDL_Log("OverlayRenderer: GPU buffer allocation failed: %s", SDL_GetError());
        return;
    }

    // `cycle=true` — see UploadDynamicMesh's comment; this buffer is
    // rewritten far less often, but a reslice still right after the
    // previous one's draw is still in flight is possible.
    void* mapped = SDL_MapGPUTransferBuffer(device_, transfer_, true);
    if (!mapped) return;
    std::memcpy(mapped, gpuVerts.data(), vbBytes);
    std::memcpy(static_cast<uint8_t*>(mapped) + vbBytes, mesh.indices.data(), ibBytes);
    SDL_UnmapGPUTransferBuffer(device_, transfer_);

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_);
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);

    SDL_GPUTransferBufferLocation vbSrc = {transfer_, 0};
    SDL_GPUBufferRegion vbDst = {vertexBuffer_, 0, vbBytes};
    SDL_UploadToGPUBuffer(copyPass, &vbSrc, &vbDst, true);

    SDL_GPUTransferBufferLocation ibSrc = {transfer_, vbBytes};
    SDL_GPUBufferRegion ibDst = {indexBuffer_, 0, ibBytes};
    SDL_UploadToGPUBuffer(copyPass, &ibSrc, &ibDst, true);

    SDL_EndGPUCopyPass(copyPass);
    SDL_SubmitGPUCommandBuffer(cmd);

    indexCount_ = static_cast<uint32_t>(mesh.indices.size());
}

void OverlayRenderer::UploadDynamicMesh(SDL_GPUCommandBuffer* cmd, const OverlayMesh& mesh) {
    dynamicIndexCount_ = 0;
    if (mesh.vertices.empty()) return;

    // Checked before the CPU-side copy below, which for a mesh this large
    // would itself be several GiB.
    if (!MeshFitsGpuLimits(static_cast<uint64_t>(mesh.vertices.size()) * sizeof(GpuVertex),
                           static_cast<uint64_t>(mesh.indices.size()) * sizeof(uint32_t))) {
        SDL_Log("OverlayRenderer: mesh too large to upload (%zu vertices, %zu indices) - not drawn.",
                mesh.vertices.size(), mesh.indices.size());
        return;
    }

    std::vector<GpuVertex> gpuVerts(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        const OverlayVertex& v = mesh.vertices[i];
        gpuVerts[i] = {v.pos.x, v.pos.y, v.pos.z, v.color.x, v.color.y, v.color.z, v.alpha};
    }

    uint32_t vbBytes = static_cast<uint32_t>(gpuVerts.size() * sizeof(GpuVertex));
    uint32_t ibBytes = static_cast<uint32_t>(mesh.indices.size() * sizeof(uint32_t));

    // Grows with 50% headroom so a particle/trail-length count that
    // creeps up over a few frames doesn't reallocate on every single one
    // of them — once it settles, this never reallocates again.
    if (vbBytes > dynamicVertexCapacityBytes_) {
        if (dynamicVertexBuffer_) SDL_ReleaseGPUBuffer(device_, dynamicVertexBuffer_);
        dynamicVertexCapacityBytes_ = CapacityWithHeadroom(vbBytes);
        SDL_GPUBufferCreateInfo vbInfo = {};
        vbInfo.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
        vbInfo.size = dynamicVertexCapacityBytes_;
        dynamicVertexBuffer_ = SDL_CreateGPUBuffer(device_, &vbInfo);
        if (!dynamicVertexBuffer_) dynamicVertexCapacityBytes_ = 0;
    }
    if (ibBytes > dynamicIndexCapacityBytes_) {
        if (dynamicIndexBuffer_) SDL_ReleaseGPUBuffer(device_, dynamicIndexBuffer_);
        dynamicIndexCapacityBytes_ = CapacityWithHeadroom(ibBytes);
        SDL_GPUBufferCreateInfo ibInfo = {};
        ibInfo.usage = SDL_GPU_BUFFERUSAGE_INDEX;
        ibInfo.size = dynamicIndexCapacityBytes_;
        dynamicIndexBuffer_ = SDL_CreateGPUBuffer(device_, &ibInfo);
        if (!dynamicIndexBuffer_) dynamicIndexCapacityBytes_ = 0;
    }

    uint32_t transferBytes = vbBytes + ibBytes;
    if (transferBytes > dynamicTransferCapacityBytes_) {
        if (dynamicTransfer_) SDL_ReleaseGPUTransferBuffer(device_, dynamicTransfer_);
        dynamicTransferCapacityBytes_ = CapacityWithHeadroom(transferBytes);
        SDL_GPUTransferBufferCreateInfo tbInfo = {};
        tbInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        tbInfo.size = dynamicTransferCapacityBytes_;
        dynamicTransfer_ = SDL_CreateGPUTransferBuffer(device_, &tbInfo);
        if (!dynamicTransfer_) dynamicTransferCapacityBytes_ = 0;
    }
    if (!dynamicVertexBuffer_ || !dynamicIndexBuffer_ || !dynamicTransfer_) {
        SDL_Log("OverlayRenderer: GPU buffer allocation failed: %s", SDL_GetError());
        return;
    }

    // `cycle=true` on both the map and the upload tells SDL_GPU it may
    // hand back fresh backing memory if the previous frame's copy of
    // this same buffer could still be in flight on the GPU, instead of
    // this code having to track that with its own fences — the
    // documented pattern for a buffer that's rewritten every frame.
    void* mapped = SDL_MapGPUTransferBuffer(device_, dynamicTransfer_, true);
    if (!mapped) return;
    std::memcpy(mapped, gpuVerts.data(), vbBytes);
    std::memcpy(static_cast<uint8_t*>(mapped) + vbBytes, mesh.indices.data(), ibBytes);
    SDL_UnmapGPUTransferBuffer(device_, dynamicTransfer_);

    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);

    SDL_GPUTransferBufferLocation vbSrc = {dynamicTransfer_, 0};
    SDL_GPUBufferRegion vbDst = {dynamicVertexBuffer_, 0, vbBytes};
    SDL_UploadToGPUBuffer(copyPass, &vbSrc, &vbDst, true);

    SDL_GPUTransferBufferLocation ibSrc = {dynamicTransfer_, vbBytes};
    SDL_GPUBufferRegion ibDst = {dynamicIndexBuffer_, 0, ibBytes};
    SDL_UploadToGPUBuffer(copyPass, &ibSrc, &ibDst, true);

    SDL_EndGPUCopyPass(copyPass);

    dynamicIndexCount_ = static_cast<uint32_t>(mesh.indices.size());
}

void OverlayRenderer::EnsureDepthTexture(uint32_t width, uint32_t height) {
    if (depthTexture_ && depthWidth_ == width && depthHeight_ == height) return;
    if (depthTexture_) SDL_ReleaseGPUTexture(device_, depthTexture_);

    SDL_GPUTextureCreateInfo info = {};
    info.type = SDL_GPU_TEXTURETYPE_2D;
    info.format = kDepthFormat;
    info.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET;
    info.width = width;
    info.height = height;
    info.layer_count_or_depth = 1;
    info.num_levels = 1;
    info.sample_count = SDL_GPU_SAMPLECOUNT_1;
    depthTexture_ = SDL_CreateGPUTexture(device_, &info);
    depthWidth_ = width;
    depthHeight_ = height;
}

void OverlayRenderer::Render(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchainTexture, uint32_t width,
                              uint32_t height, const Mat4& view, const Mat4& proj, const SDL_FColor& clearColor,
                              SDL_GPUTexture* backdropDepth, bool forceLoadColor) {
    RenderWithPipeline(pipeline_, cmd, swapchainTexture, width, height, view, proj, clearColor, backdropDepth,
                        forceLoadColor, vertexBuffer_, indexBuffer_, indexCount_);
}

void OverlayRenderer::RenderTransparent(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchainTexture, uint32_t width,
                                         uint32_t height, const Mat4& view, const Mat4& proj,
                                         const SDL_FColor& clearColor, SDL_GPUTexture* backdropDepth,
                                         bool forceLoadColor) {
    RenderWithPipeline(pipelineTransparent_, cmd, swapchainTexture, width, height, view, proj, clearColor,
                        backdropDepth, forceLoadColor, vertexBuffer_, indexBuffer_, indexCount_);
}

void OverlayRenderer::RenderTransparentDynamic(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchainTexture,
                                                uint32_t width, uint32_t height, const Mat4& view, const Mat4& proj,
                                                const SDL_FColor& clearColor, SDL_GPUTexture* backdropDepth,
                                                bool forceLoadColor) {
    RenderWithPipeline(pipelineTransparent_, cmd, swapchainTexture, width, height, view, proj, clearColor,
                        backdropDepth, forceLoadColor, dynamicVertexBuffer_, dynamicIndexBuffer_, dynamicIndexCount_);
}

void OverlayRenderer::RenderWithPipeline(SDL_GPUGraphicsPipeline* pipeline, SDL_GPUCommandBuffer* cmd,
                                          SDL_GPUTexture* swapchainTexture, uint32_t width, uint32_t height,
                                          const Mat4& view, const Mat4& proj, const SDL_FColor& clearColor,
                                          SDL_GPUTexture* backdropDepth, bool forceLoadColor,
                                          SDL_GPUBuffer* vertexBuffer, SDL_GPUBuffer* indexBuffer,
                                          uint32_t indexCount) {
    bool compositing = backdropDepth != nullptr;
    if (!compositing) EnsureDepthTexture(width, height);

    // Depth's own load/clear stays keyed on `compositing` alone (not
    // forceLoadColor) — safe even when layering two overlay draws with no
    // model backdrop, since neither this pipeline nor the opaque one ever
    // *writes* depth here when not compositing against a real one (the
    // transparent pipeline never writes depth at all; the opaque one only
    // runs standalone, never layered) — so a second draw's depth clear
    // would just reset it to the same 1.0 the first draw's clear already
    // left it at. Only the color target actually needs the override.
    SDL_GPUColorTargetInfo colorTarget = {};
    colorTarget.texture = swapchainTexture;
    colorTarget.clear_color = clearColor;
    colorTarget.load_op = (compositing || forceLoadColor) ? SDL_GPU_LOADOP_LOAD : SDL_GPU_LOADOP_CLEAR;
    colorTarget.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPUDepthStencilTargetInfo depthTarget = {};
    depthTarget.texture = compositing ? backdropDepth : depthTexture_;
    depthTarget.clear_depth = 1.0f;
    depthTarget.load_op = compositing ? SDL_GPU_LOADOP_LOAD : SDL_GPU_LOADOP_CLEAR;
    depthTarget.store_op = SDL_GPU_STOREOP_DONT_CARE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &colorTarget, 1, &depthTarget);

    if (indexCount > 0) {
        SDL_BindGPUGraphicsPipeline(pass, pipeline);

        SDL_GPUBufferBinding vbBinding = {vertexBuffer, 0};
        SDL_BindGPUVertexBuffers(pass, 0, &vbBinding, 1);
        SDL_GPUBufferBinding ibBinding = {indexBuffer, 0};
        SDL_BindGPUIndexBuffer(pass, &ibBinding, SDL_GPU_INDEXELEMENTSIZE_32BIT);

        VSUniforms vsUniforms;
        Mat4 mvp = proj * view;
        std::memcpy(vsUniforms.mvp, mvp.m, sizeof(mvp.m));
        SDL_PushGPUVertexUniformData(cmd, 0, &vsUniforms, sizeof(vsUniforms));

        SDL_DrawGPUIndexedPrimitives(pass, indexCount, 1, 0, 0, 0);
    }

    SDL_EndGPURenderPass(pass);
}
