#include "Renderer.h"
#include "GpuShaders.h"

#include <cstddef>
#include <cstring>
#include <vector>

namespace {

constexpr SDL_GPUTextureFormat kDepthFormat = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;

struct GpuVertex {
    float px, py, pz;
    float nx, ny, nz;
};

struct VSUniforms {
    float mvp[16];
    float model[16];
};

struct FSUniforms {
    float lightDir[4];
    float cameraPos[4];
    float baseColor[4];
};

// Shaders are authored once in GLSL (shaders/*.glsl) and embedded as SPIR-V and
// MSL blobs — see GpuShaders.h.
SDL_GPUShader* CompileShader(SDL_GPUDevice* device, const ShaderBlob& blob, SDL_GPUShaderStage stage,
                             uint32_t numUniformBuffers) {
    return CreateShaderFromBlob(device, blob, stage, numUniformBuffers);
}

}  // namespace

bool MeshRenderer::Init(SDL_GPUDevice* device, SDL_Window* window) {
    device_ = device;

    SDL_GPUShader* vs = CompileShader(device_, FC_SHADER_BLOB(MeshVert), SDL_GPU_SHADERSTAGE_VERTEX, 1);
    SDL_GPUShader* fs = CompileShader(device_, FC_SHADER_BLOB(MeshFrag), SDL_GPU_SHADERSTAGE_FRAGMENT, 1);
    if (!vs || !fs) {
        SDL_Log("MeshRenderer: failed to compile shaders: %s", SDL_GetError());
        if (vs) SDL_ReleaseGPUShader(device_, vs);
        if (fs) SDL_ReleaseGPUShader(device_, fs);
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
    attrs[1].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
    attrs[1].offset = offsetof(GpuVertex, nx);

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
    pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;  // STL winding/normals aren't reliably consistent
    pipelineInfo.depth_stencil_state.enable_depth_test = true;
    pipelineInfo.depth_stencil_state.enable_depth_write = true;
    pipelineInfo.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS;
    pipelineInfo.target_info.num_color_targets = 1;
    pipelineInfo.target_info.color_target_descriptions = &colorTarget;
    pipelineInfo.target_info.has_depth_stencil_target = true;
    pipelineInfo.target_info.depth_stencil_format = kDepthFormat;

    pipeline_ = SDL_CreateGPUGraphicsPipeline(device_, &pipelineInfo);
    SDL_ReleaseGPUShader(device_, vs);
    SDL_ReleaseGPUShader(device_, fs);

    if (!pipeline_) {
        SDL_Log("MeshRenderer: failed to create pipeline: %s", SDL_GetError());
        return false;
    }
    return true;
}

void MeshRenderer::Shutdown() {
    if (vertexBuffer_) SDL_ReleaseGPUBuffer(device_, vertexBuffer_);
    if (indexBuffer_) SDL_ReleaseGPUBuffer(device_, indexBuffer_);
    if (depthTexture_) SDL_ReleaseGPUTexture(device_, depthTexture_);
    if (pipeline_) SDL_ReleaseGPUGraphicsPipeline(device_, pipeline_);
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    depthTexture_ = nullptr;
    pipeline_ = nullptr;
}

void MeshRenderer::UploadMesh(const Mesh& mesh) {
    if (vertexBuffer_) { SDL_ReleaseGPUBuffer(device_, vertexBuffer_); vertexBuffer_ = nullptr; }
    if (indexBuffer_) { SDL_ReleaseGPUBuffer(device_, indexBuffer_); indexBuffer_ = nullptr; }
    indexCount_ = 0;
    if (mesh.vertices.empty()) return;

    std::vector<GpuVertex> gpuVerts(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        gpuVerts[i] = {mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z,
                       mesh.normals[i].x, mesh.normals[i].y, mesh.normals[i].z};
    }

    uint32_t vbBytes = static_cast<uint32_t>(gpuVerts.size() * sizeof(GpuVertex));
    uint32_t ibBytes = static_cast<uint32_t>(mesh.indices.size() * sizeof(uint32_t));

    SDL_GPUBufferCreateInfo vbInfo = {};
    vbInfo.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
    vbInfo.size = vbBytes;
    vertexBuffer_ = SDL_CreateGPUBuffer(device_, &vbInfo);

    SDL_GPUBufferCreateInfo ibInfo = {};
    ibInfo.usage = SDL_GPU_BUFFERUSAGE_INDEX;
    ibInfo.size = ibBytes;
    indexBuffer_ = SDL_CreateGPUBuffer(device_, &ibInfo);

    SDL_GPUTransferBufferCreateInfo tbInfo = {};
    tbInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    tbInfo.size = vbBytes + ibBytes;
    SDL_GPUTransferBuffer* transfer = SDL_CreateGPUTransferBuffer(device_, &tbInfo);

    void* mapped = SDL_MapGPUTransferBuffer(device_, transfer, false);
    std::memcpy(mapped, gpuVerts.data(), vbBytes);
    std::memcpy(static_cast<uint8_t*>(mapped) + vbBytes, mesh.indices.data(), ibBytes);
    SDL_UnmapGPUTransferBuffer(device_, transfer);

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_);
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);

    SDL_GPUTransferBufferLocation vbSrc = {transfer, 0};
    SDL_GPUBufferRegion vbDst = {vertexBuffer_, 0, vbBytes};
    SDL_UploadToGPUBuffer(copyPass, &vbSrc, &vbDst, false);

    SDL_GPUTransferBufferLocation ibSrc = {transfer, vbBytes};
    SDL_GPUBufferRegion ibDst = {indexBuffer_, 0, ibBytes};
    SDL_UploadToGPUBuffer(copyPass, &ibSrc, &ibDst, false);

    SDL_EndGPUCopyPass(copyPass);
    SDL_SubmitGPUCommandBuffer(cmd);
    SDL_ReleaseGPUTransferBuffer(device_, transfer);

    indexCount_ = static_cast<uint32_t>(mesh.indices.size());
}

void MeshRenderer::EnsureDepthTexture(uint32_t width, uint32_t height) {
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

void MeshRenderer::Render(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchainTexture, uint32_t width, uint32_t height,
                           const Mat4& view, const Mat4& proj, const Mat4& model, const Vec3& cameraPos,
                           const SDL_FColor& clearColor) {
    EnsureDepthTexture(width, height);

    SDL_GPUColorTargetInfo colorTarget = {};
    colorTarget.texture = swapchainTexture;
    colorTarget.clear_color = clearColor;
    colorTarget.load_op = SDL_GPU_LOADOP_CLEAR;
    colorTarget.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPUDepthStencilTargetInfo depthTarget = {};
    depthTarget.texture = depthTexture_;
    depthTarget.clear_depth = 1.0f;
    depthTarget.load_op = SDL_GPU_LOADOP_CLEAR;
    depthTarget.store_op = SDL_GPU_STOREOP_STORE;  // kept so a following pass (e.g. the flow overlay) can depth-test against it

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &colorTarget, 1, &depthTarget);

    if (indexCount_ > 0) {
        SDL_BindGPUGraphicsPipeline(pass, pipeline_);

        SDL_GPUBufferBinding vbBinding = {vertexBuffer_, 0};
        SDL_BindGPUVertexBuffers(pass, 0, &vbBinding, 1);
        SDL_GPUBufferBinding ibBinding = {indexBuffer_, 0};
        SDL_BindGPUIndexBuffer(pass, &ibBinding, SDL_GPU_INDEXELEMENTSIZE_32BIT);

        VSUniforms vsUniforms;
        Mat4 mvp = proj * view * model;
        std::memcpy(vsUniforms.mvp, mvp.m, sizeof(mvp.m));
        std::memcpy(vsUniforms.model, model.m, sizeof(model.m));
        SDL_PushGPUVertexUniformData(cmd, 0, &vsUniforms, sizeof(vsUniforms));

        FSUniforms fsUniforms;
        // A fixed world-space light, never revisited despite "up" (Y->Z)
        // and "flow" (X->Y) both changing meaning today — its dominant
        // component used to point along Y back when Y meant up; now Y
        // is the flow axis, so that same dominant component was
        // lighting along the flow direction instead. Rebalanced so X —
        // the camera's own default viewing axis (forward=-X, camera on
        // the +X side) — dominates instead, putting the light source on
        // the same side as the camera (source direction is -lightDir,
        // so a negative, dominant X component here means a source at
        // +X), with a smaller upward (-Z) bias and a slight (-Y) tilt
        // for a non-flat look. Not visually verified — this app's
        // native GUI isn't something this session can screenshot.
        fsUniforms.lightDir[0] = -0.75f; fsUniforms.lightDir[1] = -0.3f; fsUniforms.lightDir[2] = -0.6f; fsUniforms.lightDir[3] = 0.0f;
        fsUniforms.cameraPos[0] = cameraPos.x; fsUniforms.cameraPos[1] = cameraPos.y; fsUniforms.cameraPos[2] = cameraPos.z; fsUniforms.cameraPos[3] = 0.0f;
        fsUniforms.baseColor[0] = 0.75f; fsUniforms.baseColor[1] = 0.78f; fsUniforms.baseColor[2] = 0.82f; fsUniforms.baseColor[3] = 1.0f;
        SDL_PushGPUFragmentUniformData(cmd, 0, &fsUniforms, sizeof(fsUniforms));

        SDL_DrawGPUIndexedPrimitives(pass, indexCount_, 1, 0, 0, 0);
    }

    SDL_EndGPURenderPass(pass);
}
