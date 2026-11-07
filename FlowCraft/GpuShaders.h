#pragma once

#include <SDL3/SDL_gpu.h>

#include <cstddef>

#include "generated/ShaderBlobs.h"

// Backend-agnostic shader creation. Every shader is authored once in GLSL
// (see shaders/*.glsl), compiled offline to SPIR-V (Vulkan: Windows/Linux)
// and to MSL (Metal: macOS) by shaders/build_shaders.py, and embedded in
// generated/ShaderBlobs.h. At runtime the device's supported formats decide
// which blob is handed to SDL_GPU — there is no per-platform shader code.

constexpr SDL_GPUShaderFormat kEmbeddedShaderFormats =
    SDL_GPU_SHADERFORMAT_SPIRV | SDL_GPU_SHADERFORMAT_MSL;

struct ShaderBlob {
    const unsigned char* spirv;
    size_t spirvSize;
    const unsigned char* msl;
    size_t mslSize;  // includes the trailing NUL (SDL_GPU wants the MSL length without it)
};

#define FC_SHADER_BLOB(name) \
    ShaderBlob { name##_SPIRV, name##_SPIRVSize, name##_MSL, name##_MSLSize }

// SDL_shadercross names the MSL entry point "main0" (since "main" is reserved
// in MSL); SPIR-V keeps "main".
struct ChosenShaderFormat {
    SDL_GPUShaderFormat format = SDL_GPU_SHADERFORMAT_INVALID;
    const unsigned char* code = nullptr;
    size_t size = 0;
    const char* entrypoint = nullptr;
};

inline ChosenShaderFormat ChooseShaderFormat(SDL_GPUDevice* device, const ShaderBlob& blob) {
    SDL_GPUShaderFormat supported = SDL_GetGPUShaderFormats(device);
    ChosenShaderFormat c;
    if (supported & SDL_GPU_SHADERFORMAT_SPIRV) {
        c = {SDL_GPU_SHADERFORMAT_SPIRV, blob.spirv, blob.spirvSize, "main"};
    } else if (supported & SDL_GPU_SHADERFORMAT_MSL) {
        c = {SDL_GPU_SHADERFORMAT_MSL, blob.msl, blob.mslSize - 1, "main0"};
    }
    return c;
}

inline SDL_GPUShader* CreateShaderFromBlob(SDL_GPUDevice* device, const ShaderBlob& blob, SDL_GPUShaderStage stage,
                                           Uint32 numUniformBuffers) {
    ChosenShaderFormat c = ChooseShaderFormat(device, blob);
    if (c.format == SDL_GPU_SHADERFORMAT_INVALID) {
        SDL_SetError("No supported shader format (device supports neither SPIR-V nor MSL)");
        return nullptr;
    }
    SDL_GPUShaderCreateInfo info = {};
    info.code = c.code;
    info.code_size = c.size;
    info.entrypoint = c.entrypoint;
    info.format = c.format;
    info.stage = stage;
    info.num_uniform_buffers = numUniformBuffers;
    return SDL_CreateGPUShader(device, &info);
}
