#version 450
// SDL_GPU SPIR-V binding convention (vertex stage): uniform buffers live in set 1.
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 worldNormal;
layout(location = 1) out vec3 worldPos;

layout(set = 1, binding = 0) uniform VSUniforms {
    mat4 mvp;
    mat4 model;
} u;

void main() {
    gl_Position = u.mvp * vec4(position, 1.0);
    worldNormal = (u.model * vec4(normal, 0.0)).xyz;
    worldPos = (u.model * vec4(position, 1.0)).xyz;
}
