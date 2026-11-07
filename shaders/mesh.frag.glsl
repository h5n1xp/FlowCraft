#version 450
// SDL_GPU SPIR-V binding convention (fragment stage): uniform buffers live in set 3.
layout(location = 0) in vec3 worldNormal;
layout(location = 1) in vec3 worldPos;

layout(location = 0) out vec4 outColor;

layout(set = 3, binding = 0) uniform FSUniforms {
    vec4 lightDir;
    vec4 cameraPos;
    vec4 baseColor;
} u;

void main() {
    vec3 n = normalize(worldNormal);
    vec3 l = normalize(-u.lightDir.xyz);
    float diff = max(dot(n, l), 0.0);

    vec3 viewDir = normalize(u.cameraPos.xyz - worldPos);
    vec3 halfVec = normalize(l + viewDir);
    float spec = pow(max(dot(n, halfVec), 0.0), 32.0);

    vec3 color = u.baseColor.xyz * (0.25 + diff * 0.65) + vec3(1.0) * spec * 0.2;
    outColor = vec4(color, 1.0);
}
