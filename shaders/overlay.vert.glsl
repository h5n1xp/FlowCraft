#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 vColor;

layout(set = 1, binding = 0) uniform VSUniforms {
    mat4 mvp;
} u;

void main() {
    gl_Position = u.mvp * vec4(position, 1.0);
    vColor = color;
}
