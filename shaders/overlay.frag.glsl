#version 450
layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(vColor.rgb, 1.0);  // always fully opaque, whatever alpha the vertex carried
}
