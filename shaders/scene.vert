#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragPos;
layout(location = 3) out vec2 fragUV;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 proj;
} pc;

void main() {
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    fragPos       = worldPos.xyz;
    fragNormal    = normalize(mat3(transpose(inverse(pc.model))) * inNormal);
    fragColor     = inColor;
    fragUV        = inUV;
    gl_Position   = pc.proj * pc.view * worldPos;
}
