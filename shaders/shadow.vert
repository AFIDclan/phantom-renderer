#version 450

// Layered shadow pass: renders all 6 cube faces in a single instanced draw.
//
// gl_InstanceIndex (0..5) selects which face to render to.
// gl_Layer routes each primitive to the correct cube map layer in the
// single layered framebuffer.  Requires shaderOutputLayer (Vulkan 1.2+).

#extension GL_ARB_shader_viewport_layer_array : require

layout(location = 0) in vec3 inPosition;

// Only the per-draw model matrix is pushed; lightVP comes from the UBO.
layout(push_constant) uniform ShadowPC {
    mat4 model;
} pc;

// All 6 face view-projection matrices, updated each frame.
layout(set = 0, binding = 0) uniform ShadowFacesUBO {
    mat4 vp[6];
} faces;

void main() {
    gl_Layer    = gl_InstanceIndex;
    gl_Position = faces.vp[gl_InstanceIndex] * pc.model * vec4(inPosition, 1.0);
}
