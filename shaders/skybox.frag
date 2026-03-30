#version 450

layout(location = 0) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    vec4  lightPos;
    vec4  lightDir;
    vec4  lightColor;
    vec4  cameraPos;
    mat4  lightVP;
    vec4  cameraRight;
    vec4  cameraUp;
    vec4  cameraForward;
    vec4  skyboxParams;
    vec4  skyboxUp;
    uint  renderMode;
    float shadowNear;
    float shadowFar;
    uint  lightType;
} ubo;

layout(set = 0, binding = 3) uniform sampler2D skyboxMap;

const float PI = 3.14159265359;

vec2 skyboxUV(vec3 dir) {
    dir = normalize(dir);
    float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    float v = acos(clamp(dir.y, -1.0, 1.0)) / PI;
    return vec2(u, v);
}

vec3 orientSkybox(vec3 worldDir) {
    vec3 up = normalize(ubo.skyboxUp.xyz);
    vec3 refRight = abs(dot(up, vec3(1.0, 0.0, 0.0))) > 0.99
                  ? vec3(0.0, 1.0, 0.0)
                  : vec3(1.0, 0.0, 0.0);
    vec3 forward = normalize(cross(refRight, up));
    vec3 right   = normalize(cross(up, forward));
    return vec3(dot(worldDir, right), dot(worldDir, up), dot(worldDir, forward));
}

void main() {
    vec2 ndc = vec2(fragUV.x * 2.0 - 1.0, 1.0 - fragUV.y * 2.0);
    vec3 ray = normalize(
        ubo.cameraForward.xyz
      + ndc.x * ubo.skyboxParams.y * ubo.skyboxParams.x * ubo.cameraRight.xyz
      + ndc.y * ubo.skyboxParams.x * ubo.cameraUp.xyz
    );
        vec3 sky = texture(skyboxMap, skyboxUV(orientSkybox(ray))).rgb;
    outColor = vec4(sky, 1.0);
}