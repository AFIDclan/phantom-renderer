#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPos;
layout(location = 3) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Per-frame light + camera data.
layout(set = 0, binding = 0) uniform SceneUBO {
    vec4  lightPos;    // xyz = world pos
    vec4  lightDir;    // xyz = world direction
    vec4  lightColor;  // xyz = color, w = intensity (0 = prelit)
    vec4  cameraPos;   // xyz = world eye position
    mat4  lightVP;     // directional-light shadow matrix
    vec4  cameraRight;
    vec4  cameraUp;
    vec4  cameraForward;
    vec4  skyboxParams; // x=tanHalfFov, y=aspect, z=ambientIntensity, w=enabled
    vec4  skyboxUp;
    uint  renderMode;  // 0=unlit, 1=lambert, 2=pbr
    float shadowNear;
    float shadowFar;
    uint  lightType;   // 0=Point, 1=Directional, 2=Spot (mirrors LightType enum)
} ubo;

// Set 0 binding 1: omnidirectional cube shadow map (point light)
layout(set = 0, binding = 1) uniform samplerCubeShadow shadowCubeMap;
layout(set = 0, binding = 2) uniform sampler2DShadow shadow2DMap;
layout(set = 0, binding = 3) uniform sampler2D skyboxMap;

// Set 1: per-draw textures
layout(set = 1, binding = 0) uniform sampler2D baseColorTex;
layout(set = 1, binding = 1) uniform sampler2D metalRoughTex; // G=roughness, B=metallic

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

vec3 sampleAmbientSky(vec3 dir) {
    if (ubo.skyboxParams.w < 0.5)
        return vec3(0.03);
    return texture(skyboxMap, skyboxUV(orientSkybox(dir))).rgb * ubo.skyboxParams.z;
}

// Hardware PCF cube shadow lookup.
// d = max abs component of direction = view-space depth for the selected face.
// ref reconstructs the NDC depth using the same projection as the shadow pass.
float shadowFactor() {
    vec3  dir = fragPos - ubo.lightPos.xyz;
    float d   = max(abs(dir.x), max(abs(dir.y), abs(dir.z)));
    // Beyond the shadow far plane there is no depth data — treat as fully lit.
    if (d >= ubo.shadowFar) return 1.0;
    float ref = ubo.shadowFar * (d - ubo.shadowNear)
              / (d * (ubo.shadowFar - ubo.shadowNear));
    return texture(shadowCubeMap, vec4(dir, ref));
}

float directionalShadowFactor(float NdotL) {
    vec4 lightClip = ubo.lightVP * vec4(fragPos, 1.0);
    vec3 proj      = lightClip.xyz / max(lightClip.w, 0.0001);
    vec2 uv        = proj.xy * 0.5 + 0.5;
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) return 1.0;
    if (proj.z <= 0.0 || proj.z >= 1.0) return 1.0;

    float bias = max(0.0015 * (1.0 - NdotL), 0.0002);
    return texture(shadow2DMap, vec3(uv, proj.z - bias));
}

// --- PBR helpers (Cook-Torrance) ---

// GGX normal distribution function
float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 0.0001);
}

// Smith geometry function (Schlick-GGX)
float G_Smith(float NdotV, float NdotL, float roughness) {
    float r  = roughness + 1.0;
    float k  = (r * r) / 8.0;
    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    return gv * gl;
}

// Fresnel-Schlick approximation
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec3  baseColor = texture(baseColorTex, fragUV).rgb * fragColor;
    float intensity = ubo.lightColor.w;

    if (ubo.renderMode == 0u) {
        outColor = vec4(baseColor, 1.0);
        return;
    }

    vec3 N = normalize(fragNormal);
    vec3 ambientColor = sampleAmbientSky(N) * baseColor;

    if (intensity < 0.001) {
        vec3 color = ambientColor;
        if (ubo.renderMode == 2u) {
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0 / 2.2));
        }
        outColor = vec4(color, 1.0);
        return;
    }

    vec3  L        = vec3(0.0);
    vec3  radiance = ubo.lightColor.xyz * intensity;
    float shadow   = 1.0;

    if (ubo.lightType == 1u) {
        L = normalize(-ubo.lightDir.xyz);
    } else {
        vec3  lightVec    = ubo.lightPos.xyz - fragPos;
        float dist        = length(lightVec);
        float attenuation = 1.0 / (1.0 + 0.07 * dist + 0.017 * dist * dist);
        L                 = lightVec / max(dist, 0.0001);
        radiance         *= attenuation;
        shadow            = shadowFactor();
    }

    float NdotL = max(dot(N, L), 0.0);
    if (ubo.lightType == 1u)
        shadow = directionalShadowFactor(NdotL);

    // --- LAMBERT ---
    if (ubo.renderMode == 1u) {
        outColor = vec4(ambientColor + baseColor * (radiance * NdotL * shadow), 1.0);
        return;
    }

    // --- PBR: Cook-Torrance BRDF ---
    vec4  mr        = texture(metalRoughTex, fragUV);
    float roughness = max(mr.g, 0.04);  // avoid singularity at roughness=0
    float metallic  = mr.b;

    vec3  V     = normalize(ubo.cameraPos.xyz - fragPos);
    vec3  H     = normalize(V + L);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    // Fresnel base reflectance: dielectrics use 0.04, metals use base colour
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3  F = F_Schlick(HdotV, F0);

    // Diffuse component: metallic surfaces have no diffuse
    vec3 kD      = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kD * baseColor / PI;

    // Specular component (Cook-Torrance)
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    vec3 Lo = (diffuse + specular) * radiance * NdotL * shadow;

    // Reinhard tone mapping + gamma correction
    vec3 color = Lo + ambientColor;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}
