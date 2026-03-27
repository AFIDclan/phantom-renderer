#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPos;
layout(location = 3) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Per-frame light + camera data (std140, 64 bytes)
layout(set = 0, binding = 0) uniform SceneUBO {
    vec4  lightPos;    // xyz = world pos
    vec4  lightColor;  // xyz = color, w = intensity (0 = prelit)
    vec4  cameraPos;   // xyz = world eye position
    uint  renderMode;  // 0=unlit, 1=lambert, 2=pbr
    float shadowNear;
    float shadowFar;
    uint  lightType;   // 0=Point, 1=Directional, 2=Spot (mirrors LightType enum)
} ubo;

// Set 0 binding 1: omnidirectional cube shadow map (point light)
layout(set = 0, binding = 1) uniform samplerCubeShadow shadowCubeMap;

// Set 1: per-draw textures
layout(set = 1, binding = 0) uniform sampler2D baseColorTex;
layout(set = 1, binding = 1) uniform sampler2D metalRoughTex; // G=roughness, B=metallic

const float PI = 3.14159265359;

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

    // --- UNLIT: render mode 0, or no dynamic light in scene ---
    if (ubo.renderMode == 0u || intensity < 0.001) {
        outColor = vec4(baseColor, 1.0);
        return;
    }

    vec3  N           = normalize(fragNormal);
    vec3  L           = normalize(ubo.lightPos.xyz - fragPos);
    float dist        = length(ubo.lightPos.xyz - fragPos);
    float attenuation = 1.0 / (1.0 + 0.07 * dist + 0.017 * dist * dist);
    vec3  radiance    = ubo.lightColor.xyz * intensity * attenuation;
    float shadow      = shadowFactor();
    float NdotL       = max(dot(N, L), 0.0);

    // --- LAMBERT ---
    if (ubo.renderMode == 1u) {
        const float AMBIENT = 0.20;
        outColor = vec4(baseColor * (AMBIENT + radiance * NdotL * shadow), 1.0);
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

    // Simple ambient term (image-based lighting approximation)
    vec3 ambient = vec3(0.03) * baseColor;

    // Reinhard tone mapping + gamma correction
    vec3 color = Lo + ambient;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}
