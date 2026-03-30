#pragma once

#include "vh/Math.h"

#include <cstdint>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Vertex — interleaved per-vertex data.  Layout matches scene.vert bindings:
//   location 0 = pos, location 1 = color, location 2 = normal, location 3 = uv
// ----------------------------------------------------------------------------

struct Vertex {
    float pos[3];
    float color[3];
    float normal[3];
    float uv[2];     // texture coordinates
};

// ----------------------------------------------------------------------------
// TextureData — decoded RGBA8 image ready for GPU upload.
// Populated by GLTFLoader from the model's embedded images.
// ----------------------------------------------------------------------------

struct TextureData {
    uint32_t             width  = 0;
    uint32_t             height = 0;
    std::vector<uint8_t> pixels;   // RGBA8, row-major
};

// ----------------------------------------------------------------------------
// Light — a KHR_lights_punctual scene light.
//
// type == Point       uses position + range; omnidirectional cube shadow map.
// type == Directional uses direction only; ortho shadow map.
// type == Spot        uses position + direction + innerCone/outerCone
//                     (single-frustum shadow, not yet impl.).
// ----------------------------------------------------------------------------

enum class LightType : uint32_t {
    Point       = 0,
    Directional = 1,
    Spot        = 2,
};

struct Light {
    LightType type      = LightType::Point;
    Vec3      position  = {};           // world space (Point, Spot)
    Vec3      direction = {0,-1, 0};    // world space unit vector (Directional, Spot)
    Vec3      color     = {1, 1, 1};
    float     intensity = 1.f;
    float     range     = 0.f;          // 0 = no range limit (Point, Spot)
    float     innerCone = 0.f;          // radians (Spot only)
    float     outerCone = 0.785398f;    // radians = π/4 (Spot only)

    // Convenience constructors.
    static Light point(Vec3 pos, Vec3 col = {1,1,1}, float intensity = 1.f) {
        Light l;  l.type = LightType::Point;
        l.position = pos;  l.color = col;  l.intensity = intensity;
        return l;
    }
    static Light directional(Vec3 dir, Vec3 col = {1,1,1}, float intensity = 1.f) {
        Light l;  l.type = LightType::Directional;
        l.direction = normalize(dir);  l.color = col;  l.intensity = intensity;
        return l;
    }
    static Light spot(Vec3 pos, Vec3 dir, float inner, float outer,
                      Vec3 col = {1,1,1}, float intensity = 1.f) {
        Light l;  l.type = LightType::Spot;
        l.position = pos;  l.direction = normalize(dir);
        l.innerCone = inner;  l.outerCone = outer;
        l.color = col;  l.intensity = intensity;
        return l;
    }
};

// ----------------------------------------------------------------------------
// Mesh — raw geometry for a single mesh primitive, stored in CPU memory.
// A SceneObject owns one or more Meshes.
// ----------------------------------------------------------------------------

struct Mesh {
    std::string           name;
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
    int                   textureIdx       = -1;  // index into Scene::textures, or -1
    int                   metalRoughTexIdx = -1;  // metallic-roughness texture (G=rough, B=metal)
};

// ----------------------------------------------------------------------------
// DrawCall — references a slice of the merged GPU vertex/index buffers and
// carries the model-to-world transform for that draw.  Built fresh each frame
// by Scene::buildRenderScene() so object transforms are always current.
// ----------------------------------------------------------------------------

struct DrawCall {
    uint32_t indexCount;
    uint32_t firstIndex;         // byte offset / sizeof(uint32) into the index buffer
    int32_t  vertexOffset;       // added to every index before sampling the vertex buffer
    Mat4     transform;          // model-to-world matrix
    int32_t  textureIdx       = -1;  // index into RenderScene::textures, or -1
    int32_t  metalRoughTexIdx = -1;  // metallic-roughness texture index, or -1
};

// ----------------------------------------------------------------------------
// RenderScene — everything the GPU needs for one frame, produced by
// Scene::buildRenderScene().  Passed to HeadlessRenderer::uploadScene() when
// geometry changes, and the view/proj matrices to renderFrame() each tick.
// ----------------------------------------------------------------------------

struct RenderScene {
    std::vector<Vertex>      vertices;
    std::vector<uint32_t>    indices;
    std::vector<DrawCall>    draws;
    std::vector<TextureData> textures;  // indexed by DrawCall::textureIdx
    std::vector<Light>        lights;    // scene lights (KHR_lights_punctual)
};
