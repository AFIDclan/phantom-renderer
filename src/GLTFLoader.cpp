// tinygltf and stb implementations — defined here and nowhere else.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include "vh/GLTFLoader.h"
#include "vh/Scene.h"

#include <cstdio>
#include <string>
#include <unordered_map>

// ----------------------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------------------

// Read a VEC3 float from an accessor, handling interleaved (strided) buffers.
static Vec3 readVec3(const tinygltf::Model& model, int accIdx, size_t i) {
    const auto&    acc  = model.accessors[accIdx];
    const auto&    bv   = model.bufferViews[acc.bufferView];
    const uint8_t* base = model.buffers[bv.buffer].data.data()
                        + bv.byteOffset + acc.byteOffset;
    size_t stride = bv.byteStride > 0 ? bv.byteStride : sizeof(float) * 3;
    const float* p = reinterpret_cast<const float*>(base + i * stride);
    return {p[0], p[1], p[2]};
}

// Read a VEC2 float from an accessor (UV coordinates).
static void readVec2(const tinygltf::Model& model, int accIdx, size_t i,
                     float& u, float& v) {
    const auto&    acc  = model.accessors[accIdx];
    const auto&    bv   = model.bufferViews[acc.bufferView];
    const uint8_t* base = model.buffers[bv.buffer].data.data()
                        + bv.byteOffset + acc.byteOffset;
    size_t stride = bv.byteStride > 0 ? bv.byteStride : sizeof(float) * 2;
    const float* p = reinterpret_cast<const float*>(base + i * stride);
    u = p[0]; v = p[1];
}

// Read a scalar index (uint8, uint16, or uint32) from an accessor.
static uint32_t readIndex(const tinygltf::Model& model, int accIdx, size_t i) {
    const auto&    acc  = model.accessors[accIdx];
    const auto&    bv   = model.bufferViews[acc.bufferView];
    const uint8_t* base = model.buffers[bv.buffer].data.data()
                        + bv.byteOffset + acc.byteOffset;
    // Index buffers are tightly packed by the spec; stride is determined by type.
    switch (acc.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  return base[i];
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return reinterpret_cast<const uint16_t*>(base)[i];
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   return reinterpret_cast<const uint32_t*>(base)[i];
        default: return 0;
    }
}

// Build the local-space transform matrix for a node.
static Mat4 nodeLocalTransform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16)
        return Mat4::fromGLTF(node.matrix);
    return Mat4::fromTRS(node.translation, node.rotation, node.scale);
}

// Recursively process a scene node and all its children, appending
// SceneObjects (and possibly lights) to the scene.
static void processNode(
        const tinygltf::Model&               model,
        int                                  nodeIdx,
        const Mat4&                          parentWorld,
        Scene&                               scene,
        int&                                 objCounter,
        const std::unordered_map<int,int>&   gltfTexToSceneTex)
{
    const auto& node       = model.nodes[nodeIdx];
    Mat4        worldXform = parentWorld * nodeLocalTransform(node);

    // ---- KHR_lights_punctual -----------------------------------------------
    auto lightExtIt = node.extensions.find("KHR_lights_punctual");
    if (lightExtIt != node.extensions.end()) {
        int lightIdx = (int)lightExtIt->second.Get("light").GetNumberAsDouble();
        if (lightIdx >= 0 && lightIdx < (int)model.lights.size()) {
            const auto& l = model.lights[lightIdx];
            Vec3 col = { 1.f, 1.f, 1.f };
            if (l.color.size() >= 3)
                col = { (float)l.color[0], (float)l.color[1], (float)l.color[2] };
            float intensity = (float)l.intensity;

            if (l.type == "directional") {
                // Direction is -Z of the node's world transform.
                Vec3 dir = { -worldXform.m[2][0], -worldXform.m[2][1], -worldXform.m[2][2] };
                scene.addLight(Light::directional(dir, col, intensity));
            } else if (l.type == "spot") {
                Vec3 pos = { worldXform.m[3][0], worldXform.m[3][1], worldXform.m[3][2] };
                Vec3 dir = { -worldXform.m[2][0], -worldXform.m[2][1], -worldXform.m[2][2] };
                float inner = l.spot.innerConeAngle > 0.0 ? (float)l.spot.innerConeAngle : 0.f;
                float outer = l.spot.outerConeAngle > 0.0 ? (float)l.spot.outerConeAngle : 0.785398f;
                scene.addLight(Light::spot(pos, dir, inner, outer, col, intensity));
            } else {
                // "point" (default).
                Vec3 pos = { worldXform.m[3][0], worldXform.m[3][1], worldXform.m[3][2] };
                scene.addLight(Light::point(pos, col, intensity));
            }
        }
    }

    // ---- Mesh --------------------------------------------------------------
    if (node.mesh >= 0) {
        const auto& gltfMesh = model.meshes[node.mesh];
        for (size_t pi = 0; pi < gltfMesh.primitives.size(); ++pi) {
            const auto& prim = gltfMesh.primitives[pi];
            // Non-indexed geometry is supported: we synthesize sequential indices.

            // Base colour factor (modulates the texture, or IS the colour if no texture)
            float color[3] = {0.8f, 0.8f, 0.8f};
            int   sceneTex = -1;
            if (prim.material >= 0) {
                const auto& mat = model.materials[prim.material];
                const auto& pbr = mat.pbrMetallicRoughness;
                if (pbr.baseColorFactor.size() >= 3) {
                    color[0] = (float)pbr.baseColorFactor[0];
                    color[1] = (float)pbr.baseColorFactor[1];
                    color[2] = (float)pbr.baseColorFactor[2];
                }
                // Look up uploaded base-colour texture (if any)
                if (pbr.baseColorTexture.index >= 0) {
                    auto it = gltfTexToSceneTex.find(pbr.baseColorTexture.index);
                    if (it != gltfTexToSceneTex.end())
                        sceneTex = it->second;
                }
            }

            auto posIt = prim.attributes.find("POSITION");
            if (posIt == prim.attributes.end()) continue;

            int    posAcc  = posIt->second;
            size_t nVerts  = model.accessors[posAcc].count;

            int normAcc = -1;
            auto normIt = prim.attributes.find("NORMAL");
            if (normIt != prim.attributes.end())
                normAcc = normIt->second;

            int uvAcc = -1;
            auto uvIt = prim.attributes.find("TEXCOORD_0");
            if (uvIt != prim.attributes.end())
                uvAcc = uvIt->second;

            Mesh mesh;
            mesh.name = gltfMesh.name.empty()
                ? ("mesh_" + std::to_string(objCounter))
                : gltfMesh.name;
            mesh.textureIdx = sceneTex;
            int sceneMRTex = -1;
            if (prim.material >= 0) {
                const auto& mat = model.materials[prim.material];
                int gltfMRTex = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
                if (gltfMRTex >= 0) {
                    auto it = gltfTexToSceneTex.find(gltfMRTex);
                    if (it != gltfTexToSceneTex.end())
                        sceneMRTex = it->second;
                }
            }
            mesh.metalRoughTexIdx = sceneMRTex;

            mesh.vertices.reserve(nVerts);
            for (size_t i = 0; i < nVerts; ++i) {
                Vec3  p = readVec3(model, posAcc, i);
                Vec3  n = normAcc >= 0 ? readVec3(model, normAcc, i) : Vec3{0, 1, 0};
                float u = 0.f, v = 0.f;
                if (uvAcc >= 0) readVec2(model, uvAcc, i, u, v);
                mesh.vertices.push_back({{p.x, p.y, p.z},
                                         {color[0], color[1], color[2]},
                                         {n.x, n.y, n.z},
                                         {u, v}});
            }

            if (prim.indices >= 0) {
                size_t nIdx = model.accessors[prim.indices].count;
                mesh.indices.reserve(nIdx);
                for (size_t i = 0; i < nIdx; ++i)
                    mesh.indices.push_back(readIndex(model, prim.indices, i));
            } else {
                // Non-indexed: generate a trivial 0,1,2,...,N-1 index list.
                mesh.indices.reserve(nVerts);
                for (size_t i = 0; i < nVerts; ++i)
                    mesh.indices.push_back(static_cast<uint32_t>(i));
            }

            std::string id = mesh.name + "_" + std::to_string(objCounter++);
            scene.add(SceneObject(std::move(id), std::move(mesh), worldXform));
        }
    }

    for (int child : node.children)
        processNode(model, child, worldXform, scene, objCounter, gltfTexToSceneTex);
}

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

bool loadGLTF(const std::string& path, Scene& scene) {
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string        err, warn;

    bool ok = (path.size() >= 4 && path.substr(path.size() - 4) == ".glb")
        ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
        : loader.LoadASCIIFromFile(&model, &err, &warn, path);

    if (!warn.empty()) printf("glTF warning: %s\n", warn.c_str());
    if (!err.empty())  fprintf(stderr, "glTF error: %s\n", err.c_str());
    if (!ok)           return false;

    // ------------------------------------------------------------------
    // Pre-upload all textures to the scene texture store.
    // Deduplicates by image source index: multiple glTF textures that
    // reference the same image (e.g. different sampler settings) share
    // one scene TextureData, avoiding redundant VRAM uploads.
    // Maps glTF texture index → scene texture index.
    // ------------------------------------------------------------------
    std::unordered_map<int,int> gltfTexToSceneTex;
    std::unordered_map<int,int> imgIdxToSceneTex;   // deduplicate by image source
    for (int ti = 0; ti < (int)model.textures.size(); ++ti) {
        int imgIdx = model.textures[ti].source;
        if (imgIdx < 0 || imgIdx >= (int)model.images.size()) continue;

        // Reuse an already-uploaded scene texture if this image was seen before.
        auto cached = imgIdxToSceneTex.find(imgIdx);
        if (cached != imgIdxToSceneTex.end()) {
            gltfTexToSceneTex[ti] = cached->second;
            continue;
        }

        const auto& img = model.images[imgIdx];
        if (img.image.empty()) continue;

        TextureData td;
        td.width  = (uint32_t)img.width;
        td.height = (uint32_t)img.height;
        if (img.component == 4) {
            td.pixels.assign(img.image.begin(), img.image.end());
        } else if (img.component == 3) {
            // Convert RGB → RGBA (Vulkan requires 4-channel format)
            td.pixels.reserve((size_t)img.width * img.height * 4);
            for (int p = 0; p < img.width * img.height; ++p) {
                td.pixels.push_back(img.image[p*3+0]);
                td.pixels.push_back(img.image[p*3+1]);
                td.pixels.push_back(img.image[p*3+2]);
                td.pixels.push_back(255);
            }
        } else {
            continue; // unsupported channel count — skip
        }

        int sceneTexIdx = scene.addTexture(std::move(td));
        printf("Texture %d: %ux%u (%d ch)\n", ti, img.width, img.height, img.component);
        imgIdxToSceneTex[imgIdx]  = sceneTexIdx;
        gltfTexToSceneTex[ti]     = sceneTexIdx;
    }

    int sceneIdx = model.defaultScene >= 0 ? model.defaultScene : 0;
    int counter  = 0;
    for (int nodeIdx : model.scenes[sceneIdx].nodes)
        processNode(model, nodeIdx, Mat4::identity(), scene, counter, gltfTexToSceneTex);

    printf("Loaded %zu object(s), %zu texture(s), %zu light(s) from %s\n",
           scene.objectCount(), scene.textures().size(),
           scene.lights().size(), path.c_str());
    return true;
}

// ----------------------------------------------------------------------------
// Built-in demo scene
// ----------------------------------------------------------------------------

void loadDemoScene(Scene& scene) {
    // --- Triangle (red) ---
    {
        Mesh m; m.name = "triangle";
        m.vertices = {
            {{ 0.0f,  0.5f, 0.0f}, {0.85f, 0.15f, 0.15f}, {0, 0, 1}, {0.5f, 0.f}},
            {{-0.5f, -0.5f, 0.0f}, {0.85f, 0.15f, 0.15f}, {0, 0, 1}, {0.f,  1.f}},
            {{ 0.5f, -0.5f, 0.0f}, {0.85f, 0.15f, 0.15f}, {0, 0, 1}, {1.f,  1.f}},
        };
        m.indices = {0, 1, 2};
        Mat4 t = Mat4::translate({0.5f, 0.3f, 0.5f}) * Mat4::scale(0.7f);
        scene.add(SceneObject("triangle", std::move(m), t));
    }

    // --- Quad (blue) ---
    {
        Mesh m; m.name = "quad";
        m.vertices = {
            {{-0.9f,  0.9f, -0.5f}, {0.15f, 0.35f, 0.85f}, {0, 0, 1}, {0.f, 0.f}},
            {{-0.9f, -0.9f, -0.5f}, {0.15f, 0.35f, 0.85f}, {0, 0, 1}, {0.f, 1.f}},
            {{ 0.9f, -0.9f, -0.5f}, {0.15f, 0.35f, 0.85f}, {0, 0, 1}, {1.f, 1.f}},
            {{ 0.9f,  0.9f, -0.5f}, {0.15f, 0.35f, 0.85f}, {0, 0, 1}, {1.f, 0.f}},
        };
        m.indices = {0, 1, 2,  0, 2, 3};
        scene.add(SceneObject("quad", std::move(m), Mat4::translate({0, 0, -1.5f})));
    }

    // --- Box (green, 6 faces × 4 verts) ---
    {
        Mesh m; m.name = "box";
        m.vertices = {
            // front  +Z
            {{-0.3f,-0.3f, 0.3f},{0.2f,0.75f,0.2f},{0, 0, 1},{0.f,1.f}},
            {{ 0.3f,-0.3f, 0.3f},{0.2f,0.75f,0.2f},{0, 0, 1},{1.f,1.f}},
            {{ 0.3f, 0.3f, 0.3f},{0.2f,0.75f,0.2f},{0, 0, 1},{1.f,0.f}},
            {{-0.3f, 0.3f, 0.3f},{0.2f,0.75f,0.2f},{0, 0, 1},{0.f,0.f}},
            // back   -Z
            {{ 0.3f,-0.3f,-0.3f},{0.2f,0.75f,0.2f},{0, 0,-1},{0.f,1.f}},
            {{-0.3f,-0.3f,-0.3f},{0.2f,0.75f,0.2f},{0, 0,-1},{1.f,1.f}},
            {{-0.3f, 0.3f,-0.3f},{0.2f,0.75f,0.2f},{0, 0,-1},{1.f,0.f}},
            {{ 0.3f, 0.3f,-0.3f},{0.2f,0.75f,0.2f},{0, 0,-1},{0.f,0.f}},
            // top    +Y
            {{-0.3f, 0.3f, 0.3f},{0.3f,0.85f,0.3f},{0, 1, 0},{0.f,0.f}},
            {{ 0.3f, 0.3f, 0.3f},{0.3f,0.85f,0.3f},{0, 1, 0},{1.f,0.f}},
            {{ 0.3f, 0.3f,-0.3f},{0.3f,0.85f,0.3f},{0, 1, 0},{1.f,1.f}},
            {{-0.3f, 0.3f,-0.3f},{0.3f,0.85f,0.3f},{0, 1, 0},{0.f,1.f}},
            // bottom -Y
            {{-0.3f,-0.3f,-0.3f},{0.15f,0.6f,0.15f},{0,-1, 0},{0.f,0.f}},
            {{ 0.3f,-0.3f,-0.3f},{0.15f,0.6f,0.15f},{0,-1, 0},{1.f,0.f}},
            {{ 0.3f,-0.3f, 0.3f},{0.15f,0.6f,0.15f},{0,-1, 0},{1.f,1.f}},
            {{-0.3f,-0.3f, 0.3f},{0.15f,0.6f,0.15f},{0,-1, 0},{0.f,1.f}},
            // left   -X
            {{-0.3f,-0.3f,-0.3f},{0.2f,0.75f,0.2f},{-1, 0, 0},{0.f,1.f}},
            {{-0.3f,-0.3f, 0.3f},{0.2f,0.75f,0.2f},{-1, 0, 0},{1.f,1.f}},
            {{-0.3f, 0.3f, 0.3f},{0.2f,0.75f,0.2f},{-1, 0, 0},{1.f,0.f}},
            {{-0.3f, 0.3f,-0.3f},{0.2f,0.75f,0.2f},{-1, 0, 0},{0.f,0.f}},
            // right  +X
            {{ 0.3f,-0.3f, 0.3f},{0.2f,0.75f,0.2f},{1, 0, 0},{0.f,1.f}},
            {{ 0.3f,-0.3f,-0.3f},{0.2f,0.75f,0.2f},{1, 0, 0},{1.f,1.f}},
            {{ 0.3f, 0.3f,-0.3f},{0.2f,0.75f,0.2f},{1, 0, 0},{1.f,0.f}},
            {{ 0.3f, 0.3f, 0.3f},{0.2f,0.75f,0.2f},{1, 0, 0},{0.f,0.f}},
        };
        m.indices.reserve(36);
        for (uint32_t f = 0; f < 6; f++) {
            uint32_t b = f * 4;
            for (uint32_t idx : {b, b+1, b+2,  b, b+2, b+3})
                m.indices.push_back(idx);
        }
        Mat4 t = Mat4::translate({-0.6f, -0.3f, 0.f}) * Mat4::rotateY(0.5f) * Mat4::scale(0.8f);
        scene.add(SceneObject("box", std::move(m), t));
    }
}
