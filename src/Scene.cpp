#include "vh/Scene.h"

#include <algorithm>
#include <stdexcept>

// ----------------------------------------------------------------------------
// SceneObject
// ----------------------------------------------------------------------------

SceneObject::SceneObject(std::string id, Mesh mesh, Mat4 transform)
    : id_(std::move(id)), mesh_(std::move(mesh)), transform_(transform) {}

// ----------------------------------------------------------------------------
// Scene — object management
// ----------------------------------------------------------------------------

void Scene::add(SceneObject obj) {
    objects_.push_back(std::move(obj));
}

void Scene::remove(const std::string& id) {
    auto it = std::find_if(objects_.begin(), objects_.end(),
                           [&id](const SceneObject& o){ return o.id() == id; });
    if (it != objects_.end())
        objects_.erase(it);
}

void Scene::setTransform(const std::string& id, const Mat4& t) {
    if (auto* obj = findObject(id))
        obj->setTransform(t);
}

bool Scene::hasObject(const std::string& id) const {
    return std::any_of(objects_.begin(), objects_.end(),
                       [&id](const SceneObject& o){ return o.id() == id; });
}

SceneObject* Scene::findObject(const std::string& id) {
    auto it = std::find_if(objects_.begin(), objects_.end(),
                           [&id](const SceneObject& o){ return o.id() == id; });
    return it != objects_.end() ? &*it : nullptr;
}

// ----------------------------------------------------------------------------
// Scene — texture / light management
// ----------------------------------------------------------------------------

int Scene::addTexture(TextureData td) {
    textures_.push_back(std::move(td));
    return static_cast<int>(textures_.size()) - 1;
}

void Scene::addLight(Light light) {
    lights_.push_back(light);
}

// ----------------------------------------------------------------------------
// Scene — GPU packaging
// ----------------------------------------------------------------------------

RenderScene Scene::buildRenderScene() const {
    RenderScene rs;
    rs.textures = textures_;  // forward the texture table
    rs.lights   = lights_;    // forward scene lights

    for (const auto& obj : objects_) {
        const Mesh& mesh = obj.mesh();

        uint32_t firstIndex  = static_cast<uint32_t>(rs.indices.size());
        int32_t  vertexStart = static_cast<int32_t>(rs.vertices.size());

        rs.vertices.insert(rs.vertices.end(),
                           mesh.vertices.begin(), mesh.vertices.end());
        rs.indices.insert(rs.indices.end(),
                          mesh.indices.begin(),  mesh.indices.end());

        rs.draws.push_back({
            static_cast<uint32_t>(mesh.indices.size()),
            firstIndex,
            vertexStart,
            obj.transform(),
            mesh.textureIdx,
            mesh.metalRoughTexIdx
        });
    }
    return rs;
}
