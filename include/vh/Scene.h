#pragma once

#include "vh/Math.h"
#include "vh/Mesh.h"

#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Camera — defines the viewpoint.  Produces view and projection matrices on
// demand so callers never manipulate raw Mat4s for camera movement.
// ----------------------------------------------------------------------------

struct Camera {
    Vec3  eye    = { 2.f,  5.f, 10.f };
    Vec3  center = { 0.f,  0.f, -2.f };
    Vec3  up     = { 0.f,  1.f,  0.f };
    float fovY   = 0.85f;   // radians
    float nearZ  = 0.1f;
    float farZ   = 100.f;

    Mat4 viewMatrix() const {
        return Mat4::lookAt(eye, center, up);
    }

    Mat4 projMatrix(float aspectRatio) const {
        return Mat4::perspective(fovY, aspectRatio, nearZ, farZ);
    }
};

// ----------------------------------------------------------------------------
// SceneObject — one renderable entity: a Mesh plus a world-space transform.
//
// Designed to be extended:
//   • Physics body — attach a rigid-body handle here in the future.
//   • Animator     — swap out mesh vertices per frame for skeletal/blend shapes.
// ----------------------------------------------------------------------------

class SceneObject {
public:
    SceneObject(std::string id, Mesh mesh, Mat4 transform = Mat4::identity());

    const std::string& id()        const { return id_; }
    const Mesh&        mesh()      const { return mesh_; }
    const Mat4&        transform() const { return transform_; }

    void setTransform(const Mat4& t) { transform_ = t; }

    // Future: void setMesh(Mesh m) for animation frame swaps

private:
    std::string id_;
    Mesh        mesh_;
    Mat4        transform_;
};

// ----------------------------------------------------------------------------
// Scene — the authoritative scene graph.
//
// Objects can be added and removed at any time (future dynamic insertion of
// cars, pedestrians, etc.).  Call buildRenderScene() to produce a flat
// packed struct ready for GPU upload.
// ----------------------------------------------------------------------------

class Scene {
public:
    // ---- object management --------------------------------------------------
    void add(SceneObject obj);
    void remove(const std::string& id);
    void setTransform(const std::string& id, const Mat4& t);

    bool         hasObject(const std::string& id) const;
    SceneObject* findObject(const std::string& id);
    size_t       objectCount() const { return objects_.size(); }

    const std::vector<SceneObject>& objects() const { return objects_; }

    // ---- camera -------------------------------------------------------------
    Camera&       camera()       { return camera_; }
    const Camera& camera() const { return camera_; }

    // ---- texture store ------------------------------------------------------
    // Add a decoded RGBA8 texture image and return its index (used in Mesh::textureIdx).
    int addTexture(TextureData td);

    const std::vector<TextureData>& textures() const { return textures_; }

    // ---- light store --------------------------------------------------------
    void addLight(Light light);

    const std::vector<Light>& lights() const { return lights_; }

    // ---- GPU packaging ------------------------------------------------------
    // Merges all object geometry into a single flat RenderScene.  Cheap enough
    // to call every frame for small-to-medium scenes; cache the result if
    // profiling shows it matters.
    RenderScene buildRenderScene() const;

private:
    std::vector<SceneObject> objects_;
    Camera                   camera_;
    std::vector<TextureData> textures_;
    std::vector<Light>        lights_;
};
