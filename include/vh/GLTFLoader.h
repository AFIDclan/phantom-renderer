#pragma once

#include <string>

// Forward declaration — avoids pulling Scene.h (and its transitive includes)
// into everything that only needs to call these loader functions.
class Scene;

// ----------------------------------------------------------------------------
// loadGLTF — loads a .gltf or .glb file and appends the contained meshes as
// SceneObjects into the given scene.  Returns false on error.
//
// Node hierarchy, TRS transforms, matrix transforms, interleaved vertex data,
// uint16/uint32 index buffers, and baseColorFactor materials are all handled.
// ----------------------------------------------------------------------------
bool loadGLTF(const std::string& path, Scene& scene);

// ----------------------------------------------------------------------------
// loadDemoScene — populates the scene with a hard-coded triangle + quad + box,
// useful for quick smoke-tests without a glTF file on disk.
// ----------------------------------------------------------------------------
void loadDemoScene(Scene& scene);
