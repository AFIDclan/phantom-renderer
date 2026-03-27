# vulcan-headless

Headless Vulkan renderer for the Daxos vision pipeline. Renders glTF 2.0 scenes offscreen to RGBA frames — no display or swapchain required.

**Destination**: will be wired into `/git/metavision` as a synthetic video source, replacing the CUDA renderer in `extern/cuda-renderer`.

---

## Platform

- **Hardware**: Jetson Orin (nvgpu), arm64, Ubuntu 22.04 Jammy
- **Vulkan**: instance 1.3.204 / device 1.3.251, no swapchain (offscreen only)
- **Push constant limit**: spec guarantees 128 bytes; Orin supports 192 bytes (3 × mat4) — confirmed working, do not increase without re-testing
- **Shader compiler**: `glslangValidator` (from `glslang-tools`). `glslc` is also installed and detected by CMake, which prefers it. Both produce compatible SPIR-V.

---

## Build & Run

```bash
# First time (installs deps + cmake configure + build)
bash build.sh

# Subsequent builds
cmake --build build/

# Render Scene.gltf → build/render_output.png
cmake --build build/ --target run

# Run manually with explicit arguments
./build/vulcan-headless \
  --scene Scene.gltf \
  --output out.png \
  --spirv-dir build/shaders \
  --width 1280 --height 720
```

SPIR-V files are compiled by CMake into `build/shaders/`. The `run` target passes `--spirv-dir` automatically. tinygltf headers are downloaded to `build/vendor/tinygltf/` at configure time — no network needed after the first configure.

---

## File Structure

```
include/vh/
  Math.h          — Vec3 helpers + Mat4 (column-major, m[col][row]); fully inline
  Mesh.h          — Vertex, Mesh, DrawCall, RenderScene (POD data types only)
  Scene.h         — Camera, SceneObject, Scene (scene graph)
  GLTFLoader.h    — loadGLTF(), loadDemoScene() declarations
  Renderer.h      — HeadlessRenderer (all Vulkan)
  FrameOutput.h   — IFrameOutput interface + PngFileOutput

src/
  Scene.cpp       — Scene graph implementation
  GLTFLoader.cpp  — tinygltf impl (TINYGLTF_IMPLEMENTATION defined here only)
  Renderer.cpp    — All Vulkan setup, upload, and per-frame rendering
  FrameOutput.cpp — PNG writer (libpng)
  main.cpp        — CLI entry point (~100 lines)

shaders/
  scene.vert / scene.frag  — GLSL sources; compiled to build/shaders/*.spv
```

**Do not** define `TINYGLTF_IMPLEMENTATION`, `STB_IMAGE_IMPLEMENTATION`, or `STB_IMAGE_WRITE_IMPLEMENTATION` in any file other than `src/GLTFLoader.cpp`.

---

## Key Architectural Decisions

### Geometry pipeline
- All mesh primitives are merged into a single GPU vertex + index buffer per `uploadScene()` call.
- Per-draw transforms are pushed with a `PushConstants` struct (model + view + proj mat4).
- Index buffers are always `VK_INDEX_TYPE_UINT32`; uint8/uint16 indices from glTF are upconverted in `GLTFLoader.cpp`.
- NORMAL accessor stride is read from `bufferView.byteStride` — handles Blender's interleaved output.

### Camera
- `Camera` lives on `Scene` and produces view/proj matrices via `viewMatrix()` / `projMatrix(aspect)`.
- main.cpp sets the camera after loading the scene — override per use-case there.

### Output abstraction (`IFrameOutput`)
- `HeadlessRenderer::renderFrame()` returns raw `std::vector<uint8_t>` (RGBA, row-major).
- `main.cpp` holds a `std::unique_ptr<IFrameOutput>` and calls `writeFrame()`.
- **Metavision integration point**: implement `MetavisionFrameOutput : public IFrameOutput` and swap it in `main.cpp`. The renderer needs no changes.

### Dynamic scene
- `Scene::add()` / `remove()` / `setTransform()` are the public API for scene mutation.
- After mutating, call `renderer.uploadScene(scene.buildRenderScene())` to re-upload geometry, or just update push-constant transforms if only poses changed (future optimisation).

---

## Planned Extensions (not yet implemented — keep compatible)

1. **Physics** — `SceneObject` is the attachment point for a rigid-body handle; `setTransform()` would be driven by the physics tick.
2. **Dynamic objects** — `Scene::add/remove` already exists; future work is batching re-uploads efficiently.
3. **Animation** — skeletal/blend-shape animation would swap or modify `Mesh` vertices per frame inside `SceneObject`; the render path is unchanged.

---

## Conventions

- Headers use `#pragma once`; no `.hpp` extension.
- All Vulkan calls are wrapped with `VK_CHECK()` (defined in `Renderer.cpp`).
- Math is self-contained in `Math.h` — do not add GLM or similar without discussion.
- `buildRenderScene()` is intentionally simple (full rebuild). Profile before caching.
