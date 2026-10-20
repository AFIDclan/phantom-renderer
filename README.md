# phantom-renderer

Headless Vulkan renderer for glTF 2.0 scenes. It renders offscreen to RGBA8 frames with no
display, window or swapchain, and is the synthetic camera behind metavision's `RenderSource`
on the Daxos virtual platforms. A small CLI renders a scene to a PNG for testing.

Target hardware is the Jetson Orin (arm64, Ubuntu 22.04, Vulkan 1.3), but nothing in the code
is Tegra specific beyond the push-constant budget noted in `AGENTS.md`.

## Features

- glTF 2.0 and GLB loading through tinygltf: node hierarchy with TRS or matrix transforms,
  indexed and non-indexed triangles, `POSITION`, `NORMAL`, `TEXCOORD_0`, uint8/16/32 indices.
- Materials: `baseColorFactor`, `baseColorTexture`, `metallicRoughnessTexture`, and
  `alphaMode` (`OPAQUE`, `MASK` with `alphaCutoff`, `BLEND`). Masked materials are cut out in
  the main pass; blended ones are drawn after all opaque geometry, back to front, without depth
  writes. Everything is drawn double-sided.
- Lights from `KHR_lights_punctual`: one point light (cube shadow map with PCF) or one
  directional light (fitted orthographic shadow map).
- Three shading modes selected at runtime: `unlit` (base colour only, for pre-baked scenes),
  `lambert`, and `pbr` (Cook-Torrance with Reinhard tone mapping).
- Equirectangular HDR skybox (`resources/skybox-1.hdr` by default) drawn as a background fill
  and used as ambient light in the lit modes. The world up axis is configurable.
- MSAA (up to 4x), 32-bit float depth with reversed-Z, so far planes of several kilometres are fine.
- Two output paths: a CPU readback (`renderFrame`) and a GPU-resident readback buffer whose
  memory can be exported as a POSIX fd and imported into CUDA (`renderNoReadback` +
  `exportReadbackMemFd`), which is how metavision consumes it without a copy through host memory.
- SPIR-V is compiled by CMake and embedded into the library, so the binary works without a
  shader directory; `--spirv-dir` overrides it with shaders from disk for quick iteration.

## Building

Dependencies: a C++17 compiler, CMake, `libvulkan-dev`, `libpng-dev`, and `glslang-tools`
(or `glslc`). tinygltf and stb are downloaded into `build/vendor/` at configure time.

```bash
bash build.sh                 # installs the apt packages, configures and builds into build/
cmake --build build/          # rebuild
cmake --build build/ --target run   # renders Scene.gltf (if present) to build/render_output.png
```

Inside metavision the repo is a git submodule under `extern/phantom-renderer` and is built by
the parent project's CMake; the CLI lands in the parent's `build/bin/phantom-renderer`.

`cmake --install` puts `libvh.a`, the CLI and `resources/` under the usual prefix
(`share/phantom-renderer/resources`). The skybox is looked up in `$PHANTOM_RENDERER_RESOURCES_DIR`,
then `./resources`, then the build, source and install resource directories.

## Command line

```bash
build/phantom-renderer \
  --scene /etc/daxos/virtual-scenes/campus_556.glb \
  --render-mode unlit \
  --eye 0 0 1 --center 0 10 1 --up 0 0 1 \
  --far 10000 \
  --width 1280 --height 720 \
  --warmup 1 --frames 1 \
  --output campus.png
```

| Flag | Meaning |
|---|---|
| `--scene <path>` | glTF or GLB file. Omit it for a built-in demo scene. |
| `--output <png>` | Output image (default `render_output.png`). |
| `--width`, `--height` | Render size in pixels (default 1280x720). |
| `--render-mode unlit\|lambert\|pbr` | Shading mode (default `pbr`). |
| `--eye x y z`, `--center x y z`, `--up x y z` | Camera pose. The default up is +Y; Daxos scenes are Z-up, so pass `--up 0 0 1`. |
| `--yaw-deg <deg>` | Alternative to `--center`: look horizontally from `--eye`. |
| `--fov <rad>` | Vertical field of view (default 0.85). |
| `--near <m>`, `--far <m>` | Clip planes (default 0.1 and 100). |
| `--skybox <hdr>`, `--skybox-intensity <f>` | Environment map and its ambient multiplier. |
| `--spirv-dir <dir>` | Load `*.spv` from disk instead of the embedded blobs. |
| `--warmup <n>`, `--frames <n>` | Frames rendered before and during timing (defaults 5 and 30). The last timed frame is written. |

The CLI prints CPU and GPU timings for the shadow and scene passes.

## Library use

```cpp
#include "vh/GLTFLoader.h"
#include "vh/Renderer.h"
#include "vh/Scene.h"

Scene scene;
loadGLTF("scene.glb", scene);

HeadlessRenderer renderer(1920, 1080, /*spirvDir=*/"");
renderer.setSkyboxUp({0.f, 0.f, 1.f});
renderer.init();
renderer.setRenderMode(0);                      // unlit
renderer.uploadScene(scene.buildRenderScene()); // call again after scene edits

Mat4 view = Mat4::lookAt({0, 0, 1}, {0, 10, 1}, {0, 0, 1});
Mat4 proj = Mat4::perspective(0.85f, 16.f / 9.f, 0.05f, 10000.f);
std::vector<uint8_t> rgba = renderer.renderFrame(view, proj);
```

`Scene` holds `SceneObject`s (one `Mesh` plus a world transform) and supports `add`, `remove`
and `setTransform`; `buildRenderScene()` flattens it into the merged vertex and index buffers,
draw list, textures and lights the renderer uploads. `IFrameOutput` in `vh/FrameOutput.h` is the
sink abstraction; `PngFileOutput` is the only implementation here.

Matrices are column-major (`m[col][row]`). `Mat4::perspective` produces a Vulkan projection with
the Y flip and the reversed-Z mapping the main pipeline expects (depth clears to 0, test is
`GREATER`). A hand-built projection must use the same mapping: `m[2][2] = n / (f - n)`,
`m[3][2] = f n / (f - n)`, `m[2][3] = -1`.

## Layout

```
include/vh/   Math.h, Mesh.h, Scene.h, GLTFLoader.h, Renderer.h, FrameOutput.h
src/          Scene.cpp, GLTFLoader.cpp, Renderer.cpp, FrameOutput.cpp, main.cpp
shaders/      scene.*, shadow.*, skybox.* (GLSL, compiled to build/shaders/*.spv and embedded)
resources/    default skybox
example_scenes/  Scene.gltf, ABeautifulGame.gltf, Sponza.glb (Sponza has alpha-masked plants)
cmake/        spv_to_header.cmake (SPIR-V embedding)
```

`AGENTS.md` records the architectural decisions and the constraints to respect when changing
the code (push-constant size, where tinygltf is instantiated, math conventions).

## Known limitations

- Only `TEXCOORD_0`; glTF samplers, normal, occlusion and emissive maps, `KHR_materials_*`
  extensions, cameras, skins and animations are ignored. 1- and 2-channel images are skipped.
- Textures are uploaded as RGBA8 UNORM without mipmaps, so fine detail shimmers at distance.
- One light per scene. The shadow pass does not alpha-test, so masked materials cast the shadow
  of their full quad in the lit modes.
- No culling of any kind: every draw is issued every frame.
