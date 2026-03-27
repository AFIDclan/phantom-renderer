/**
 * vulcan-headless: entry point
 *
 * Parses CLI arguments, loads a scene (glTF or built-in demo),
 * renders one frame with the headless Vulkan renderer, and writes
 * the result through the configured IFrameOutput.
 *
 * Usage:
 *   vulcan-headless [--scene <path.gltf>] [--output <out.png>]
 *                  [--width <px>] [--height <px>] [--spirv-dir <dir>]
 *                  [--render-mode unlit|lambert|pbr]
 *
 * --spirv-dir is optional; omitting it uses the SPIR-V shaders compiled into
 * the binary at build time.  Supplying it overrides with shaders from disk.
 */

#include "vh/FrameOutput.h"
#include "vh/GLTFLoader.h"
#include "vh/Renderer.h"
#include "vh/Scene.h"

#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    uint32_t    width      = 1280;
    uint32_t    height     = 720;
    std::string outputPath = "render_output.png";
    std::string spirvDir   = "";   // empty = use embedded SPIR-V blobs
    std::string scenePath  = "";
    uint32_t    renderMode = 2;  // 0=unlit, 1=lambert, 2=pbr (default)
    // Optional camera overrides (all three must be given together to take effect)
    float eyeX   = 0, eyeY   = 0, eyeZ   = 0;
    float centX  = 0, centY  = 0, centZ  = 0;
    float fovY   = 0;
    bool  hasEye = false, hasCenter = false, hasFov = false;
    // Benchmark parameters
    int warmupFrames    = 5;   // frames rendered before timing starts
    int benchmarkFrames = 30;  // frames included in the timed average

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--scene"       && i+1 < argc) scenePath  = argv[++i];
        else if (a == "--output"      && i+1 < argc) outputPath = argv[++i];
        else if (a == "--width"       && i+1 < argc) width      = std::stoul(argv[++i]);
        else if (a == "--height"      && i+1 < argc) height     = std::stoul(argv[++i]);
        else if (a == "--spirv-dir"   && i+1 < argc) spirvDir   = argv[++i];
        else if (a == "--render-mode" && i+1 < argc) {
            std::string m = argv[++i];
            if      (m == "unlit")   renderMode = 0;
            else if (m == "lambert") renderMode = 1;
            else if (m == "pbr")     renderMode = 2;
            else { fprintf(stderr, "Unknown render mode '%s'; use unlit|lambert|pbr\n", m.c_str()); return 1; }
        }
        else if (a == "--eye"    && i+3 < argc) { eyeX  = std::stof(argv[++i]); eyeY  = std::stof(argv[++i]); eyeZ  = std::stof(argv[++i]); hasEye    = true; }
        else if (a == "--center" && i+3 < argc) { centX = std::stof(argv[++i]); centY = std::stof(argv[++i]); centZ = std::stof(argv[++i]); hasCenter = true; }
        else if (a == "--fov"    && i+1 < argc) { fovY  = std::stof(argv[++i]); hasFov = true; }
        else if (a == "--warmup"  && i+1 < argc) { warmupFrames    = std::stoi(argv[++i]); }
        else if (a == "--frames"  && i+1 < argc) { benchmarkFrames = std::stoi(argv[++i]); }
    }

    // Build scene
    Scene scene;
    if (scenePath.empty()) {
        printf("No scene specified — using built-in demo.\n");
        loadDemoScene(scene);
        // Demo camera: close-up view of the objects near the origin
        scene.camera() = Camera{
            .eye    = {0.f, 0.5f, 3.f},
            .center = {0.f, 0.f,  0.f},
            .up     = {0.f, 1.f,  0.f},
            .fovY   = 0.85f,
            .nearZ  = 0.1f,
            .farZ   = 100.f,
        };
    } else {
        printf("Loading scene: %s\n", scenePath.c_str());
        if (!loadGLTF(scenePath, scene)) {
            fprintf(stderr, "Failed to load scene — falling back to demo.\n");
            loadDemoScene(scene);
        }
        // Camera tuned for Scene.gltf: cube(0,0,0), cylinder(-2,0,-3), disc(2,0,-4), floor
        scene.camera() = Camera{
            .eye    = { 2.f, 5.f, 10.f},
            .center = { 0.f, 0.f, -2.f},
            .up     = { 0.f, 1.f,  0.f},
            .fovY   = 0.85f,
            .nearZ  = 0.1f,
            .farZ   = 100.f,
        };
    }

    // Apply CLI camera overrides if provided
    if (hasEye)    { scene.camera().eye    = {eyeX, eyeY, eyeZ}; }
    if (hasCenter) { scene.camera().center = {centX, centY, centZ}; }
    if (hasFov)    { scene.camera().fovY   = fovY; }

    if (scene.objectCount() == 0) {
        fprintf(stderr, "Scene has no objects — nothing to render.\n");
        return 1;
    }

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    Mat4 view = scene.camera().viewMatrix();
    Mat4 proj = scene.camera().projMatrix(aspect);

    try {
        // Output sink — swap this for a MetavisionFrameOutput in the future
        std::unique_ptr<IFrameOutput> output = std::make_unique<PngFileOutput>(outputPath);

        printf("Initialising Vulkan headless renderer (%ux%u)...\n", width, height);
        HeadlessRenderer renderer(width, height, spirvDir);
        renderer.init();
        renderer.setRenderMode(renderMode);
        const char* modeNames[] = {"unlit", "lambert", "pbr"};
        printf("Render mode: %s\n", modeNames[renderMode < 3 ? renderMode : 2]);

        printf("Uploading %zu object(s)...\n", scene.objectCount());
        renderer.uploadScene(scene.buildRenderScene());

        // Warmup — let the GPU JIT shaders and reach steady state.
        printf("Warmup (%d frame(s))...\n", warmupFrames);
        for (int i = 0; i < warmupFrames; ++i)
            renderer.renderFrame(view, proj);

        // Timed benchmark loop.
        printf("Benchmark (%d frame(s))...\n", benchmarkFrames);
        double sumCpuMs    = 0, sumShadowMs = 0, sumSceneMs = 0, sumGpuMs = 0;
        double minGpuMs    = 1e9, maxGpuMs  = 0;
        std::vector<uint8_t> pixels;
        for (int i = 0; i < benchmarkFrames; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            pixels   = renderer.renderFrame(view, proj);
            auto t1  = std::chrono::high_resolution_clock::now();
            sumCpuMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
            const auto& tim = renderer.lastTimings();
            sumShadowMs += tim.shadowPassMs;
            sumSceneMs  += tim.scenePassMs;
            sumGpuMs    += tim.totalGpuMs;
            minGpuMs     = std::min(minGpuMs, tim.totalGpuMs);
            maxGpuMs     = std::max(maxGpuMs, tim.totalGpuMs);
        }
        double n = static_cast<double>(benchmarkFrames);
        printf("Results (%d frames, %d warmup):\n", benchmarkFrames, warmupFrames);
        printf("  CPU (submit+readback) avg : %6.3f ms\n",  sumCpuMs    / n);
        printf("  GPU shadow pass      avg : %6.3f ms\n",  sumShadowMs / n);
        printf("  GPU scene pass       avg : %6.3f ms\n",  sumSceneMs  / n);
        printf("  GPU total            avg : %6.3f ms  |  min %.3f ms  |  max %.3f ms\n",
               sumGpuMs / n, minGpuMs, maxGpuMs);
        printf("  Theoretical max FPS      : %.1f\n", 1000.0 / (sumGpuMs / n));

        output->writeFrame(pixels.data(), width, height);
        printf("Written to: %s\n", outputPath.c_str());

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
