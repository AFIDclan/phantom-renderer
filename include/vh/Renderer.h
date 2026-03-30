#pragma once

#include "vh/Mesh.h"

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

// ----------------------------------------------------------------------------
// PushConstants — per-draw-call data pushed directly to the vertex shader.
// Must stay in sync with the layout in shaders/scene.vert.
// 3 × mat4 = 192 bytes (exceeds the 128-byte minimum guarantee; confirmed
// working on Tegra Orin).
// ----------------------------------------------------------------------------

struct PushConstants {
    Mat4 model;
    Mat4 view;
    Mat4 proj;
};

// Shadow pass push constants — model only (64 bytes). lightVP now comes from
// the ShadowFacesUBO, indexed by gl_InstanceIndex in the vertex shader.
struct ShadowPushConstants {
    Mat4 model;
};

// 6 face view-projection matrices for the layered shadow pass (std140, 384 bytes).
// Updated every frame and uploaded to shadowFacesUBO_.
struct ShadowFacesUBOData {
    Mat4 vp[6];
};

// CPU mirror of the per-frame UBO sent to the GPU.
struct SceneUBOData {
    float    lightPos[4];    // vec4, offset  0, size 16  (xyz = world light pos)
    float    lightDir[4];    // vec4, offset 16, size 16  (xyz = world light direction)
    float    lightColor[4];  // vec4, offset 32, size 16  (w = intensity; 0 = prelit)
    float    cameraPos[4];   // vec4, offset 48, size 16  (xyz = world eye pos)
    Mat4     lightVP;        // mat4, offset 64, size 64 (directional shadow matrix)
    float    cameraRight[4];   // vec4, offset 128, size 16 (world-space camera basis)
    float    cameraUp[4];      // vec4, offset 144, size 16 (world-space camera basis)
    float    cameraForward[4]; // vec4, offset 160, size 16 (world-space camera basis)
    float    skyboxParams[4];  // vec4, offset 176, size 16 (tanHalfFov, aspect, ambientIntensity, enabled)
    float    skyboxUp[4];      // vec4, offset 192, size 16 (world-space up for environment sampling)
    uint32_t renderMode;       // uint, offset 208, size  4  (0=unlit, 1=lambert, 2=pbr)
    float    shadowNear;       // float,offset 212, size  4  (shadow near plane)
    float    shadowFar;        // float,offset 216, size  4  (shadow far plane)
    uint32_t lightType;        // uint, offset 220, size  4  (mirrors LightType enum)
};

// Per-frame GPU timing results.
struct FrameTimings {
    double shadowPassMs = 0.0;
    double scenePassMs  = 0.0;
    double totalGpuMs   = 0.0;
};

// ----------------------------------------------------------------------------
// HeadlessRenderer — offscreen Vulkan renderer.
//
// Features:
//   • 4× MSAA resolve pass (no swapchain)
//   • 1024×1024 omnidirectional shadow map from the first scene Light (Point)
//   • Per-draw base-colour texture (white 1×1 fallback for untextured draws)
//   • GPU timestamp queries for per-pass timing
//
// Lifecycle:
//   1. Construct with resolution + SPIR-V directory.
//   2. Call init() once to set up the full Vulkan stack.
//   3. Call uploadScene() whenever geometry changes.
//   4. Call renderFrame() each tick to get RGBA pixels back on the CPU.
//   5. Destructor cleans up everything automatically.
// ----------------------------------------------------------------------------

class HeadlessRenderer {
public:
    HeadlessRenderer(uint32_t width, uint32_t height, const std::string& spirvDir);
    ~HeadlessRenderer();

    HeadlessRenderer(const HeadlessRenderer&)            = delete;
    HeadlessRenderer& operator=(const HeadlessRenderer&) = delete;

    // Build the full Vulkan stack (call once).
    void init();

    // Upload scene geometry + textures + lights; safe to call again.
    // Blocks until any in-flight GPU work finishes before replacing resources.
    void uploadScene(const RenderScene& scene);

    // Record + submit + readback one frame.  Geometry must have been uploaded.
    std::vector<uint8_t> renderFrame(const Mat4& view, const Mat4& proj);

    // Record + submit without CPU readback.  The rendered RGBA pixels are kept
    // in the GPU readback buffer.  Use exportReadbackMemFd() to import the
    // underlying memory into CUDA and read the result directly on the GPU.
    void renderNoReadback(const Mat4& view, const Mat4& proj);

    // Override the environment map path. Empty selects the packaged default.
    // Safe to call before or after init(); after init it hot-reloads the texture.
    void setSkyboxPath(const std::string& path);

    // Scalar multiplier for ambient lighting sampled from the skybox.
    void setSkyboxIntensity(float intensity) { skyboxIntensity_ = intensity; }

    // Select which world-space direction maps to the skybox's vertical axis.
    void setSkyboxUp(Vec3 up);

    // Export the readback buffer's VkDeviceMemory as an opaque POSIX fd
    // (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR).  The caller must
    // close the fd after importing it via cudaImportExternalMemory().
    // The buffer holds width()*height()*4 bytes, tightly packed RGBA8.
    int exportReadbackMemFd() const;

    // Select shading mode: 0=unlit, 1=lambert, 2=pbr (default).
    void setRenderMode(uint32_t mode) { renderMode_ = mode; }

    const FrameTimings& lastTimings() const { return lastTimings_; }

    uint32_t width()  const { return width_; }
    uint32_t height() const { return height_; }

private:
    // ---- Vulkan init --------------------------------------------------------
    void createInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createShadowAttachments();     // depth cube map + per-face views
    void createShadowRenderPass();
    void createShadowFramebuffers();    // creates shadowLayeredFB_ (layered, 6 layers)
    void createShadowFacesDescriptors(); // UBO + descriptor set for face VP matrices
    void createRenderPass();            // MSAA colour + depth + resolve
    void createOffscreenAttachments();
    void createFramebuffer();
    void createDescriptorSetLayouts();  // global (set 0) + tex (set 1)
    void createDescriptorPool();
    void createSceneUBO();
    void allocateGlobalDescriptorSet(); // binds UBO + shadow sampler
    void createShadowPipeline();
    void createSkyboxTexture();
    void destroySkyboxTexture();
    void updateSkyboxDescriptor();
    void createSkyboxPipeline();
    void createGraphicsPipeline();
    void createCommandPool();
    void createReadbackBuffer();
    void allocateCommandBuffer();
    void createTimestampPool();
    void createWhiteTexture();              // 1×1 white RGBA fallback image
    void createDefaultMetalRoughTexture();  // 1×1 default metalRough image + whiteTexDescSet_

    // ---- Geometry + texture management ------------------------------------
    void destroyGeometryBuffers();
    void destroyTextureResources();     // frees per-upload texture GPU objects
    void uploadTexturesToGPU(const std::vector<TextureData>& textures);
    // pool == VK_NULL_HANDLE uses the fixed init-time descriptorPool_.
    void createTextureDescriptorSet(VkImageView baseColorView, VkImageView metalRoughView,
                                    VkDescriptorSet& outSet,
                                    VkDescriptorPool pool = VK_NULL_HANDLE);

    // ---- Vulkan utilities ---------------------------------------------------
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags memProps,
                      VkBuffer& buffer, VkDeviceMemory& memory);
    void uploadToDevice(VkBuffer dst, const void* data, VkDeviceSize size);
    void createImage(uint32_t w, uint32_t h, VkFormat fmt,
                     VkImageUsageFlags usage, VkMemoryPropertyFlags memProps,
                     VkImage& img, VkDeviceMemory& mem,
                     VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);
    VkImageView    createImageView(VkImage img, VkFormat fmt, VkImageAspectFlags aspect);
    VkShaderModule loadShader(const std::string& path);
    VkShaderModule loadShaderFromMemory(const uint32_t* code, size_t byteSize);
    void transitionImageLayout(VkImage img, VkImageLayout oldLayout,
                               VkImageLayout newLayout, VkImageAspectFlags aspect);
    void uploadImageToGPU(VkImage dst, const void* data, uint32_t w, uint32_t h);

    // ---- Per-frame ----------------------------------------------------------
    void                 recordCommandBuffer(const Mat4& view, const Mat4& proj);
    void                 submitAndWait();
    std::vector<uint8_t> readbackPixels();
    void                 readTimestamps();

    void cleanup();

    // ---- state --------------------------------------------------------------
    uint32_t    width_, height_;
    std::string spirvDir_;
    std::string skyboxPath_;

    std::vector<DrawCall>       drawCalls_;
    SceneUBOData                uboData_         = {};
    uint32_t                    renderMode_      = 2;  // 0=unlit, 1=lambert, 2=pbr
    float                       skyboxIntensity_ = 0.25f;
    Vec3                        skyboxUp_        = {0.f, 1.f, 0.f};
    FrameTimings                lastTimings_;
    float                       timestampPeriod_ = 1.f;
    bool                        hasTimestamps_   = false;
    VkSampleCountFlagBits       msaaSamples_     = VK_SAMPLE_COUNT_4_BIT;
    static constexpr uint32_t   SHADOW_MAP_SIZE  = 1024;
    static constexpr float      SHADOW_NEAR      = 0.005f;
    static constexpr float      SHADOW_FAR       = 10.0f;

    // Vulkan core
    VkInstance            instance_    = VK_NULL_HANDLE;
    VkPhysicalDevice      physDev_     = VK_NULL_HANDLE;
    VkDevice              device_      = VK_NULL_HANDLE;
    uint32_t              queueFamily_ = 0;
    VkQueue               queue_       = VK_NULL_HANDLE;

    // Shadow cube map (omnidirectional point light)
    VkImage               shadowCubeImage_          = VK_NULL_HANDLE;
    VkDeviceMemory        shadowCubeImageMem_        = VK_NULL_HANDLE;
    VkImageView           shadowCubeView_            = VK_NULL_HANDLE;  // cube view for sampling
    VkImageView           shadowLayeredView_          = VK_NULL_HANDLE;  // 2D-array view for layered FB
    VkImage               shadowDirImage_            = VK_NULL_HANDLE;
    VkDeviceMemory        shadowDirImageMem_         = VK_NULL_HANDLE;
    VkImageView           shadowDirView_             = VK_NULL_HANDLE;  // 2D view for directional shadows
    VkSampler             shadowSampler_             = VK_NULL_HANDLE;
    VkSampler             skyboxSampler_             = VK_NULL_HANDLE;
    VkRenderPass          shadowRenderPass_          = VK_NULL_HANDLE;
    VkFramebuffer         shadowLayeredFB_           = VK_NULL_HANDLE;  // single layered framebuffer
    VkFramebuffer         shadowDirFB_               = VK_NULL_HANDLE;  // single 2D framebuffer
    VkBuffer              shadowFacesUBO_            = VK_NULL_HANDLE;  // 6 × mat4 face VPs
    VkDeviceMemory        shadowFacesUBOMem_         = VK_NULL_HANDLE;
    VkDescriptorSetLayout shadowFacesLayout_         = VK_NULL_HANDLE;
    VkDescriptorSet       shadowFacesDescSet_        = VK_NULL_HANDLE;
    VkPipelineLayout      shadowPipelineLayout_      = VK_NULL_HANDLE;
    VkPipeline            shadowPipeline_            = VK_NULL_HANDLE;
    VkImage               skyboxImage_               = VK_NULL_HANDLE;
    VkDeviceMemory        skyboxImageMem_            = VK_NULL_HANDLE;
    VkImageView           skyboxView_                = VK_NULL_HANDLE;

    // Main render pass (MSAA)
    VkRenderPass          renderPass_     = VK_NULL_HANDLE;
    VkFramebuffer         framebuffer_    = VK_NULL_HANDLE;
    VkImage               msaaColorImage_    = VK_NULL_HANDLE;
    VkDeviceMemory        msaaColorImageMem_ = VK_NULL_HANDLE;
    VkImageView           msaaColorView_     = VK_NULL_HANDLE;
    VkImage               msaaDepthImage_    = VK_NULL_HANDLE;
    VkDeviceMemory        msaaDepthImageMem_ = VK_NULL_HANDLE;
    VkImageView           msaaDepthView_     = VK_NULL_HANDLE;
    VkImage               resolveImage_      = VK_NULL_HANDLE;
    VkDeviceMemory        resolveImageMem_   = VK_NULL_HANDLE;
    VkImageView           resolveImageView_  = VK_NULL_HANDLE;

    // Readback
    VkBuffer              readbackBuffer_    = VK_NULL_HANDLE;
    VkDeviceMemory        readbackBufferMem_ = VK_NULL_HANDLE;

    // Vulkan external-memory extension (loaded after device creation)
    PFN_vkGetMemoryFdKHR  vkGetMemoryFdKHR_  = nullptr;

    // Persistent fence (reused every frame — no create/destroy overhead)
    VkFence               frameFence_        = VK_NULL_HANDLE;

    // Descriptor set layouts + pools
    VkDescriptorSetLayout globalSetLayout_  = VK_NULL_HANDLE; // set 0
    VkDescriptorSetLayout texSetLayout_     = VK_NULL_HANDLE; // set 1
    VkDescriptorPool      descriptorPool_   = VK_NULL_HANDLE; // fixed init-time pool (3 sets)
    VkDescriptorPool      sceneDescPool_    = VK_NULL_HANDLE; // per-scene pool, rebuilt each uploadScene()

    // Scene UBO  (set 0, binding 0)
    VkBuffer              sceneUBO_    = VK_NULL_HANDLE;
    VkDeviceMemory        sceneUBOMem_ = VK_NULL_HANDLE;

    // Global descriptor set (set 0)
    VkDescriptorSet       globalDescSet_ = VK_NULL_HANDLE;

    // White 1×1 fallback base-colour texture
    VkImage               whiteTexImage_    = VK_NULL_HANDLE;
    VkDeviceMemory        whiteTexImageMem_ = VK_NULL_HANDLE;
    VkImageView           whiteTexView_     = VK_NULL_HANDLE;
    VkSampler             texSampler_       = VK_NULL_HANDLE; // shared by all textures
    VkDescriptorSet       whiteTexDescSet_  = VK_NULL_HANDLE; // fallback: white + default MR

    // Default 1×1 metallic-roughness texture (G=255 rough=1, B=0 metallic=0)
    VkImage               defaultMetalRoughImage_    = VK_NULL_HANDLE;
    VkDeviceMemory        defaultMetalRoughMem_      = VK_NULL_HANDLE;
    VkImageView           defaultMetalRoughView_     = VK_NULL_HANDLE;

    // Per-scene textures (managed by uploadScene)
    std::vector<VkImage>          texImages_;
    std::vector<VkDeviceMemory>   texImageMems_;
    std::vector<VkImageView>      texImageViews_;

    // Per-draw descriptor sets (one per draw call: binding 0=baseColor, 1=metalRough)
    std::vector<VkDescriptorSet>  drawDescSets_;

    // Main graphics pipeline
    VkPipelineLayout      pipelineLayout_   = VK_NULL_HANDLE;
    VkPipeline            pipeline_         = VK_NULL_HANDLE;
    VkPipelineLayout      skyboxPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline            skyboxPipeline_       = VK_NULL_HANDLE;

    // Commands
    VkCommandPool         cmdPool_   = VK_NULL_HANDLE;
    VkCommandBuffer       cmdBuffer_ = VK_NULL_HANDLE;

    // Geometry buffers
    VkBuffer              vertexBuffer_    = VK_NULL_HANDLE;
    VkDeviceMemory        vertexBufferMem_ = VK_NULL_HANDLE;
    VkBuffer              indexBuffer_     = VK_NULL_HANDLE;
    VkDeviceMemory        indexBufferMem_  = VK_NULL_HANDLE;

    // GPU timestamp query pool (4 slots: shadowStart/End, sceneStart/End)
    VkQueryPool           queryPool_ = VK_NULL_HANDLE;
};
