#include "vh/Renderer.h"
#include "vh/SpirvBlobs.h"

#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Vulkan error-checking macro
// ----------------------------------------------------------------------------

#define VK_CHECK(expr)                                                              \
    do {                                                                             \
        VkResult _r = (expr);                                                        \
        if (_r != VK_SUCCESS)                                                        \
            throw std::runtime_error(std::string(#expr) + " failed, VkResult=" +    \
                                     std::to_string(static_cast<int>(_r)));          \
    } while (0)

// ----------------------------------------------------------------------------
// File-scope utilities
// ----------------------------------------------------------------------------

static std::vector<uint32_t> readSPIRV(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Cannot open SPIR-V file: " + path);
    auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint32_t> buf(sz / 4);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    return buf;
}

static uint32_t findMemoryType(VkPhysicalDevice pd,
                                uint32_t typeBits,
                                VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("No compatible Vulkan memory type found");
}

static VkSampleCountFlagBits getMaxUsableSampleCount(VkPhysicalDevice pd) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts
                              & props.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
    if (counts & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;
    return VK_SAMPLE_COUNT_1_BIT;
}

// ----------------------------------------------------------------------------
// Constructor / destructor
// ----------------------------------------------------------------------------

HeadlessRenderer::HeadlessRenderer(uint32_t width, uint32_t height,
                                   const std::string& spirvDir)
    : width_(width), height_(height), spirvDir_(spirvDir) {}

HeadlessRenderer::~HeadlessRenderer() {
    cleanup();
}

// ----------------------------------------------------------------------------
// Public — init
// ----------------------------------------------------------------------------

void HeadlessRenderer::init() {
    createInstance();
    pickPhysicalDevice();
    createLogicalDevice();
    createShadowAttachments();
    createShadowRenderPass();
    createShadowFramebuffers();
    createRenderPass();
    createOffscreenAttachments();
    createFramebuffer();
    createCommandPool();
    createDescriptorSetLayouts();
    createDescriptorPool();
    createSceneUBO();
    createWhiteTexture();
    createDefaultMetalRoughTexture();
    allocateGlobalDescriptorSet();
    createShadowFacesDescriptors();
    createShadowPipeline();
    createGraphicsPipeline();
    createReadbackBuffer();
    allocateCommandBuffer();
    createTimestampPool();

    // Persistent fence (pre-signaled so the first vkResetFences works cleanly).
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(device_, &fci, nullptr, &frameFence_));
}

// ----------------------------------------------------------------------------
// Public — uploadScene
// ----------------------------------------------------------------------------

void HeadlessRenderer::uploadScene(const RenderScene& scene) {
    if (device_)
        vkDeviceWaitIdle(device_);

    destroyGeometryBuffers();
    destroyTextureResources();

    // ---- Geometry -----------------------------------------------------------
    {
        VkDeviceSize sz = sizeof(Vertex) * scene.vertices.size();
        createBuffer(sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     vertexBuffer_, vertexBufferMem_);
        uploadToDevice(vertexBuffer_, scene.vertices.data(), sz);
    }
    {
        VkDeviceSize sz = sizeof(uint32_t) * scene.indices.size();
        createBuffer(sz,
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     indexBuffer_, indexBufferMem_);
        uploadToDevice(indexBuffer_, scene.indices.data(), sz);
    }
    drawCalls_ = scene.draws;

    // ---- Textures -----------------------------------------------------------
    uploadTexturesToGPU(scene.textures);

    // ---- Descriptor sets (deduplicated by unique texture pair) --------------
    // Build a map of unique (baseColorIdx, metalRoughIdx) pairs → descriptor set.
    // Many draw calls often share the same material; allocating one set per unique
    // pair instead of one per draw call keeps the descriptor pool proportional to
    // material count, not object count.
    {
        std::map<std::pair<int32_t,int32_t>, VkDescriptorSet> pairToSet;
        for (const auto& dc : drawCalls_) {
            // Only pairs with at least one real texture need a dedicated set;
            // (-1,-1) falls back to whiteTexDescSet_ at draw time.
            if (dc.textureIdx >= 0 || dc.metalRoughTexIdx >= 0)
                pairToSet[{dc.textureIdx, dc.metalRoughTexIdx}] = VK_NULL_HANDLE;
        }

        uint32_t numUnique = static_cast<uint32_t>(pairToSet.size());
        if (numUnique > 0) {
            // Create a per-scene descriptor pool sized exactly for this scene.
            VkDescriptorPoolSize poolSize{};
            poolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            poolSize.descriptorCount = 2 * numUnique;  // 2 samplers per set

            VkDescriptorPoolCreateInfo pci{};
            pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.flags         = 0;  // pool destruction frees all sets; no need for FREE_BIT
            pci.maxSets       = numUnique;
            pci.poolSizeCount = 1;
            pci.pPoolSizes    = &poolSize;
            VK_CHECK(vkCreateDescriptorPool(device_, &pci, nullptr, &sceneDescPool_));

            for (auto& [key, set] : pairToSet) {
                VkImageView baseView = (key.first >= 0 && key.first < (int)texImageViews_.size())
                                     ? texImageViews_[key.first] : whiteTexView_;
                VkImageView mrView   = (key.second >= 0 && key.second < (int)texImageViews_.size())
                                     ? texImageViews_[key.second] : defaultMetalRoughView_;
                createTextureDescriptorSet(baseView, mrView, set, sceneDescPool_);
            }
        }

        drawDescSets_.resize(drawCalls_.size(), VK_NULL_HANDLE);
        for (size_t i = 0; i < drawCalls_.size(); ++i) {
            const DrawCall& dc = drawCalls_[i];
            auto it = pairToSet.find({dc.textureIdx, dc.metalRoughTexIdx});
            drawDescSets_[i] = (it != pairToSet.end()) ? it->second : VK_NULL_HANDLE;
        }
    }

    // ---- UBO (light data) ---------------------------------------------------
    memset(&uboData_, 0, sizeof(uboData_));
    uboData_.shadowNear = SHADOW_NEAR;
    uboData_.shadowFar  = SHADOW_FAR;
    if (!scene.lights.empty()) {
        const Light& l = scene.lights[0];
        uboData_.lightPos[0]   = l.position.x;
        uboData_.lightPos[1]   = l.position.y;
        uboData_.lightPos[2]   = l.position.z;
        uboData_.lightPos[3]   = 0.f;
        uboData_.lightColor[0] = l.color.x;
        uboData_.lightColor[1] = l.color.y;
        uboData_.lightColor[2] = l.color.z;
        uboData_.lightColor[3] = l.intensity;
        uboData_.lightType     = static_cast<uint32_t>(l.type);
    }
    {
        void* mapped;
        VK_CHECK(vkMapMemory(device_, sceneUBOMem_, 0, sizeof(SceneUBOData), 0, &mapped));
        memcpy(mapped, &uboData_, sizeof(SceneUBOData));
        vkUnmapMemory(device_, sceneUBOMem_);
    }
}

// ----------------------------------------------------------------------------
// Public — renderFrame
// ----------------------------------------------------------------------------

std::vector<uint8_t> HeadlessRenderer::renderFrame(const Mat4& view, const Mat4& proj) {
    recordCommandBuffer(view, proj);
    submitAndWait();
    readTimestamps();
    return readbackPixels();
}

void HeadlessRenderer::renderNoReadback(const Mat4& view, const Mat4& proj) {
    recordCommandBuffer(view, proj);
    submitAndWait();
    readTimestamps();
}

int HeadlessRenderer::exportReadbackMemFd() const {
    VkMemoryGetFdInfoKHR info{};
    info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    info.memory     = readbackBufferMem_;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    int fd = -1;
    VK_CHECK(vkGetMemoryFdKHR_(device_, &info, &fd));
    return fd;
}

// ----------------------------------------------------------------------------
// Private — Vulkan setup
// ----------------------------------------------------------------------------

void HeadlessRenderer::createInstance() {
    VkApplicationInfo ai{};
    ai.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "phantom-renderer";
    ai.apiVersion       = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
}

void HeadlessRenderer::pickPhysicalDevice() {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    if (!n) throw std::runtime_error("No Vulkan-capable GPU found");
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance_, &n, devs.data());
    physDev_ = devs[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDev_, &props);
    printf("GPU: %s\n", props.deviceName);

    timestampPeriod_ = props.limits.timestampPeriod;
    msaaSamples_     = getMaxUsableSampleCount(physDev_);
    printf("MSAA: %dx\n", (int)msaaSamples_);
}

void HeadlessRenderer::createLogicalDevice() {
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDev_, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qps(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(physDev_, &qn, qps.data());

    queueFamily_ = UINT32_MAX;
    for (uint32_t i = 0; i < qn; i++)
        if (qps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queueFamily_   = i;
            hasTimestamps_ = (qps[i].timestampValidBits > 0);
            break;
        }
    if (queueFamily_ == UINT32_MAX)
        throw std::runtime_error("No graphics queue family found");

    float pri = 1.f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queueFamily_;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &pri;

    // Enable shaderOutputLayer (Vulkan 1.2 core) so the shadow vertex shader
    // can write gl_Layer directly without a geometry shader.
    VkPhysicalDeviceVulkan12Features vk12{};
    vk12.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12.shaderOutputLayer = VK_TRUE;

    // VK_KHR_external_memory_fd — needed to export the readback buffer memory
    // as a POSIX fd for CUDA interop.  VK_KHR_external_memory is Vulkan 1.1
    // core so no explicit request needed; only the fd variant requires an ext.
    const char* devExts[] = { VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME };

    VkDeviceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext                   = &vk12;
    ci.queueCreateInfoCount    = 1;
    ci.pQueueCreateInfos       = &qci;
    ci.enabledExtensionCount   = 1;
    ci.ppEnabledExtensionNames = devExts;
    VK_CHECK(vkCreateDevice(physDev_, &ci, nullptr, &device_));
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);

    vkGetMemoryFdKHR_ = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
    if (!vkGetMemoryFdKHR_)
        throw std::runtime_error("vkGetMemoryFdKHR not available on this device");
}

// ---- Shadow-pass resources -------------------------------------------------

void HeadlessRenderer::createShadowAttachments() {
    // Depth cube map: 6 array layers, cube-compatible.
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_D32_SFLOAT;
    ici.extent        = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 6;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device_, &ici, nullptr, &shadowCubeImage_));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, shadowCubeImage_, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(physDev_, req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &shadowCubeImageMem_));
    VK_CHECK(vkBindImageMemory(device_, shadowCubeImage_, shadowCubeImageMem_, 0));

    // Cube image view for sampling in the fragment shader.
    VkImageViewCreateInfo vci{};
    vci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image                           = shadowCubeImage_;
    vci.viewType                        = VK_IMAGE_VIEW_TYPE_CUBE;
    vci.format                          = VK_FORMAT_D32_SFLOAT;
    vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    vci.subresourceRange.baseMipLevel   = 0;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount     = 6;
    VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &shadowCubeView_));

    // 2D-array view for the layered framebuffer (all 6 layers as one attachment).
    VkImageViewCreateInfo lci{};
    lci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    lci.image                           = shadowCubeImage_;
    lci.viewType                        = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    lci.format                          = VK_FORMAT_D32_SFLOAT;
    lci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    lci.subresourceRange.baseMipLevel   = 0;
    lci.subresourceRange.levelCount     = 1;
    lci.subresourceRange.baseArrayLayer = 0;
    lci.subresourceRange.layerCount     = 6;
    VK_CHECK(vkCreateImageView(device_, &lci, nullptr, &shadowLayeredView_));

    // Sampler with depth comparison (cube maps use CLAMP_TO_EDGE, no border needed).
    VkSamplerCreateInfo sci{};
    sci.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter     = VK_FILTER_LINEAR;
    sci.minFilter     = VK_FILTER_LINEAR;
    sci.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.compareEnable = VK_TRUE;
    sci.compareOp     = VK_COMPARE_OP_LESS;
    VK_CHECK(vkCreateSampler(device_, &sci, nullptr, &shadowSampler_));
}

void HeadlessRenderer::createShadowRenderPass() {
    VkAttachmentDescription depth{};
    depth.format         = VK_FORMAT_D32_SFLOAT;
    depth.samples        = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference dRef{ 0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription sub{};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.pDepthStencilAttachment = &dRef;

    VkSubpassDependency dep{};
    dep.srcSubpass      = 0;
    dep.dstSubpass      = VK_SUBPASS_EXTERNAL;
    dep.srcStageMask    = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep.dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dep.srcAccessMask   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = 1;
    ci.pAttachments    = &depth;
    ci.subpassCount    = 1;
    ci.pSubpasses      = &sub;
    ci.dependencyCount = 1;
    ci.pDependencies   = &dep;
    VK_CHECK(vkCreateRenderPass(device_, &ci, nullptr, &shadowRenderPass_));
}

void HeadlessRenderer::createShadowFramebuffers() {
    // Single layered framebuffer — all 6 cube faces as array layers.
    // gl_Layer in the vertex shader routes each primitive to the right face.
    VkFramebufferCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.renderPass      = shadowRenderPass_;
    ci.attachmentCount = 1;
    ci.pAttachments    = &shadowLayeredView_;
    ci.width           = SHADOW_MAP_SIZE;
    ci.height          = SHADOW_MAP_SIZE;
    ci.layers          = 6;
    VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &shadowLayeredFB_));
}

// ---- Main render pass (MSAA) -----------------------------------------------

void HeadlessRenderer::createRenderPass() {
    VkAttachmentDescription color{};
    color.format         = VK_FORMAT_R8G8B8A8_UNORM;
    color.samples        = msaaSamples_;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depth{};
    depth.format         = VK_FORMAT_D32_SFLOAT;
    depth.samples        = msaaSamples_;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription resolve{};
    resolve.format         = VK_FORMAT_R8G8B8A8_UNORM;
    resolve.samples        = VK_SAMPLE_COUNT_1_BIT;
    resolve.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolve.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    resolve.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    resolve.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    resolve.finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference cRef     { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference dRef     { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    VkAttachmentReference resolveRef{ 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription sub{};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = 1;
    sub.pColorAttachments       = &cRef;
    sub.pDepthStencilAttachment = &dRef;
    sub.pResolveAttachments     = &resolveRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = 0;
    dep.dstSubpass    = VK_SUBPASS_EXTERNAL;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    std::array<VkAttachmentDescription, 3> atts = {color, depth, resolve};
    VkRenderPassCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = static_cast<uint32_t>(atts.size());
    ci.pAttachments    = atts.data();
    ci.subpassCount    = 1;
    ci.pSubpasses      = &sub;
    ci.dependencyCount = 1;
    ci.pDependencies   = &dep;
    VK_CHECK(vkCreateRenderPass(device_, &ci, nullptr, &renderPass_));
}

void HeadlessRenderer::createImage(uint32_t w, uint32_t h, VkFormat fmt,
                                    VkImageUsageFlags usage,
                                    VkMemoryPropertyFlags memProps,
                                    VkImage& img, VkDeviceMemory& mem,
                                    VkSampleCountFlagBits samples) {
    VkImageCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = fmt;
    ci.extent        = {w, h, 1};
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.samples       = samples;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = usage;
    ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device_, &ci, nullptr, &img));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, img, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(physDev_, req.memoryTypeBits, memProps);
    VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &mem));
    VK_CHECK(vkBindImageMemory(device_, img, mem, 0));
}

VkImageView HeadlessRenderer::createImageView(VkImage img, VkFormat fmt,
                                               VkImageAspectFlags aspect) {
    VkImageViewCreateInfo ci{};
    ci.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ci.image                       = img;
    ci.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    ci.format                      = fmt;
    ci.subresourceRange.aspectMask = aspect;
    ci.subresourceRange.levelCount = 1;
    ci.subresourceRange.layerCount = 1;
    VkImageView v;
    VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &v));
    return v;
}

void HeadlessRenderer::createOffscreenAttachments() {
    createImage(width_, height_, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                msaaColorImage_, msaaColorImageMem_, msaaSamples_);
    msaaColorView_ = createImageView(msaaColorImage_, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_ASPECT_COLOR_BIT);

    createImage(width_, height_, VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                msaaDepthImage_, msaaDepthImageMem_, msaaSamples_);
    msaaDepthView_ = createImageView(msaaDepthImage_, VK_FORMAT_D32_SFLOAT,
                                     VK_IMAGE_ASPECT_DEPTH_BIT);

    createImage(width_, height_, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                resolveImage_, resolveImageMem_);
    resolveImageView_ = createImageView(resolveImage_, VK_FORMAT_R8G8B8A8_UNORM,
                                        VK_IMAGE_ASPECT_COLOR_BIT);
}

void HeadlessRenderer::createFramebuffer() {
    std::array<VkImageView, 3> atts = {msaaColorView_, msaaDepthView_, resolveImageView_};
    VkFramebufferCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.renderPass      = renderPass_;
    ci.attachmentCount = static_cast<uint32_t>(atts.size());
    ci.pAttachments    = atts.data();
    ci.width           = width_;
    ci.height          = height_;
    ci.layers          = 1;
    VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffer_));
}

// ---- Descriptors -----------------------------------------------------------

void HeadlessRenderer::createDescriptorSetLayouts() {
    // Set 0: UBO (vert+frag) + shadow sampler (frag)
    {
        VkDescriptorSetLayoutBinding b0{};
        b0.binding        = 0;
        b0.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b0.descriptorCount= 1;
        b0.stageFlags     = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding b1{};
        b1.binding        = 1;
        b1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b1.descriptorCount= 1;
        b1.stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {b0, b1};
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = static_cast<uint32_t>(bindings.size());
        ci.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &globalSetLayout_));
    }

    // Set 1: base-colour + metallic-roughness texture samplers (frag)
    {
        VkDescriptorSetLayoutBinding binds[2] = {};
        binds[0].binding         = 0;
        binds[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binds[0].descriptorCount = 1;
        binds[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        binds[1].binding         = 1;
        binds[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binds[1].descriptorCount = 1;
        binds[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 2;
        ci.pBindings    = binds;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &texSetLayout_));
    }
}

void HeadlessRenderer::createDescriptorPool() {
    // Fixed-set pool for exactly 3 sets that live for the renderer's lifetime:
    //   set 0 (global):      1 UBO + 1 shadow sampler (combined-image-sampler)
    //   set 0 (shadowFaces): 1 UBO
    //   set 1 (white):       2 combined-image-samplers (baseColor + metalRough fallback)
    // Per-draw/per-scene sets are managed in sceneDescPool_, rebuilt on uploadScene().
    std::array<VkDescriptorPoolSize, 2> sizes{};
    sizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         2 };  // sceneUBO + shadowFacesUBO
    sizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 };  // 1 shadow + 2 white-fallback

    VkDescriptorPoolCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets       = 3;  // global + shadowFaces + whiteFallback
    ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
    ci.pPoolSizes    = sizes.data();
    VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descriptorPool_));
}

void HeadlessRenderer::createSceneUBO() {
    createBuffer(sizeof(SceneUBOData),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 sceneUBO_, sceneUBOMem_);
}

void HeadlessRenderer::createShadowFacesDescriptors() {
    // Layout: set 0, binding 0 = uniform buffer (6 face VP matrices, vertex stage).
    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 1;
    lci.pBindings    = &b;
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &shadowFacesLayout_));

    // Buffer: ShadowFacesUBOData = 6 × Mat4 = 384 bytes.
    createBuffer(sizeof(ShadowFacesUBOData),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 shadowFacesUBO_, shadowFacesUBOMem_);

    // Allocate descriptor set.
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = descriptorPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &shadowFacesLayout_;
    VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &shadowFacesDescSet_));

    // Bind the buffer to set 0, binding 0.
    VkDescriptorBufferInfo bi{};
    bi.buffer = shadowFacesUBO_;
    bi.offset = 0;
    bi.range  = sizeof(ShadowFacesUBOData);

    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = shadowFacesDescSet_;
    w.dstBinding      = 0;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w.pBufferInfo     = &bi;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
}

void HeadlessRenderer::allocateGlobalDescriptorSet() {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = descriptorPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &globalSetLayout_;
    VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &globalDescSet_));

    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = sceneUBO_;
    bufInfo.offset = 0;
    bufInfo.range  = sizeof(SceneUBOData);

    VkDescriptorImageInfo shadowInfo{};
    shadowInfo.sampler     = shadowSampler_;
    shadowInfo.imageView   = shadowCubeView_;
    shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = globalDescSet_;
    writes[0].dstBinding      = 0;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo     = &bufInfo;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = globalDescSet_;
    writes[1].dstBinding      = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo      = &shadowInfo;

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

void HeadlessRenderer::createWhiteTexture() {
    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VK_CHECK(vkCreateSampler(device_, &sci, nullptr, &texSampler_));

    createImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                whiteTexImage_, whiteTexImageMem_);
    uint8_t white[4] = {255, 255, 255, 255};
    uploadImageToGPU(whiteTexImage_, white, 1, 1);
    whiteTexView_ = createImageView(whiteTexImage_, VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_IMAGE_ASPECT_COLOR_BIT);
    // whiteTexDescSet_ is created in createDefaultMetalRoughTexture() once both
    // fallback views are available.
}

void HeadlessRenderer::createDefaultMetalRoughTexture() {
    // Default metallic-roughness: R=255 G=255 B=0 A=255
    // → roughness (G channel) = 1.0, metallic (B channel) = 0.0
    createImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                defaultMetalRoughImage_, defaultMetalRoughMem_);
    uint8_t mr[4] = {255, 255, 0, 255};
    uploadImageToGPU(defaultMetalRoughImage_, mr, 1, 1);
    defaultMetalRoughView_ = createImageView(defaultMetalRoughImage_, VK_FORMAT_R8G8B8A8_UNORM,
                                             VK_IMAGE_ASPECT_COLOR_BIT);
    // Now both fallback views exist — create the white/default descriptor set.
    createTextureDescriptorSet(whiteTexView_, defaultMetalRoughView_, whiteTexDescSet_,
                               descriptorPool_);
}

// ---- Pipelines -------------------------------------------------------------

VkShaderModule HeadlessRenderer::loadShader(const std::string& path) {
    auto code = readSPIRV(path);
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size() * 4;
    ci.pCode    = code.data();
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(device_, &ci, nullptr, &mod));
    return mod;
}

VkShaderModule HeadlessRenderer::loadShaderFromMemory(const uint32_t* code, size_t byteSize) {
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = byteSize;
    ci.pCode    = code;
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(device_, &ci, nullptr, &mod));
    return mod;
}

void HeadlessRenderer::createShadowPipeline() {
    VkShaderModule vert = spirvDir_.empty()
        ? loadShaderFromMemory(k_shadow_vert_spv, k_shadow_vert_spv_size)
        : loadShader(spirvDir_ + "/shadow.vert.spv");
    VkShaderModule frag = spirvDir_.empty()
        ? loadShaderFromMemory(k_shadow_frag_spv, k_shadow_frag_spv_size)
        : loadShader(spirvDir_ + "/shadow.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_VERTEX_BIT,   vert, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

    VkVertexInputBindingDescription bind{};
    bind.binding   = 0;
    bind.stride    = sizeof(Vertex);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attr{};
    attr.location = 0;
    attr.binding  = 0;
    attr.format   = VK_FORMAT_R32G32B32_SFLOAT;
    attr.offset   = offsetof(Vertex, pos);

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &bind;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions    = &attr;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0, 0, (float)SHADOW_MAP_SIZE, (float)SHADOW_MAP_SIZE, 0.f, 1.f};
    VkRect2D   sc{{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
    VkPipelineViewportStateCreateInfo vs{};
    vs.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vs.viewportCount = 1; vs.pViewports = &vp;
    vs.scissorCount  = 1; vs.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode             = VK_POLYGON_MODE_FILL;
    rs.cullMode                = VK_CULL_MODE_BACK_BIT;
    rs.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth               = 1.f;
    rs.depthBiasEnable         = VK_TRUE;
    rs.depthBiasConstantFactor = 2.0f;
    rs.depthBiasSlopeFactor    = 1.5f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(ShadowPushConstants);  // 64 bytes: model only

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &shadowFacesLayout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(device_, &plci, nullptr, &shadowPipelineLayout_));

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vs;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState   = &ms;
    pci.pDepthStencilState  = &ds;
    pci.pColorBlendState    = &cb;
    pci.layout              = shadowPipelineLayout_;
    pci.renderPass          = shadowRenderPass_;
    pci.subpass             = 0;
    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pci, nullptr, &shadowPipeline_));

    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
}

void HeadlessRenderer::createGraphicsPipeline() {
    VkShaderModule vert = spirvDir_.empty()
        ? loadShaderFromMemory(k_scene_vert_spv, k_scene_vert_spv_size)
        : loadShader(spirvDir_ + "/scene.vert.spv");
    VkShaderModule frag = spirvDir_.empty()
        ? loadShaderFromMemory(k_scene_frag_spv, k_scene_frag_spv_size)
        : loadShader(spirvDir_ + "/scene.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_VERTEX_BIT,   vert, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

    VkVertexInputBindingDescription bind{};
    bind.binding   = 0;
    bind.stride    = sizeof(Vertex);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 4> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
    attrs[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
    attrs[3] = {3, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)};

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &bind;
    vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vi.pVertexAttributeDescriptions    = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0, 0, (float)width_, (float)height_, 0.f, 1.f};
    VkRect2D   sc{{0, 0}, {width_, height_}};
    VkPipelineViewportStateCreateInfo vs{};
    vs.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vs.viewportCount = 1; vs.pViewports = &vp;
    vs.scissorCount  = 1; vs.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = msaaSamples_;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &blendAtt;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(PushConstants);

    std::array<VkDescriptorSetLayout, 2> setLayouts = {globalSetLayout_, texSetLayout_};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = static_cast<uint32_t>(setLayouts.size());
    plci.pSetLayouts            = setLayouts.data();
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(device_, &plci, nullptr, &pipelineLayout_));

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vs;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState   = &ms;
    pci.pDepthStencilState  = &ds;
    pci.pColorBlendState    = &cb;
    pci.layout              = pipelineLayout_;
    pci.renderPass          = renderPass_;
    pci.subpass             = 0;
    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_));

    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
}

// ---- Commands + readback ---------------------------------------------------

void HeadlessRenderer::createCommandPool() {
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = queueFamily_;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmdPool_));
}

void HeadlessRenderer::createReadbackBuffer() {
    VkDeviceSize sz = static_cast<VkDeviceSize>(width_) * height_ * 4;

    // Create the buffer (same as before).
    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = sz;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &readbackBuffer_));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, readbackBuffer_, &req);

    // Tag the allocation as exportable so the memory can be imported into CUDA
    // via cudaImportExternalMemory(cudaExternalMemoryHandleTypeOpaqueFd).
    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext           = &exportInfo;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(physDev_, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &readbackBufferMem_));
    VK_CHECK(vkBindBufferMemory(device_, readbackBuffer_, readbackBufferMem_, 0));
}

void HeadlessRenderer::allocateCommandBuffer() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = cmdPool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cmdBuffer_));
}

void HeadlessRenderer::createTimestampPool() {
    if (!hasTimestamps_) return;
    VkQueryPoolCreateInfo ci{};
    ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = 4;
    VK_CHECK(vkCreateQueryPool(device_, &ci, nullptr, &queryPool_));
}

// ----------------------------------------------------------------------------
// Private — geometry + texture management
// ----------------------------------------------------------------------------

void HeadlessRenderer::destroyGeometryBuffers() {
    if (vertexBuffer_)    { vkDestroyBuffer(device_, vertexBuffer_,    nullptr); vertexBuffer_    = VK_NULL_HANDLE; }
    if (vertexBufferMem_) { vkFreeMemory   (device_, vertexBufferMem_, nullptr); vertexBufferMem_ = VK_NULL_HANDLE; }
    if (indexBuffer_)     { vkDestroyBuffer(device_, indexBuffer_,     nullptr); indexBuffer_     = VK_NULL_HANDLE; }
    if (indexBufferMem_)  { vkFreeMemory   (device_, indexBufferMem_,  nullptr); indexBufferMem_  = VK_NULL_HANDLE; }
}

void HeadlessRenderer::destroyTextureResources() {
    // Per-draw sets all live in sceneDescPool_; destroying the pool frees them all atomically.
    drawDescSets_.clear();
    if (sceneDescPool_) {
        vkDestroyDescriptorPool(device_, sceneDescPool_, nullptr);
        sceneDescPool_ = VK_NULL_HANDLE;
    }

    for (auto v : texImageViews_) if (v) vkDestroyImageView(device_, v, nullptr);
    texImageViews_.clear();

    for (size_t i = 0; i < texImages_.size(); ++i) {
        if (texImages_[i])    vkDestroyImage (device_, texImages_[i],    nullptr);
        if (texImageMems_[i]) vkFreeMemory   (device_, texImageMems_[i], nullptr);
    }
    texImages_.clear();
    texImageMems_.clear();
}

void HeadlessRenderer::uploadTexturesToGPU(const std::vector<TextureData>& textures) {
    if (textures.empty()) return;

    texImages_.resize(textures.size(),     VK_NULL_HANDLE);
    texImageMems_.resize(textures.size(),  VK_NULL_HANDLE);
    texImageViews_.resize(textures.size(), VK_NULL_HANDLE);

    // Create all device-local images up-front so they are ready to receive data.
    for (size_t i = 0; i < textures.size(); ++i) {
        createImage(textures[i].width, textures[i].height, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    texImages_[i], texImageMems_[i]);
    }

    // Upload in chunks capped at MAX_STAGING_BYTES so a single staging buffer
    // never consumes more than that even when the total texture set is large.
    // On Jetson Orin (UMA), staging and device memory draw from the same pool,
    // so keeping staging small prevents a temporary 2× spike.
    constexpr VkDeviceSize MAX_STAGING_BYTES = 256ULL * 1024 * 1024;  // 256 MB

    size_t first = 0;
    while (first < textures.size()) {
        // Greedily pack textures into the chunk until we would exceed the limit.
        // A single texture that exceeds the limit gets its own chunk.
        VkDeviceSize chunkBytes = 0;
        size_t last = first;
        while (last < textures.size()) {
            VkDeviceSize sz = (VkDeviceSize)textures[last].width * textures[last].height * 4;
            if (last > first && chunkBytes + sz > MAX_STAGING_BYTES) break;
            chunkBytes += sz;
            ++last;
        }
        // Textures [first, last) form the current chunk.

        // Allocate one staging buffer for the whole chunk.
        VkBuffer       stagingBuf;
        VkDeviceMemory stagingMem;
        createBuffer(chunkBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuf, stagingMem);

        void* mapped;
        VK_CHECK(vkMapMemory(device_, stagingMem, 0, chunkBytes, 0, &mapped));
        VkDeviceSize writeOffset = 0;
        for (size_t i = first; i < last; ++i) {
            VkDeviceSize sz = (VkDeviceSize)textures[i].width * textures[i].height * 4;
            memcpy(static_cast<uint8_t*>(mapped) + writeOffset,
                   textures[i].pixels.data(), sz);
            writeOffset += sz;
        }
        vkUnmapMemory(device_, stagingMem);

        // Record all layout transitions + copies for this chunk in one command buffer.
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cb));

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &beginInfo));

        size_t n = last - first;
        std::vector<VkImageMemoryBarrier> barriers(n);

        // Batch barrier: UNDEFINED → TRANSFER_DST for all images in chunk.
        for (size_t j = 0; j < n; ++j) {
            barriers[j] = {};
            barriers[j].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[j].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[j].newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barriers[j].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[j].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[j].image               = texImages_[first + j];
            barriers[j].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barriers[j].srcAccessMask       = 0;
            barriers[j].dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr,
                             static_cast<uint32_t>(n), barriers.data());

        // Copy staging → each image.
        VkDeviceSize bufOff = 0;
        for (size_t j = 0; j < n; ++j) {
            VkDeviceSize sz = (VkDeviceSize)textures[first+j].width * textures[first+j].height * 4;
            VkBufferImageCopy region{};
            region.bufferOffset                    = bufOff;
            region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount     = 1;
            region.imageExtent                     = {textures[first+j].width, textures[first+j].height, 1};
            vkCmdCopyBufferToImage(cb, stagingBuf, texImages_[first + j],
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            bufOff += sz;
        }

        // Batch barrier: TRANSFER_DST → SHADER_READ_ONLY for all images in chunk.
        for (size_t j = 0; j < n; ++j) {
            barriers[j].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barriers[j].newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barriers[j].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[j].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        }
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr,
                             static_cast<uint32_t>(n), barriers.data());

        VK_CHECK(vkEndCommandBuffer(cb));

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue_));

        vkFreeCommandBuffers(device_, cmdPool_, 1, &cb);
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);

        first = last;
    }

    // Create image views for all textures after upload is complete.
    for (size_t i = 0; i < textures.size(); ++i) {
        texImageViews_[i] = createImageView(texImages_[i], VK_FORMAT_R8G8B8A8_UNORM,
                                            VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

void HeadlessRenderer::createTextureDescriptorSet(VkImageView baseColorView,
                                                    VkImageView metalRoughView,
                                                    VkDescriptorSet& outSet,
                                                    VkDescriptorPool pool) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &texSetLayout_;
    VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &outSet));

    VkDescriptorImageInfo img0{};
    img0.sampler     = texSampler_;
    img0.imageView   = baseColorView;
    img0.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo img1{};
    img1.sampler     = texSampler_;
    img1.imageView   = metalRoughView;
    img1.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = outSet;
    writes[0].dstBinding      = 0;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo      = &img0;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = outSet;
    writes[1].dstBinding      = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo      = &img1;

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

// ----------------------------------------------------------------------------
// Private — Vulkan utilities
// ----------------------------------------------------------------------------

void HeadlessRenderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                     VkMemoryPropertyFlags memProps,
                                     VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo ci{};
    ci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size        = size;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device_, &ci, nullptr, &buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(physDev_, req.memoryTypeBits, memProps);
    VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device_, buffer, memory, 0));
}

void HeadlessRenderer::uploadToDevice(VkBuffer dst, const void* data, VkDeviceSize size) {
    VkBuffer       stg;
    VkDeviceMemory stgMem;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stg, stgMem);

    void* mapped;
    VK_CHECK(vkMapMemory(device_, stgMem, 0, size, 0, &mapped));
    memcpy(mapped, data, size);
    vkUnmapMemory(device_, stgMem);

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = cmdPool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cb));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cb, stg, dst, 1, &region);
    vkEndCommandBuffer(cb);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue_));

    vkFreeCommandBuffers(device_, cmdPool_, 1, &cb);
    vkDestroyBuffer(device_, stg, nullptr);
    vkFreeMemory(device_, stgMem, nullptr);
}

void HeadlessRenderer::transitionImageLayout(VkImage img, VkImageLayout oldLayout,
                                              VkImageLayout newLayout,
                                              VkImageAspectFlags aspect) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = cmdPool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cb));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = img;
    barrier.subresourceRange    = {aspect, 0, 1, 0, 1};

    VkPipelineStageFlags src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dst = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dst = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dst = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }

    vkCmdPipelineBarrier(cb, src, dst, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkEndCommandBuffer(cb);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue_));
    vkFreeCommandBuffers(device_, cmdPool_, 1, &cb);
}

void HeadlessRenderer::uploadImageToGPU(VkImage dst, const void* data,
                                         uint32_t w, uint32_t h) {
    VkDeviceSize sz = (VkDeviceSize)w * h * 4;

    VkBuffer       stg;
    VkDeviceMemory stgMem;
    createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stg, stgMem);

    void* mapped;
    VK_CHECK(vkMapMemory(device_, stgMem, 0, sz, 0, &mapped));
    memcpy(mapped, data, sz);
    vkUnmapMemory(device_, stgMem);

    transitionImageLayout(dst, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_ASPECT_COLOR_BIT);

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = cmdPool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cb));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel       = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount     = 1;
    region.imageExtent                     = {w, h, 1};
    vkCmdCopyBufferToImage(cb, stg, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    vkEndCommandBuffer(cb);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue_));
    vkFreeCommandBuffers(device_, cmdPool_, 1, &cb);

    transitionImageLayout(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_IMAGE_ASPECT_COLOR_BIT);

    vkDestroyBuffer(device_, stg, nullptr);
    vkFreeMemory(device_, stgMem, nullptr);
}

// ----------------------------------------------------------------------------
// Private — per-frame rendering
// ----------------------------------------------------------------------------

void HeadlessRenderer::recordCommandBuffer(const Mat4& view, const Mat4& proj) {
    // Update per-frame UBO fields: camera position (extracted from view) + render mode.
    // view = R * T  →  eye_world = -R^T * translation
    {
        float t0 = view.m[3][0], t1 = view.m[3][1], t2 = view.m[3][2];
        uboData_.cameraPos[0] = -(view.m[0][0]*t0 + view.m[0][1]*t1 + view.m[0][2]*t2);
        uboData_.cameraPos[1] = -(view.m[1][0]*t0 + view.m[1][1]*t1 + view.m[1][2]*t2);
        uboData_.cameraPos[2] = -(view.m[2][0]*t0 + view.m[2][1]*t1 + view.m[2][2]*t2);
        uboData_.cameraPos[3] = 0.f;
        uboData_.renderMode   = renderMode_;
        void* mapped;
        VK_CHECK(vkMapMemory(device_, sceneUBOMem_, 0, sizeof(SceneUBOData), 0, &mapped));
        memcpy(mapped, &uboData_, sizeof(SceneUBOData));
        vkUnmapMemory(device_, sceneUBOMem_);
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmdBuffer_, &bi));

    if (queryPool_)
        vkCmdResetQueryPool(cmdBuffer_, queryPool_, 0, 4);

    // ---- Shadow pass -------------------------------------------------------
    if (queryPool_)
        vkCmdWriteTimestamp(cmdBuffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, 0);

    {
        // Layered omnidirectional cube shadow pass.
        //
        // All 6 faces are rendered in a single render pass using a layered framebuffer
        // (shadowLayeredFB_, layers=6).  The vertex shader uses gl_InstanceIndex to
        // select the face and writes gl_Layer to route each primitive to the right layer.
        // This replaces the old 6-pass loop (6 BeginRenderPass + 294 draw calls) with
        // 1 BeginRenderPass + 49 draw calls (instanceCount=6 each).
        static const struct { Vec3 fwd; Vec3 up; } kFaces[6] = {
            {{ 1, 0, 0}, {0,-1, 0}},  // +X
            {{-1, 0, 0}, {0,-1, 0}},  // -X
            {{ 0, 1, 0}, {0, 0, 1}},  // +Y
            {{ 0,-1, 0}, {0, 0,-1}},  // -Y
            {{ 0, 0, 1}, {0,-1, 0}},  // +Z
            {{ 0, 0,-1}, {0,-1, 0}},  // -Z
        };

        Vec3 lightPos = {uboData_.lightPos[0], uboData_.lightPos[1], uboData_.lightPos[2]};
        Mat4 faceProj = Mat4::perspectiveCube(1.5707963f, 1.f, SHADOW_NEAR, SHADOW_FAR);

        // Upload the 6 face VP matrices to the UBO read by the vertex shader.
        {
            ShadowFacesUBOData facesData;
            for (int f = 0; f < 6; ++f) {
                Vec3 target = add(lightPos, kFaces[f].fwd);
                facesData.vp[f] = faceProj * Mat4::lookAt(lightPos, target, kFaces[f].up);
            }
            void* mapped;
            VK_CHECK(vkMapMemory(device_, shadowFacesUBOMem_, 0, sizeof(facesData), 0, &mapped));
            memcpy(mapped, &facesData, sizeof(facesData));
            vkUnmapMemory(device_, shadowFacesUBOMem_);
        }

        VkClearValue cv; cv.depthStencil = {1.f, 0};
        VkRenderPassBeginInfo rpi{};
        rpi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpi.renderPass        = shadowRenderPass_;
        rpi.framebuffer       = shadowLayeredFB_;
        rpi.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
        rpi.clearValueCount   = 1;
        rpi.pClearValues      = &cv;

        vkCmdBeginRenderPass(cmdBuffer_, &rpi, VK_SUBPASS_CONTENTS_INLINE);

        if (uboData_.lightColor[3] > 0.f) {
            vkCmdBindPipeline(cmdBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_);
            vkCmdBindDescriptorSets(cmdBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    shadowPipelineLayout_, 0, 1, &shadowFacesDescSet_, 0, nullptr);

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmdBuffer_, 0, 1, &vertexBuffer_, &offset);
            vkCmdBindIndexBuffer(cmdBuffer_, indexBuffer_, 0, VK_INDEX_TYPE_UINT32);

            for (const auto& dc : drawCalls_) {
                ShadowPushConstants spc;
                spc.model = dc.transform;
                vkCmdPushConstants(cmdBuffer_, shadowPipelineLayout_,
                                   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(spc), &spc);
                // instanceCount=6: gl_InstanceIndex (0..5) selects the cube face
                vkCmdDrawIndexed(cmdBuffer_, dc.indexCount, 6, dc.firstIndex,
                                 dc.vertexOffset, 0);
            }
        }
        vkCmdEndRenderPass(cmdBuffer_);
    }

    if (queryPool_)
        vkCmdWriteTimestamp(cmdBuffer_, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, queryPool_, 1);

    // ---- Main pass (MSAA) --------------------------------------------------
    if (queryPool_)
        vkCmdWriteTimestamp(cmdBuffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, 2);

    {
        std::array<VkClearValue, 3> clears{};
        clears[0].color        = {{0.08f, 0.08f, 0.1f, 1.0f}};
        clears[1].depthStencil = {1.0f, 0};
        clears[2].color        = {{0.08f, 0.08f, 0.1f, 1.0f}};

        VkRenderPassBeginInfo rpi{};
        rpi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpi.renderPass        = renderPass_;
        rpi.framebuffer       = framebuffer_;
        rpi.renderArea.extent = {width_, height_};
        rpi.clearValueCount   = static_cast<uint32_t>(clears.size());
        rpi.pClearValues      = clears.data();

        vkCmdBeginRenderPass(cmdBuffer_, &rpi, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmdBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        // Set 0: global (UBO + shadow map)
        vkCmdBindDescriptorSets(cmdBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout_, 0, 1, &globalDescSet_, 0, nullptr);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmdBuffer_, 0, 1, &vertexBuffer_, &offset);
        vkCmdBindIndexBuffer(cmdBuffer_, indexBuffer_, 0, VK_INDEX_TYPE_UINT32);

        for (size_t i = 0; i < drawCalls_.size(); ++i) {
            const auto& dc = drawCalls_[i];
            // Set 1: per-draw (baseColor + metalRough); fallback to white/default
            VkDescriptorSet texSet = (i < drawDescSets_.size() && drawDescSets_[i] != VK_NULL_HANDLE)
                                   ? drawDescSets_[i] : whiteTexDescSet_;
            vkCmdBindDescriptorSets(cmdBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout_, 1, 1, &texSet, 0, nullptr);

            PushConstants pc;
            pc.model = dc.transform;
            pc.view  = view;
            pc.proj  = proj;
            vkCmdPushConstants(cmdBuffer_, pipelineLayout_,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            vkCmdDrawIndexed(cmdBuffer_, dc.indexCount, 1, dc.firstIndex,
                             dc.vertexOffset, 0);
        }

        vkCmdEndRenderPass(cmdBuffer_);
    }

    if (queryPool_)
        vkCmdWriteTimestamp(cmdBuffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, 3);

    // Copy resolved image to CPU readback buffer
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent                 = {width_, height_, 1};
    vkCmdCopyImageToBuffer(cmdBuffer_, resolveImage_,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readbackBuffer_, 1, &region);

    VK_CHECK(vkEndCommandBuffer(cmdBuffer_));
}

void HeadlessRenderer::submitAndWait() {
    // Reuse the persistent fence: reset it, submit, wait.
    VK_CHECK(vkResetFences(device_, 1, &frameFence_));

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmdBuffer_;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, frameFence_));
    VK_CHECK(vkWaitForFences(device_, 1, &frameFence_, VK_TRUE, UINT64_MAX));
}

std::vector<uint8_t> HeadlessRenderer::readbackPixels() {
    VkDeviceSize sz = static_cast<VkDeviceSize>(width_) * height_ * 4;
    void* mapped;
    VK_CHECK(vkMapMemory(device_, readbackBufferMem_, 0, sz, 0, &mapped));
    std::vector<uint8_t> pixels(sz);
    memcpy(pixels.data(), mapped, sz);
    vkUnmapMemory(device_, readbackBufferMem_);
    return pixels;
}

void HeadlessRenderer::readTimestamps() {
    lastTimings_ = {};
    if (!queryPool_) return;

    uint64_t ts[4] = {};
    VkResult r = vkGetQueryPoolResults(device_, queryPool_, 0, 4,
                                       sizeof(ts), ts, sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) return;

    double ns = static_cast<double>(timestampPeriod_);
    lastTimings_.shadowPassMs = static_cast<double>(ts[1] - ts[0]) * ns * 1e-6;
    lastTimings_.scenePassMs  = static_cast<double>(ts[3] - ts[2]) * ns * 1e-6;
    lastTimings_.totalGpuMs   = static_cast<double>(ts[3] - ts[0]) * ns * 1e-6;
}

// ----------------------------------------------------------------------------
// Private — cleanup
// ----------------------------------------------------------------------------

void HeadlessRenderer::cleanup() {
    if (!device_) return;
    vkDeviceWaitIdle(device_);

    destroyGeometryBuffers();
    destroyTextureResources();

#define D(h, fn)   if (h) { fn(device_, h, nullptr); h = VK_NULL_HANDLE; }
#define FM(h)      if (h) { vkFreeMemory(device_, h, nullptr); h = VK_NULL_HANDLE; }

    D(whiteTexView_,    vkDestroyImageView)
    D(whiteTexImage_,   vkDestroyImage)
    FM(whiteTexImageMem_)
    D(defaultMetalRoughView_,  vkDestroyImageView)
    D(defaultMetalRoughImage_, vkDestroyImage)
    FM(defaultMetalRoughMem_)
    D(texSampler_,      vkDestroySampler)
    D(shadowSampler_,   vkDestroySampler)

    if (globalDescSet_)   { vkFreeDescriptorSets(device_, descriptorPool_, 1, &globalDescSet_);   globalDescSet_   = VK_NULL_HANDLE; }
    if (whiteTexDescSet_) { vkFreeDescriptorSets(device_, descriptorPool_, 1, &whiteTexDescSet_); whiteTexDescSet_ = VK_NULL_HANDLE; }
    D(descriptorPool_,      vkDestroyDescriptorPool)
    D(globalSetLayout_,     vkDestroyDescriptorSetLayout)
    D(texSetLayout_,        vkDestroyDescriptorSetLayout)
    D(shadowFacesLayout_,   vkDestroyDescriptorSetLayout)

    D(sceneUBO_,         vkDestroyBuffer)
    FM(sceneUBOMem_)
    D(shadowFacesUBO_,   vkDestroyBuffer)
    FM(shadowFacesUBOMem_)

    D(pipeline_,             vkDestroyPipeline)
    D(pipelineLayout_,       vkDestroyPipelineLayout)
    D(shadowPipeline_,       vkDestroyPipeline)
    D(shadowPipelineLayout_, vkDestroyPipelineLayout)

    D(framebuffer_,       vkDestroyFramebuffer)
    D(shadowLayeredFB_,   vkDestroyFramebuffer)

    D(frameFence_,    vkDestroyFence)

    D(msaaColorView_,    vkDestroyImageView)
    D(msaaDepthView_,    vkDestroyImageView)
    D(resolveImageView_, vkDestroyImageView)
    D(msaaColorImage_,   vkDestroyImage)
    D(msaaDepthImage_,   vkDestroyImage)
    D(resolveImage_,     vkDestroyImage)
    FM(msaaColorImageMem_)
    FM(msaaDepthImageMem_)
    FM(resolveImageMem_)

    D(shadowCubeView_,     vkDestroyImageView)
    D(shadowLayeredView_,  vkDestroyImageView)
    D(shadowCubeImage_,    vkDestroyImage)
    FM(shadowCubeImageMem_)

    if (shadowFacesDescSet_) { vkFreeDescriptorSets(device_, descriptorPool_, 1, &shadowFacesDescSet_); shadowFacesDescSet_ = VK_NULL_HANDLE; }

    D(renderPass_,       vkDestroyRenderPass)
    D(shadowRenderPass_, vkDestroyRenderPass)

    D(readbackBuffer_,   vkDestroyBuffer)
    FM(readbackBufferMem_)

    D(queryPool_,        vkDestroyQueryPool)

    D(cmdPool_,          vkDestroyCommandPool)

#undef D
#undef FM

    vkDestroyDevice(device_, nullptr);    device_   = VK_NULL_HANDLE;
    vkDestroyInstance(instance_, nullptr); instance_ = VK_NULL_HANDLE;
}
