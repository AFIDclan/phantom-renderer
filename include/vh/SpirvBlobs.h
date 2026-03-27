#pragma once
// Embedded SPIR-V blobs — generated at build time from shaders/*.spv.
// Include this header only from Renderer.cpp (or wherever loadShader lives).
//
// Provides four arrays:
//   kSceneVertSpv[]  / kSceneVertSpv_size
//   kSceneFragSpv[]  / kSceneFragSpv_size
//   kShadowVertSpv[] / kShadowVertSpv_size
//   kShadowFragSpv[] / kShadowFragSpv_size
//
// The header is auto-generated into the build tree; the include path is set
// by CMakeLists.txt (target_include_directories).

#include "spirv_blobs_generated.h"
