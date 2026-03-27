cmake_minimum_required(VERSION 3.16)
project(vulcan-headless LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

find_package(Vulkan REQUIRED)
find_package(PNG    REQUIRED)

# ---------------------------------------------------------------------------
# tinygltf (header-only) + nlohmann/json (required by tinygltf)
# Downloaded into the build directory so no git dependency is needed.
# ---------------------------------------------------------------------------

set(TINYGLTF_DIR "${CMAKE_BINARY_DIR}/vendor/tinygltf")
file(MAKE_DIRECTORY "${TINYGLTF_DIR}")

foreach(_file
        "tiny_gltf.h|https://raw.githubusercontent.com/syoyo/tinygltf/v2.8.21/tiny_gltf.h"
        "json.hpp|https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp"
        "stb_image.h|https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
        "stb_image_write.h|https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h")
    string(REPLACE "|" ";" _pair "${_file}")
    list(GET _pair 0 _name)
    list(GET _pair 1 _url)
    if(NOT EXISTS "${TINYGLTF_DIR}/${_name}")
        message(STATUS "Downloading ${_name}...")
        file(DOWNLOAD "${_url}" "${TINYGLTF_DIR}/${_name}"
             STATUS _status TLS_VERIFY ON)
        list(GET _status 0 _code)
        if(NOT _code EQUAL 0)
            message(FATAL_ERROR "Failed to download ${_name}. Check network connectivity.")
        endif()
    endif()
endforeach()

# ---------------------------------------------------------------------------
# Shader compilation
# Requires glslangValidator or glslc — installed by the build script.
# SPIR-V files are written into ${CMAKE_BINARY_DIR}/shaders/ and also
# embedded into the binary's working directory at install/run time.
# ---------------------------------------------------------------------------

find_program(GLSLANG_VALIDATOR glslangValidator)
find_program(GLSLC glslc)

set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

function(compile_shader SRC_FILE OUT_FILE)
    if(GLSLC)
        add_custom_command(
            OUTPUT  ${OUT_FILE}
            COMMAND ${GLSLC} -o ${OUT_FILE} ${SRC_FILE}
            DEPENDS ${SRC_FILE}
            COMMENT "Compiling shader ${SRC_FILE} with glslc"
        )
    elseif(GLSLANG_VALIDATOR)
        add_custom_command(
            OUTPUT  ${OUT_FILE}
            COMMAND ${GLSLANG_VALIDATOR} -V -o ${OUT_FILE} ${SRC_FILE}
            DEPENDS ${SRC_FILE}
            COMMENT "Compiling shader ${SRC_FILE} with glslangValidator"
        )
    else()
        message(FATAL_ERROR
            "No GLSL compiler found. Install glslang-tools:\n"
            "  sudo apt install glslang-tools")
    endif()
endfunction()

compile_shader(
    ${SHADER_SRC_DIR}/scene.vert
    ${SHADER_OUT_DIR}/scene.vert.spv
)
compile_shader(
    ${SHADER_SRC_DIR}/scene.frag
    ${SHADER_OUT_DIR}/scene.frag.spv
)
compile_shader(
    ${SHADER_SRC_DIR}/shadow.vert
    ${SHADER_OUT_DIR}/shadow.vert.spv
)
compile_shader(
    ${SHADER_SRC_DIR}/shadow.frag
    ${SHADER_OUT_DIR}/shadow.frag.spv
)

add_custom_target(shaders DEPENDS
    ${SHADER_OUT_DIR}/scene.vert.spv
    ${SHADER_OUT_DIR}/scene.frag.spv
    ${SHADER_OUT_DIR}/shadow.vert.spv
    ${SHADER_OUT_DIR}/shadow.frag.spv
)

# ---------------------------------------------------------------------------
# Embed SPIR-V into a generated C++ header so the renderer can work without
# an external --spirv-dir.  Each .spv is converted to a uint32_t[] array by
# cmake/spv_to_header.cmake and the four arrays are collected into a single
# spirv_blobs_generated.h that Renderer.cpp includes via SpirvBlobs.h.
# ---------------------------------------------------------------------------

set(SPIRV_BLOBS_HEADER "${SHADER_OUT_DIR}/spirv_blobs_generated.h")

# Helper: emit one array per SPV file, then concatenate into the final header.
set(_spv_headers "")
foreach(_shader scene.vert scene.frag shadow.vert shadow.frag)
    string(REPLACE "." "_" _var "k_${_shader}_spv")   # e.g. k_scene_vert_spv
    set(_spv  "${SHADER_OUT_DIR}/${_shader}.spv")
    set(_hdr  "${SHADER_OUT_DIR}/${_shader}.spv.h")
    add_custom_command(
        OUTPUT  "${_hdr}"
        COMMAND ${CMAKE_COMMAND}
                -DSPV=${_spv}
                -DVAR=${_var}
                -DOUT=${_hdr}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/spv_to_header.cmake"
        DEPENDS "${_spv}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/spv_to_header.cmake"
        COMMENT "Embedding ${_shader}.spv into ${_shader}.spv.h"
    )
    list(APPEND _spv_headers "${_hdr}")
endforeach()

# Concatenate all per-shader headers into spirv_blobs_generated.h.
add_custom_command(
    OUTPUT  "${SPIRV_BLOBS_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E cat ${_spv_headers} > "${SPIRV_BLOBS_HEADER}"
    DEPENDS ${_spv_headers}
    COMMENT "Generating spirv_blobs_generated.h"
)

add_custom_target(spirv_blobs DEPENDS "${SPIRV_BLOBS_HEADER}")

# ---------------------------------------------------------------------------
# Library (for embedding into other projects, e.g. metavision)
# ---------------------------------------------------------------------------

add_library(vh STATIC
    src/Scene.cpp
    src/GLTFLoader.cpp
    src/Renderer.cpp
    src/FrameOutput.cpp
)

add_dependencies(vh shaders spirv_blobs)

# tinygltf pulls in stb which generates many warnings — suppress for that TU only
set_source_files_properties(src/GLTFLoader.cpp PROPERTIES COMPILE_OPTIONS "-w")

target_include_directories(vh PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
    ${TINYGLTF_DIR}
    ${SHADER_OUT_DIR}          # so Renderer.cpp can find spirv_blobs_generated.h
)

target_link_libraries(vh PUBLIC
    ${Vulkan_LIBRARIES}
    ${PNG_LIBRARIES}
)

target_compile_options(vh PRIVATE
    -Wall -Wextra -O2 -fPIC
)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------

add_executable(vulcan-headless
    src/main.cpp
)

add_dependencies(vulcan-headless shaders)

target_link_libraries(vulcan-headless PRIVATE
    vh
)

target_compile_options(vulcan-headless PRIVATE
    -Wall -Wextra -O2
)

target_compile_options(vulcan-headless PRIVATE
    -Wall -Wextra -O2
)

# ---------------------------------------------------------------------------
# Run helper: pass the shader directory automatically
# ---------------------------------------------------------------------------

set(SPIRV_OUTPUT_DIR ${SHADER_OUT_DIR})

# Pass --scene if a Scene.gltf exists next to CMakeLists.txt
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Scene.gltf")
    set(SCENE_ARGS --scene "${CMAKE_CURRENT_SOURCE_DIR}/Scene.gltf")
else()
    set(SCENE_ARGS "")
endif()

add_custom_target(run
    COMMAND vulcan-headless
        --spirv-dir ${SPIRV_OUTPUT_DIR}
        --output ${CMAKE_BINARY_DIR}/render_output.png
        --width 1280
        --height 720
        ${SCENE_ARGS}
    DEPENDS vulcan-headless shaders
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running vulcan-headless renderer"
)
