#!/bin/bash
# build.sh — Install dependencies and build vulcan-headless
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== vulcan-headless build script ==="

# ---------------------------------------------------------------------------
# Install dependencies if not already present
# ---------------------------------------------------------------------------

install_if_missing() {
    local pkg=$1
    if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        echo "Installing $pkg..."
        sudo apt-get install -y "$pkg"
    else
        echo "  $pkg already installed"
    fi
}

echo ""
echo "--- Checking dependencies ---"
install_if_missing libvulkan-dev
install_if_missing glslang-tools
install_if_missing libpng-dev

# ---------------------------------------------------------------------------
# Configure and build
# ---------------------------------------------------------------------------

echo ""
echo "--- Configuring with CMake ---"
mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo ""
echo "--- Building ---"
cmake --build "$BUILD_DIR" -- -j$(nproc)

echo ""
echo "=== Build complete ==="
echo ""
echo "To run:"
echo "  $BUILD_DIR/vulcan-headless --spirv-dir $BUILD_DIR/shaders --output render_output.png"
echo ""
echo "Or use the cmake 'run' target:"
echo "  cmake --build $BUILD_DIR --target run"
