#pragma once

#include <cstdint>
#include <string>

// ----------------------------------------------------------------------------
// IFrameOutput — abstract sink for rendered RGBA frames.
//
// Implement this interface to connect the renderer to any consumer:
//   • PngFileOutput  — write frames to a PNG file (implemented here)
//   • MetavisionFrameOutput — feed frames into the metavision pipeline
//   • RingBufferOutput — push frames into a shared-memory ring for IPC
// ----------------------------------------------------------------------------

class IFrameOutput {
public:
    virtual ~IFrameOutput() = default;

    // Called once per rendered frame.  `rgba` points to width*height*4 bytes,
    // row-major, top-to-bottom, R8 G8 B8 A8 unpremultiplied.
    virtual void writeFrame(const uint8_t* rgba, uint32_t width, uint32_t height) = 0;
};

// ----------------------------------------------------------------------------
// PngFileOutput — writes each frame as a PNG file at the given path.
// If multiple frames are produced the file is overwritten each time.
// ----------------------------------------------------------------------------

class PngFileOutput : public IFrameOutput {
public:
    explicit PngFileOutput(std::string path);
    void writeFrame(const uint8_t* rgba, uint32_t width, uint32_t height) override;

private:
    std::string path_;
};
