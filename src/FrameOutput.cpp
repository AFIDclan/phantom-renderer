#include "vh/FrameOutput.h"

#include <png.h>
#include <stdexcept>
#include <string>

// ----------------------------------------------------------------------------
// PngFileOutput
// ----------------------------------------------------------------------------

PngFileOutput::PngFileOutput(std::string path)
    : path_(std::move(path)) {}

void PngFileOutput::writeFrame(const uint8_t* rgba, uint32_t width, uint32_t height) {
    FILE* fp = fopen(path_.c_str(), "wb");
    if (!fp)
        throw std::runtime_error("PngFileOutput: cannot open " + path_);

    png_structp png  = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                               nullptr, nullptr, nullptr);
    png_infop   info = png_create_info_struct(png);

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw std::runtime_error("PngFileOutput: libpng error while writing " + path_);
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info,
                 width, height, 8,
                 PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    for (uint32_t y = 0; y < height; y++)
        png_write_row(png,
            // const_cast is safe: libpng promises not to modify row data on write
            const_cast<png_bytep>(rgba + static_cast<size_t>(y) * width * 4));

    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}
