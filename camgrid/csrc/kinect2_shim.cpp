// Minimal C ABI over libfreenect2 for camgrid.
//
// One global Freenect2 context (libfreenect2 requires it to outlive devices).
// Each opened device gets a SyncMultiFrameListener; k2_wait() blocks for the
// next color+depth pair and copies them into caller-provided buffers.

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/config.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <string>
#include <dirent.h>

namespace {

std::mutex g_mutex;
libfreenect2::Freenect2* g_ctx = nullptr;

libfreenect2::Freenect2& ctx() {
    if (!g_ctx) {
        // libfreenect2 prefers VA-API for JPEG decoding when libva is present,
        // which fails on systems without a working VA driver (e.g. NVIDIA) and
        // then yields no color frames at all. Force the TurboJPEG decoder
        // unless the user explicitly opts in with CAMGRID_K2_VAAPI=1.
        const char* va = std::getenv("CAMGRID_K2_VAAPI");
        if (!(va && va[0] == '1')) setenv("LIBVA_DRIVER_NAME", "disabled", 1); // overrides desktop env (e.g. "nvidia")
        libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
        g_ctx = new libfreenect2::Freenect2();
    }
    return *g_ctx;
}

// True if at least one OpenCL ICD is registered (otherwise the OpenCL
// pipeline has no platform to run on).
bool have_opencl_icd() {
    DIR* d = opendir("/etc/OpenCL/vendors");
    if (!d) return false;
    bool found = false;
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() > 4 && n.compare(n.size() - 4, 4, ".icd") == 0) { found = true; break; }
    }
    closedir(d);
    return found;
}

// Pick a depth-processing pipeline. Default: OpenCL if an ICD exists, else
// OpenGL, else CPU. CAMGRID_K2_PIPELINE=cpu|gl|cl forces one.
libfreenect2::PacketPipeline* make_pipeline(const char** chosen) {
    const char* want = std::getenv("CAMGRID_K2_PIPELINE");
    std::string w = want ? want : "";
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
    if (w == "cl" || (w.empty() && have_opencl_icd())) { *chosen = "opencl"; return new libfreenect2::OpenCLPacketPipeline(); }
#endif
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
    if (w.empty() || w == "gl") { *chosen = "opengl"; return new libfreenect2::OpenGLPacketPipeline(); }
#endif
    *chosen = "cpu";
    return new libfreenect2::CpuPacketPipeline();
}

struct Handle {
    libfreenect2::Freenect2Device* dev = nullptr;
    libfreenect2::SyncMultiFrameListener listener{libfreenect2::Frame::Color | libfreenect2::Frame::Depth};
    const char* pipeline = "";
};

} // namespace

extern "C" {

// Writes newline-separated serials into `out` (NUL-terminated). Returns device count.
int k2_enumerate(char* out, size_t cap) {
    std::lock_guard<std::mutex> lock(g_mutex);
    int n = ctx().enumerateDevices();
    std::string s;
    for (int i = 0; i < n; i++) {
        if (i) s += '\n';
        s += ctx().getDeviceSerialNumber(i);
    }
    if (out && cap) {
        std::strncpy(out, s.c_str(), cap - 1);
        out[cap - 1] = 0;
    }
    return n;
}

// Opens and starts streaming. Returns NULL on failure.
void* k2_open(const char* serial) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Handle* h = new Handle();
    libfreenect2::PacketPipeline* pipe = make_pipeline(&h->pipeline);
    h->dev = ctx().openDevice(serial, pipe); // device takes ownership of pipe
    if (!h->dev) { delete h; return nullptr; }
    h->dev->setColorFrameListener(&h->listener);
    h->dev->setIrAndDepthFrameListener(&h->listener);
    if (!h->dev->start()) {
        h->dev->close();
        delete h->dev;
        delete h;
        return nullptr;
    }
    return h;
}

const char* k2_pipeline(void* hp) { return static_cast<Handle*>(hp)->pipeline; }

// Waits up to timeout_ms for a new frame set. Copies color as packed RGB
// (1920x1080x3) and depth as float millimetres (512x424). Returns 1 on
// success, 0 on timeout, -1 on error.
int k2_wait(void* hp, int timeout_ms, uint8_t* rgb_out, float* depth_out) {
    Handle* h = static_cast<Handle*>(hp);
    libfreenect2::FrameMap frames;
    if (!h->listener.waitForNewFrame(frames, timeout_ms)) return 0;

    libfreenect2::Frame* c = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame* d = frames[libfreenect2::Frame::Depth];
    int ok = 1;
    if (c && c->data && c->width == 1920 && c->height == 1080 && c->bytes_per_pixel == 4) {
        const uint8_t* src = c->data;
        const bool bgrx = (c->format != libfreenect2::Frame::RGBX);
        const size_t n = 1920u * 1080u;
        for (size_t i = 0; i < n; i++, src += 4) {
            if (bgrx) { rgb_out[i*3+0] = src[2]; rgb_out[i*3+1] = src[1]; rgb_out[i*3+2] = src[0]; }
            else      { rgb_out[i*3+0] = src[0]; rgb_out[i*3+1] = src[1]; rgb_out[i*3+2] = src[2]; }
        }
    } else if (c && c->status != 0) {
        ok = -1;
    }
    if (d && d->data && d->width == 512 && d->height == 424) {
        std::memcpy(depth_out, d->data, 512u * 424u * sizeof(float));
    } else if (d && d->status != 0) {
        ok = -1;
    }
    h->listener.release(frames);
    return ok;
}

void k2_close(void* hp) {
    Handle* h = static_cast<Handle*>(hp);
    if (!h) return;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (h->dev) { h->dev->stop(); h->dev->close(); delete h->dev; }
    }
    delete h;
}

} // extern "C"
