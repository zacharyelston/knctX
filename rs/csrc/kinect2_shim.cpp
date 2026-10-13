// Minimal C ABI over libfreenect2 for the kinect3d crate (roommap).
//
// One global Freenect2 context; per device a SyncMultiFrameListener plus a
// Registration so k2_wait_cloud() can hand back a registered XYZ+RGB cloud
// in the depth camera's frame (metres; x right, y down, z forward).

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/config.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <mutex>
#include <string>

namespace {

std::mutex g_mutex;
libfreenect2::Freenect2* g_ctx = nullptr;

libfreenect2::Freenect2& ctx() {
    if (!g_ctx) {
        // libfreenect2 prefers VA-API JPEG decoding when libva exists; on
        // machines without a working VA driver (NVIDIA) that yields no color
        // frames at all. Force TurboJPEG unless KINECT2_VAAPI=1.
        const char* va = std::getenv("KINECT2_VAAPI");
        if (!(va && va[0] == '1')) setenv("LIBVA_DRIVER_NAME", "disabled", 1);
        libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
        g_ctx = new libfreenect2::Freenect2();
    }
    return *g_ctx;
}

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

// OpenCL if an ICD exists, else OpenGL, else CPU. KINECT2_PIPELINE=cpu|gl|cl forces one.
libfreenect2::PacketPipeline* make_pipeline(const char** chosen) {
    const char* want = std::getenv("KINECT2_PIPELINE");
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
    libfreenect2::Registration* reg = nullptr;
    libfreenect2::Frame undistorted{512, 424, 4};
    libfreenect2::Frame registered{512, 424, 4};
    const char* pipeline = "";
};

} // namespace

extern "C" {

int k2_enumerate(char* out, size_t cap) {
    std::lock_guard<std::mutex> lock(g_mutex);
    int n = ctx().enumerateDevices();
    std::string s;
    for (int i = 0; i < n; i++) {
        if (i) s += '\n';
        s += ctx().getDeviceSerialNumber(i);
    }
    if (out && cap) { std::strncpy(out, s.c_str(), cap - 1); out[cap - 1] = 0; }
    return n;
}

// Opens and starts streaming; empty serial = first device. NULL on failure.
void* k2_open(const char* serial) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Handle* h = new Handle();
    libfreenect2::PacketPipeline* pipe = make_pipeline(&h->pipeline);
    std::string s = serial ? serial : "";
    if (s.empty()) {
        if (ctx().enumerateDevices() == 0) { delete pipe; delete h; return nullptr; }
        s = ctx().getDefaultDeviceSerialNumber();
    }
    h->dev = ctx().openDevice(s, pipe); // device takes ownership of pipe
    if (!h->dev) { delete h; return nullptr; }
    h->dev->setColorFrameListener(&h->listener);
    h->dev->setIrAndDepthFrameListener(&h->listener);
    if (!h->dev->start()) {
        h->dev->close();
        delete h->dev;
        delete h;
        return nullptr;
    }
    h->reg = new libfreenect2::Registration(h->dev->getIrCameraParams(), h->dev->getColorCameraParams());
    return h;
}

const char* k2_pipeline(void* hp) { return static_cast<Handle*>(hp)->pipeline; }

// Depth-camera intrinsics: fx, fy, cx, cy.
void k2_ir_params(void* hp, float* out4) {
    libfreenect2::Freenect2Device::IrCameraParams p = static_cast<Handle*>(hp)->dev->getIrCameraParams();
    out4[0] = p.fx; out4[1] = p.fy; out4[2] = p.cx; out4[3] = p.cy;
}

// Waits up to timeout_ms for a frame set and writes a registered cloud:
// xyz_out: 512*424*3 floats (metres; invalid points have z = 0),
// rgb_out: 512*424*3 bytes. Returns 1 ok, 0 timeout, -1 error.
int k2_wait_cloud(void* hp, int timeout_ms, float* xyz_out, uint8_t* rgb_out) {
    Handle* h = static_cast<Handle*>(hp);
    libfreenect2::FrameMap frames;
    if (!h->listener.waitForNewFrame(frames, timeout_ms)) return 0;
    libfreenect2::Frame* c = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame* d = frames[libfreenect2::Frame::Depth];
    int ok = 1;
    if (c && d && c->status == 0 && d->status == 0) {
        h->reg->apply(c, d, &h->undistorted, &h->registered, true);
        const uint8_t* reg = h->registered.data;
        for (int r = 0; r < 424; r++) {
            for (int col = 0; col < 512; col++) {
                float x, y, z, rgb;
                h->reg->getPointXYZRGB(&h->undistorted, &h->registered, r, col, x, y, z, rgb);
                size_t i = (size_t)r * 512 + col;
                if (std::isfinite(z) && z > 0.0f) { xyz_out[i*3] = x; xyz_out[i*3+1] = y; xyz_out[i*3+2] = z; }
                else { xyz_out[i*3] = 0; xyz_out[i*3+1] = 0; xyz_out[i*3+2] = 0; }
                const uint8_t* px = reg + i * 4; // BGRX
                rgb_out[i*3] = px[2]; rgb_out[i*3+1] = px[1]; rgb_out[i*3+2] = px[0];
            }
        }
    } else {
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
    delete h->reg;
    delete h;
}

} // extern "C"
