//! Kinect v2 (Xbox One) support through libfreenect2, via the C shim in csrc/.
//!
//! One device thread produces two tiles: the 1920x1080 color camera and the
//! 512x424 depth camera (rendered as a color ramp, near = warm, far = cool).

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::capture::{sleep_while_running, Camera, Frame, Slot, Worker};

extern "C" {
    fn k2_enumerate(out: *mut c_char, cap: usize) -> c_int;
    fn k2_open(serial: *const c_char) -> *mut c_void;
    fn k2_pipeline(h: *mut c_void) -> *const c_char;
    fn k2_wait(h: *mut c_void, timeout_ms: c_int, rgb_out: *mut u8, depth_out: *mut f32) -> c_int;
    fn k2_close(h: *mut c_void);
}

pub const COLOR_W: usize = 1920;
pub const COLOR_H: usize = 1080;
pub const DEPTH_W: usize = 512;
pub const DEPTH_H: usize = 424;

/// Serials of all Kinect v2 sensors currently on the USB bus.
pub fn enumerate() -> Vec<String> {
    let mut buf = vec![0u8; 4096];
    let n = unsafe { k2_enumerate(buf.as_mut_ptr() as *mut c_char, buf.len()) };
    if n <= 0 { return Vec::new(); }
    let s = CStr::from_bytes_until_nul(&buf).map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
    s.lines().filter(|l| !l.is_empty()).map(str::to_string).collect()
}

/// "kinect2:<serial>/color" -> Some("<serial>")
pub fn serial_of(path: &str) -> Option<&str> {
    path.strip_prefix("kinect2:").and_then(|r| r.split('/').next())
}

struct Device(*mut c_void);
unsafe impl Send for Device {}
impl Drop for Device {
    fn drop(&mut self) { unsafe { k2_close(self.0) } }
}

/// Open the sensor, start streaming, and return its color and depth tiles.
pub fn open(serial: &str) -> std::io::Result<Vec<Camera>> {
    let cserial = CString::new(serial).map_err(|_| std::io::Error::other("bad serial"))?;
    let h = unsafe { k2_open(cserial.as_ptr()) };
    if h.is_null() {
        return Err(std::io::Error::other(
            "libfreenect2 could not open the sensor (needs USB 3 and udev access to 045e:02c4/02d8/02d9)",
        ));
    }
    let pipeline = unsafe { CStr::from_ptr(k2_pipeline(h)) }.to_string_lossy().into_owned();
    let dev = Device(h);

    let color = Slot::new();
    let depth = Slot::new();
    let (c2, d2) = (color.clone(), depth.clone());
    let path = format!("kinect2:{serial}");
    let p2 = path.clone();
    let worker = Worker::spawn(&format!("k2-{serial}"), move |w| run(dev, c2, d2, p2, w))?;

    Ok(vec![
        Camera {
            path: format!("{path}/color"),
            name: "Kinect v2 color".into(),
            fourcc: "BGRX".into(),
            width: COLOR_W as u32,
            height: COLOR_H as u32,
            slot: color,
            worker: worker.attach(),
        },
        Camera {
            path: format!("{path}/depth"),
            name: format!("Kinect v2 depth ({pipeline})"),
            fourcc: "DEPTH".into(),
            width: DEPTH_W as u32,
            height: DEPTH_H as u32,
            slot: depth,
            worker: worker.attach(),
        },
    ])
}

fn run(dev: Device, color: Arc<Slot>, depth: Arc<Slot>, path: String, worker: Arc<Worker>) {
    let mut rgb = vec![0u8; COLOR_W * COLOR_H * 3];
    let mut mm = vec![0f32; DEPTH_W * DEPTH_H];
    let mut window_start = Instant::now();
    let mut window_frames = 0u32;
    let mut timeouts = 0u32;

    while worker.is_running() {
        let r = unsafe { k2_wait(dev.0, 1000, rgb.as_mut_ptr(), mm.as_mut_ptr()) };
        match r {
            1 => {
                timeouts = 0;
                color.publish(Frame { width: COLOR_W, height: COLOR_H, rgb: rgb.clone() });
                depth.publish(Frame { width: DEPTH_W, height: DEPTH_H, rgb: colorize_depth(&mm) });
                color.set_status("live");
                depth.set_status("live");
                window_frames += 1;
                let dt = window_start.elapsed();
                if dt >= Duration::from_secs(1) {
                    let fps = (window_frames as f64 / dt.as_secs_f64()).round() as u64;
                    color.fps.store(fps, Ordering::Relaxed);
                    depth.fps.store(fps, Ordering::Relaxed);
                    window_start = Instant::now();
                    window_frames = 0;
                }
            }
            0 => {
                timeouts += 1;
                let msg = "no frames from sensor (USB 3 bandwidth or unplugged?)";
                color.set_status(msg);
                depth.set_status(msg);
                color.fps.store(0, Ordering::Relaxed);
                depth.fps.store(0, Ordering::Relaxed);
                if timeouts >= 10 { break; } // ~10 s silent: let the watcher re-open it
            }
            _ => {
                color.set_status("sensor error");
                depth.set_status("sensor error");
                sleep_while_running(&worker, Duration::from_millis(200));
            }
        }
    }
    if worker.is_running() { eprintln!("{path}: {}", color.status()); }
    // `dev` drops here: stop + close on this thread.
}

/// Map depth in millimetres to RGB. 0 (no reading) is black; 500 mm is red,
/// sweeping through yellow, green, cyan to blue at 4500 mm.
fn colorize_depth(mm: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; mm.len() * 3];
    for (i, &d) in mm.iter().enumerate() {
        if !(d > 0.0) { continue; }
        let t = ((d - 500.0) / 4000.0).clamp(0.0, 1.0);
        let h = t * 240.0; // hue degrees, red -> blue
        let (r, g, b) = hsv(h, 1.0, 1.0);
        out[i * 3] = r;
        out[i * 3 + 1] = g;
        out[i * 3 + 2] = b;
    }
    out
}

fn hsv(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (((r + m) * 255.0) as u8, ((g + m) * 255.0) as u8, ((b + m) * 255.0) as u8)
}
