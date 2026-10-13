//! Kinect v2 (Xbox One) through libfreenect2: a background thread that keeps
//! the latest registered XYZ+RGB cloud (512x424, depth camera frame).
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::math::V3;

extern "C" {
    fn k2_enumerate(out: *mut c_char, cap: usize) -> c_int;
    fn k2_open(serial: *const c_char) -> *mut c_void;
    fn k2_pipeline(h: *mut c_void) -> *const c_char;
    fn k2_ir_params(h: *mut c_void, out4: *mut f32);
    fn k2_wait_cloud(h: *mut c_void, timeout_ms: c_int, xyz: *mut f32, rgb: *mut u8) -> c_int;
    fn k2_close(h: *mut c_void);
}

pub const W: usize = 512;
pub const H: usize = 424;

/// Valid points only, camera frame (metres; x right, y down, z forward), with colours.
pub struct Cloud {
    pub pts: Vec<V3>,
    pub rgb: Vec<[u8; 3]>,
    /// Pixel index (row * W + col) of each point, for stride sub-sampling.
    pub pix: Vec<u32>,
}

pub fn enumerate() -> Vec<String> {
    let mut buf = vec![0u8; 4096];
    let n = unsafe { k2_enumerate(buf.as_mut_ptr() as *mut c_char, buf.len()) };
    if n <= 0 { return Vec::new(); }
    let s = CStr::from_bytes_until_nul(&buf).map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
    s.lines().filter(|l| !l.is_empty()).map(str::to_string).collect()
}

pub struct Kinect2 {
    latest: Mutex<Option<Arc<Cloud>>>,
    status: Mutex<String>,
    pub seq: AtomicU64,
    pub intrinsics: Mutex<[f32; 4]>,
    running: AtomicBool,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Kinect2 {
    /// Start a thread that opens the first Kinect v2 (retrying until one appears).
    pub fn start() -> Arc<Kinect2> {
        let k = Arc::new(Kinect2 {
            latest: Mutex::new(None),
            status: Mutex::new("searching".into()),
            seq: AtomicU64::new(0),
            intrinsics: Mutex::new([365.0, 365.0, 256.0, 212.0]),
            running: AtomicBool::new(true),
            thread: Mutex::new(None),
        });
        let k2 = k.clone();
        *k.thread.lock().unwrap() = Some(std::thread::spawn(move || k2.run()));
        k
    }

    pub fn cloud(&self) -> Option<Arc<Cloud>> { self.latest.lock().unwrap().clone() }
    pub fn status(&self) -> String { self.status.lock().unwrap().clone() }
    fn set_status(&self, s: impl Into<String>) { *self.status.lock().unwrap() = s.into(); }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(t) = self.thread.lock().unwrap().take() { let _ = t.join(); }
    }

    fn run(&self) {
        let mut xyz = vec![0f32; W * H * 3];
        let mut rgb = vec![0u8; W * H * 3];
        while self.running.load(Ordering::SeqCst) {
            let serial = match enumerate().into_iter().next() {
                Some(s) => s,
                None => { self.set_status("no Kinect v2 found"); std::thread::sleep(Duration::from_secs(2)); continue; }
            };
            let cs = CString::new(serial.clone()).unwrap();
            let h = unsafe { k2_open(cs.as_ptr()) };
            if h.is_null() {
                self.set_status("Kinect v2 open failed (USB 3? udev?)");
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
            let pipe = unsafe { CStr::from_ptr(k2_pipeline(h)) }.to_string_lossy().into_owned();
            let mut p = [0f32; 4];
            unsafe { k2_ir_params(h, p.as_mut_ptr()) };
            *self.intrinsics.lock().unwrap() = p;
            self.set_status(format!("Kinect v2 {serial} ({pipe})"));
            let mut timeouts = 0;
            while self.running.load(Ordering::SeqCst) {
                match unsafe { k2_wait_cloud(h, 1000, xyz.as_mut_ptr(), rgb.as_mut_ptr()) } {
                    1 => {
                        timeouts = 0;
                        let mut pts = Vec::with_capacity(W * H / 2);
                        let mut col = Vec::with_capacity(W * H / 2);
                        let mut pix = Vec::with_capacity(W * H / 2);
                        for i in 0..W * H {
                            let z = xyz[i * 3 + 2];
                            if z > 0.4 && z < 8.0 {
                                pts.push([xyz[i * 3], xyz[i * 3 + 1], z]);
                                col.push([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]]);
                                pix.push(i as u32);
                            }
                        }
                        *self.latest.lock().unwrap() = Some(Arc::new(Cloud { pts, rgb: col, pix }));
                        self.seq.fetch_add(1, Ordering::SeqCst);
                    }
                    0 => {
                        timeouts += 1;
                        self.set_status("Kinect v2: no frames");
                        if timeouts >= 5 { break; }
                    }
                    _ => { self.set_status("Kinect v2: frame error"); std::thread::sleep(Duration::from_millis(100)); }
                }
            }
            unsafe { k2_close(h) };
            *self.latest.lock().unwrap() = None;
        }
    }
}
