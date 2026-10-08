//! Frame sources: the real Kinect (libfreenect FFI), a synthetic spinning person, and a recording player.
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const W: usize = 640;
pub const H: usize = 480;
pub type Rgb = Arc<Vec<u8>>; // W*H*3, RGB
pub type Depth = Arc<Vec<u16>>; // W*H, millimetres (0 = no data)

/// Latest frames + control state shared between a producer thread and the UI.
pub struct Store {
    frames: Mutex<(Option<Rgb>, Option<Depth>)>,
    pub seq: AtomicU64,
    pub tilt: AtomicI32,
    pub running: AtomicBool,
}

impl Store {
    pub fn new() -> Arc<Store> {
        Arc::new(Store {
            frames: Mutex::new((None, None)),
            seq: AtomicU64::new(0),
            tilt: AtomicI32::new(0),
            running: AtomicBool::new(true),
        })
    }
    pub fn put_rgb(&self, rgb: Vec<u8>) { self.frames.lock().unwrap().0 = Some(Arc::new(rgb)); }
    pub fn put_depth(&self, d: Vec<u16>) {
        self.frames.lock().unwrap().1 = Some(Arc::new(d));
        self.seq.fetch_add(1, Ordering::SeqCst);
    }
    pub fn put_both(&self, rgb: Vec<u8>, d: Vec<u16>) {
        *self.frames.lock().unwrap() = (Some(Arc::new(rgb)), Some(Arc::new(d)));
        self.seq.fetch_add(1, Ordering::SeqCst);
    }
    pub fn clear(&self) { *self.frames.lock().unwrap() = (None, None); }
}

pub trait Source: Send + Sync {
    fn store(&self) -> &Arc<Store>;
    fn start(&self);
    fn stop(&self);
    fn frames(&self) -> Option<(Rgb, Depth)> {
        let g = self.store().frames.lock().unwrap();
        match (&g.0, &g.1) {
            (Some(r), Some(d)) => Some((r.clone(), d.clone())),
            _ => None,
        }
    }
    fn seq(&self) -> u64 { self.store().seq.load(Ordering::SeqCst) }
    fn tilt_target(&self) -> i32 { self.store().tilt.load(Ordering::SeqCst) }
    fn set_tilt(&self, deg: i32) { self.store().tilt.store(deg.clamp(-30, 30), Ordering::SeqCst); }
}

/// Source with no sensor (viewer-only mode).
pub struct NoCam(Arc<Store>);
impl NoCam {
    pub fn new() -> Self { NoCam(Store::new()) }
}
impl Source for NoCam {
    fn store(&self) -> &Arc<Store> { &self.0 }
    fn start(&self) {}
    fn stop(&self) {}
}

// ------------------------------------------------------------------------- libfreenect FFI

#[repr(C)]
#[derive(Clone, Copy)]
struct FrameMode {
    reserved: u32,
    resolution: i32,
    format: i32,
    bytes: i32,
    width: i16,
    height: i16,
    data_bits: i8,
    padding_bits: i8,
    framerate: i8,
    is_valid: i8,
}

#[repr(C)]
struct TimeVal {
    sec: i64,
    usec: i64,
}

type Cb = extern "C" fn(*mut c_void, *mut c_void, u32);

extern "C" {
    fn freenect_init(ctx: *mut *mut c_void, usb: *mut c_void) -> i32;
    fn freenect_shutdown(ctx: *mut c_void) -> i32;
    fn freenect_set_log_level(ctx: *mut c_void, level: i32);
    fn freenect_open_device(ctx: *mut c_void, dev: *mut *mut c_void, index: i32) -> i32;
    fn freenect_close_device(dev: *mut c_void) -> i32;
    fn freenect_process_events_timeout(ctx: *mut c_void, tv: *mut TimeVal) -> i32;
    fn freenect_set_user(dev: *mut c_void, user: *mut c_void);
    fn freenect_get_user(dev: *mut c_void) -> *mut c_void;
    fn freenect_set_depth_callback(dev: *mut c_void, cb: Cb);
    fn freenect_set_video_callback(dev: *mut c_void, cb: Cb);
    fn freenect_find_video_mode(res: i32, fmt: i32) -> FrameMode;
    fn freenect_find_depth_mode(res: i32, fmt: i32) -> FrameMode;
    fn freenect_set_video_mode(dev: *mut c_void, mode: FrameMode) -> i32;
    fn freenect_set_depth_mode(dev: *mut c_void, mode: FrameMode) -> i32;
    fn freenect_start_depth(dev: *mut c_void) -> i32;
    fn freenect_start_video(dev: *mut c_void) -> i32;
    fn freenect_stop_depth(dev: *mut c_void) -> i32;
    fn freenect_stop_video(dev: *mut c_void) -> i32;
    fn freenect_set_tilt_degs(dev: *mut c_void, angle: f64) -> i32;
}

struct Shared {
    store: Arc<Store>,
    last_rgb: Mutex<Instant>,
    last_depth: Mutex<Instant>,
    last_ok: Mutex<Instant>,
}

impl Shared {
    /// Healthy only while both streams have delivered within the last second.
    fn mark_ok(&self) {
        let (r, d) = (*self.last_rgb.lock().unwrap(), *self.last_depth.lock().unwrap());
        let now = Instant::now();
        if now.duration_since(r) < Duration::from_secs(1) && now.duration_since(d) < Duration::from_secs(1) {
            *self.last_ok.lock().unwrap() = now;
        }
    }
}

extern "C" fn on_video(dev: *mut c_void, data: *mut c_void, _ts: u32) {
    let sh = unsafe { &*(freenect_get_user(dev) as *const Shared) };
    let rgb = unsafe { std::slice::from_raw_parts(data as *const u8, W * H * 3) }.to_vec();
    sh.store.put_rgb(rgb);
    *sh.last_rgb.lock().unwrap() = Instant::now();
    sh.mark_ok();
}

extern "C" fn on_depth(dev: *mut c_void, data: *mut c_void, _ts: u32) {
    let sh = unsafe { &*(freenect_get_user(dev) as *const Shared) };
    let d = unsafe { std::slice::from_raw_parts(data as *const u16, W * H) }.to_vec();
    sh.store.put_depth(d);
    *sh.last_depth.lock().unwrap() = Instant::now();
    sh.mark_ok();
}

pub struct Kinect {
    store: Arc<Store>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Kinect {
    pub fn new() -> Self { Kinect { store: Store::new(), thread: Mutex::new(None) } }

    fn run(store: Arc<Store>) {
        while store.running.load(Ordering::SeqCst) {
            store.clear();
            let sh = Arc::new(Shared { store: store.clone(), last_rgb: Mutex::new(Instant::now()), last_depth: Mutex::new(Instant::now()), last_ok: Mutex::new(Instant::now()) });
            let user = Arc::into_raw(sh.clone()) as *mut c_void;
            let mut restart = false;
            unsafe {
                let mut ctx = std::ptr::null_mut();
                if freenect_init(&mut ctx, std::ptr::null_mut()) < 0 {
                    std::thread::sleep(Duration::from_secs(1));
                    drop(Arc::from_raw(user as *const Shared));
                    continue;
                }
                freenect_set_log_level(ctx, 1);
                let mut dev = std::ptr::null_mut();
                if freenect_open_device(ctx, &mut dev, 0) < 0 || dev.is_null() {
                    freenect_shutdown(ctx);
                    drop(Arc::from_raw(user as *const Shared));
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
                freenect_set_user(dev, user);
                freenect_set_video_mode(dev, freenect_find_video_mode(1, 0)); // medium, RGB
                freenect_set_depth_mode(dev, freenect_find_depth_mode(1, 4)); // medium, registered mm
                freenect_set_depth_callback(dev, on_depth);
                freenect_set_video_callback(dev, on_video);
                freenect_start_depth(dev);
                freenect_start_video(dev);
                let mut sent: Option<i32> = None;
                while store.running.load(Ordering::SeqCst) && !restart {
                    let mut tv = TimeVal { sec: 0, usec: 100_000 };
                    if freenect_process_events_timeout(ctx, &mut tv) < 0 {
                        restart = true;
                        break;
                    }
                    let want = store.tilt.load(Ordering::SeqCst);
                    if sent != Some(want) {
                        freenect_set_tilt_degs(dev, want as f64);
                        sent = Some(want);
                    }
                    // streams sometimes fail to come up on USB 2: restart if either stalls
                    if Instant::now().duration_since(*sh.last_ok.lock().unwrap()) > Duration::from_secs(3) {
                        restart = true;
                    }
                }
                freenect_stop_depth(dev);
                freenect_stop_video(dev);
                freenect_close_device(dev);
                freenect_shutdown(ctx);
                drop(Arc::from_raw(user as *const Shared));
            }
            if restart {
                eprintln!("Kinect stream stalled, restarting...");
            }
        }
    }
}

impl Source for Kinect {
    fn store(&self) -> &Arc<Store> { &self.store }
    fn start(&self) {
        let s = self.store.clone();
        *self.thread.lock().unwrap() = Some(std::thread::spawn(move || Kinect::run(s)));
    }
    fn stop(&self) {
        self.store.running.store(false, Ordering::SeqCst);
        if let Some(t) = self.thread.lock().unwrap().take() {
            let _ = t.join();
        }
    }
}

// ------------------------------------------------------------------------- recording player

pub struct Player {
    store: Arc<Store>,
    names: Vec<String>,
    times: Vec<f64>,
    folder: std::path::PathBuf,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Player {
    pub fn new(folder: &str) -> Result<Self, String> {
        let folder = std::path::PathBuf::from(folder);
        let mut names: Vec<String> = std::fs::read_dir(&folder)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.strip_prefix("depth_").and_then(|s| s.strip_suffix(".png")).map(|s| s.to_string())
            })
            .collect();
        names.sort();
        if names.is_empty() {
            return Err(format!("no frames found in {}", folder.display()));
        }
        let times = std::fs::read_to_string(folder.join("timestamps.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<f64>>(&s).ok())
            .unwrap_or_else(|| (0..names.len()).map(|i| i as f64 / 30.0).collect());
        Ok(Player { store: Store::new(), names, times, folder, thread: Mutex::new(None) })
    }
}

impl Source for Player {
    fn store(&self) -> &Arc<Store> { &self.store }
    fn start(&self) {
        let (store, names, times, folder) = (self.store.clone(), self.names.clone(), self.times.clone(), self.folder.clone());
        *self.thread.lock().unwrap() = Some(std::thread::spawn(move || {
            while store.running.load(Ordering::SeqCst) {
                let t0 = Instant::now();
                for (i, n) in names.iter().enumerate() {
                    let wait = times.get(i).copied().unwrap_or(0.0) - t0.elapsed().as_secs_f64();
                    if wait > 0.0 {
                        std::thread::sleep(Duration::from_secs_f64(wait));
                    }
                    if !store.running.load(Ordering::SeqCst) {
                        return;
                    }
                    if let Some((rgb, d)) = crate::io::load_frame(&folder, &format!("rgb_{n}.jpg"), &format!("depth_{n}.png")) {
                        store.put_both(rgb, d);
                    }
                }
            }
        }));
    }
    fn stop(&self) {
        self.store.running.store(false, Ordering::SeqCst);
        if let Some(t) = self.thread.lock().unwrap().take() {
            let _ = t.join();
        }
    }
}
