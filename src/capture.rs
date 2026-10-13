//! Camera discovery and per-camera capture threads.
//!
//! Every V4L2 video-capture node under /dev is probed. For each one that
//! actually produces pictures (metadata-only nodes are skipped) a thread is
//! spawned that streams frames via mmap, converts them to packed RGB, and
//! publishes the newest frame into a shared [`Slot`].

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use v4l::buffer::Type;
use v4l::capability::Flags as CapFlags;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::{Format, FourCC};

/// Preferred capture size. Every tile is downscaled for the grid anyway, so a
/// modest size keeps USB bandwidth low when several cameras share a hub.
const TARGET_W: u32 = 640;
const TARGET_H: u32 = 480;

/// One decoded frame in packed RGB (3 bytes per pixel).
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Shared state between a capture thread and the UI.
pub struct Slot {
    frame: Mutex<Option<Arc<Frame>>>,
    status: Mutex<String>,
    last_frame: Mutex<Instant>,
    pub seq: AtomicU64,
    pub fps: AtomicU64,
    pub running: AtomicBool,
}

impl Slot {
    pub fn new() -> Arc<Slot> {
        Arc::new(Slot {
            frame: Mutex::new(None),
            status: Mutex::new("starting".into()),
            last_frame: Mutex::new(Instant::now()),
            seq: AtomicU64::new(0),
            fps: AtomicU64::new(0),
            running: AtomicBool::new(true),
        })
    }
    pub fn frame(&self) -> Option<Arc<Frame>> { self.frame.lock().unwrap().clone() }
    pub fn status(&self) -> String { self.status.lock().unwrap().clone() }
    /// Time since the most recent decoded frame (or since the camera was opened).
    pub fn since_last_frame(&self) -> Duration { self.last_frame.lock().unwrap().elapsed() }
    pub fn set_status(&self, s: impl Into<String>) { *self.status.lock().unwrap() = s.into(); }
    pub fn publish(&self, f: Frame) {
        *self.frame.lock().unwrap() = Some(Arc::new(f));
        *self.last_frame.lock().unwrap() = Instant::now();
        self.seq.fetch_add(1, Ordering::Relaxed);
    }
}

/// The thread behind one or more cameras. Dropping the last `Camera` that
/// refers to it asks the thread to stop and joins it.
pub struct Worker {
    pub running: AtomicBool,
    handle: Mutex<Option<JoinHandle<()>>>,
    tiles: AtomicUsize,
}

impl Worker {
    pub fn spawn<F: FnOnce(Arc<Worker>) + Send + 'static>(name: &str, f: F) -> std::io::Result<Arc<Worker>> {
        let w = Arc::new(Worker { running: AtomicBool::new(true), handle: Mutex::new(None), tiles: AtomicUsize::new(0) });
        let wt = w.clone();
        let h = std::thread::Builder::new().name(name.to_string()).spawn(move || f(wt))?;
        *w.handle.lock().unwrap() = Some(h);
        Ok(w)
    }
    pub fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
    /// Register one more tile fed by this worker (see `Camera::drop`).
    pub fn attach(self: &Arc<Self>) -> Arc<Self> { self.tiles.fetch_add(1, Ordering::SeqCst); self.clone() }
    pub fn is_finished(&self) -> bool { self.handle.lock().unwrap().as_ref().map_or(true, |h| h.is_finished()) }
}

/// A camera tile: a frame slot plus the worker thread that fills it.
pub struct Camera {
    /// Unique id: a /dev path for V4L2 devices, "kinect2:<serial>/<stream>" for Kinect v2.
    pub path: String,
    pub name: String,
    pub fourcc: String,
    pub width: u32,
    pub height: u32,
    pub slot: Arc<Slot>,
    pub worker: Arc<Worker>,
}

impl Camera {
    /// True once the capture thread has exited (device unplugged or gave up).
    pub fn is_dead(&self) -> bool { self.worker.is_finished() }

    /// Short device id for the label (e.g. "video3" or "kinect2").
    pub fn short_id(&self) -> String {
        if let Some(rest) = self.path.strip_prefix("/dev/") { return rest.to_string(); }
        self.path.split(':').next().unwrap_or(&self.path).to_string()
    }

    pub fn label(&self) -> String {
        format!("{}  {}  {}x{} {}", self.short_id(), self.name, self.width, self.height, self.fourcc)
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.slot.running.store(false, Ordering::Relaxed);
        // Last tile on this worker: stop the thread and wait for it so the
        // device is released before anything tries to reopen it.
        if self.worker.tiles.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.worker.running.store(false, Ordering::Relaxed);
            if let Some(h) = self.worker.handle.lock().unwrap().take() { let _ = h.join(); }
        }
    }
}

/// Pixel formats we know how to convert, in order of preference.
const SUPPORTED: [&[u8; 4]; 4] = [b"MJPG", b"YUYV", b"UYVY", b"GREY"];

/// All /dev/videoN nodes, sorted by N.
pub fn video_paths() -> Vec<String> {
    let mut paths: Vec<String> = std::fs::read_dir("/dev")
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with("video") && n.len() > 5 && n[5..].chars().all(|c| c.is_ascii_digit()))
                .map(|n| format!("/dev/{n}"))
                .collect()
        })
        .unwrap_or_default();
    paths.sort_by_key(|p| dev_index(p));
    paths
}

fn dev_index(path: &str) -> u32 { path.trim_start_matches("/dev/video").parse().unwrap_or(u32::MAX) }

/// Keeps a camera list in sync with the device nodes that currently exist.
///
/// Call [`Watcher::poll`] periodically. Cameras whose node vanished or whose
/// capture thread died are dropped (releasing the device); new nodes are
/// probed and, if they produce pictures, appended. Nodes that turned out not
/// to be cameras are remembered and not re-probed until they disappear and
/// reappear; nodes whose probe failed (e.g. udev permissions not applied yet)
/// are retried after a short back-off.
#[derive(Default)]
pub struct Watcher {
    not_camera: HashSet<String>,
    failed: HashMap<String, Instant>,
    #[cfg_attr(not(feature = "kinect2"), allow(dead_code))]
    last_k2_scan: Option<Instant>,
    #[cfg_attr(not(feature = "kinect2"), allow(dead_code))]
    k2_misses: HashMap<String, u32>,
}

impl Watcher {
    /// Reconcile `cams` with /dev. Returns true if the list changed.
    pub fn poll(&mut self, cams: &mut Vec<Camera>) -> bool {
        let present = video_paths();
        let present_set: HashSet<&String> = present.iter().collect();
        let mut changed = false;

        // Forget bookkeeping for nodes that went away.
        self.not_camera.retain(|p| present_set.contains(p));
        self.failed.retain(|p, _| present_set.contains(p));

        // Drop cameras that were unplugged or whose thread exited.
        let before = cams.len();
        cams.retain(|c| {
            if !c.path.starts_with("/dev/") { return true; } // not ours (Kinect v2 etc.)
            let keep = present_set.contains(&c.path) && !c.is_dead();
            if !keep { eprintln!("{}: removed ({})", c.path, if c.is_dead() { "stream ended" } else { "device gone" }); }
            keep
        });
        changed |= cams.len() != before;

        // Probe anything new.
        for path in &present {
            if cams.iter().any(|c| &c.path == path) || self.not_camera.contains(path) { continue; }
            if let Some(t) = self.failed.get(path) {
                if t.elapsed() < Duration::from_secs(3) { continue; }
            }
            match probe(path) {
                Ok(Some(c)) => {
                    eprintln!("{}: added  {}", path, c.label());
                    cams.push(c);
                    self.failed.remove(path);
                    changed = true;
                }
                Ok(None) => { self.not_camera.insert(path.clone()); }
                Err(e) => {
                    if !self.failed.contains_key(path) { eprintln!("{path}: probe failed ({}), will retry", explain(&e)); }
                    self.failed.insert(path.clone(), Instant::now());
                }
            }
        }
        changed |= self.poll_kinect2(cams);

        if changed { cams.sort_by(|a, b| sort_key(&a.path).cmp(&sort_key(&b.path))); }
        changed
    }

    /// Kinect v2 sensors speak libusb, not V4L2, so they are enumerated
    /// separately (every 3 s; a libusb scan is heavier than a readdir).
    #[cfg(feature = "kinect2")]
    fn poll_kinect2(&mut self, cams: &mut Vec<Camera>) -> bool {
        if self.last_k2_scan.map_or(false, |t| t.elapsed() < Duration::from_secs(3)) { return false; }
        self.last_k2_scan = Some(Instant::now());
        let serials = crate::kinect2::enumerate();
        if std::env::var_os("CAMGRID_DEBUG").is_some() { eprintln!("kinect2 scan: {serials:?}"); }
        let mut changed = false;

        // A libusb scan can miss an open device transiently; only drop a
        // sensor after two consecutive misses (or if its thread died).
        let before = cams.len();
        cams.retain(|c| {
            let Some(serial) = crate::kinect2::serial_of(&c.path) else { return true };
            let present = serials.iter().any(|s| s == serial);
            let misses = self.k2_misses.entry(serial.to_string()).or_insert(0);
            *misses = if present { 0 } else { *misses + 1 };
            let keep = *misses < 2 && !c.is_dead();
            if !keep { eprintln!("{}: removed ({})", c.path, if c.is_dead() { "stream ended" } else { "device gone" }); }
            keep
        });
        changed |= cams.len() != before;
        self.failed.retain(|p, _| !p.starts_with("kinect2:") || serials.iter().any(|s| p == &format!("kinect2:{s}")));

        for serial in &serials {
            let key = format!("kinect2:{serial}");
            if cams.iter().any(|c| crate::kinect2::serial_of(&c.path) == Some(serial)) { continue; }
            if let Some(t) = self.failed.get(&key) {
                if t.elapsed() < Duration::from_secs(5) { continue; }
            }
            match crate::kinect2::open(serial) {
                Ok(new) => {
                    for c in &new { eprintln!("{}: added  {}", c.path, c.label()); }
                    cams.extend(new);
                    self.failed.remove(&key);
                    changed = true;
                }
                Err(e) => {
                    if !self.failed.contains_key(&key) { eprintln!("{key}: open failed ({e}), will retry"); }
                    self.failed.insert(key, Instant::now());
                }
            }
        }
        changed
    }

    #[cfg(not(feature = "kinect2"))]
    fn poll_kinect2(&mut self, _cams: &mut Vec<Camera>) -> bool { false }
}

/// V4L2 devices first (by number), then Kinect v2 streams (by serial, color before depth).
fn sort_key(path: &str) -> (u8, u32, String) {
    if path.starts_with("/dev/") { (0, dev_index(path), String::new()) } else { (1, 0, path.to_string()) }
}

/// Open a device, pick a format, and spawn its capture thread.
/// Returns Ok(None) for nodes that are not picture sources.
fn probe(path: &str) -> std::io::Result<Option<Camera>> {
    let dev = Device::with_path(path)?;
    let caps = dev.query_caps()?;
    if !caps.capabilities.contains(CapFlags::VIDEO_CAPTURE) || !caps.capabilities.contains(CapFlags::STREAMING) {
        return Ok(None);
    }
    let formats = dev.enum_formats()?;
    if formats.is_empty() {
        return Ok(None);
    }

    // Choose the best supported pixel format and the size closest to TARGET.
    let mut choice: Option<(FourCC, u32, u32)> = None;
    for want in SUPPORTED {
        let fcc = FourCC::new(want);
        if !formats.iter().any(|f| f.fourcc == fcc) { continue; }
        let sizes: Vec<(u32, u32)> = dev
            .enum_framesizes(fcc)?
            .into_iter()
            .flat_map(|fs| fs.size.to_discrete().into_iter().map(|d| (d.width, d.height)).collect::<Vec<_>>())
            .collect();
        let (w, h) = sizes
            .iter()
            .copied()
            .min_by_key(|&(w, h)| (w as i64 - TARGET_W as i64).abs() + (h as i64 - TARGET_H as i64).abs())
            .unwrap_or((TARGET_W, TARGET_H));
        choice = Some((fcc, w, h));
        break;
    }
    let Some((fcc, w, h)) = choice else {
        let list: Vec<String> = formats.iter().map(|f| f.fourcc.to_string()).collect();
        eprintln!("{path}: no supported pixel format (offers {})", list.join(", "));
        return Ok(None);
    };

    let fmt = dev.set_format(&Format::new(w, h, fcc))?;
    let _ = dev.set_params(&v4l::video::capture::Parameters::with_fps(30));
    let name = caps.card.clone();
    let slot = Slot::new();

    let t_slot = slot.clone();
    let t_path = path.to_string();
    let worker = Worker::spawn(&format!("cap-{}", path.trim_start_matches("/dev/")), move |w| capture_loop(dev, fmt, t_slot, t_path, w))?;

    Ok(Some(Camera {
        path: path.to_string(),
        name,
        fourcc: fmt.fourcc.to_string(),
        width: fmt.width,
        height: fmt.height,
        slot,
        worker: worker.attach(),
    }))
}

/// Human-readable explanation for the errors cameras commonly return.
fn explain(e: &std::io::Error) -> String {
    match e.raw_os_error() {
        Some(28) => "USB bus out of bandwidth (ENOSPC): try another USB port".into(),
        Some(16) => "device busy: in use by another program".into(),
        Some(22) => "stream failed to start (EINVAL): usually no USB bandwidth, try another port".into(),
        Some(19) => "device unplugged".into(),
        Some(110) => "timeout: no frames from device".into(),
        Some(5) => "I/O error from device".into(),
        _ => e.to_string(),
    }
}

fn is_gone(e: &std::io::Error) -> bool { e.raw_os_error() == Some(19) } // ENODEV

/// Capture thread body. Keeps trying to (re)start the stream until the
/// device disappears or the camera is dropped, so a transient condition
/// (bus bandwidth, another program holding the device) shows as a status
/// on the tile and recovers by itself when it clears.
fn capture_loop(dev: Device, fmt: Format, slot: Arc<Slot>, path: String, worker: Arc<Worker>) {
    let (w, h) = (fmt.width as usize, fmt.height as usize);
    let fourcc = fmt.fourcc;
    let retry = Duration::from_secs(2);

    'outer: while worker.is_running() {
        let mut stream = match MmapStream::with_buffers(&dev, Type::VideoCapture, 4) {
            Ok(s) => s,
            Err(e) => {
                slot.set_status(explain(&e));
                if is_gone(&e) { break; }
                sleep_while_running(&worker, retry);
                continue;
            }
        };
        let mut window_start = Instant::now();
        let mut window_frames = 0u32;
        let mut consecutive_errors = 0u32;

        while worker.is_running() {
            let (buf, meta) = match stream.next() {
                Ok(x) => x,
                Err(e) => {
                    slot.set_status(explain(&e));
                    slot.fps.store(0, Ordering::Relaxed);
                    if is_gone(&e) { break 'outer; }
                    consecutive_errors += 1;
                    if consecutive_errors >= 3 {
                        // Tear the stream down and start over after a pause.
                        drop(stream);
                        sleep_while_running(&worker, retry);
                        continue 'outer;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
            };
            let used = (meta.bytesused as usize).min(buf.len());
            if used == 0 { continue; }
            let data = &buf[..used];

            let rgb = match &fourcc.repr {
                b"MJPG" => decode_mjpeg(data, w, h),
                b"YUYV" => Ok(yuv422_to_rgb(data, w, h, 0, 1, 3)),
                b"UYVY" => Ok(yuv422_to_rgb(data, w, h, 1, 0, 2)),
                b"GREY" => Ok(data.iter().take(w * h).flat_map(|&g| [g, g, g]).collect()),
                other => Err(format!("unsupported {}", String::from_utf8_lossy(other))),
            };
            match rgb {
                Ok(rgb) => {
                    consecutive_errors = 0;
                    slot.publish(Frame { width: w, height: h, rgb });
                    slot.set_status("live");
                    window_frames += 1;
                    let dt = window_start.elapsed();
                    if dt >= Duration::from_secs(1) {
                        slot.fps.store((window_frames as f64 / dt.as_secs_f64()).round() as u64, Ordering::Relaxed);
                        window_start = Instant::now();
                        window_frames = 0;
                    }
                }
                Err(e) => {
                    // A corrupt frame now and then is normal for MJPEG; just skip it.
                    slot.set_status(format!("decode: {e}"));
                }
            }
        }
    }
    slot.fps.store(0, Ordering::Relaxed);
    if worker.is_running() {
        eprintln!("{path}: {}", slot.status());
    }
}

pub fn sleep_while_running(worker: &Worker, d: Duration) {
    let end = Instant::now() + d;
    while worker.is_running() && Instant::now() < end {
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Decode one MJPEG frame to RGB. Resizes/letterboxes if the JPEG's size
/// differs from the negotiated format (some cameras lie).
fn decode_mjpeg(data: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB).set_strict_mode(false);
    let attempt = |bytes: &[u8]| -> Result<(Vec<u8>, usize, usize), String> {
        let mut dec = JpegDecoder::new_with_options(std::io::Cursor::new(bytes), opts);
        let px = dec.decode().map_err(|e| e.to_string())?;
        let info = dec.info().ok_or("no header")?;
        Ok((px, info.width as usize, info.height as usize))
    };
    let (px, iw, ih) = match attempt(data) {
        Ok(x) => x,
        Err(e) if !has_dht(data) => attempt(&inject_dht(data)).map_err(|_| e)?,
        Err(e) => return Err(e),
    };
    if iw == w && ih == h {
        return Ok(px);
    }
    // Copy into the expected size (crop or pad).
    let mut out = vec![0u8; w * h * 3];
    let cw = iw.min(w);
    for y in 0..ih.min(h) {
        out[y * w * 3..y * w * 3 + cw * 3].copy_from_slice(&px[y * iw * 3..y * iw * 3 + cw * 3]);
    }
    Ok(out)
}

/// True if the JPEG stream carries its own Huffman tables (DHT marker).
fn has_dht(data: &[u8]) -> bool {
    let mut i = 2;
    while i + 3 < data.len() {
        if data[i] != 0xFF { return false; }
        let m = data[i + 1];
        if m == 0xC4 { return true; }
        if m == 0xDA || m == 0xD9 { return false; }
        let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
        i += 2 + len;
    }
    false
}

/// Insert the standard Motion-JPEG Huffman tables right after SOI.
fn inject_dht(data: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(data.len() + MJPEG_DHT.len());
    v.extend_from_slice(&data[..2.min(data.len())]);
    v.extend_from_slice(&MJPEG_DHT);
    v.extend_from_slice(&data[2.min(data.len())..]);
    v
}

/// Packed 4:2:2 to RGB. `yo` is the byte offset of Y0 within a 4-byte macro
/// pixel, `uo` the offset of U, `vo` the offset of V.
fn yuv422_to_rgb(data: &[u8], w: usize, h: usize, yo: usize, uo: usize, vo: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    let pairs = (w / 2) * h;
    for (i, mp) in data.chunks_exact(4).take(pairs).enumerate() {
        let y0 = mp[yo] as i32;
        let y1 = mp[yo + 2] as i32;
        let u = mp[uo] as i32 - 128;
        let v = mp[vo] as i32 - 128;
        let o = i * 6;
        yuv_px(y0, u, v, &mut out[o..o + 3]);
        yuv_px(y1, u, v, &mut out[o + 3..o + 6]);
    }
    out
}

#[inline]
fn yuv_px(y: i32, u: i32, v: i32, out: &mut [u8]) {
    // BT.601 full-range integer approximation.
    let c = y - 16;
    let r = (298 * c + 409 * v + 128) >> 8;
    let g = (298 * c - 100 * u - 208 * v + 128) >> 8;
    let b = (298 * c + 516 * u + 128) >> 8;
    out[0] = r.clamp(0, 255) as u8;
    out[1] = g.clamp(0, 255) as u8;
    out[2] = b.clamp(0, 255) as u8;
}

/// Standard JPEG Huffman tables (ITU T.81 Annex K.3), as a DHT segment.
/// Used for MJPEG streams that omit them.
const MJPEG_DHT: [u8; 420] = [
    0xFF, 0xC4, 0x01, 0xA2,
    0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
    0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
    0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
    0x11, 0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77,
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];
