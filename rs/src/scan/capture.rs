//! Capture (tracked, masked frames), raw saving, rescan from raw, and the full model build.
use super::fuse::{fuse, FrameRef, FuseCfg};
use super::mesh::{extract_mesh, Mesh};
use super::rig::Figure;
use super::segment::Segmenter;
use super::track::*;
use super::world::*;
use crate::io;
use crate::math::*;
use crate::source::{Depth, Rgb, H, W};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

/// Everything that differs between the full-body and the seated-bust booth.
#[derive(Clone, Copy)]
pub struct Profile {
    pub name: &'static str,
    pub min_mask_px: usize,
    pub min_pts: usize,
    pub track_pts: usize,
    /// Tracking points are limited to this world-height band (None = the segmenter's own gate).
    pub track_y: Option<(f32, f32)>,
    pub icp_max_d: f32,
    pub icp_rms: f64,
    /// Rotation between tracking keyframes (degrees).
    pub key_deg: f64,
    pub loops: LoopCfg,
    pub voxel: f32,
    pub trunc: f32,
    pub radius: f32,
    pub max_frames: usize,
    pub min_extent: f32,
    pub smooth: usize,
}

impl Profile {
    pub fn body() -> Profile {
        Profile { name: "body", min_mask_px: 3000, min_pts: 1500, track_pts: 3000, track_y: Some((0.15, 1.9)), icp_max_d: 0.10, icp_rms: 0.04, key_deg: 12.0,
                  loops: LoopCfg { max_d: 0.12, rms: 0.04, shift: 0.25 }, voxel: 0.015, trunc: 0.06, radius: 0.95, max_frames: 110, min_extent: 0.3, smooth: 8 }
    }
    pub fn bust() -> Profile {
        Profile { name: "bust", min_mask_px: 6000, min_pts: 3000, track_pts: 6000, track_y: None, icp_max_d: 0.08, icp_rms: 0.025, key_deg: 6.0,
                  loops: LoopCfg { max_d: 0.06, rms: 0.025, shift: 0.12 }, voxel: 0.006, trunc: 0.025, radius: 0.40, max_frames: 150, min_extent: 0.25, smooth: 4 }
    }
    pub fn by_name(n: &str) -> Profile { if n == "bust" { Profile::bust() } else { Profile::body() } }
}

pub struct Cap {
    pub depth: Vec<u16>, // masked
    pub rgb: Rgb,
    pub pts: Vec<V3>,
    pub pose: Pose,
}

/// Turns raw (rgb, depth) frames into tracked, masked captures ready for fusion.
pub struct Capture {
    pub profile: Profile,
    pub seg: Segmenter,
    pub trk: Tracker,
    pub caps: Vec<Cap>,
    pub mask_px: Vec<usize>,
}

impl Capture {
    pub fn new(seg: Segmenter) -> Self { Capture::with(seg, Profile::body()) }
    pub fn with(seg: Segmenter, profile: Profile) -> Self {
        Capture { profile, seg, trk: Tracker::with(profile.icp_max_d, profile.icp_rms, profile.key_deg), caps: Vec::new(), mask_px: Vec::new() }
    }

    /// Returns (mask, used). `used` is false when the person wasn't usable in this frame.
    pub fn add(&mut self, rgb: &Rgb, depth: &[u16]) -> (Vec<bool>, bool) {
        let (m, pw) = self.seg.mask(depth);
        let npx = m.iter().filter(|b| **b).count();
        if npx < self.profile.min_mask_px {
            return (m, false);
        }
        // no floor / overhead: static points bias the spin tracker
        let (ylo, yhi) = self.profile.track_y.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
        let p: Vec<V3> = (0..W * H).filter(|&i| m[i] && pw[i][1] > ylo && pw[i][1] < yhi).map(|i| pw[i]).collect();
        if p.len() < self.profile.min_pts {
            return (m, false);
        }
        let step = (p.len() / self.profile.track_pts).max(1);
        let pts: Vec<V3> = p.iter().step_by(step).copied().collect();
        let pose = self.trk.add(&pts);
        let masked: Vec<u16> = (0..W * H).map(|i| if m[i] { depth[i] } else { 0 }).collect();
        self.caps.push(Cap { depth: masked, rgb: rgb.clone(), pts, pose });
        self.mask_px.push(npx);
        (m, true)
    }
}

/// Writes every captured frame (unmasked rgb jpg + 16-bit depth png) so scans can be re-processed.
pub struct RawSaver {
    dir: PathBuf,
    tx: Option<mpsc::SyncSender<(usize, Rgb, Depth)>>,
    handle: Option<std::thread::JoinHandle<()>>,
    n: usize,
    times: Vec<f64>,
    t0: Instant,
}

impl RawSaver {
    pub fn new(folder: &Path) -> RawSaver {
        let dir = folder.join("raw");
        let _ = std::fs::create_dir_all(&dir);
        let (tx, rx) = mpsc::sync_channel::<(usize, Rgb, Depth)>(400);
        let d2 = dir.clone();
        let handle = std::thread::spawn(move || {
            while let Ok((i, rgb, depth)) = rx.recv() {
                io::save_rgb_jpg(&d2.join(format!("rgb_{i:05}.jpg")), &rgb);
                io::save_depth_png(&d2.join(format!("depth_{i:05}.png")), &depth);
            }
        });
        RawSaver { dir, tx: Some(tx), handle: Some(handle), n: 0, times: Vec::new(), t0: Instant::now() }
    }

    pub fn add(&mut self, rgb: &Rgb, depth: &Depth) {
        if let Some(tx) = &self.tx {
            if tx.try_send((self.n, rgb.clone(), depth.clone())).is_ok() {
                self.times.push(self.t0.elapsed().as_secs_f64());
                self.n += 1;
            }
        }
    }

    pub fn finish(&mut self, world: &World, center: Option<[f32; 2]>, extra: serde_json::Value) {
        self.tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let mut info = serde_json::json!({
            "frames": self.n, "times": self.times,
            "world_R": world.r.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
            "world_t": world.t.to_vec(), "floor_found": world.floor_found,
            "center": center.map(|c| c.to_vec()),
            "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
        });
        if let (Some(o), Some(e)) = (info.as_object_mut(), extra.as_object()) {
            for (k, v) in e {
                o.insert(k.clone(), v.clone());
            }
        }
        let _ = std::fs::write(self.dir.join("capture.json"), info.to_string());
    }
}

/// Pixels with valid depth at least `r` pixels away from any invalid pixel (separable erosion).
fn interior(depth: &[u16], r: usize) -> Vec<bool> {
    let mut h = vec![false; W * H];
    for y in 0..H {
        for x in 0..W {
            h[y * W + x] = x >= r && x + r < W && (x - r..=x + r).all(|k| depth[y * W + k] > 0);
        }
    }
    let mut out = vec![false; W * H];
    for y in r..H.saturating_sub(r) {
        for x in 0..W {
            out[y * W + x] = (y - r..=y + r).all(|k| h[k * W + x]);
        }
    }
    out
}

pub fn build_model(caps: &[Cap], world: &World, center: [f32; 2], prof: &Profile, y_range: (f32, f32), progress: &dyn Fn(&str, f32), log: &mut dyn FnMut(String)) -> Result<Mesh, String> {
    let pts: Vec<&[V3]> = caps.iter().map(|c| c.pts.as_slice()).collect();
    let poses: Vec<Pose> = caps.iter().map(|c| c.pose).collect();
    let poses = close_loops(&pts, &poses, prof.loops, log);
    let step = ((caps.len() as f32 / prof.max_frames as f32).ceil() as usize).max(1);
    let use_idx: Vec<usize> = (0..caps.len()).step_by(step).collect();
    let col_ok: Vec<Vec<bool>> = use_idx.iter().map(|&i| interior(&caps[i].depth, 3)).collect();
    let frames: Vec<FrameRef> = use_idx.iter().zip(&col_ok).map(|(&i, ok)| FrameRef { depth: &caps[i].depth, col_ok: ok, rgb: &caps[i].rgb }).collect();
    let up: Vec<Pose> = use_idx.iter().map(|&i| poses[i]).collect();
    let cfg = FuseCfg { voxel: prof.voxel, trunc: prof.trunc, radius: prof.radius, y0: y_range.0, height: y_range.1 - y_range.0 };
    let vol = fuse(&frames, &up, world, center, cfg, &|k, n| progress("Fusing depth", 0.1 + 0.7 * k as f32 / n as f32));
    progress("Building mesh", 0.85);
    extract_mesh(&vol, 2.0, prof.smooth, prof.min_extent)
}

pub struct RescanStats {
    pub frames: usize,
    pub used: usize,
    pub turns: f64,
    pub quality: f64,
}

/// Rebuild a model from a saved raw capture (`<scan>/raw`). Returns (out_folder, stats, mesh).
pub fn rescan(folder: &Path, out: Option<&Path>, log: &mut dyn FnMut(String), refloor: bool) -> Result<(PathBuf, RescanStats, Mesh), String> {
    let raw = folder.join("raw");
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(raw.join("capture.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let n = cfg["frames"].as_u64().ok_or("bad capture.json")? as usize;
    let load = |i: usize| io::load_frame(&raw, &format!("rgb_{i:05}.jpg"), &format!("depth_{i:05}.png"));
    let (_, d0) = load(0).ok_or("cannot read frame 0")?;
    let f = |v: &serde_json::Value| v.as_f64().unwrap_or(0.0) as f32;
    let mut world = World {
        r: [0, 1, 2].map(|i| [0, 1, 2].map(|j| f(&cfg["world_R"][i][j]))),
        t: [0, 1, 2].map(|i| f(&cfg["world_t"][i])),
        floor_found: cfg["floor_found"].as_bool().unwrap_or(false),
    };
    if refloor {
        match fit_floor(&d0, 0) {
            Some(w) => {
                log(format!("refit floor: camera height {:.2} m", w.t[1]));
                world = w;
            }
            None => log("refit floor: FAILED, keeping the saved one".into()),
        }
    }
    let prof = Profile::by_name(cfg["profile"].as_str().unwrap_or("body"));
    let mut seg = Segmenter::new(world);
    let mut y_range = (0.0f32, 2.1f32);
    if prof.name == "bust" {
        let (lo, hi) = (f(&cfg["y_lo"]), f(&cfg["y_hi"]));
        seg.y_lo = lo;
        seg.y_hi = hi;
        seg.bust_defaults();
        y_range = (lo, hi);
    }
    if let (Some(c), false) = (cfg["center"].as_array(), refloor) {
        seg.center = Some([f(&c[0]), f(&c[1])]);
    } else if prof.name == "body" && seg.find_person(&d0).is_none() {
        return Err("no person found in the first raw frame".into());
    }
    let mut cap = Capture::with(seg, prof);
    for i in 0..n {
        if let Some((rgb, depth)) = load(i) {
            cap.add(&std::sync::Arc::new(rgb), &depth);
        }
    }
    if cap.caps.is_empty() {
        return Err("no usable frames".into());
    }
    let stats = RescanStats { frames: n, used: cap.caps.len(), turns: cap.trk.yaw().abs() / std::f64::consts::TAU, quality: cap.trk.quality };
    log(format!("rescan: {} frames, {} used, {:.2} turns, quality {:.2}", stats.frames, stats.used, stats.turns, stats.quality));
    let c0 = {
        let p = &cap.caps[0].pts;
        let nn = p.len() as f32;
        [p.iter().map(|q| q[0]).sum::<f32>() / nn, p.iter().map(|q| q[2]).sum::<f32>() / nn]
    };
    let mesh = build_model(&cap.caps, &world, c0, &prof, y_range, &|_, _| {}, log)?;
    let out = out.map(|p| p.to_path_buf()).unwrap_or_else(|| folder.join(format!("rescan_{}", timestamp("%H%M%S"))));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    io::save_ply(&out.join("model.ply"), &mesh.verts, &mesh.faces, &mesh.rgb).map_err(|e| e.to_string())?;
    io::save_obj(&out.join("model.obj"), &mesh.verts, &mesh.faces, &mesh.rgb).map_err(|e| e.to_string())?;
    Ok((out, stats, mesh))
}

/// Local time via libc `strftime`-free formatting (only the formats used here).
pub fn timestamp(fmt: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let off = local_offset_secs();
    let t = secs + off;
    let (days, rem) = (t.div_euclid(86400), t.rem_euclid(86400));
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    fmt.replace("%Y", &format!("{y:04}")).replace("%m", &format!("{m:02}")).replace("%d", &format!("{d:02}"))
        .replace("%H", &format!("{h:02}")).replace("%M", &format!("{mi:02}")).replace("%S", &format!("{s:02}"))
}

fn local_offset_secs() -> i64 {
    #[repr(C)]
    struct Tm {
        sec: i32, min: i32, hour: i32, mday: i32, mon: i32, year: i32, wday: i32, yday: i32, isdst: i32, gmtoff: i64, zone: *const i8,
    }
    extern "C" {
        fn localtime_r(t: *const i64, out: *mut Tm) -> *mut Tm;
    }
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let mut tm = Tm { sec: 0, min: 0, hour: 0, mday: 0, mon: 0, year: 0, wday: 0, yday: 0, isdst: 0, gmtoff: 0, zone: std::ptr::null() };
    unsafe {
        localtime_r(&now, &mut tm);
    }
    tm.gmtoff
}

pub fn safe_name(name: &str) -> String {
    let k: String = name.trim().chars().filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '_' || *c == '-').take(30).collect();
    let k = k.trim().to_string();
    if k.is_empty() { "visitor".into() } else { k }
}

pub fn new_scan_folder(root: &Path, name: &str) -> PathBuf {
    let f = root.join(format!("{}_{}", safe_name(name).replace(' ', "_"), timestamp("%Y%m%d_%H%M%S")));
    let _ = std::fs::create_dir_all(&f);
    f
}

pub fn save_scan(folder: &Path, name: &str, mesh: &Mesh, fig: Option<&Figure>, thumb: Option<(&[u32], usize, usize)>) {
    let _ = io::save_ply(&folder.join("model.ply"), &mesh.verts, &mesh.faces, &mesh.rgb);
    let _ = io::save_obj(&folder.join("model.obj"), &mesh.verts, &mesh.faces, &mesh.rgb);
    let mut meta = serde_json::json!({"name": name, "created": timestamp("%Y-%m-%d %H:%M:%S"), "vertices": mesh.verts.len(), "faces": mesh.faces.len()});
    if let (Some(f), Some(o)) = (fig, meta.as_object_mut()) {
        if let Some(m) = f.meta_json().as_object() {
            for (k, v) in m {
                o.insert(k.clone(), v.clone());
            }
        }
    }
    let _ = std::fs::write(folder.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap());
    if let Some((buf, w, h)) = thumb {
        io::save_rgb_png(&folder.join("preview.png"), buf, w, h);
    }
}
