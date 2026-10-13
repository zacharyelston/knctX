//! roommap: fuse the Kinect v1 and Kinect v2 depth cameras into one 3D map of the room.
//!
//! World frame = Kinect v1 depth camera frame (metres; x right, y down, z forward).
//! The Kinect v2 cloud is placed with an adjustable rigid pose (saved to roommap_pose.json)
//! that can be refined automatically with ICP against the v1 cloud. Every frame from both
//! sensors is merged into a voxel map which is rendered with an orbit camera.
//!
//! Mouse: drag = orbit, wheel = zoom.   Esc/Q quit   H help
//! View:  W/S dolly, A/D strafe, R/F up/down, Home reset, 1/2 toggle v1/v2, M map on/off,
//!        L live clouds on/off, X mirror, P point size
//! v2 pose: arrows = move x/z, PgUp/PgDn = move y, Z/C yaw, T/G pitch, V/B roll
//!        (hold Shift for fine steps), I = ICP refine, O = reset pose, K = save pose
//! Map:   N clear map, E export PLY to ../pointclouds/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use kinect3d::gfx::{Canvas, Col};
use kinect3d::kinect2::{Cloud, Kinect2};
use kinect3d::math::*;
use kinect3d::scan::world::cam_point;
use kinect3d::source::{Kinect, Source, H as H1, W as W1};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};

const CW: usize = 1280;
const CH: usize = 800;
const VOXEL: f32 = 0.03; // metres
const MAX_RANGE: f32 = 7.0;
const POSE_FILE: &str = "roommap_pose.json";

// ----------------------------------------------------------------------------- rigid poses

#[derive(Clone, Copy)]
struct Pose {
    r: M3,
    t: V3,
}

impl Pose {
    const IDENT: Pose = Pose { r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], t: [0.0; 3] };
    fn apply(&self, p: V3) -> V3 { add(mat_vec(&self.r, p), self.t) }
    /// self ∘ other  (apply `other` first, then `self`)
    fn compose(&self, other: &Pose) -> Pose { Pose { r: mat_mul(&self.r, &other.r), t: self.apply(other.t) } }
    /// Rotate about the pose's own position (keeps t fixed).
    fn rotate_local(&mut self, rx: f32, ry: f32, rz: f32) { self.r = mat_mul(&euler(rx, ry, rz), &self.r); }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "r": self.r, "t": self.t })
    }
    fn from_json(v: &serde_json::Value) -> Option<Pose> {
        let f = |x: &serde_json::Value| x.as_f64().map(|f| f as f32);
        let r = v.get("r")?.as_array()?;
        let t = v.get("t")?.as_array()?;
        let mut m = [[0.0f32; 3]; 3];
        for (i, row) in r.iter().enumerate().take(3) {
            for (j, x) in row.as_array()?.iter().enumerate().take(3) { m[i][j] = f(x)?; }
        }
        Some(Pose { r: m, t: [f(&t[0])?, f(&t[1])?, f(&t[2])?] })
    }
}

/// Starting guess for the v2 pose: beside the v1 looking the same way.
fn default_v2_pose() -> Pose { Pose { r: Pose::IDENT.r, t: [0.3, 0.0, 0.0] } }

// ----------------------------------------------------------------------------- ICP

/// Uniform-grid spatial hash for nearest-neighbour queries.
struct Grid {
    cell: f32,
    map: HashMap<(i32, i32, i32), Vec<u32>>,
    pts: Vec<V3>,
}

impl Grid {
    fn new(pts: Vec<V3>, cell: f32) -> Grid {
        let mut map: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, p) in pts.iter().enumerate() {
            map.entry(key(*p, cell)).or_default().push(i as u32);
        }
        Grid { cell, map, pts }
    }
    fn nearest(&self, q: V3, max_d: f32) -> Option<(V3, f32)> {
        let k = key(q, self.cell);
        let reach = (max_d / self.cell).ceil() as i32;
        let mut best: Option<(V3, f32)> = None;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    if let Some(ids) = self.map.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                        for &i in ids {
                            let p = self.pts[i as usize];
                            let d = norm(sub(p, q));
                            if d <= max_d && best.map_or(true, |b| d < b.1) { best = Some((p, d)); }
                        }
                    }
                }
            }
        }
        best
    }
}

#[inline]
fn key(p: V3, cell: f32) -> (i32, i32, i32) {
    ((p[0] / cell).floor() as i32, (p[1] / cell).floor() as i32, (p[2] / cell).floor() as i32)
}

/// Keep one point per `cell`-sized voxel.
fn voxel_downsample(pts: &[V3], cell: f32) -> Vec<V3> {
    let mut seen = std::collections::HashSet::new();
    pts.iter().copied().filter(|p| seen.insert(key(*p, cell))).collect()
}

/// Best rigid transform mapping `src[i]` onto `dst[i]` (Horn's quaternion method).
fn kabsch(src: &[V3], dst: &[V3]) -> Pose {
    let n = src.len() as f32;
    let cs = scale(src.iter().fold([0.0; 3], |a, p| add(a, *p)), 1.0 / n);
    let cd = scale(dst.iter().fold([0.0; 3], |a, p| add(a, *p)), 1.0 / n);
    let mut s = [[0.0f64; 3]; 3]; // cross-covariance Σ (src-cs)(dst-cd)^T
    for (p, q) in src.iter().zip(dst) {
        let a = sub(*p, cs);
        let b = sub(*q, cd);
        for i in 0..3 { for j in 0..3 { s[i][j] += a[i] as f64 * b[j] as f64; } }
    }
    let (sxx, sxy, sxz) = (s[0][0], s[0][1], s[0][2]);
    let (syx, syy, syz) = (s[1][0], s[1][1], s[1][2]);
    let (szx, szy, szz) = (s[2][0], s[2][1], s[2][2]);
    let nm = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    // Largest eigenvector by power iteration on (N + shift I).
    let shift: f64 = nm.iter().flatten().map(|x| x.abs()).sum::<f64>();
    let mut q = [1.0f64, 0.0, 0.0, 0.0];
    for _ in 0..200 {
        let mut v = [0.0f64; 4];
        for i in 0..4 { v[i] = shift * q[i] + (0..4).map(|j| nm[i][j] * q[j]).sum::<f64>(); }
        let l = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        q = [v[0] / l, v[1] / l, v[2] / l, v[3] / l];
    }
    let (w, x, y, z) = (q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32);
    let r: M3 = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ];
    Pose { r, t: sub(cd, mat_vec(&r, cs)) }
}

/// Point-to-point ICP: refine `init` so that init(src) lands on `tgt`.
/// Returns the pose, final RMS residual (m) and the inlier fraction.
fn icp(src: &[V3], tgt: &[V3], init: Pose, log: &mut dyn FnMut(String)) -> (Pose, f32, f32) {
    let src = voxel_downsample(src, 0.05);
    let tgt = voxel_downsample(tgt, 0.04);
    let mut pose = init;
    let mut rms = f32::NAN;
    let mut frac = 0.0;
    let schedule = [0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05];
    for (round, &max_d) in schedule.iter().enumerate() {
        // cell = search radius, so a query touches at most 27 cells
        let grid = Grid::new(tgt.clone(), max_d);
        for _ in 0..6 {
            let mut a = Vec::new();
            let mut b = Vec::new();
            let mut se = 0.0f32;
            for p in &src {
                let w = pose.apply(*p);
                if let Some((q, d)) = grid.nearest(w, max_d) {
                    a.push(*p);
                    b.push(q);
                    se += d * d;
                }
            }
            if a.len() < 50 { log(format!("ICP: too few matches at {max_d:.2} m ({})", a.len())); return (pose, rms, frac); }
            pose = kabsch(&a, &b);
            rms = (se / a.len() as f32).sqrt();
            frac = a.len() as f32 / src.len() as f32;
        }
        log(format!("ICP round {}: max {:.2} m, rms {:.3} m, {:.0}% matched", round + 1, max_d, rms, frac * 100.0));
    }
    (pose, rms, frac)
}

// ----------------------------------------------------------------------------- voxel map

struct Voxel {
    sum: [u32; 3],
    n: u32,
}

#[derive(Default)]
struct VoxelMap {
    cells: HashMap<(i32, i32, i32), Voxel>,
}

impl VoxelMap {
    fn insert(&mut self, p: V3, c: [u8; 3]) {
        let v = self.cells.entry(key(p, VOXEL)).or_insert(Voxel { sum: [0; 3], n: 0 });
        if v.n >= 64 { // keep colours responsive to change
            for i in 0..3 { v.sum[i] = v.sum[i] * 63 / 64; }
            v.n = 63;
        }
        for i in 0..3 { v.sum[i] += c[i] as u32; }
        v.n += 1;
    }
    fn points(&self) -> impl Iterator<Item = (V3, [u8; 3])> + '_ {
        self.cells.iter().map(|(k, v)| {
            let p = [(k.0 as f32 + 0.5) * VOXEL, (k.1 as f32 + 0.5) * VOXEL, (k.2 as f32 + 0.5) * VOXEL];
            let c = [(v.sum[0] / v.n) as u8, (v.sum[1] / v.n) as u8, (v.sum[2] / v.n) as u8];
            (p, c)
        })
    }
    fn save_ply(&self, path: &std::path::Path) -> std::io::Result<usize> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let n = self.cells.len();
        write!(f, "ply\nformat ascii 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")?;
        for (p, c) in self.points() {
            // y-up, z-back for viewers (Blender/MeshLab style): flip y and z.
            writeln!(f, "{:.3} {:.3} {:.3} {} {} {}", p[0], -p[1], -p[2], c[0], c[1], c[2])?;
        }
        Ok(n)
    }
}

// ----------------------------------------------------------------------------- live clouds

/// Kinect v1: registered depth + colour -> camera-frame points.
fn v1_cloud(rgb: &[u8], depth: &[u16], stride: usize) -> (Vec<V3>, Vec<[u8; 3]>) {
    let mut pts = Vec::with_capacity(W1 * H1 / (stride * stride));
    let mut col = Vec::with_capacity(pts.capacity());
    for v in (0..H1).step_by(stride) {
        for u in (0..W1).step_by(stride) {
            let d = depth[v * W1 + u];
            if d == 0 { continue; }
            let z = d as f32 * 0.001;
            if z > MAX_RANGE { continue; }
            pts.push(cam_point(u, v, z));
            let i = (v * W1 + u) * 3;
            col.push([rgb[i], rgb[i + 1], rgb[i + 2]]);
        }
    }
    (pts, col)
}

/// Kinect v2 cloud sub-sampled by pixel stride.
fn v2_cloud(c: &Cloud, stride: usize) -> (Vec<V3>, Vec<[u8; 3]>) {
    let w = kinect3d::kinect2::W as u32;
    let s = stride as u32;
    let mut pts = Vec::new();
    let mut col = Vec::new();
    for (i, &pix) in c.pix.iter().enumerate() {
        if pix % w % s == 0 && pix / w % s == 0 && c.pts[i][2] <= MAX_RANGE {
            pts.push(c.pts[i]);
            col.push(c.rgb[i]);
        }
    }
    (pts, col)
}

// ----------------------------------------------------------------------------- renderer

struct View {
    yaw: f32,
    pitch: f32,
    dist: f32,
    focus: V3,
    drag: Option<(f32, f32)>,
}

impl View {
    fn reset(&mut self) { *self = View::default(); }
    /// Camera basis for a y-down world: returns (eye, right, down, forward).
    fn basis(&self) -> (V3, V3, V3, V3) {
        let dir = [self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos()];
        let eye = sub(self.focus, scale(dir, self.dist));
        let f = normalize(dir);
        let r = normalize(cross([0.0, 1.0, 0.0], f));
        let d = cross(f, r);
        (eye, r, d, f)
    }
}

impl Default for View {
    fn default() -> Self { View { yaw: -0.6, pitch: 0.45, dist: 6.0, focus: [0.0, 0.3, 2.5], drag: None } }
}

struct Projector {
    eye: V3,
    r: V3,
    d: V3,
    f: V3,
    fl: f32,
    cx: f32,
    cy: f32,
}

impl Projector {
    fn new(view: &View, w: usize, h: usize) -> Projector {
        let (eye, r, d, f) = view.basis();
        Projector { eye, r, d, f, fl: h as f32 * 0.9, cx: w as f32 / 2.0, cy: h as f32 / 2.0 }
    }
    #[inline]
    fn project(&self, p: V3) -> Option<(f32, f32, f32)> {
        let q = sub(p, self.eye);
        let z = dot(self.f, q);
        if z < 0.05 { return None; }
        Some((self.cx + self.fl * dot(self.r, q) / z, self.cy + self.fl * dot(self.d, q) / z, z))
    }
}

struct Frame3D {
    c: Canvas,
    z: Vec<f32>,
    proj: Projector,
}

impl Frame3D {
    fn new(view: &View, w: usize, h: usize) -> Frame3D {
        let mut c = Canvas::new(w, h);
        c.fill([18, 18, 22]);
        Frame3D { c, z: vec![f32::INFINITY; w * h], proj: Projector::new(view, w, h) }
    }
    fn splat(&mut self, p: V3, col: Col, size: i32) {
        let Some((sx, sy, z)) = self.proj.project(p) else { return };
        let (w, h) = (self.c.w as i32, self.c.h as i32);
        let s = (size as f32 * (2.0 / z).clamp(0.5, 2.5)).round() as i32;
        let (x0, y0) = (sx as i32 - s / 2, sy as i32 - s / 2);
        for y in y0..y0 + s.max(1) {
            if y < 0 || y >= h { continue; }
            for x in x0..x0 + s.max(1) {
                if x < 0 || x >= w { continue; }
                let i = (y * w + x) as usize;
                if z < self.z[i] {
                    self.z[i] = z;
                    self.c.buf[i] = kinect3d::gfx::pack(col);
                }
            }
        }
    }
    fn line3(&mut self, a: V3, b: V3, col: Col) {
        if let (Some(pa), Some(pb)) = (self.proj.project(a), self.proj.project(b)) {
            self.c.line_blend(pa.0 as i32, pa.1 as i32, pb.0 as i32, pb.1 as i32, col, 0.9);
        }
    }
    /// Camera frustum (0.4 m deep) at `pose` with the sensor's field of view.
    fn frustum(&mut self, pose: &Pose, hfov: f32, vfov: f32, col: Col) {
        let d = 0.4;
        let x = d * (hfov / 2.0).tan();
        let y = d * (vfov / 2.0).tan();
        let o = pose.apply([0.0, 0.0, 0.0]);
        let corners = [[-x, -y, d], [x, -y, d], [x, y, d], [-x, y, d]].map(|c| pose.apply(c));
        for i in 0..4 {
            self.line3(o, corners[i], col);
            self.line3(corners[i], corners[(i + 1) % 4], col);
        }
        // up tick
        self.line3(pose.apply([0.0, -y, d]), pose.apply([0.0, -y * 1.4, d]), col);
    }
    fn floor_grid(&mut self, y: f32, col: Col) {
        for i in -6..=6 {
            let k = i as f32;
            self.line3([k, y, -2.0], [k, y, 10.0], col);
            self.line3([-6.0, y, k + 4.0], [6.0, y, k + 4.0], col);
        }
    }
}

// ----------------------------------------------------------------------------- app

struct App {
    v1: Kinect,
    v2: Arc<Kinect2>,
    v2_pose: Pose,
    map: VoxelMap,
    view: View,
    show_v1: bool,
    show_v2: bool,
    show_map: bool,
    show_live: bool,
    mirror: bool,
    point_size: i32,
    help: bool,
    log: Vec<(Instant, String)>,
    v1_seq: u64,
    v2_seq: u64,
    v1_last: Option<(Vec<V3>, Vec<[u8; 3]>)>,
    v2_last: Option<(Vec<V3>, Vec<[u8; 3]>)>,
    icp_pending: bool,
    /// Run ICP once automatically when both sensors are live (no saved pose).
    auto_icp: Option<Instant>,
}

impl App {
    fn say(&mut self, s: String) {
        eprintln!("{s}");
        self.log.push((Instant::now(), s));
        if self.log.len() > 6 { self.log.remove(0); }
    }

    fn load_pose(&mut self) {
        if let Ok(txt) = std::fs::read_to_string(POSE_FILE) {
            if let Some(p) = serde_json::from_str::<serde_json::Value>(&txt).ok().and_then(|v| Pose::from_json(&v)) {
                self.v2_pose = p;
                self.auto_icp = None;
                self.say(format!("loaded v2 pose from {POSE_FILE}"));
            }
        }
    }
    fn save_pose(&mut self) {
        match std::fs::write(POSE_FILE, serde_json::to_string_pretty(&self.v2_pose.to_json()).unwrap()) {
            Ok(_) => self.say(format!("saved v2 pose to {POSE_FILE}")),
            Err(e) => self.say(format!("save failed: {e}")),
        }
    }

    /// Pull new frames from both sensors and merge them into the map.
    fn ingest(&mut self) {
        if self.v1.seq() != self.v1_seq {
            if let Some((rgb, depth)) = self.v1.frames() {
                self.v1_seq = self.v1.seq();
                let (pts, col) = v1_cloud(&rgb, &depth, 2);
                if self.show_v1 { for (p, c) in pts.iter().zip(&col) { self.map.insert(*p, *c); } }
                self.v1_last = Some((pts, col));
            }
        }
        if self.v2.seq.load(std::sync::atomic::Ordering::SeqCst) != self.v2_seq {
            if let Some(cloud) = self.v2.cloud() {
                self.v2_seq = self.v2.seq.load(std::sync::atomic::Ordering::SeqCst);
                let (pts, col) = v2_cloud(&cloud, 2);
                if self.show_v2 { for (p, c) in pts.iter().zip(&col) { self.map.insert(self.v2_pose.apply(*p), *c); } }
                self.v2_last = Some((pts, col));
            }
        }
    }

    fn run_icp(&mut self) {
        let (Some(v1), Some(v2)) = (&self.v1_last, &self.v2_last) else { self.say("ICP: need frames from both sensors".into()); return };
        let (src, tgt) = (v2.0.clone(), v1.0.clone());
        let init = self.v2_pose;
        let mut lines = Vec::new();
        let t0 = Instant::now();
        let (pose, rms, frac) = icp(&src, &tgt, init, &mut |s| lines.push(s));
        for l in lines { self.say(l); }
        if rms.is_finite() && rms < 0.12 && frac > 0.15 {
            self.v2_pose = pose;
            self.map.cells.clear();
            self.say(format!("ICP done in {:.1} s: rms {:.3} m, map cleared", t0.elapsed().as_secs_f32(), rms));
            self.save_pose();
        } else {
            self.say("ICP rejected (too little overlap); adjust the pose by hand and retry".into());
        }
    }

    fn handle_keys(&mut self, win: &Window) {
        let fine = win.is_key_down(Key::LeftShift) || win.is_key_down(Key::RightShift);
        let step = if fine { 0.01 } else { 0.05 };
        let ang = if fine { 0.5f32.to_radians() } else { 2.0f32.to_radians() };
        let (r, d, f) = { let (_, r, d, f) = self.view.basis(); (r, d, f) };
        for k in win.get_keys_pressed(KeyRepeat::Yes) {
            match k {
                // view
                Key::W => self.view.focus = add(self.view.focus, scale(f, 0.2)),
                Key::S => self.view.focus = sub(self.view.focus, scale(f, 0.2)),
                Key::A => self.view.focus = sub(self.view.focus, scale(r, 0.2)),
                Key::D => self.view.focus = add(self.view.focus, scale(r, 0.2)),
                Key::R => self.view.focus = sub(self.view.focus, scale(d, 0.2)),
                Key::F => self.view.focus = add(self.view.focus, scale(d, 0.2)),
                Key::Home => self.view.reset(),
                // v2 pose: translate in world axes
                Key::Left => self.v2_pose.t[0] -= step,
                Key::Right => self.v2_pose.t[0] += step,
                Key::Up => self.v2_pose.t[2] += step,
                Key::Down => self.v2_pose.t[2] -= step,
                Key::PageUp => self.v2_pose.t[1] -= step,
                Key::PageDown => self.v2_pose.t[1] += step,
                Key::Z => self.v2_pose.rotate_local(0.0, -ang, 0.0),
                Key::C => self.v2_pose.rotate_local(0.0, ang, 0.0),
                Key::T => self.v2_pose.rotate_local(-ang, 0.0, 0.0),
                Key::G => self.v2_pose.rotate_local(ang, 0.0, 0.0),
                Key::V => self.v2_pose.rotate_local(0.0, 0.0, -ang),
                Key::B => self.v2_pose.rotate_local(0.0, 0.0, ang),
                _ => {}
            }
        }
        for k in win.get_keys_pressed(KeyRepeat::No) {
            match k {
                Key::Key1 => { self.show_v1 = !self.show_v1; }
                Key::Key2 => { self.show_v2 = !self.show_v2; }
                Key::M => self.show_map = !self.show_map,
                Key::L => self.show_live = !self.show_live,
                Key::X => self.mirror = !self.mirror,
                Key::P => self.point_size = if self.point_size >= 4 { 1 } else { self.point_size + 1 },
                Key::H => self.help = !self.help,
                Key::I => self.icp_pending = true,
                Key::O => { self.v2_pose = default_v2_pose(); self.map.cells.clear(); self.say("v2 pose reset".into()); }
                Key::K => self.save_pose(),
                Key::N => { self.map.cells.clear(); self.say("map cleared".into()); }
                Key::E => self.export(),
                _ => {}
            }
        }
    }

    fn export(&mut self) {
        let dir = std::path::Path::new("../pointclouds");
        let _ = std::fs::create_dir_all(dir);
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let path = dir.join(format!("roommap-{ts}.ply"));
        match self.map.save_ply(&path) {
            Ok(n) => self.say(format!("wrote {n} points to {}", path.display())),
            Err(e) => self.say(format!("export failed: {e}")),
        }
    }

    fn draw(&self, w: usize, h: usize) -> Canvas {
        let mut fr = Frame3D::new(&self.view, w, h);
        let mx = |p: V3| if self.mirror { [-p[0], p[1], p[2]] } else { p };
        if self.show_map {
            for (p, c) in self.map.points() { fr.splat(mx(p), c, self.point_size + 1); }
        }
        if self.show_live {
            if self.show_v1 { if let Some((pts, col)) = &self.v1_last { for (p, c) in pts.iter().zip(col) { fr.splat(mx(*p), *c, self.point_size); } } }
            if self.show_v2 { if let Some((pts, col)) = &self.v2_last { for (p, c) in pts.iter().zip(col) { fr.splat(mx(self.v2_pose.apply(*p)), *c, self.point_size); } } }
        }
        // floor guess: 97th percentile of map y (y is down)
        let mut ys: Vec<f32> = self.map.points().map(|(p, _)| p[1]).collect();
        if ys.len() > 100 {
            let fy = percentile(&mut ys, 97.0);
            fr.floor_grid(fy, [45, 45, 55]);
        }
        let v1_pose = if self.mirror { Pose { r: [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], t: [0.0; 3] } } else { Pose::IDENT };
        let v2_pose = if self.mirror { v1_pose.compose(&self.v2_pose) } else { self.v2_pose };
        fr.frustum(&v1_pose, 57f32.to_radians(), 43f32.to_radians(), [80, 200, 255]);
        fr.frustum(&v2_pose, 70f32.to_radians(), 60f32.to_radians(), [255, 170, 60]);

        let mut c = fr.c;
        self.hud(&mut c);
        c
    }

    fn hud(&self, c: &mut Canvas) {
        let sc = 0.45;
        let v1_ok = self.v1.frames().is_some();
        let st = format!(
            "v1 {}  |  {}  |  map {} voxels ({:.0} MB)  |  v2 pose t=({:.2},{:.2},{:.2})  |  {}{}{}",
            if v1_ok { "live" } else { "no frames" },
            self.v2.status(),
            self.map.cells.len(),
            self.map.cells.len() as f32 * 48.0 / 1e6,
            self.v2_pose.t[0], self.v2_pose.t[1], self.v2_pose.t[2],
            if self.show_v1 { "[1]v1 " } else { "[1]-- " },
            if self.show_v2 { "[2]v2 " } else { "[2]-- " },
            if self.show_map { "[M]map " } else { "[M]live-only " },
        );
        c.panel(0, 0, c.w as i32, 26, 0.6);
        c.text(&st, 8, 5, sc, [230, 230, 230], false);
        c.text("H help", c.w as i32 - 60, 5, sc, [160, 160, 160], false);
        let mut y = c.h as i32 - 8 - 18 * self.log.len() as i32;
        for (t, s) in &self.log {
            let age = t.elapsed().as_secs_f32();
            if age < 12.0 { c.text(s, 8, y, sc, [220, 200, 120], true); }
            y += 18;
        }
        if self.help {
            let lines = [
                "Mouse drag: orbit   wheel: zoom   W/S/A/D/R/F: move view   Home: reset view",
                "1 / 2: include v1 / v2      M: show map      L: show live clouds      X: mirror      P: point size",
                "v2 pose:  arrows = move x/z   PgUp/PgDn = move y   Z/C yaw   T/G pitch   V/B roll   (Shift = fine)",
                "I: ICP refine v2 pose against v1 (needs overlap)    O: reset pose    K: save pose",
                "N: clear map      E: export PLY to ../pointclouds/      Esc/Q: quit",
            ];
            let hh = 20 * lines.len() as i32 + 20;
            c.panel(20, 40, c.w as i32 - 20, 40 + hh, 0.75);
            for (i, l) in lines.iter().enumerate() { c.text(l, 32, 50 + 20 * i as i32, sc, [235, 235, 235], false); }
        }
    }
}

fn main() {
    let v1 = Kinect::new();
    v1.start();
    let v2 = Kinect2::start();
    let mut app = App {
        v1,
        v2,
        v2_pose: default_v2_pose(),
        map: VoxelMap::default(),
        view: View::default(),
        show_v1: true,
        show_v2: true,
        show_map: true,
        show_live: true,
        mirror: false,
        point_size: 2,
        help: true,
        log: Vec::new(),
        v1_seq: 0,
        v2_seq: 0,
        v1_last: None,
        v2_last: None,
        icp_pending: false,
        auto_icp: Some(Instant::now()),
    };
    app.load_pose();
    if app.auto_icp.is_some() { app.say("no saved pose: ICP will align the Kinect v2 automatically once both sensors are live".into()); }

    let mut win = Window::new("Room Map", CW, CH, WindowOptions { resize: true, scale_mode: minifb::ScaleMode::Stretch, ..Default::default() }).expect("window");
    win.set_target_fps(30);
    let (mut w, mut h) = (CW, CH);

    while win.is_open() && !win.is_key_down(Key::Escape) && !win.is_key_down(Key::Q) {
        let (ww, wh) = win.get_size();
        if ww >= 64 && wh >= 64 { w = ww; h = wh; }

        let pos = win.get_mouse_pos(MouseMode::Clamp);
        let down = win.get_mouse_down(MouseButton::Left);
        match (pos, down) {
            (Some(p), true) => {
                if let Some(d) = app.view.drag {
                    app.view.yaw -= (p.0 - d.0) * 0.006;
                    app.view.pitch = (app.view.pitch + (p.1 - d.1) * 0.006).clamp(-1.4, 1.4);
                }
                app.view.drag = Some(p);
            }
            _ => app.view.drag = None,
        }
        if let Some((_, wheel)) = win.get_scroll_wheel() {
            if wheel != 0.0 { app.view.dist = (app.view.dist * if wheel > 0.0 { 0.9 } else { 1.1 }).clamp(0.5, 30.0); }
        }
        app.handle_keys(&win);
        app.ingest();
        if let Some(t0) = app.auto_icp {
            // wait until both sensors have delivered and the v1 has settled for a moment
            if app.v1_last.is_some() && app.v2_last.is_some() && t0.elapsed().as_secs_f32() > 3.0 {
                app.auto_icp = None;
                app.icp_pending = true;
            }
        }
        if app.icp_pending { app.icp_pending = false; app.run_icp(); }

        let c = app.draw(w, h);
        win.update_with_buffer(&c.buf, w, h).expect("update");
    }
    app.v1.stop();
    app.v2.stop();
}
