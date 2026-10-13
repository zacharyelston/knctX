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
//!        (hold Shift for fine steps), I = align on room planes, U = ICP refine, Y = flip v2 mirror, J = tint clouds
//!        per sensor to judge alignment (or start with --tint), O = reset pose, K = save pose
//!        3 = let the v2 win where both sensors see (default; --both to start off)
//! Map:   N clear map, E export PLY to ../pointclouds/
//! Args:  --tint, --both, --no-polish (skip the ICP after the plane alignment),
//!        --baseline 27in (fix the camera separation), --dump DIR (write both clouds and exit)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use kinect3d::gfx::{Canvas, Col};
use kinect3d::kinect2::{Cloud, Kinect2};
use kinect3d::math::*;
use kinect3d::scan::world::{CX as CX1, CY as CY1, FX as FX1_DEFAULT};
use kinect3d::source::{Depth, Kinect, Rgb, Source, H as H1, W as W1};
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
    lum: Vec<f32>,
}

impl Grid {
    fn new(pts: Vec<V3>, lum: Vec<f32>, cell: f32) -> Grid {
        let mut map: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, p) in pts.iter().enumerate() {
            map.entry(key(*p, cell)).or_default().push(i as u32);
        }
        Grid { cell, map, pts, lum }
    }
    /// Nearest point within `max_d`; with `lum = Some((l, tol))` only points whose
    /// (normalised) luminance is within `tol` of `l` qualify, so a dark picture frame
    /// cannot be matched to the bright wall around it.
    fn nearest(&self, q: V3, max_d: f32, lum: Option<(f32, f32)>) -> Option<(V3, f32)> {
        let k = key(q, self.cell);
        let reach = (max_d / self.cell).ceil() as i32;
        let mut best: Option<(V3, f32)> = None;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    if let Some(ids) = self.map.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                        for &i in ids {
                            if let Some((l, tol)) = lum { if (self.lum[i as usize] - l).abs() > tol { continue; } }
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

/// Keep one point (and its luminance) per `cell`-sized voxel.
fn voxel_downsample(pts: &[V3], lum: &[f32], cell: f32) -> (Vec<V3>, Vec<f32>) {
    let mut seen = std::collections::HashSet::new();
    let mut p_out = Vec::new();
    let mut l_out = Vec::new();
    for (p, l) in pts.iter().zip(lum) {
        if seen.insert(key(*p, cell)) { p_out.push(*p); l_out.push(*l); }
    }
    (p_out, l_out)
}

/// Luminance (0..1) of each point, shifted so the cloud's mean is 0.5: the two
/// sensors have different exposure and white balance, only contrast is comparable.
fn luminance(col: &[[u8; 3]]) -> Vec<f32> {
    let l: Vec<f32> = col.iter().map(|c| (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) / 255.0).collect();
    let mean = l.iter().sum::<f32>() / l.len().max(1) as f32;
    l.iter().map(|v| (v - mean + 0.5).clamp(0.0, 1.0)).collect()
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
fn icp(src: &[V3], src_col: &[[u8; 3]], tgt: &[V3], tgt_col: &[[u8; 3]], init: Pose, fine_only: bool, log: &mut dyn FnMut(String)) -> (Pose, f32, f32) {
    let (src, src_l) = voxel_downsample(src, &luminance(src_col), 0.05);
    let (tgt, tgt_l) = voxel_downsample(tgt, &luminance(tgt_col), 0.04);
    let mut pose = init;
    let mut rms = f32::NAN;
    let mut frac = 0.0;
    let schedule: &[f32] = if fine_only { &[0.2, 0.15, 0.1, 0.08, 0.06, 0.05] } else { &[0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05] };
    for (round, &max_d) in schedule.iter().enumerate() {
        // cell = search radius, so a query touches at most 27 cells
        let grid = Grid::new(tgt.clone(), tgt_l.clone(), max_d);
        // geometry first; once roughly aligned, also require similar brightness
        let tol = if max_d <= 0.2 { Some(0.18) } else { None };
        for _ in 0..6 {
            let mut a = Vec::new();
            let mut b = Vec::new();
            let mut se = 0.0f32;
            for (p, l) in src.iter().zip(&src_l) {
                let w = pose.apply(*p);
                if let Some((q, d)) = grid.nearest(w, max_d, tol.map(|t| (*l, t))) {
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
        if round == 0 || round == schedule.len() - 1 || round % 3 == 2 {
            log(format!("ICP round {}: max {:.2} m, rms {:.3} m, {:.0}% matched{}", round + 1, max_d, rms, frac * 100.0, if tol.is_some() { ", colour-gated" } else { "" }));
        }
    }
    (pose, rms, frac)
}

// ----------------------------------------------------------------------------- room-plane alignment

/// Tiny deterministic xorshift RNG (no rand dependency needed).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

/// Rotation by `a` radians about the unit axis `u` (Rodrigues).
fn axis_angle(u: V3, a: f32) -> M3 {
    let (c, s) = (a.cos(), a.sin());
    let t = 1.0 - c;
    [
        [c + u[0] * u[0] * t, u[0] * u[1] * t - u[2] * s, u[0] * u[2] * t + u[1] * s],
        [u[1] * u[0] * t + u[2] * s, c + u[1] * u[1] * t, u[1] * u[2] * t - u[0] * s],
        [u[2] * u[0] * t - u[1] * s, u[2] * u[1] * t + u[0] * s, c + u[2] * u[2] * t],
    ]
}

fn transpose(m: &M3) -> M3 { [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]] }

/// Plane n·p + d = 0 with |n| = 1.
#[derive(Clone, Copy)]
struct Plane { n: V3, d: f32, inliers: usize }

/// Least-squares plane through `pts` (normal = smallest-variance direction, by power iteration on the inverse).
fn fit_plane(pts: &[V3]) -> Option<Plane> {
    if pts.len() < 3 { return None; }
    let n = pts.len() as f32;
    let c = scale(pts.iter().fold([0.0; 3], |a, p| add(a, *p)), 1.0 / n);
    let mut cov = [[0.0f32; 3]; 3];
    for p in pts { let q = sub(*p, c); for i in 0..3 { for j in 0..3 { cov[i][j] += q[i] * q[j]; } } }
    // smallest eigenvector: power iteration on (tr(cov) I - cov)
    let tr = cov[0][0] + cov[1][1] + cov[2][2];
    let m: M3 = [[tr - cov[0][0], -cov[0][1], -cov[0][2]], [-cov[1][0], tr - cov[1][1], -cov[1][2]], [-cov[2][0], -cov[2][1], tr - cov[2][2]]];
    let mut v = [0.3f32, 0.5, 0.8];
    for _ in 0..100 { v = normalize(mat_vec(&m, v)); }
    Some(Plane { n: v, d: -dot(v, c), inliers: pts.len() })
}

/// RANSAC plane whose normal is within `max_ang` (radians) of `axis` (either sign), refined by least squares.
fn ransac_plane(pts: &[V3], axis: V3, max_ang: f32, tol: f32, iters: usize, rng: &mut Rng) -> Option<Plane> {
    if pts.len() < 50 { return None; }
    let min_cos = max_ang.cos();
    // score candidates on a subsample so that many iterations stay cheap
    let probe: Vec<V3> = if pts.len() > 2500 { (0..2500).map(|_| pts[rng.below(pts.len())]).collect() } else { pts.to_vec() };
    let mut best: Option<(usize, V3, f32)> = None;
    for _ in 0..iters {
        let a = pts[rng.below(pts.len())];
        // pick the other two near the first so that small planes get sampled too
        let (mut b, mut c) = (a, a);
        for _ in 0..40 { let q = pts[rng.below(pts.len())]; if norm(sub(q, a)) < 1.5 { if b == a { b = q; } else { c = q; break; } } }
        let n = cross(sub(b, a), sub(c, a));
        if norm(n) < 1e-6 { continue; }
        let n = normalize(n);
        if dot(n, axis).abs() < min_cos { continue; }
        let d = -dot(n, a);
        let k = probe.iter().filter(|p| (dot(n, **p) + d).abs() < tol).count();
        if best.map_or(true, |b| k > b.0) { best = Some((k, n, d)); }
    }
    let (k, n, d) = best?;
    if k * pts.len() / probe.len() < 50 { return None; }
    // refit on the inliers until the inlier set stops growing
    let mut pl = Plane { n, d, inliers: 0 };
    for _ in 0..10 {
        let inl: Vec<V3> = pts.iter().copied().filter(|p| (dot(pl.n, *p) + pl.d).abs() < tol).collect();
        let mut nxt = fit_plane(&inl)?;
        if dot(nxt.n, pl.n) < 0.0 { nxt.n = scale(nxt.n, -1.0); nxt.d = -nxt.d; }
        nxt.inliers = pts.iter().filter(|p| (dot(nxt.n, **p) + nxt.d).abs() < tol).count();
        if nxt.inliers <= pl.inliers { break; }
        pl = nxt;
    }
    Some(pl)
}

/// A sensor's position/orientation relative to the room's horizontal plane (ceiling or floor)
/// and its dominant vertical wall: `r` maps camera -> room axes (right, up, forward=toward wall),
/// `c` is the camera position in that frame with lateral coordinate 0 (unknown).
struct RoomFrame { r: M3, c: V3, ceiling: bool, info: String, up: V3, d_h: f32 }

/// `tol` is the plane inlier tolerance: a few cm for the Kinect v2, more for the Kinect v1
/// whose depth is quantised in 10-20 cm steps beyond 4 m.
fn room_frame(pts: &[V3], tol: f32, rng: &mut Rng) -> Result<RoomFrame, String> {
    // horizontal plane: normal within 25 deg of the camera's y axis
    let h = ransac_plane(pts, [0.0, 1.0, 0.0], 25f32.to_radians(), tol * 0.6, 1500, rng).ok_or("no ceiling/floor plane")?;
    // orient the normal upward (-y in this y-down frame); then d < 0 means the plane is above (ceiling)
    let (mut n, mut d) = (h.n, h.d);
    if n[1] > 0.0 { n = scale(n, -1.0); d = -d; }
    let up = n;
    let ceiling = d < 0.0;
    let rest: Vec<V3> = pts.iter().copied().filter(|p| (dot(up, *p) + d).abs() > tol * 1.5).collect();
    // vertical wall: normal perpendicular to `up` (within 8 deg), in front of the camera
    let mut w = ransac_plane(&rest, up, (90f32 - 8.0).to_radians(), tol, 8000, rng).ok_or("no wall plane")?;
    if w.d > 0.0 { w.n = scale(w.n, -1.0); w.d = -w.d; } // normal points away from the camera, toward the wall
    let fwd = normalize(sub(w.n, scale(up, dot(w.n, up))));
    let right = normalize(cross(fwd, up));
    let r: M3 = [right, up, fwd];
    let c = [0.0, d, w.d];
    let info = format!(
        "{} {:.2} m {}, wall {:.2} m ahead, yaw {:+.1} deg, pitch {:+.1} deg ({} / {} pts)",
        if ceiling { "ceiling" } else { "floor" }, d.abs(), if ceiling { "above" } else { "below" },
        -w.d, fwd[0].atan2(fwd[2]).to_degrees(), (-up[2]).asin().to_degrees(), h.inliers, w.inliers
    );
    Ok(RoomFrame { r, c, ceiling, info, up, d_h: d })
}

/// Align the v2 cloud to the v1 cloud through the room's planes: the ceiling (or floor) fixes
/// pitch, roll and height, the dominant wall fixes yaw and distance, and the remaining lateral
/// offset along the wall is found by scanning for the best colour-gated point overlap.
/// Returns the pose and the implied distance between the two cameras.
fn room_align(src: &[V3], src_col: &[[u8; 3]], tgt: &[V3], tgt_col: &[[u8; 3]], baseline: Option<f32>, log: &mut dyn FnMut(String)) -> Result<(Pose, f32), String> {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (src, src_l) = voxel_downsample(src, &luminance(src_col), 0.04);
    let (tgt, tgt_l) = voxel_downsample(tgt, &luminance(tgt_col), 0.04);
    let f2 = room_frame(&src, 0.05, &mut rng).map_err(|e| format!("v2: {e}"))?;
    let f1 = room_frame(&tgt, 0.12, &mut rng).map_err(|e| format!("v1: {e}"))?;
    log(format!("v2: {}", f2.info));
    log(format!("v1: {}", f1.info));
    if f1.ceiling != f2.ceiling { return Err("sensors locked onto different horizontal planes (ceiling vs floor)".into()); }
    // p1 = R1^T (R2 p2 + c2 - c1 + lateral)
    let frame = |f1: &RoomFrame| {
        let r1t = transpose(&f1.r);
        (mat_mul(&r1t, &f2.r), mat_vec(&r1t, sub(f2.c, f1.c)), mat_vec(&r1t, [1.0, 0.0, 0.0]))
    };
    let (r, base, lateral) = frame(&f1);
    // Scan over every v2 point except the ceiling/floor, which slides freely and would only add
    // noise: the wall segment pins the heading, the picture and the furniture pin the lateral offset.
    let on_h = |p: &V3| (dot(f2.up, *p) + f2.d_h).abs() < 0.08;
    let feats: Vec<(V3, f32)> = src.iter().zip(&src_l).filter(|(p, _)| !on_h(p)).map(|(p, l)| (*p, *l)).collect();
    if feats.len() < 50 { return Err(format!("too few distinctive v2 points for the lateral scan ({})", feats.len())); }
    log(format!("heading/lateral scan over {} v2 points", feats.len()));
    let grid = Grid::new(tgt.clone(), tgt_l.clone(), 0.06);
    // The v1's wall normal is not trustworthy either (its depth bows), so the v2's heading is
    // searched too: rotate the v2 about the vertical axis through its own position.
    let up1 = f1.up;
    let with_yaw = |dyaw: f32| mat_mul(&axis_angle(up1, dyaw), &r);
    let score = |ry: &M3, dx: f32| -> usize {
        let pose = Pose { r: *ry, t: add(base, scale(lateral, dx)) };
        feats.iter().filter(|(p, l)| grid.nearest(pose.apply(*p), 0.06, Some((*l, 0.15))).is_some()).count()
    };
    let mut best = (0.0f32, f32::NAN, 0usize); // (yaw, lateral, matches)
    let mut yaw = -16f32.to_radians();
    while yaw <= 16.01f32.to_radians() {
        let ry = with_yaw(yaw);
        let mut dx = -1.5f32;
        while dx <= 1.5 {
            let k = score(&ry, dx);
            if k > best.2 { best = (yaw, dx, k); }
            dx += 0.04;
        }
        yaw += 2f32.to_radians();
    }
    if best.2 < 50 { return Err("no lateral overlap found".into()); }
    // refine: 0.5 deg and 1 cm around the best
    let coarse = best;
    let mut yaw = coarse.0 - 2f32.to_radians();
    while yaw <= coarse.0 + 2.01f32.to_radians() {
        let ry = with_yaw(yaw);
        let mut dx = coarse.1 - 0.04;
        while dx <= coarse.1 + 0.04 {
            let k = score(&ry, dx);
            if k > best.2 { best = (yaw, dx, k); }
            dx += 0.01;
        }
        yaw += 0.5f32.to_radians();
    }
    let pct = |k: usize| k as f32 * 100.0 / feats.len() as f32;
    log(format!("heading {:+.1} deg off the wall-plane estimate, lateral {:+.2} m along the wall ({:.0}% of v2 points matched within 6 cm)", best.0.to_degrees(), best.1, pct(best.2)));
    let r = with_yaw(best.0);
    let score = |dx: f32| score(&r, dx);
    let mut dx = best.1;
    if let Some(b) = baseline {
        // keep the planes, but pick the lateral offset that puts the cameras `b` apart
        let bb = dot(base, lateral);
        let disc = bb * bb - (dot(base, base) - b * b);
        if disc < 0.0 {
            log(format!("cannot place the cameras {b:.2} m apart with these planes (closest {:.2} m); using the colour scan", (dot(base, base) - bb * bb).max(0.0).sqrt()));
        } else {
            let (d1, d2) = (-bb - disc.sqrt(), -bb + disc.sqrt());
            let (k1, k2) = (score(d1), score(d2));
            dx = if k1 >= k2 { d1 } else { d2 };
            log(format!("baseline {b:.2} m: lateral {:+.2} m ({:.0}% matched) chosen over {:+.2} m ({:.0}% matched)", dx, pct(k1.max(k2)), if k1 >= k2 { d2 } else { d1 }, pct(k1.min(k2))));
        }
    }
    let pose = Pose { r, t: add(base, scale(lateral, dx)) };
    Ok((pose, norm(pose.t)))
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

/// Kinect v1: registered depth + colour -> camera-frame points, with focal length `fx`.
fn v1_cloud(rgb: &[u8], depth: &[u16], stride: usize, fx: f32) -> (Vec<V3>, Vec<[u8; 3]>) {
    let mut pts = Vec::with_capacity(W1 * H1 / (stride * stride));
    let mut col = Vec::with_capacity(pts.capacity());
    for v in (0..H1).step_by(stride) {
        for u in (0..W1).step_by(stride) {
            let d = depth[v * W1 + u];
            if d == 0 { continue; }
            let z = d as f32 * 0.001;
            if z > MAX_RANGE { continue; }
            pts.push([(u as f32 - CX1) / fx * z, (v as f32 - CY1) / fx * z, z]);
            let i = (v * W1 + u) * 3;
            col.push([rgb[i], rgb[i + 1], rgb[i + 2]]);
        }
    }
    (pts, col)
}

/// Kinect v2 cloud sub-sampled by pixel stride. libfreenect2 delivers mirrored
/// images, so by default (`mirror` = true) x is negated to match the Kinect v1.
fn v2_cloud(c: &Cloud, stride: usize, mirror: bool) -> (Vec<V3>, Vec<[u8; 3]>) {
    let w = kinect3d::kinect2::W as u32;
    let s = stride as u32;
    let sx = if mirror { -1.0 } else { 1.0 };
    let mut pts = Vec::new();
    let mut col = Vec::new();
    for (i, &pix) in c.pix.iter().enumerate() {
        if pix % w % s == 0 && pix / w % s == 0 && c.pts[i][2] <= MAX_RANGE {
            let p = c.pts[i];
            pts.push([sx * p[0], p[1], p[2]]);
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
    /// Kinect v1 focal length (px) for the registered depth; calibrated against the v2.
    v1_fx: f32,
    v1_raw: Option<(Rgb, Depth)>,
    /// Flip the v2 cloud's x axis (libfreenect2 images are mirror images).
    v2_mirror: bool,
    v2_raw: Option<Arc<Cloud>>,
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
    align_pending: bool,
    /// Run the room alignment once automatically when both sensors are live (no saved pose).
    auto_icp: Option<Instant>,
    /// Draw v1 points bluish and v2 points orange instead of their colours.
    tint: bool,
    /// Known distance between the two cameras (m), from --baseline.
    baseline: Option<f32>,
    /// Follow the plane alignment with a short ICP (off with --no-polish).
    polish: bool,
    /// Where the Kinect v2 has a valid depth, drop the Kinect v1's points (the v1 is coarse
    /// and bowed beyond ~4 m, so letting both in doubles every surface in the overlap).
    v2_first: bool,
    /// Kinect v2 depth image (512x424, metres, 0 = none) of the latest frame, for the overlap test.
    v2_depth: Vec<f32>,
    /// Which v1 points survive the overlap test (parallel to v1_last).
    v1_keep: Vec<bool>,
}

impl App {
    fn say(&mut self, s: String) {
        eprintln!("{s}");
        self.log.push((Instant::now(), s));
        if self.log.len() > 6 { self.log.remove(0); }
    }

    fn load_pose(&mut self) {
        if let Ok(txt) = std::fs::read_to_string(POSE_FILE) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                if let Some(p) = Pose::from_json(&v) {
                    self.v2_pose = p;
                    self.auto_icp = None;
                    if let Some(fx) = v.get("v1_fx").and_then(|f| f.as_f64()) { self.v1_fx = fx as f32; }
                    if let Some(m) = v.get("v2_mirror").and_then(|f| f.as_bool()) { self.v2_mirror = m; }
                    self.say(format!("loaded v2 pose from {POSE_FILE} (v1 fx {:.0} px, v2 mirror {})", self.v1_fx, self.v2_mirror));
                }
            }
        }
    }
    fn save_pose(&mut self) {
        let mut j = self.v2_pose.to_json();
        j["v1_fx"] = serde_json::json!(self.v1_fx);
        j["v2_mirror"] = serde_json::json!(self.v2_mirror);
        match std::fs::write(POSE_FILE, serde_json::to_string_pretty(&j).unwrap()) {
            Ok(_) => self.say(format!("saved v2 pose to {POSE_FILE}")),
            Err(e) => self.say(format!("save failed: {e}")),
        }
    }

    /// Pull new frames from both sensors and merge them into the map.
    fn ingest(&mut self) {
        if self.v2.seq.load(std::sync::atomic::Ordering::SeqCst) != self.v2_seq {
            if let Some(cloud) = self.v2.cloud() {
                self.v2_seq = self.v2.seq.load(std::sync::atomic::Ordering::SeqCst);
                let (pts, col) = v2_cloud(&cloud, 2, self.v2_mirror);
                if self.show_v2 { for (p, c) in pts.iter().zip(&col) { self.map.insert(self.v2_pose.apply(*p), *c); } }
                self.v2_last = Some((pts, col));
                self.v2_depth = vec![0.0; kinect3d::kinect2::W * kinect3d::kinect2::H];
                for (i, &pix) in cloud.pix.iter().enumerate() { self.v2_depth[pix as usize] = cloud.pts[i][2]; }
                self.v2_raw = Some(cloud);
            }
        }
        if self.v1.seq() != self.v1_seq {
            if let Some((rgb, depth)) = self.v1.frames() {
                self.v1_seq = self.v1.seq();
                let (pts, col) = v1_cloud(&rgb, &depth, 2, self.v1_fx);
                self.v1_keep = self.overlap_mask(&pts);
                if self.show_v1 {
                    for ((p, c), keep) in pts.iter().zip(&col).zip(&self.v1_keep) { if *keep { self.map.insert(*p, *c); } }
                }
                self.v1_last = Some((pts, col));
                self.v1_raw = Some((rgb, depth));
            }
        }
    }

    /// For each v1 point: true if it should be kept, false if it falls on a pixel where the
    /// Kinect v2 has its own (better) depth measurement.
    fn overlap_mask(&self, pts: &[V3]) -> Vec<bool> {
        if !self.v2_first || self.v2_depth.is_empty() { return vec![true; pts.len()]; }
        let [fx, fy, cx, cy] = *self.v2.intrinsics.lock().unwrap();
        let (w, h) = (kinect3d::kinect2::W as i32, kinect3d::kinect2::H as i32);
        let rt = transpose(&self.v2_pose.r);
        let sx = if self.v2_mirror { -1.0 } else { 1.0 };
        pts.iter().map(|p| {
            let q = mat_vec(&rt, sub(*p, self.v2_pose.t)); // into the v2 camera frame
            if q[2] < 0.3 { return true; }
            let u = (sx * q[0] / q[2] * fx + cx - 0.5).round() as i32;
            let v = (q[1] / q[2] * fy + cy - 0.5).round() as i32;
            if u < 0 || v < 0 || u >= w || v >= h { return true; }
            self.v2_depth[(v * w + u) as usize] <= 0.0
        }).collect()
    }

    /// Plane-based alignment (ceiling/floor + wall + lateral colour scan).
    fn run_room_align(&mut self) {
        let (Some(v1), Some(v2)) = (self.v1_last.clone(), self.v2_last.clone()) else { self.say("align: need frames from both sensors".into()); return };
        let mut lines = Vec::new();
        let t0 = Instant::now();
        let res = room_align(&v2.0, &v2.1, &v1.0, &v1.1, self.baseline, &mut |s| lines.push(s));
        for l in lines { self.say(l); }
        match res {
            Ok((mut pose, _)) => {
                if self.polish {
                    // settle the details (picture, furniture) with a short colour-gated ICP
                    let mut lines = Vec::new();
                    let (p2, rms, frac) = icp(&v2.0, &v2.1, &v1.0, &v1.1, pose, true, &mut |s| lines.push(s));
                    for l in lines { self.say(l); }
                    if rms.is_finite() && rms < 0.12 && frac > 0.15 { pose = p2; } else { self.say("ICP polish rejected, keeping the plane alignment".into()); }
                }
                self.v2_pose = pose;
                self.map.cells.clear();
                let sep = norm(pose.t);
                let yaw = pose.r[0][2].atan2(pose.r[2][2]).to_degrees();
                self.say(format!("aligned in {:.1} s: cameras {:.2} m ({:.0} in) apart, v2 yawed {:+.1} deg; map cleared", t0.elapsed().as_secs_f32(), sep, sep / 0.0254, yaw));
                self.save_pose();
            }
            Err(e) => self.say(format!("room alignment failed: {e}; press U for plain ICP or adjust by hand")),
        }
    }

    /// Point-to-point ICP refinement from the current pose.
    fn run_icp(&mut self) {
        let (Some(v1), Some(v2)) = (&self.v1_last, &self.v2_last) else { self.say("ICP: need frames from both sensors".into()); return };
        let init = self.v2_pose;
        let mut lines = Vec::new();
        let t0 = Instant::now();
        let (pose, rms, frac) = icp(&v2.0, &v2.1, &v1.0, &v1.1, init, false, &mut |s| lines.push(s));
        for l in lines { self.say(l); }
        if rms.is_finite() && rms < 0.12 && frac > 0.15 {
            self.v2_pose = pose;
            self.map.cells.clear();
            self.say(format!("ICP done in {:.1} s: rms {:.3} m, cameras {:.2} m apart, map cleared", t0.elapsed().as_secs_f32(), rms, norm(pose.t)));
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
                Key::Key3 => {
                    self.v2_first = !self.v2_first;
                    self.map.cells.clear();
                    self.say(format!("v2 takes priority in the overlap: {}; map cleared", if self.v2_first { "on" } else { "off (both sensors everywhere)" }));
                }
                Key::M => self.show_map = !self.show_map,
                Key::L => self.show_live = !self.show_live,
                Key::X => self.mirror = !self.mirror,
                Key::P => self.point_size = if self.point_size >= 4 { 1 } else { self.point_size + 1 },
                Key::H => self.help = !self.help,
                Key::I => self.align_pending = true,
                Key::U => self.icp_pending = true,
                Key::J => { self.tint = !self.tint; }
                Key::Y => {
                    self.v2_mirror = !self.v2_mirror;
                    if let Some(c) = &self.v2_raw { self.v2_last = Some(v2_cloud(c, 2, self.v2_mirror)); }
                    self.map.cells.clear();
                    self.say(format!("v2 mirror {}", if self.v2_mirror { "on" } else { "off" }));
                }
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

    /// Write both live clouds (sensor frames, full resolution) as PLY for offline analysis.
    fn dump(&mut self, dir: &str) {
        let dir = std::path::Path::new(dir);
        let _ = std::fs::create_dir_all(dir);
        let write = |path: &std::path::Path, pts: &[V3], col: &[[u8; 3]]| -> std::io::Result<()> {
            use std::io::Write;
            let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
            write!(f, "ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n", pts.len())?;
            for (p, c) in pts.iter().zip(col) { writeln!(f, "{:.4} {:.4} {:.4} {} {} {}", p[0], p[1], p[2], c[0], c[1], c[2])?; }
            Ok(())
        };
        let mut n = 0;
        if let Some((rgb, depth)) = &self.v1_raw {
            let (pts, col) = v1_cloud(rgb, depth, 1, self.v1_fx);
            if write(&dir.join("dump-v1.ply"), &pts, &col).is_ok() { n += pts.len(); }
        }
        if let Some(c) = &self.v2_raw {
            let (pts, col) = v2_cloud(c, 1, self.v2_mirror);
            if write(&dir.join("dump-v2.ply"), &pts, &col).is_ok() { n += pts.len(); }
        }
        self.say(format!("dumped {n} points to {}", dir.display()));
    }

    fn draw(&self, w: usize, h: usize) -> Canvas {
        let mut fr = Frame3D::new(&self.view, w, h);
        let mx = |p: V3| if self.mirror { [-p[0], p[1], p[2]] } else { p };
        if self.show_map && !self.tint {
            for (p, c) in self.map.points() { fr.splat(mx(p), c, self.point_size + 1); }
        }
        if self.show_live || self.tint {
            let tint = |c: &[u8; 3], hue: [f32; 3]| -> Col {
                let l = (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) * 0.6 + 90.0;
                [(l * hue[0]) as u8, (l * hue[1]) as u8, (l * hue[2]) as u8]
            };
            if self.show_v1 { if let Some((pts, col)) = &self.v1_last { for (i, (p, c)) in pts.iter().zip(col).enumerate() { if self.v1_keep.get(i).copied().unwrap_or(true) { fr.splat(mx(*p), if self.tint { tint(c, [0.45, 0.7, 1.0]) } else { *c }, self.point_size); } } } }
            if self.show_v2 { if let Some((pts, col)) = &self.v2_last { for (p, c) in pts.iter().zip(col) { fr.splat(mx(self.v2_pose.apply(*p)), if self.tint { tint(c, [1.0, 0.65, 0.3]) } else { *c }, self.point_size); } } }
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
            "v1 {}  |  {}{}  |  map {} voxels ({:.0} MB)  |  v2 at ({:.2},{:.2},{:.2}) = {:.2} m from v1  |  {}{}{}{}{}",
            if v1_ok { "live" } else { "no frames" }, self.v2.status(), if self.v2_mirror { " (mirror)" } else { "" },
            self.map.cells.len(),
            self.map.cells.len() as f32 * 48.0 / 1e6,
            self.v2_pose.t[0], self.v2_pose.t[1], self.v2_pose.t[2], norm(self.v2_pose.t),
            if self.show_v1 { "[1]v1 " } else { "[1]-- " },
            if self.show_v2 { "[2]v2 " } else { "[2]-- " },
            if self.v2_first { "[3]v2-first " } else { "[3]both " },
            if self.show_map { "[M]map " } else { "[M]live-only " },
            if self.tint { "[J]tint " } else { "" },
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
                "1 / 2: include v1 / v2    3: v2 wins where both see    M: show map    L: show live clouds    X: mirror    P: point size",
                "v2 pose:  arrows = move x/z   PgUp/PgDn = move y   Z/C yaw   T/G pitch   V/B roll   (Shift = fine)",
                "I: align v2 on ceiling + wall planes    U: ICP refine    Y: flip v2 mirror    J: tint clouds (v1 blue / v2 orange) to judge alignment    O: reset pose    K: save pose",
                "N: clear map      E: export PLY to ../pointclouds/      Esc/Q: quit",
            ];
            let hh = 20 * lines.len() as i32 + 20;
            c.panel(20, 40, c.w as i32 - 20, 40 + hh, 0.75);
            for (i, l) in lines.iter().enumerate() { c.text(l, 32, 50 + 20 * i as i32, sc, [235, 235, 235], false); }
        }
    }
}

/// "27in", "27\"", "686mm", "68.6cm" or plain metres.
fn parse_length(s: &str) -> Option<f32> {
    let s = s.trim().to_lowercase();
    let (num, unit) = s.split_at(s.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(s.len()));
    let v: f32 = num.parse().ok()?;
    Some(match unit.trim() {
        "" | "m" => v,
        "in" | "\"" | "inch" | "inches" => v * 0.0254,
        "mm" => v / 1000.0,
        "cm" => v / 100.0,
        "ft" | "'" => v * 0.3048,
        _ => return None,
    })
}

fn main() {
    let v1 = Kinect::new();
    v1.start();
    let v2 = Kinect2::start();
    let mut app = App {
        v1,
        v2,
        v2_pose: default_v2_pose(),
        v1_fx: FX1_DEFAULT,
        v1_raw: None,
        v2_mirror: true,
        v2_raw: None,
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
        align_pending: false,
        auto_icp: Some(Instant::now()),
        tint: std::env::args().any(|a| a == "--tint"),
        baseline: None,
        polish: !std::env::args().any(|a| a == "--no-polish"),
        v2_first: !std::env::args().any(|a| a == "--both"),
        v2_depth: Vec::new(),
        v1_keep: Vec::new(),
    };
    app.load_pose();
    let args: Vec<String> = std::env::args().collect();
    let dump_dir = args.iter().position(|a| a == "--dump").and_then(|i| args.get(i + 1).cloned());
    if let Some(b) = args.iter().position(|a| a == "--baseline").and_then(|i| args.get(i + 1)) {
        app.baseline = parse_length(b);
        match app.baseline {
            Some(m) => app.say(format!("camera baseline fixed at {m:.3} m ({:.1} in)", m / 0.0254)),
            None => app.say(format!("could not parse --baseline {b:?} (use e.g. 27in, 0.686 or 686mm)")),
        }
    }
    let started = Instant::now();
    if app.auto_icp.is_some() { app.say("no saved pose: the Kinect v2 will be aligned on the room planes once both sensors are live".into()); }

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
                app.align_pending = true;
            }
        }
        if app.align_pending { app.align_pending = false; app.run_room_align(); }
        if app.icp_pending { app.icp_pending = false; app.run_icp(); }
        if let Some(dir) = &dump_dir {
            if app.v1_last.is_some() && app.v2_last.is_some() && started.elapsed().as_secs_f32() > 4.0 {
                app.dump(dir);
                break;
            }
        }

        let c = app.draw(w, h);
        win.update_with_buffer(&c.buf, w, h).expect("update");
    }
    app.v1.stop();
    app.v2.stop();
}
