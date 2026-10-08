//! Camera model, world frame (x right, y up with floor = 0, z toward the camera) and floor fitting.
use crate::math::*;
use crate::source::{H, W};

// Approximate Kinect v1 intrinsics, 640x480 registered depth.
pub const FX: f32 = 580.0;
pub const FY: f32 = 580.0;
pub const CX: f32 = 319.5;
pub const CY: f32 = 239.5;

/// Camera-space point (x right, y down, z forward, metres) for pixel (u, v) at depth `z` metres.
#[inline]
pub fn cam_point(u: usize, v: usize, z: f32) -> V3 { [(u as f32 - CX) / FX * z, (v as f32 - CY) / FY * z, z] }

/// Rigid camera->world transform: p_w = R p_c + t (gravity-aligned via the floor plane).
#[derive(Clone, Copy, Debug)]
pub struct World {
    pub r: M3,
    pub t: V3,
    pub floor_found: bool,
}

impl World {
    pub fn default_world() -> World { World { r: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], t: [0.0, 1.0, 0.0], floor_found: false } }
    #[inline]
    pub fn to_world(&self, pc: V3) -> V3 { add(mat_vec(&self.r, pc), self.t) }
    #[inline]
    pub fn to_camera(&self, pw: V3) -> V3 { mat_t_vec(&self.r, sub(pw, self.t)) }
}

impl Default for World {
    fn default() -> Self { World::default_world() }
}

/// Find the floor: the *lowest* strong horizontal surface.
///
/// RANSAC finds the dominant roughly-horizontal plane (it may be a table), which fixes the up
/// direction. All points are then binned by height along that direction and the lowest
/// well-supported level is taken as the floor, so desks and tables above it are ignored.
pub fn fit_floor(depth: &[u16], seed: u64) -> Option<World> {
    let mut pts: Vec<V3> = Vec::new();
    for v in (0..H).step_by(4) {
        for u in (0..W).step_by(4) {
            let z = depth[v * W + u] as f32 / 1000.0;
            if z > 1.0 && z < 5.0 {
                pts.push(cam_point(u, v, z));
            }
        }
    }
    if pts.len() < 2000 {
        return None;
    }
    let mut rng = Rng::new(seed);
    let up: V3 = [0.0, -1.0, 0.0];
    let mut best: Option<(usize, V3, f32)> = None;
    for _ in 0..300 {
        let s = [pts[rng.below(pts.len())], pts[rng.below(pts.len())], pts[rng.below(pts.len())]];
        let mut n = cross(sub(s[1], s[0]), sub(s[2], s[0]));
        let nn = norm(n);
        if nn < 1e-6 {
            continue;
        }
        n = scale(n, 1.0 / nn);
        if dot(n, up) < 0.0 {
            n = scale(n, -1.0);
        }
        if dot(n, up) < 0.85 {
            continue;
        }
        let off = dot(n, s[0]);
        let cnt = pts.iter().filter(|p| (dot(**p, n) - off).abs() < 0.03).count();
        if best.map_or(true, |b| cnt > b.0) {
            best = Some((cnt, n, off));
        }
    }
    let (cnt, n0, off0) = best?;
    if (cnt as f32) < 0.03 * pts.len() as f32 {
        return None;
    }
    let inl: Vec<V3> = pts.iter().copied().filter(|p| (dot(*p, n0) - off0).abs() < 0.03).collect();
    let (mut n, _) = plane_fit(&inl);
    if dot(n, up) < 0.0 {
        n = scale(n, -1.0);
    }
    // lowest well-supported horizontal level along n
    let h: Vec<f32> = pts.iter().map(|p| dot(*p, n)).collect();
    let (hmin, hmax) = (h.iter().cloned().fold(f32::INFINITY, f32::min), h.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    let nb = (((hmax + 0.02 - hmin) / 0.02).ceil() as usize).max(1);
    let mut hist = vec![0usize; nb];
    for &x in &h {
        hist[(((x - hmin) / 0.02) as usize).min(nb - 1)] += 1;
    }
    let thr = 150usize.max((0.012 * pts.len() as f32) as usize + 1);
    let mut floor_c = None;
    for i in 0..nb {
        let c = hmin + (i as f32 + 0.5) * 0.02;
        if hist[i] >= thr && hist[i] >= hist[i.saturating_sub(1)] && hist[i] >= hist[(i + 1).min(nb - 1)] && c > -2.5 && c < -0.4 {
            floor_c = Some(c);
            break;
        }
    }
    let floor_c = floor_c?;
    let level: Vec<V3> = pts.iter().zip(&h).filter(|(_, hh)| (**hh - floor_c).abs() < 0.04).map(|(p, _)| *p).collect();
    if level.len() < 300 {
        return None;
    }
    let (mut n2, mean) = plane_fit(&level);
    if dot(n2, up) < 0.0 {
        n2 = scale(n2, -1.0);
    }
    if dot(n2, n) > 0.97 {
        n = n2;
    }
    let c = dot(n, mean);
    if !(c > -2.5 && c < -0.4) {
        return None;
    }
    let ex: V3 = [1.0, 0.0, 0.0];
    let xw = normalize(sub(ex, scale(n, dot(ex, n))));
    let zw = cross(xw, n);
    Some(World { r: [xw, n, zw], t: [0.0, -c, 0.0], floor_found: true })
}

/// Distance (m) to the nearest sizeable object in the middle of the view, or None.
pub fn nearest_distance(depth: &[u16]) -> Option<f32> {
    let mut v: Vec<f32> = Vec::new();
    for r in 0..380 {
        for c in 120..520 {
            let d = depth[r * W + c];
            if d > 1000 {
                v.push(d as f32);
            }
        }
    }
    if v.len() < 4000 {
        return None;
    }
    let z0 = percentile(&mut v.clone(), 8.0);
    let mut blob: Vec<f32> = Vec::new();
    for r in 0..380 {
        for c in 120..520 {
            let d = depth[r * W + c];
            if d > 0 && (d as f32 - z0).abs() < 250.0 {
                blob.push(d as f32);
            }
        }
    }
    if blob.len() > 3000 {
        Some(median(&mut blob) / 1000.0)
    } else {
        None
    }
}

/// World frame from a gravity reading (the sensor's accelerometer): y up, camera 1 m above the
/// origin. Use when the floor isn't visible (close-up seated scans). `accel` is the raw reading in
/// sensor axes. Returns None if it is implausible.
pub fn world_from_accel(accel: [f64; 3]) -> Option<World> {
    // sensor axes -> camera axes (x right, y down, z forward): the sensor's y points up (verified
    // against floor fits at several tilts)
    let a = [accel[0] as f32, -accel[1] as f32, accel[2] as f32];
    let m = norm(a);
    if !(8.0..11.5).contains(&m) {
        return None;
    }
    let u = scale(a, 1.0 / m); // up, in camera axes (x right, y down, z forward)
    let ex: V3 = [1.0, 0.0, 0.0];
    let xw = normalize(sub(ex, scale(u, dot(ex, u))));
    let zw = cross(xw, u);
    Some(World { r: [xw, u, zw], t: [0.0, 1.0, 0.0], floor_found: true })
}
