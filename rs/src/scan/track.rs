//! 2D rigid poses (yaw + floor translation), yaw ICP, keyframe spin tracker and loop closure.
use crate::math::*;
use rayon::prelude::*;
use std::f64::consts::PI;

/// (phi, tx, tz): rotation about the vertical axis plus translation in the floor plane.
pub type Pose = [f64; 3];

pub fn rot2(phi: f64) -> [[f64; 2]; 2] {
    let (c, s) = (phi.cos(), phi.sin());
    [[c, -s], [s, c]]
}
fn mv(m: &[[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] { [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]] }

#[inline]
pub fn pose_apply_pt(p: &Pose, q: V3) -> V3 {
    let (c, s) = (p[0].cos() as f32, p[0].sin() as f32);
    [c * q[0] - s * q[2] + p[1] as f32, q[1], s * q[0] + c * q[2] + p[2] as f32]
}

/// a after b.
pub fn pose_compose(a: &Pose, b: &Pose) -> Pose {
    let r = mv(&rot2(a[0]), [b[1], b[2]]);
    [a[0] + b[0], r[0] + a[1], r[1] + a[2]]
}

pub fn pose_inverse(a: &Pose) -> Pose {
    let r = mv(&rot2(-a[0]), [a[1], a[2]]);
    [-a[0], -r[0], -r[1]]
}

pub fn wrap_pi(a: f64) -> f64 { (a + PI).rem_euclid(2.0 * PI) - PI }

/// Uniform-grid nearest neighbour index over a point set.
pub struct Grid {
    pts: Vec<V3>,
    origin: V3,
    cell: f32,
    dims: [usize; 3],
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Grid {
    pub fn new(pts: &[V3], cell: f32) -> Grid {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in pts {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let origin = [lo[0] - cell, lo[1] - cell, lo[2] - cell];
        let dims = [0, 1, 2].map(|k| ((hi[k] - origin[k]) / cell) as usize + 2);
        let ncell = dims[0] * dims[1] * dims[2];
        let cid = |p: &V3| -> usize {
            let c = [0, 1, 2].map(|k| ((p[k] - origin[k]) / cell) as usize);
            (c[0] * dims[1] + c[1]) * dims[2] + c[2]
        };
        let mut start = vec![0u32; ncell + 1];
        for p in pts {
            start[cid(p) + 1] += 1;
        }
        for i in 0..ncell {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; pts.len()];
        for (i, p) in pts.iter().enumerate() {
            let c = cid(p);
            items[fill[c] as usize] = i as u32;
            fill[c] += 1;
        }
        Grid { pts: pts.to_vec(), origin, cell, dims, start, items }
    }

    /// Nearest stored point within `max_d` of `q`: (index, distance).
    pub fn nearest(&self, q: V3, max_d: f32) -> Option<(usize, f32)> {
        let lo = [0, 1, 2].map(|k| ((q[k] - max_d - self.origin[k]) / self.cell).floor() as i64);
        let hi = [0, 1, 2].map(|k| ((q[k] + max_d - self.origin[k]) / self.cell).floor() as i64);
        let mut best: Option<(usize, f32)> = None;
        let mut best2 = max_d * max_d;
        for x in lo[0].max(0)..=hi[0].min(self.dims[0] as i64 - 1) {
            for y in lo[1].max(0)..=hi[1].min(self.dims[1] as i64 - 1) {
                for z in lo[2].max(0)..=hi[2].min(self.dims[2] as i64 - 1) {
                    let id = (x as usize * self.dims[1] + y as usize) * self.dims[2] + z as usize;
                    for &it in &self.items[self.start[id] as usize..self.start[id + 1] as usize] {
                        let p = self.pts[it as usize];
                        let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                        if d2 < best2 {
                            best2 = d2;
                            best = Some((it as usize, d2.sqrt()));
                        }
                    }
                }
            }
        }
        best
    }
}

/// Align `src` to `tgt` with a yaw rotation + floor translation. Returns (pose, rms, inlier_frac).
pub fn icp_yaw(src: &[V3], tgt: &[V3], grid: &Grid, init: Pose, iters: usize, max_d: f32) -> (Pose, f64, f64) {
    let mut pose = init;
    let mut last_rms = 1.0;
    let mut last_frac = 0.0;
    for _ in 0..iters {
        // (src idx, tgt idx, d)
        let mut pairs: Vec<(usize, usize, f32)> = src
            .par_iter()
            .enumerate()
            .filter_map(|(i, s)| grid.nearest(pose_apply_pt(&pose, *s), max_d).map(|(j, d)| (i, j, d)))
            .collect();
        if pairs.len() < 100 {
            return (pose, 1.0, 0.0);
        }
        let mut ds: Vec<f32> = pairs.iter().map(|p| p.2).collect();
        let cut = percentile(&mut ds, 85.0);
        pairs.retain(|p| p.2 <= cut);
        last_rms = (pairs.iter().map(|p| (p.2 as f64).powi(2)).sum::<f64>() / pairs.len() as f64).sqrt();
        last_frac = pairs.len() as f64 / src.len() as f64;
        let n = pairs.len() as f64;
        let (mut pc, mut qc) = ([0.0f64; 2], [0.0f64; 2]);
        let moved: Vec<V3> = pairs.iter().map(|p| pose_apply_pt(&pose, src[p.0])).collect();
        for (k, p) in pairs.iter().enumerate() {
            pc[0] += moved[k][0] as f64;
            pc[1] += moved[k][2] as f64;
            qc[0] += tgt[p.1][0] as f64;
            qc[1] += tgt[p.1][2] as f64;
        }
        for k in 0..2 {
            pc[k] /= n;
            qc[k] /= n;
        }
        let (mut cr, mut dt) = (0.0f64, 0.0f64);
        for (k, p) in pairs.iter().enumerate() {
            let a = [moved[k][0] as f64 - pc[0], moved[k][2] as f64 - pc[1]];
            let b = [tgt[p.1][0] as f64 - qc[0], tgt[p.1][2] as f64 - qc[1]];
            cr += a[0] * b[1] - a[1] * b[0];
            dt += a[0] * b[0] + a[1] * b[1];
        }
        let dphi = cr.atan2(dt);
        let r = mv(&rot2(dphi), pc);
        let tr = [qc[0] - r[0], qc[1] - r[1]];
        pose = pose_compose(&[dphi, tr[0], tr[1]], &pose);
        if dphi.abs() < 2e-4 && (tr[0] * tr[0] + tr[1] * tr[1]).sqrt() < 3e-4 {
            break;
        }
    }
    (pose, last_rms, last_frac)
}

/// Keyframe-based yaw tracking of a spinning person.
///
/// Aligning every frame to the previous one fails at small steps: a few degrees per frame moves
/// the body a centimetre, below the sensor's depth quantisation, and ICP then biases toward
/// "hardly moved". Instead each frame is aligned to the last *keyframe*, which is only replaced
/// after ~KEY_DEG of rotation, so every alignment sees a motion well above the noise.
pub struct Tracker {
    key: Option<Vec<V3>>,
    grid: Option<Grid>,
    key_pose: Pose,
    pub pose: Pose,
    vel: f64,
    last_phi: f64,
    pub quality: f64,
}

const KEY_DEG: f64 = 12.0;

impl Tracker {
    pub fn new() -> Self { Tracker { key: None, grid: None, key_pose: [0.0; 3], pose: [0.0; 3], vel: 0.0, last_phi: 0.0, quality: 1.0 } }
    pub fn yaw(&self) -> f64 { self.pose[0] }

    pub fn add(&mut self, pts: &[V3]) -> Pose {
        let key = match &self.key {
            None => {
                self.set_key(pts, [0.0; 3]);
                return self.pose;
            }
            Some(k) => k,
        };
        let c = {
            let n = key.len() as f64;
            [key.iter().map(|p| p[0] as f64).sum::<f64>() / n, key.iter().map(|p| p[2] as f64).sum::<f64>() / n]
        };
        let guess_phi = self.last_phi - self.key_pose[0] + self.vel;
        let r = mv(&rot2(guess_phi), c);
        let init = [guess_phi, c[0] - r[0], c[1] - r[1]];
        let (rel, rms, frac) = icp_yaw(pts, key, self.grid.as_ref().unwrap(), init, 25, 0.10);
        let ok = rms < 0.04 && frac > 0.35 && rel[0].abs() < 60f64.to_radians();
        self.quality = 0.85 * self.quality + 0.15 * if ok { 1.0 } else { 0.0 };
        let rel = if ok { rel } else { init }; // coast on the velocity estimate until tracking recovers
        self.pose = pose_compose(&self.key_pose, &rel);
        if ok {
            self.vel = 0.6 * self.vel + 0.4 * (self.pose[0] - self.last_phi);
        }
        self.last_phi = self.pose[0];
        if ok && rel[0].abs() > KEY_DEG.to_radians() {
            let p = self.pose;
            self.set_key(pts, p);
        }
        self.pose
    }

    fn set_key(&mut self, pts: &[V3], pose: Pose) {
        self.key = Some(pts.to_vec());
        self.grid = Some(Grid::new(pts, 0.12));
        self.key_pose = pose;
        self.pose = pose;
        self.last_phi = pose[0];
    }
}

/// Distribute accumulated drift so each full revolution lands back on frame 0.
///
/// The frame that completes revolution j is found by geometry, not by the (biased) yaw estimate:
/// candidates near j*360 degrees are ICP-matched against frame 0 from several starting angles and
/// the best fit wins. Drift is then spread linearly between anchors.
pub fn close_loops(pts_list: &[&[V3]], poses: &[Pose], log: &mut dyn FnMut(String)) -> Vec<Pose> {
    let k_n = poses.len();
    if k_n == 0 {
        return vec![];
    }
    let sign = if poses[k_n - 1][0] >= 0.0 { 1.0 } else { -1.0 };
    let yaw: Vec<f64> = poses.iter().map(|p| sign * p[0]).collect();
    let revs = ((yaw[k_n - 1] + 40f64.to_radians()) / (2.0 * PI)).floor() as i64;
    if revs < 1 || k_n < 8 {
        return poses.to_vec();
    }
    let grid0 = Grid::new(pts_list[0], 0.12);
    let mut anchors: Vec<usize> = vec![0];
    let mut corr: Vec<Pose> = vec![[0.0; 3]];
    for j in 1..=revs {
        let cands: Vec<usize> = (anchors[anchors.len() - 1] + 1..k_n).filter(|&k| (yaw[k] - j as f64 * 2.0 * PI).abs() < 40f64.to_radians()).collect();
        let picks: Vec<usize> = if cands.len() > 1 { cands.iter().step_by(2).copied().collect() } else { cands.clone() };
        let mut best: Option<(f64, usize, Pose, f64, f64)> = None;
        for &k in &picks {
            for off in [0.0f64, -20.0, 20.0, -35.0, 35.0] {
                let mut init = poses[k];
                init[0] = wrap_pi(init[0] + off.to_radians());
                let (new, rms, frac) = icp_yaw(pts_list[k], pts_list[0], &grid0, init, 30, 0.12);
                let score = rms / frac.max(1e-3);
                if frac > 0.5 && best.map_or(true, |b| score < b.0) {
                    best = Some((score, k, new, rms, frac));
                }
            }
        }
        let Some((_, k, new, rms, _frac)) = best else {
            log(format!("loop {j}: no match found"));
            continue;
        };
        let mut e = pose_compose(&new, &pose_inverse(&poses[k]));
        e[0] = wrap_pi(e[0]);
        let n0 = pts_list[0].len() as f64;
        let c0 = [pts_list[0].iter().map(|p| p[0] as f64).sum::<f64>() / n0, pts_list[0].iter().map(|p| p[2] as f64).sum::<f64>() / n0];
        let r = mv(&rot2(e[0]), c0);
        let shift = ((r[0] + e[1] - c0[0]).powi(2) + (r[1] + e[2] - c0[1]).powi(2)).sqrt();
        if rms > 0.04 || e[0].abs() > 60f64.to_radians() || shift > 0.25 {
            log(format!("loop {j}: rejected (rms {rms:.3}, yaw {:+.1})", e[0].to_degrees()));
            continue;
        }
        log(format!("loop {j}: frame {k}, drift {:+.1} deg, {:.1} cm, rms {:.1} cm", e[0].to_degrees(), shift * 100.0, rms * 100.0));
        anchors.push(k);
        corr.push(e);
    }
    let mut out = poses.to_vec();
    for k in 0..k_n {
        let i = anchors.partition_point(|&a| a < k);
        let e = if i >= anchors.len() {
            corr[corr.len() - 1]
        } else if i == 0 || anchors[i] == k {
            corr[i]
        } else {
            let w = (k - anchors[i - 1]) as f64 / (anchors[i] - anchors[i - 1]) as f64;
            [0, 1, 2].map(|c| corr[i - 1][c] + w * (corr[i][c] - corr[i - 1][c]))
        };
        out[k] = pose_compose(&e, &poses[k]);
    }
    out
}
