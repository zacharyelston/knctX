//! Auto-rig (16 joints from mesh slices + proportions), skinning weights, poses and linear blend skinning.
use crate::math::*;
use rayon::prelude::*;

pub const JOINTS: [&str; 16] = [
    "pelvis", "spine", "neck", "head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist", "l_hip", "l_knee", "l_ankle", "r_hip",
    "r_knee", "r_ankle",
];
pub const PARENT: [i32; 16] = [-1, 0, 1, 2, 1, 4, 5, 1, 7, 8, 0, 10, 11, 0, 13, 14];
pub const POSES: [&str; 6] = ["Rest", "T-pose", "Hands up", "Wave", "Walk", "Dance"];

pub fn ji(name: &str) -> usize { JOINTS.iter().position(|n| *n == name).unwrap() }

pub type Params = [[f32; 3]; 16];

/// Place a 16-joint skeleton from the mesh (assumes an upright A-pose-ish scan facing +z).
pub fn estimate_rig(verts: &[V3]) -> ([V3; 16], f32) {
    let top = verts.iter().map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max);
    let hh = top;
    let (mut mx, mut mz): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    for v in verts {
        if v[1] > 0.5 * hh && v[1] < 0.8 * hh {
            mx.push(v[0]);
            mz.push(v[2]);
        }
    }
    let (cx, cz) = (median(&mut mx), median(&mut mz));
    let mut j = [[0.0f32; 3]; 16];
    j[ji("pelvis")] = [cx, 0.53 * hh, cz];
    j[ji("spine")] = [cx, 0.68 * hh, cz];
    j[ji("neck")] = [cx, 0.855 * hh, cz];
    j[ji("head")] = [cx, 0.89 * hh, cz];
    for (side, s) in [("l", 1.0f32), ("r", -1.0)] {
        j[ji(&format!("{side}_hip"))] = [cx + s * 0.055 * hh, 0.53 * hh, cz];
        for (nm, lo, hi) in [("knee", 0.25f32, 0.33f32), ("ankle", 0.04, 0.10)] {
            let sel: Vec<&V3> = verts.iter().filter(|v| v[1] > lo * hh && v[1] < hi * hh && s * (v[0] - cx) > 0.0).collect();
            let id = ji(&format!("{side}_{nm}"));
            if sel.len() > 30 {
                let n = sel.len() as f32;
                j[id] = [sel.iter().map(|v| v[0]).sum::<f32>() / n, (lo + hi) / 2.0 * hh, sel.iter().map(|v| v[2]).sum::<f32>() / n];
            } else {
                j[id] = [cx + s * 0.055 * hh, if nm == "knee" { 0.285 } else { 0.045 } * hh, cz];
            }
        }
        let sh = [cx + s * 0.12 * hh, 0.81 * hh, cz];
        let arm: Vec<V3> = verts.iter().copied().filter(|v| v[1] > 0.3 * hh && v[1] < 0.86 * hh && s * (v[0] - cx) > 0.13 * hh).collect();
        let mut tip: Option<V3> = None;
        if arm.len() > 200 {
            let d: Vec<f32> = arm.iter().map(|p| norm(sub(*p, sh))).collect();
            let thr = percentile(&mut d.clone(), 97.0);
            let far: Vec<&V3> = arm.iter().zip(&d).filter(|(_, dd)| **dd >= thr).map(|(p, _)| p).collect();
            let n = far.len() as f32;
            let m = [far.iter().map(|p| p[0]).sum::<f32>() / n, far.iter().map(|p| p[1]).sum::<f32>() / n, far.iter().map(|p| p[2]).sum::<f32>() / n];
            if norm(sub(m, sh)) > 0.25 * hh {
                tip = Some(m);
            }
        }
        let tip = tip.unwrap_or_else(|| {
            let a = 25f32.to_radians();
            add(sh, scale([s * a.sin(), -a.cos(), 0.0], 0.44 * hh))
        });
        j[ji(&format!("{side}_shoulder"))] = sh;
        j[ji(&format!("{side}_elbow"))] = add(sh, scale(sub(tip, sh), 0.42));
        j[ji(&format!("{side}_wrist"))] = add(sh, scale(sub(tip, sh), 0.755));
    }
    (j, top)
}

/// (start, end) per joint; each joint drives the bone from itself to its child/tip.
pub fn bone_segments(j: &[V3; 16], top: f32) -> [[V3; 2]; 16] {
    let p = |n: &str| j[ji(n)];
    let mut b = [[[0.0f32; 3]; 2]; 16];
    b[ji("pelvis")] = [p("pelvis"), p("spine")];
    b[ji("spine")] = [p("spine"), p("neck")];
    b[ji("neck")] = [p("neck"), p("head")];
    b[ji("head")] = [p("head"), [p("head")[0], top, p("head")[2]]];
    for side in ["l", "r"] {
        let (sh, el, wr) = (p(&format!("{side}_shoulder")), p(&format!("{side}_elbow")), p(&format!("{side}_wrist")));
        b[ji(&format!("{side}_shoulder"))] = [sh, el];
        b[ji(&format!("{side}_elbow"))] = [el, wr];
        b[ji(&format!("{side}_wrist"))] = [wr, add(wr, scale(sub(wr, el), 0.45))];
        let (hp, kn, an) = (p(&format!("{side}_hip")), p(&format!("{side}_knee")), p(&format!("{side}_ankle")));
        b[ji(&format!("{side}_hip"))] = [hp, kn];
        b[ji(&format!("{side}_knee"))] = [kn, an];
        b[ji(&format!("{side}_ankle"))] = [an, add(an, scale([0.0, -0.02, 0.10], top))];
    }
    b
}

/// Inverse-distance (^4) skinning weights to the nearest k bones.
pub fn skin_weights(verts: &[V3], j: &[V3; 16], top: f32) -> (Vec<[u8; 4]>, Vec<[f32; 4]>) {
    let b = bone_segments(j, top);
    let cx = j[ji("spine")][0];
    let damp: Vec<usize> = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"].iter().map(|n| ji(n)).collect();
    let res: Vec<([u8; 4], [f32; 4])> = verts
        .par_iter()
        .map(|v| {
            let mut w = [0.0f32; 16];
            for (bi, seg) in b.iter().enumerate() {
                let ab = sub(seg[1], seg[0]);
                let t = (dot(sub(*v, seg[0]), ab) / dot(ab, ab).max(1e-9)).clamp(0.0, 1.0);
                let d = norm(sub(*v, add(seg[0], scale(ab, t))));
                w[bi] = 1.0 / (d + 0.015).powi(4);
            }
            if (v[0] - cx).abs() < 0.10 * top && v[1] > 0.55 * top {
                for &d in &damp {
                    w[d] *= 0.02;
                }
            }
            let mut order: Vec<usize> = (0..16).collect();
            order.sort_by(|a, b| w[*b].partial_cmp(&w[*a]).unwrap());
            let mut idx = [0u8; 4];
            let mut wk = [0.0f32; 4];
            let mut sum = 0.0;
            for c in 0..4 {
                idx[c] = order[c] as u8;
                wk[c] = w[order[c]];
                sum += wk[c];
            }
            for c in 0..4 {
                wk[c] /= sum;
            }
            (idx, wk)
        })
        .collect();
    res.into_iter().unzip()
}

pub struct Figure {
    pub rest: Vec<V3>,
    pub faces: Vec<[u32; 3]>,
    pub rgb: Vec<[u8; 3]>,
    pub j: [V3; 16],
    pub top: f32,
    idx: Vec<[u8; 4]>,
    w: Vec<[f32; 4]>,
    pub rest_normals: Vec<V3>,
    arm_rest: [f32; 2], // l, r
}

impl Figure {
    pub fn new(verts: Vec<V3>, faces: Vec<[u32; 3]>, rgb: Vec<[u8; 3]>) -> Figure {
        let (j, top) = estimate_rig(&verts);
        let (idx, w) = skin_weights(&verts, &j, top);
        let rest_normals = super::mesh::vertex_normals(&verts, &faces);
        let mut arm_rest = [0.0f32; 2];
        for (n, (side, s)) in [("l", 1.0f32), ("r", -1.0)].iter().enumerate() {
            let v = sub(j[ji(&format!("{side}_elbow"))], j[ji(&format!("{side}_shoulder"))]);
            arm_rest[n] = (s * v[0]).atan2(-v[1]).to_degrees();
        }
        Figure { rest: verts, faces, rgb, j, top, idx, w, rest_normals, arm_rest }
    }

    /// Joint (rx, ry, rz) degrees for a named pose at time t (seconds).
    pub fn pose_params(&self, name: &str, t: f32) -> Params {
        let mut p: Params = [[0.0; 3]; 16];
        let arm = |p: &mut Params, side: usize, deg: f32| {
            let (nm, s, rest) = if side == 0 { ("l_shoulder", 1.0, self.arm_rest[0]) } else { ("r_shoulder", -1.0, self.arm_rest[1]) };
            p[ji(nm)] = [0.0, 0.0, s * (deg - rest)];
        };
        let w = std::f32::consts::TAU * t;
        match name {
            "T-pose" => {
                arm(&mut p, 0, 90.0);
                arm(&mut p, 1, 90.0);
            }
            "Hands up" => {
                arm(&mut p, 0, 165.0);
                arm(&mut p, 1, 165.0);
            }
            "Wave" => {
                arm(&mut p, 0, 100.0);
                p[ji("l_elbow")] = [0.0, 0.0, 55.0 + 28.0 * (2.0 * w).sin()];
                arm(&mut p, 1, 15.0);
            }
            "Walk" => {
                let a = w.sin();
                p[ji("l_hip")] = [-28.0 * a, 0.0, 0.0];
                p[ji("r_hip")] = [28.0 * a, 0.0, 0.0];
                p[ji("l_knee")] = [(40.0 * w.cos()).max(0.0), 0.0, 0.0];
                p[ji("r_knee")] = [(-40.0 * w.cos()).max(0.0), 0.0, 0.0];
                p[ji("l_shoulder")] = [25.0 * a, 0.0, 0.0];
                p[ji("r_shoulder")] = [-25.0 * a, 0.0, 0.0];
                p[ji("l_elbow")] = [-25.0 * (1.0 - a), 0.0, 0.0];
                p[ji("r_elbow")] = [-25.0 * (1.0 + a), 0.0, 0.0];
                p[ji("spine")] = [0.0, 8.0 * a, 0.0];
            }
            "Dance" => {
                let a = (2.0 * w).sin();
                arm(&mut p, 0, 120.0 + 35.0 * a);
                arm(&mut p, 1, 120.0 - 35.0 * a);
                p[ji("l_elbow")] = [0.0, 0.0, 25.0 * a];
                p[ji("r_elbow")] = [0.0, 0.0, 25.0 * a];
                p[ji("pelvis")] = [0.0, 18.0 * w.sin(), 0.0];
                p[ji("spine")] = [0.0, -12.0 * w.sin(), 0.0];
                p[ji("head")] = [0.0, 0.0, 10.0 * (2.0 * w).sin()];
                p[ji("l_knee")] = [(20.0 * a).max(0.0), 0.0, 0.0];
                p[ji("r_knee")] = [(-20.0 * a).max(0.0), 0.0, 0.0];
            }
            _ => {}
        }
        p
    }

    /// Forward kinematics + linear blend skinning. Returns (verts, normals).
    pub fn posed(&self, params: &Params) -> (Vec<V3>, Vec<V3>) {
        let mut rw: [M3; 16] = [[[0.0; 3]; 3]; 16];
        let mut pw: [V3; 16] = [[0.0; 3]; 16];
        for jn in 0..16 {
            let rl = euler(params[jn][0], params[jn][1], params[jn][2]);
            let par = PARENT[jn];
            if par < 0 {
                rw[jn] = rl;
                pw[jn] = self.j[jn];
            } else {
                let p = par as usize;
                rw[jn] = mat_mul(&rw[p], &rl);
                pw[jn] = add(pw[p], mat_vec(&rw[p], sub(self.j[jn], self.j[p])));
            }
        }
        let ts: Vec<V3> = (0..16).map(|jn| sub(pw[jn], mat_vec(&rw[jn], self.j[jn]))).collect();
        let out: Vec<(V3, V3)> = (0..self.rest.len())
            .into_par_iter()
            .map(|v| {
                let (mut o, mut n) = ([0.0f32; 3], [0.0f32; 3]);
                for c in 0..4 {
                    let jn = self.idx[v][c] as usize;
                    let wc = self.w[v][c];
                    o = add(o, scale(add(mat_vec(&rw[jn], self.rest[v]), ts[jn]), wc));
                    n = add(n, scale(mat_vec(&rw[jn], self.rest_normals[v]), wc));
                }
                (o, normalize(n))
            })
            .collect();
        out.into_iter().unzip()
    }

    pub fn meta_json(&self) -> serde_json::Value {
        let mut joints = serde_json::Map::new();
        for (i, n) in JOINTS.iter().enumerate() {
            joints.insert(n.to_string(), serde_json::json!(self.j[i].iter().map(|x| (*x as f64 * 1e4).round() / 1e4).collect::<Vec<_>>()));
        }
        serde_json::json!({"height_m": (self.top as f64 * 1000.0).round() / 1000.0, "joints": joints})
    }
}
