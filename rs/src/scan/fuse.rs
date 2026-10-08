//! TSDF fusion of masked depth frames into a volume in frame-0 world coordinates.
use super::track::Pose;
use super::world::*;
use crate::math::*;
use crate::source::{H, W};
use rayon::prelude::*;

pub struct Volume {
    pub tsdf: Vec<f32>,
    pub wt: Vec<f32>,
    pub col: Vec<[f32; 3]>,
    /// Colour sample count per voxel (separate from depth weight: edge pixels give depth only).
    pub cwt: Vec<f32>,
    pub dims: [usize; 3],
    pub origin: V3,
    pub voxel: f32,
}

pub struct FrameRef<'a> {
    pub depth: &'a [u16],
    /// Pixels far enough from the silhouette edge to trust their colour (edges bleed background).
    pub col_ok: &'a [bool],
    pub rgb: &'a [u8],
}

/// Volume layout: voxel size, truncation, horizontal half-width, and the vertical span [y0, y0+height].
#[derive(Clone, Copy)]
pub struct FuseCfg {
    pub voxel: f32,
    pub trunc: f32,
    pub radius: f32,
    pub y0: f32,
    pub height: f32,
}

pub fn fuse(frames: &[FrameRef], poses: &[Pose], world: &World, center: [f32; 2], cfg: FuseCfg, progress: &dyn Fn(usize, usize)) -> Volume {
    let FuseCfg { voxel, trunc, radius, y0: ybase, height } = cfg;
    let n_of = |len: f32| (len / voxel).ceil() as usize;
    let dims = [n_of(2.0 * radius), n_of(height), n_of(2.0 * radius)];
    let origin = [center[0] - radius + voxel / 2.0, ybase + voxel / 2.0, center[1] - radius + voxel / 2.0];
    let n = dims[0] * dims[1] * dims[2];
    let mut vol = Volume { tsdf: vec![1.0; n], wt: vec![0.0; n], col: vec![[0.0; 3]; n], cwt: vec![0.0; n], dims, origin, voxel };
    let slab = dims[1] * dims[2];
    for (k, f) in frames.iter().enumerate() {
        let [phi, tx, tz] = poses[k];
        let (c, s) = (phi.cos() as f32, phi.sin() as f32);
        let (tx, tz) = (tx as f32, tz as f32);
        let (ny, nz) = (dims[1], dims[2]);
        let view_w = view_weights(f.depth);
        vol.tsdf
            .par_chunks_mut(slab)
            .zip(vol.wt.par_chunks_mut(slab))
            .zip(vol.col.par_chunks_mut(slab))
            .zip(vol.cwt.par_chunks_mut(slab))
            .enumerate()
            .for_each(|(i, (((ts, ws), cs), cw))| {
                let vx = origin[0] + i as f32 * voxel;
                for j in 0..ny {
                    let vy = origin[1] + j as f32 * voxel;
                    for kk in 0..nz {
                        let vz = origin[2] + kk as f32 * voxel;
                        let (dx, dz) = (vx - tx, vz - tz);
                        let pw = [c * dx + s * dz, vy, -s * dx + c * dz]; // model -> frame-k world
                        let pc = world.to_camera(pw);
                        if pc[2] <= 0.3 {
                            continue;
                        }
                        let u = (FX * pc[0] / pc[2] + CX).round();
                        let v = (FY * pc[1] / pc[2] + CY).round();
                        if u < 0.0 || v < 0.0 || u >= W as f32 || v >= H as f32 {
                            continue;
                        }
                        let px = v as usize * W + u as usize;
                        let d = f.depth[px] as f32 / 1000.0;
                        if d <= 0.0 {
                            continue;
                        }
                        let sdf = d - pc[2];
                        if sdf <= -trunc {
                            continue;
                        }
                        let sv = (sdf / trunc).clamp(-1.0, 1.0);
                        let id = j * nz + kk;
                        let w0 = ws[id];
                        let wd = view_w[px].sqrt().max(0.15);
                        ts[id] = (ts[id] * w0 + sv * wd) / (w0 + wd);
                        if f.col_ok[px] && sdf.abs() < trunc * 0.6 {
                            let c0 = cw[id];
                            let w = view_w[px];
                            for ch in 0..3 {
                                cs[id][ch] = (cs[id][ch] * c0 + f.rgb[px * 3 + ch] as f32 * w) / (c0 + w);
                            }
                            cw[id] = c0 + w;
                        }
                        ws[id] = w0 + wd;
                    }
                }
            });
        progress(k + 1, frames.len());
    }
    vol
}

/// Colour weight per pixel: face-on surfaces count far more than grazing ones (cos^4 of the view angle).
fn view_weights(depth: &[u16]) -> Vec<f32> {
    let mut out = vec![0.0f32; W * H];
    for y in 1..H - 1 {
        for x in 1..W - 1 {
            let d = depth[y * W + x] as f32;
            if d <= 0.0 {
                continue;
            }
            let g = |a: usize, b: usize| {
                let (da, db) = (depth[a] as f32, depth[b] as f32);
                if da > 0.0 && db > 0.0 { (db - da) / 2.0 / (d / FX) } else { 0.0 }
            };
            let sx = g(y * W + x - 1, y * W + x + 1);
            let sy = g((y - 1) * W + x, (y + 1) * W + x);
            let c2 = 1.0 / (1.0 + sx * sx + sy * sy);
            out[y * W + x] = c2 * c2 + 0.02;
        }
    }
    out
}
