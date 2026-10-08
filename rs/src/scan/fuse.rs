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
    pub dims: [usize; 3],
    pub origin: V3,
    pub voxel: f32,
}

pub struct FrameRef<'a> {
    pub depth: &'a [u16],
    pub rgb: &'a [u8],
}

pub fn fuse(frames: &[FrameRef], poses: &[Pose], world: &World, center: [f32; 2], progress: &dyn Fn(usize, usize)) -> Volume {
    let (voxel, trunc, radius, height) = (0.015f32, 0.06f32, 0.95f32, 2.1f32);
    let n_of = |len: f32| (len / voxel).ceil() as usize;
    let dims = [n_of(2.0 * radius), n_of(height), n_of(2.0 * radius)];
    let origin = [center[0] - radius + voxel / 2.0, voxel / 2.0, center[1] - radius + voxel / 2.0];
    let n = dims[0] * dims[1] * dims[2];
    let mut vol = Volume { tsdf: vec![1.0; n], wt: vec![0.0; n], col: vec![[0.0; 3]; n], dims, origin, voxel };
    let slab = dims[1] * dims[2];
    for (k, f) in frames.iter().enumerate() {
        let [phi, tx, tz] = poses[k];
        let (c, s) = (phi.cos() as f32, phi.sin() as f32);
        let (tx, tz) = (tx as f32, tz as f32);
        let (ny, nz) = (dims[1], dims[2]);
        vol.tsdf
            .par_chunks_mut(slab)
            .zip(vol.wt.par_chunks_mut(slab))
            .zip(vol.col.par_chunks_mut(slab))
            .enumerate()
            .for_each(|(i, ((ts, ws), cs))| {
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
                        ts[id] = (ts[id] * w0 + sv) / (w0 + 1.0);
                        for ch in 0..3 {
                            cs[id][ch] = (cs[id][ch] * w0 + f.rgb[px * 3 + ch] as f32) / (w0 + 1.0);
                        }
                        ws[id] = w0 + 1.0;
                    }
                }
            });
        progress(k + 1, frames.len());
    }
    vol
}
