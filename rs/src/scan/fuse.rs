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

/// Silhouette mask with small holes closed (hair and glasses lose depth inside the head outline).
fn closed_mask(depth: &[u16], gap: usize) -> Vec<bool> {
    let mut m: Vec<bool> = depth.iter().map(|d| *d > 0).collect();
    let close_run = |m: &mut Vec<bool>, idx: &dyn Fn(usize) -> usize, len: usize, lines: usize| {
        for l in 0..lines {
            let mut last: Option<usize> = None;
            for k in 0..len {
                let p = idx(l * len + k);
                if m[p] {
                    if let Some(a) = last {
                        if k - a > 1 && k - a <= gap {
                            for q in a + 1..k {
                                m[idx(l * len + q)] = true;
                            }
                        }
                    }
                    last = Some(k);
                }
            }
        }
    };
    let snapshot = m.clone();
    close_run(&mut m, &|i| i, W, H);
    let rows = m.clone();
    m = snapshot;
    close_run(&mut m, &|i| (i % H) * W + i / H, H, W);
    for (a, b) in m.iter_mut().zip(rows) {
        *a = *a && b;
    }
    m
}

/// Fill the head out to its silhouettes: hair returns little infrared and its depth reads too deep,
/// which carves the head thin. A voxel seen inside the silhouette in (nearly) every frame is solid;
/// grow the fused surface into such voxels, at most `reach` metres.
pub fn hull_fill(vol: &mut Volume, frames: &[FrameRef], poses: &[Pose], world: &World, center: [f32; 2], head_h: f32, reach: f32, min_frac: f32) -> usize {
    let [nx, ny, nz] = vol.dims;
    let slab = ny * nz;
    // top of the fused solid
    let mut top = 0;
    for j in 0..ny {
        let n = (0..nx).map(|i| (0..nz).filter(|k| vol.tsdf[i * slab + j * nz + k] < 0.0 && vol.wt[i * slab + j * nz + k] >= 2.0).count()).sum::<usize>();
        if n >= 60 {
            top = j;
        }
    }
    let j0 = top.saturating_sub((head_h / vol.voxel) as usize);
    let masks: Vec<Vec<bool>> = frames.iter().map(|f| closed_mask(f.depth, 30)).collect();
    let (origin, voxel) = (vol.origin, vol.voxel);
    let inside: Vec<bool> = (0..nx * ny * nz)
        .into_par_iter()
        .map(|id| {
            let (i, j, k) = (id / slab, (id / nz) % ny, id % nz);
            if j < j0 || j > top {
                return false;
            }
            let (vx, vy, vz) = (origin[0] + i as f32 * voxel, origin[1] + j as f32 * voxel, origin[2] + k as f32 * voxel);
            let (mut seen, mut hit) = (0u32, 0u32);
            for (f, m) in poses.iter().zip(&masks) {
                let (c, s) = (f[0].cos() as f32, f[0].sin() as f32);
                let (dx, dz) = (vx - f[1] as f32, vz - f[2] as f32);
                let pc = world.to_camera([c * dx + s * dz, vy, -s * dx + c * dz]);
                if pc[2] <= 0.3 {
                    continue;
                }
                let u = (FX * pc[0] / pc[2] + CX).round();
                let v = (FY * pc[1] / pc[2] + CY).round();
                if u < 0.0 || v < 0.0 || u >= W as f32 || v >= H as f32 {
                    continue;
                }
                seen += 1;
                if m[v as usize * W + u as usize] {
                    hit += 1;
                }
            }
            seen >= 20 && hit as f32 >= min_frac * seen as f32
        })
        .collect();
    let mut solid: Vec<bool> = (0..nx * ny * nz).map(|id| vol.tsdf[id] < 0.0 && vol.wt[id] >= 2.0).collect();
    let mut added = 0;
    for _ in 0..(reach / voxel).ceil() as usize {
        let mut grow = Vec::new();
        for i in 1..nx - 1 {
            for j in j0.max(1)..=top.min(ny - 2) {
                for k in 1..nz - 1 {
                    let id = i * slab + j * nz + k;
                    if solid[id] || !inside[id] {
                        continue;
                    }
                    if solid[id - slab] || solid[id + slab] || solid[id - nz] || solid[id + nz] || solid[id - 1] || solid[id + 1] {
                        grow.push(id);
                    }
                }
            }
        }
        for id in grow {
            solid[id] = true;
            vol.tsdf[id] = -0.3;
            vol.wt[id] = vol.wt[id].max(2.5);
            added += 1;
        }
    }
    let _ = center;
    added
}

/// Make the head left-right symmetric: find its sagittal plane (the long axis of the head's
/// footprint, through its centroid) and make every head voxel solid if its mirror image is.
pub fn symmetrize_head(vol: &mut Volume, head_h: f32) -> usize {
    let [nx, ny, nz] = vol.dims;
    let slab = ny * nz;
    let solid = |v: &Volume, id: usize| v.tsdf[id] < 0.0 && v.wt[id] >= 2.0;
    let mut top = 0;
    for j in 0..ny {
        let n = (0..nx).map(|i| (0..nz).filter(|k| solid(vol, i * slab + j * nz + k)).count()).sum::<usize>();
        if n >= 60 {
            top = j;
        }
    }
    let vox = vol.voxel;
    let (jl, jh) = (top.saturating_sub((head_h / vox) as usize), top.saturating_sub((0.03 / vox) as usize));
    // footprint of the head proper (skull band, below the hair tuft)
    let (mut n, mut sx, mut sz) = (0.0f64, 0.0f64, 0.0f64);
    let (jb0, jb1) = (top.saturating_sub((0.17 / vox) as usize), top.saturating_sub((0.05 / vox) as usize));
    for i in 0..nx {
        for j in jb0..=jb1 {
            for k in 0..nz {
                if solid(vol, i * slab + j * nz + k) {
                    n += 1.0;
                    sx += i as f64;
                    sz += k as f64;
                }
            }
        }
    }
    if n < 200.0 {
        return 0;
    }
    let (cx, cz) = (sx / n, sz / n);
    let (mut sxx, mut szz, mut sxz) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..nx {
        for j in jb0..=jb1 {
            for k in 0..nz {
                if solid(vol, i * slab + j * nz + k) {
                    let (a, b) = (i as f64 - cx, k as f64 - cz);
                    sxx += a * a;
                    szz += b * b;
                    sxz += a * b;
                }
            }
        }
    }
    // minor axis of the footprint = left-right direction
    let ang = 0.5 * (2.0 * sxz).atan2(sxx - szz); // major axis angle
    let (nxl, nzl) = (-ang.sin(), ang.cos()); // lateral unit normal (perpendicular to the major axis)
    let mut add = Vec::new();
    for i in 0..nx {
        for j in jl..=jh {
            for k in 0..nz {
                let id = i * slab + j * nz + k;
                if solid(vol, id) {
                    let d = (i as f64 - cx) * nxl + (k as f64 - cz) * nzl;
                    let (mi, mk) = ((i as f64 - 2.0 * d * nxl).round(), (k as f64 - 2.0 * d * nzl).round());
                    if mi >= 0.0 && mk >= 0.0 && (mi as usize) < nx && (mk as usize) < nz {
                        let mid = mi as usize * slab + j * nz + mk as usize;
                        if !solid(vol, mid) {
                            add.push(mid);
                        }
                    }
                }
            }
        }
    }
    let n_add = add.len();
    for id in add {
        vol.tsdf[id] = -0.3;
        vol.wt[id] = vol.wt[id].max(2.5);
    }
    n_add
}
