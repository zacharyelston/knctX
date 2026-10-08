//! Find the nearest person-sized object and mask it (floor and background removed).
use super::world::*;
use crate::math::*;
use crate::source::{H, W};

pub struct Segmenter {
    pub world: World,
    pub center: Option<[f32; 2]>, // (x, z) world metres
    pub radius: f32,
    /// Height gate (world y, metres) applied by `mask`.
    pub y_lo: f32,
    pub y_hi: f32,
    /// Smallest detached piece (as a fraction of the main blob) that is still kept, e.g. a head.
    pub min_comp_frac: f32,
    pub max_extra: usize,
}

impl Segmenter {
    pub fn new(world: World) -> Self { Segmenter { world, center: None, radius: 0.95, y_lo: 0.08, y_hi: 2.3, min_comp_frac: 0.03, max_extra: 5 } }

    /// Return (center_xz, height_m) of the nearest large upright object, or None.
    pub fn find_person(&mut self, depth: &[u16]) -> Option<([f32; 2], f32)> {
        let mut pw: Vec<V3> = Vec::new();
        let mut zc: Vec<f32> = Vec::new();
        for v in (0..H).step_by(2) {
            for u in (0..W).step_by(2) {
                let z = depth[v * W + u] as f32 / 1000.0;
                if z > 1.0 && z < 4.5 {
                    let p = self.world.to_world(cam_point(u, v, z));
                    if p[1] > 0.25 && p[1] < 2.1 {
                        pw.push(p);
                        zc.push(z);
                    }
                }
            }
        }
        if pw.len() < 1200 {
            return None;
        }
        let mut lo = 1.0f32;
        while lo < 4.4 {
            let n_win = zc.iter().filter(|&&z| z >= lo && z < lo + 0.15).count();
            if n_win >= 450 {
                let (mut xs, mut zs) = (Vec::new(), Vec::new());
                for (p, &z) in pw.iter().zip(&zc) {
                    if z >= lo - 0.05 && z < lo + 0.6 {
                        xs.push(p[0]);
                        zs.push(p[2]);
                    }
                }
                let c = [median(&mut xs), median(&mut zs)];
                let mut ys: Vec<f32> = pw.iter().filter(|p| (p[0] - c[0]).abs() < 0.6 && (p[2] - c[1]).abs() < 0.6).map(|p| p[1]).collect();
                let height = if ys.is_empty() { 0.0 } else { percentile(&mut ys, 99.0) };
                if height >= 1.0 {
                    self.center = Some(c);
                    return Some((c, height));
                }
            }
            lo += 0.05;
        }
        None
    }

    /// Settings for seated close-ups: small detached pieces (the head behind a depth step) matter.
    pub fn bust_defaults(&mut self) {
        self.radius = 0.45;
        self.min_comp_frac = 0.002;
        self.max_extra = 12;
    }

    /// Seated close-up: find the nearest large blob 0.5-1.9 m away. Sets the centre, a 0.45 m
    /// radius and a height gate covering the top `bust_h` metres of the subject (head down to
    /// mid-chest). Returns (centre_xz, top_y).
    pub fn find_bust(&mut self, depth: &[u16], bust_h: f32) -> Option<([f32; 2], f32)> {
        let mut pw: Vec<V3> = Vec::new();
        let mut zc: Vec<f32> = Vec::new();
        for v in (0..H).step_by(2) {
            for u in (100..540).step_by(2) {
                let z = depth[v * W + u] as f32 / 1000.0;
                if z > 0.5 && z < 1.9 {
                    pw.push(self.world.to_world(cam_point(u, v, z)));
                    zc.push(z);
                }
            }
        }
        if pw.len() < 1500 {
            return None;
        }
        let mut lo = 0.5f32;
        while lo < 1.8 {
            let n_win = zc.iter().filter(|&&z| z >= lo && z < lo + 0.15).count();
            if n_win >= 900 {
                let (mut xs, mut zs) = (Vec::new(), Vec::new());
                for (p, &z) in pw.iter().zip(&zc) {
                    if z >= lo - 0.05 && z < lo + 0.45 {
                        xs.push(p[0]);
                        zs.push(p[2]);
                    }
                }
                let c = [median(&mut xs), median(&mut zs)];
                let mut ys: Vec<f32> = pw.iter().filter(|p| (p[0] - c[0]).abs() < 0.45 && (p[2] - c[1]).abs() < 0.45).map(|p| p[1]).collect();
                if ys.len() < 900 {
                    lo += 0.05;
                    continue;
                }
                let top = percentile(&mut ys, 99.5);
                self.center = Some(c);
                self.bust_defaults();
                self.y_hi = top + 0.05;
                self.y_lo = top - bust_h;
                return Some((c, top));
            }
            lo += 0.05;
        }
        None
    }

    /// Mask (W*H bool) of the tracked person plus per-pixel world points.
    ///
    /// Pixels are gated by height above the floor and distance from the person, then cut along
    /// depth discontinuities so chairs/desks behind them don't join the person.
    pub fn mask(&mut self, depth: &[u16]) -> (Vec<bool>, Vec<V3>) {
        let mut pw = vec![[0.0f32; 3]; W * H];
        let mut m = vec![false; W * H];
        for v in 0..H {
            for u in 0..W {
                let i = v * W + u;
                let d = depth[i];
                if d == 0 {
                    continue;
                }
                let p = self.world.to_world(cam_point(u, v, d as f32 / 1000.0));
                pw[i] = p;
                let mut ok = p[1] > self.y_lo && p[1] < self.y_hi;
                if let Some(c) = self.center {
                    ok &= (p[0] - c[0]).abs() < self.radius && (p[2] - c[1]).abs() < self.radius;
                }
                m[i] = ok;
            }
        }
        // cut along depth discontinuities
        let mut cut = vec![false; W * H];
        for v in 0..H {
            for u in 0..W {
                let i = v * W + u;
                if u + 1 < W && (depth[i] as i32 - depth[i + 1] as i32).abs() > 50 {
                    cut[i] = true;
                    cut[i + 1] = true;
                }
                if v + 1 < H && (depth[i] as i32 - depth[i + W] as i32).abs() > 50 {
                    cut[i] = true;
                    cut[i + W] = true;
                }
            }
        }
        let base: Vec<bool> = m.iter().zip(&cut).map(|(a, b)| *a && !*b).collect();
        // 3x3 erosion (pixels outside the image do not erode)
        let mut core = vec![false; W * H];
        for v in 0..H {
            for u in 0..W {
                let mut all = true;
                'o: for dv in -1i32..=1 {
                    for du in -1i32..=1 {
                        let (vv, uu) = (v as i32 + dv, u as i32 + du);
                        if vv < 0 || uu < 0 || vv >= H as i32 || uu >= W as i32 {
                            continue;
                        }
                        if !base[vv as usize * W + uu as usize] {
                            all = false;
                            break 'o;
                        }
                    }
                }
                core[v * W + u] = all;
            }
        }
        let (lab, areas) = components(&core);
        let mut keep = vec![false; W * H];
        if !areas.is_empty() {
            let mut order: Vec<usize> = (0..areas.len()).collect();
            order.sort_by(|a, b| areas[*b].cmp(&areas[*a]));
            let big = order[0];
            let mut idx = 0usize;
            let (mut sx, mut sz, mut cnt) = (0.0f32, 0.0f32, 0usize);
            for i in 0..W * H {
                if lab[i] == big as i32 {
                    keep[i] = true;
                    if idx % 7 == 0 {
                        sx += pw[i][0];
                        sz += pw[i][2];
                        cnt += 1;
                    }
                    idx += 1;
                }
            }
            let cb = [sx / cnt.max(1) as f32, sz / cnt.max(1) as f32];
            for &ci in order.iter().skip(1).take(self.max_extra) {
                if (areas[ci] as f32) < self.min_comp_frac * areas[big] as f32 {
                    break;
                }
                let (mut ax, mut az, mut n, mut idx) = (0.0f32, 0.0f32, 0usize, 0usize);
                for i in 0..W * H {
                    if lab[i] == ci as i32 {
                        if idx % 5 == 0 {
                            ax += pw[i][0];
                            az += pw[i][2];
                            n += 1;
                        }
                        idx += 1;
                    }
                }
                let cc = [ax / n.max(1) as f32, az / n.max(1) as f32];
                if ((cc[0] - cb[0]).powi(2) + (cc[1] - cb[1]).powi(2)).sqrt() < 0.5 {
                    for i in 0..W * H {
                        if lab[i] == ci as i32 {
                            keep[i] = true;
                        }
                    }
                }
            }
        }
        // Win back thin features (hands, fingers) that the depth-edge cut and erosion removed:
        // grow into gated pixels whose depth continues smoothly from an already-kept neighbour.
        for _ in 0..3 {
            let mut add = Vec::new();
            for v in 1..H - 1 {
                for u in 1..W - 1 {
                    let i = v * W + u;
                    if keep[i] || !m[i] {
                        continue;
                    }
                    let d = depth[i] as i32;
                    for j in [i - 1, i + 1, i - W, i + W] {
                        if keep[j] && (depth[j] as i32 - d).abs() <= 30 {
                            add.push(i);
                            break;
                        }
                    }
                }
            }
            for i in add {
                keep[i] = true;
            }
        }
        let total = keep.iter().filter(|b| **b).count();
        if total > 500 {
            let (mut sx, mut sz, mut n, mut idx) = (0.0f32, 0.0f32, 0usize, 0usize);
            for i in 0..W * H {
                if keep[i] {
                    if idx % 7 == 0 {
                        sx += pw[i][0];
                        sz += pw[i][2];
                        n += 1;
                    }
                    idx += 1;
                }
            }
            let c = [sx / n as f32, sz / n as f32];
            self.center = Some(match self.center {
                None => c,
                Some(o) => [0.8 * o[0] + 0.2 * c[0], 0.8 * o[1] + 0.2 * c[1]],
            });
        }
        (keep, pw)
    }
}

/// 8-connected components of a boolean image: (label per pixel or -1, area per label).
pub fn components(img: &[bool]) -> (Vec<i32>, Vec<usize>) {
    let mut lab = vec![-1i32; W * H];
    let mut areas = Vec::new();
    let mut stack = Vec::new();
    for s in 0..W * H {
        if !img[s] || lab[s] >= 0 {
            continue;
        }
        let id = areas.len() as i32;
        let mut area = 0;
        lab[s] = id;
        stack.push(s);
        while let Some(i) = stack.pop() {
            area += 1;
            let (u, v) = ((i % W) as i32, (i / W) as i32);
            for dv in -1i32..=1 {
                for du in -1i32..=1 {
                    let (uu, vv) = (u + du, v + dv);
                    if uu < 0 || vv < 0 || uu >= W as i32 || vv >= H as i32 {
                        continue;
                    }
                    let j = vv as usize * W + uu as usize;
                    if img[j] && lab[j] < 0 {
                        lab[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
        areas.push(area);
    }
    (lab, areas)
}
