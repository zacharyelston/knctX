//! Software triangle rasteriser for the posed figure (z-buffered, back-face culled, headlight shading).
use crate::gfx::Canvas;
use crate::math::*;

pub struct Orbit3D {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub drag: Option<(f32, f32)>,
}

impl Orbit3D {
    pub fn new() -> Self { Orbit3D { yaw: 0.3, pitch: 0.1, zoom: 1.0, drag: None } }
    pub fn reset(&mut self) { self.yaw = 0.3; self.pitch = 0.1; self.zoom = 1.0; }
    pub fn mouse(&mut self, pos: Option<(f32, f32)>, down: bool, wheel: f32) {
        match (pos, down) {
            (Some(p), true) => {
                if let Some(d) = self.drag {
                    self.yaw += (p.0 - d.0) * 0.01;
                    self.pitch = (self.pitch + (p.1 - d.1) * 0.01).clamp(-1.3, 1.3);
                }
                self.drag = Some(p);
            }
            _ => self.drag = None,
        }
        if wheel != 0.0 {
            self.zoom = (self.zoom * if wheel > 0.0 { 0.9 } else { 1.1 }).clamp(0.5, 2.5);
        }
    }
}

pub fn render_mesh(c: &mut Canvas, verts: &[V3], normals: &[V3], faces: &[[u32; 3]], rgb: &[[u8; 3]], orbit: &Orbit3D, top: f32) {
    let (wc, hc) = (c.w as i32, c.h as i32);
    for y in 0..c.h {
        let g = (46.0 + (20.0 - 46.0) * y as f32 / (c.h - 1) as f32) as u8;
        let p = crate::gfx::pack([g, g, g]);
        c.buf[y * c.w..(y + 1) * c.w].iter_mut().for_each(|q| *q = p);
    }
    let f = 0.95 * c.h as f32;
    let n = verts.len().max(1) as f32;
    let focus = [verts.iter().map(|v| v[0]).sum::<f32>() / n, 0.5 * top, verts.iter().map(|v| v[2]).sum::<f32>() / n];
    let (cy, sy, cp, sp) = (orbit.yaw.cos(), orbit.yaw.sin(), orbit.pitch.cos(), orbit.pitch.sin());
    let ry: M3 = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rx: M3 = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]];
    let rv = mat_mul(&rx, &ry);
    let dist = 3.3 * top / 1.75 / orbit.zoom;
    let project = |p: V3| -> (f32, f32, f32) {
        let q = mat_vec(&rv, sub(p, focus));
        let depth = dist - q[2];
        let d = depth.max(0.05);
        (wc as f32 / 2.0 + f * q[0] / d, hc as f32 / 2.0 - f * q[1] / d, depth)
    };
    // floor ring for context
    let mut prev: Option<(i32, i32)> = None;
    for i in 0..=72 {
        let a = i as f32 / 72.0 * std::f32::consts::TAU;
        let (x, y, _) = project([focus[0] + 0.7 * a.cos(), 0.0, focus[2] + 0.7 * a.sin()]);
        let cur = (x as i32, y as i32);
        if let Some(p) = prev {
            c.line(p.0, p.1, cur.0, cur.1, [90, 90, 90]);
        }
        prev = Some(cur);
    }
    let proj: Vec<(f32, f32, f32)> = verts.iter().map(|v| project(*v)).collect();
    let nv: Vec<V3> = normals.iter().map(|n| mat_vec(&rv, *n)).collect();
    let mut zbuf = vec![f32::INFINITY; c.w * c.h];
    for fc in faces {
        let (a, b, d) = (fc[0] as usize, fc[1] as usize, fc[2] as usize);
        let fnv = normalize(add(add(nv[a], nv[b]), nv[d]));
        let facing = fnv[2];
        if facing <= 0.02 {
            continue;
        }
        let (p0, p1, p2) = (proj[a], proj[b], proj[d]);
        let minx = p0.0.min(p1.0).min(p2.0).floor().max(0.0) as i32;
        let maxx = (p0.0.max(p1.0).max(p2.0).ceil() as i32).min(wc - 1);
        let miny = p0.1.min(p1.1).min(p2.1).floor().max(0.0) as i32;
        let maxy = (p0.1.max(p1.1).max(p2.1).ceil() as i32).min(hc - 1);
        if minx > maxx || miny > maxy {
            continue;
        }
        let den = (p1.1 - p2.1) * (p0.0 - p2.0) + (p2.0 - p1.0) * (p0.1 - p2.1);
        if den.abs() < 1e-6 {
            continue;
        }
        let sh = 0.35 + 0.65 * facing;
        let col = {
            let mut o = [0u8; 3];
            for k in 0..3 {
                o[k] = (((rgb[a][k] as f32 + rgb[b][k] as f32 + rgb[d][k] as f32) / 3.0) * sh).min(255.0) as u8;
            }
            o
        };
        let packed = crate::gfx::pack(col);
        for y in miny..=maxy {
            for x in minx..=maxx {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let l0 = ((p1.1 - p2.1) * (px - p2.0) + (p2.0 - p1.0) * (py - p2.1)) / den;
                let l1 = ((p2.1 - p0.1) * (px - p2.0) + (p0.0 - p2.0) * (py - p2.1)) / den;
                let l2 = 1.0 - l0 - l1;
                if l0 >= -0.05 && l1 >= -0.05 && l2 >= -0.05 {
                    let z = l0 * p0.2 + l1 * p1.2 + l2 * p2.2;
                    let i = y as usize * c.w + x as usize;
                    if z < zbuf[i] {
                        zbuf[i] = z;
                        c.buf[i] = packed;
                    }
                }
            }
        }
    }
}
