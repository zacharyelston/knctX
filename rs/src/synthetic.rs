//! Fake Kinect: a procedural person spinning on the spot, with floor and back wall.
use crate::math::*;
use crate::scan::world::{CX, CY, FX, FY};
use crate::source::{Source, Store, H, W};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const CAM_H: f32 = 1.0;
const PITCH_DEG: f32 = -2.5;
const PERSON_Z: f32 = -3.0;
const WALL_Z: f32 = -4.8;

type Part = (Vec<V3>, Vec<[f32; 3]>);

fn sphere(rng: &mut Rng) -> V3 { normalize([rng.normal(), rng.normal(), rng.normal()]) }

fn ellipsoid(rng: &mut Rng, c: V3, r: V3, n: usize, col: [f32; 3]) -> Part {
    let pts = (0..n).map(|_| { let u = sphere(rng); [c[0] + u[0] * r[0], c[1] + u[1] * r[1], c[2] + u[2] * r[2]] }).collect();
    (pts, vec![col; n])
}

fn capsule(rng: &mut Rng, a: V3, b: V3, ra: f32, rb: f32, n: usize, col: [f32; 3]) -> Part {
    let axis = normalize(sub(b, a));
    let pts = (0..n)
        .map(|_| {
            let t = rng.f32();
            let mut u = sphere(rng);
            u = sub(u, scale(axis, dot(u, axis)));
            u = normalize(u);
            let r = ra + (rb - ra) * t;
            add(add(a, scale(sub(b, a), t)), scale(u, r))
        })
        .collect();
    (pts, vec![col; n])
}

pub fn make_body(seed: u64) -> (Vec<V3>, Vec<[u8; 3]>) {
    let mut rng = Rng::new(seed);
    let (skin, shirt, pants, shoe) = ([225.0, 175.0, 150.0], [200.0, 70.0, 60.0], [50.0, 60.0, 80.0], [40.0, 40.0, 40.0]);
    let mut parts: Vec<Part> = vec![
        ellipsoid(&mut rng, [0.0, 1.18, 0.0], [0.17, 0.27, 0.10], 9000, shirt),
        ellipsoid(&mut rng, [0.0, 0.92, 0.0], [0.16, 0.12, 0.10], 4000, pants),
        ellipsoid(&mut rng, [0.0, 1.63, 0.0], [0.095, 0.115, 0.10], 4000, skin),
        ellipsoid(&mut rng, [0.0, 1.62, 0.115], [0.02, 0.03, 0.03], 400, skin), // nose: breaks symmetry
        capsule(&mut rng, [0.0, 1.45, 0.0], [0.0, 1.52, 0.0], 0.05, 0.05, 600, skin),
        capsule(&mut rng, [-0.14, 1.41, 0.0], [0.14, 1.41, 0.0], 0.06, 0.06, 1500, shirt), // shoulders
    ];
    for s in [1.0f32, -1.0] {
        parts.push(capsule(&mut rng, [s * 0.09, 0.90, 0.0], [s * 0.10, 0.50, 0.01], 0.075, 0.055, 4500, pants));
        parts.push(capsule(&mut rng, [s * 0.10, 0.50, 0.01], [s * 0.10, 0.08, 0.0], 0.055, 0.045, 4000, pants));
        parts.push(ellipsoid(&mut rng, [s * 0.10, 0.04, 0.05], [0.045, 0.04, 0.11], 1500, shoe));
        parts.push(capsule(&mut rng, [s * 0.20, 1.40, 0.0], [s * 0.35, 1.14, 0.02], 0.05, 0.04, 2500, shirt));
        parts.push(capsule(&mut rng, [s * 0.35, 1.14, 0.02], [s * 0.48, 0.915, 0.04], 0.04, 0.032, 2500, skin));
        parts.push(ellipsoid(&mut rng, [s * 0.495, 0.88, 0.045], [0.03, 0.05, 0.02], 500, skin));
    }
    let (mut pts, mut cols) = (Vec::new(), Vec::new());
    for (p, c) in parts {
        pts.extend(p);
        cols.extend(c);
    }
    let out_cols = pts
        .iter()
        .zip(&cols)
        .map(|(p, c)| {
            let stripe = (p[1] * 40.0).sin() > 0.6 && p[0].abs() < 0.2 && p[1] > 1.0 && p[1] < 1.45;
            if stripe {
                [230, 230, 230]
            } else {
                let mut o = [0u8; 3];
                for k in 0..3 {
                    o[k] = (c[k] + rng.normal() * 8.0).clamp(0.0, 255.0) as u8;
                }
                o
            }
        })
        .collect();
    (pts, out_cols)
}

fn background() -> (Vec<u16>, Vec<u8>) {
    let (cg, sg) = (PITCH_DEG.to_radians().cos(), PITCH_DEG.to_radians().sin());
    let mut depth = vec![0u16; W * H];
    let mut rgb = vec![0u8; W * H * 3];
    for v in 0..H {
        for u in 0..W {
            let (rx, ry) = ((u as f32 - CX) / FX, (v as f32 - CY) / FY);
            let (y_c, z_c, x_c) = (ry * cg - sg, ry * sg + cg, rx);
            let (dy_s, dz_s) = (-y_c, -z_c);
            let t_floor = if dy_s < -1e-6 { -CAM_H / dy_s } else { f32::INFINITY };
            let t_wall = if dz_s < -1e-6 { (-WALL_Z) / (-dz_s) } else { f32::INFINITY };
            let t = t_floor.min(t_wall);
            let i = v * W + u;
            if !(t.is_finite() && t < 8.0) {
                continue;
            }
            depth[i] = (t * 1000.0) as u16;
            if t_floor <= t_wall {
                let (xs, zs) = (x_c * t, dz_s * t);
                let tile = (((xs * 2.0).floor() + (zs * 2.0).floor()).rem_euclid(2.0) * 30.0) as i32;
                let g = (110 + tile).clamp(0, 255) as u8;
                rgb[i * 3..i * 3 + 3].copy_from_slice(&[g, g, g]);
            } else {
                rgb[i * 3..i * 3 + 3].copy_from_slice(&[170, 175, 165]);
            }
        }
    }
    (depth, rgb)
}

pub struct SyntheticKinect {
    store: Arc<Store>,
    body: Arc<(Vec<V3>, Vec<[u8; 3]>)>,
    bg: Arc<(Vec<u16>, Vec<u8>)>,
    spt: f32,
    fps: f32,
    noise: f32,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl SyntheticKinect {
    pub fn new(seconds_per_turn: f32, fps: f32, noise_mm: f32) -> Self {
        SyntheticKinect {
            store: Store::new(),
            body: Arc::new(make_body(1)),
            bg: Arc::new(background()),
            spt: seconds_per_turn,
            fps,
            noise: noise_mm,
            thread: Mutex::new(None),
        }
    }
}

/// Render the body at spin angle `theta` over the static background.
pub fn render(body: &(Vec<V3>, Vec<[u8; 3]>), bg: &(Vec<u16>, Vec<u8>), theta: f32, noise: f32, rng: &mut Rng) -> (Vec<u8>, Vec<u16>) {
    let (c, s) = (theta.cos(), theta.sin());
    let (cg, sg) = (PITCH_DEG.to_radians().cos(), PITCH_DEG.to_radians().sin());
    let mut depth = bg.0.clone();
    let mut rgb = bg.1.clone();
    let mut zbuf = vec![f32::INFINITY; W * H];
    for (i, p) in body.0.iter().enumerate() {
        let (x, y) = (c * p[0] + s * p[2], p[1]);
        let z = -s * p[0] + c * p[2] + PERSON_Z;
        let (xl, yl, zl) = (x, -(y - CAM_H), -z);
        let (xc, yc, zc) = (xl, yl * cg + zl * sg, -yl * sg + zl * cg);
        if zc < 0.3 {
            continue;
        }
        let u = (FX * xc / zc + CX).round() as i32;
        let v = (FY * yc / zc + CY).round() as i32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (uu, vv) = (u + dx, v + dy);
                if uu < 0 || uu >= W as i32 || vv < 0 || vv >= H as i32 {
                    continue;
                }
                let idx = vv as usize * W + uu as usize;
                let bgd = bg.0[idx];
                if zc < zbuf[idx] && (bgd == 0 || zc * 1000.0 < bgd as f32) {
                    zbuf[idx] = zc;
                    depth[idx] = (zc * 1000.0) as u16;
                    rgb[idx * 3..idx * 3 + 3].copy_from_slice(&body.1[i]);
                }
            }
        }
    }
    if noise > 0.0 {
        for (i, d) in depth.iter_mut().enumerate() {
            *d = if bg.0[i] == 0 { 0 } else { (*d as f32 + rng.normal() * noise).clamp(0.0, 65000.0) as u16 };
        }
    }
    (rgb, depth)
}

impl Source for SyntheticKinect {
    fn store(&self) -> &Arc<Store> { &self.store }
    fn start(&self) {
        let (store, body, bg, spt, fps, noise) = (self.store.clone(), self.body.clone(), self.bg.clone(), self.spt, self.fps, self.noise);
        *self.thread.lock().unwrap() = Some(std::thread::spawn(move || {
            let t0 = Instant::now();
            let mut rng = Rng::new(3);
            while store.running.load(Ordering::SeqCst) {
                let tic = Instant::now();
                let theta = std::f32::consts::TAU * (tic - t0).as_secs_f32() / spt;
                let (rgb, depth) = render(&body, &bg, theta, noise, &mut rng);
                store.put_both(rgb, depth);
                let left = Duration::from_secs_f32(1.0 / fps).saturating_sub(tic.elapsed());
                std::thread::sleep(left);
            }
        }));
    }
    fn stop(&self) {
        self.store.running.store(false, Ordering::SeqCst);
        if let Some(t) = self.thread.lock().unwrap().take() {
            let _ = t.join();
        }
    }
}

/// The static floor + wall background (for headless tests).
pub fn background_pub() -> (Vec<u16>, Vec<u8>) { background() }
