//! Xbox 360 Kinect viewer: RGB scene with a colorized depth map overlaid.
//!
//! Uses libfreenect's registered depth (aligned to the RGB camera, in mm).
//!
//! Keys:  + / - opacity   [ / ] near/far limit   c colormap   v view   e edges   g grid
//!        m mirror   Up/Down tilt   s snapshot   p point cloud (.ply)   r record   0 reset 3d   q/Esc quit
//! 3d view: drag to orbit, scroll to zoom.   Replay: kinect_overlay --play recordings/<timestamp>
use kinect3d::gfx::*;
use kinect3d::io;
use kinect3d::scan::capture::timestamp;
use kinect3d::scan::world::{CX, CY, FX, FY};
use kinect3d::source::*;
use minifb::{InputCallback, Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const COLORMAPS: [&str; 5] = ["TURBO", "JET", "INFERNO", "VIRIDIS", "BONE"];
const GRIDS: [Option<(usize, usize)>; 5] = [None, Some((3, 3)), Some((5, 4)), Some((8, 6)), Some((12, 9))];
const VIEWS: [&str; 5] = ["overlay", "side-by-side", "depth", "rgb", "3d"];
const PANEL_W: usize = 200;

fn poly(c: [[f32; 3]; 7], t: f32) -> [f32; 3] {
    let mut o = [0.0; 3];
    for k in 0..3 {
        let mut v = c[6][k];
        for i in (0..6).rev() {
            v = v * t + c[i][k];
        }
        o[k] = v;
    }
    o
}

fn lerp_table(pts: &[(f32, f32)], x: f32) -> f32 {
    for w in pts.windows(2) {
        if x <= w[1].0 {
            return w[0].1 + (w[1].1 - w[0].1) * (x - w[0].0) / (w[1].0 - w[0].0);
        }
    }
    pts[pts.len() - 1].1
}

/// 256-entry colormap table (RGB).
fn colormap(name: &str) -> Vec<Col> {
    (0..256)
        .map(|i| {
            let t = i as f32 / 255.0;
            let rgb = match name {
                "TURBO" => [
                    0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))),
                    0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))),
                    0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))),
                ],
                "JET" => {
                    let f = |c: f32| (1.5 - (4.0 * t - c).abs()).clamp(0.0, 1.0);
                    [f(3.0), f(2.0), f(1.0)]
                }
                "INFERNO" => poly(
                    [[0.0002189, 0.001651, -0.019481], [0.106513, 0.563956, 3.932712], [11.602493, -3.972854, -15.942394], [-41.703996, 17.436399, 44.354145],
                     [77.162936, -33.402359, -81.807309], [-71.319428, 32.626064, 73.20952], [25.131126, -12.242669, -23.070325]], t),
                "VIRIDIS" => poly(
                    [[0.277727, 0.005407, 0.3341], [0.105093, 1.404614, 1.38459], [-0.330862, 0.214848, 0.095095], [-4.634230, -5.799101, -19.332441],
                     [6.228270, 14.179933, 56.690553], [4.776385, -13.745145, -65.353033], [-5.435456, 4.645853, 26.312435]], t),
                _ => [
                    lerp_table(&[(0.0, 0.0), (0.746, 0.652), (1.0, 1.0)], t),
                    lerp_table(&[(0.0, 0.0), (0.365, 0.319), (0.746, 0.778), (1.0, 1.0)], t),
                    lerp_table(&[(0.0, 0.0), (0.365, 0.444), (1.0, 1.0)], t),
                ],
            };
            [0, 1, 2].map(|k| (rgb[k].clamp(0.0, 1.0) * 255.0) as u8)
        })
        .collect()
}

/// Depth (mm) -> RGB (near = hot); invalid pixels are flagged in `valid`.
fn colorize(depth: &[u16], near: f32, far: f32, lut: &[Col], out: &mut [Col], valid: &mut [bool]) {
    for (i, &d) in depth.iter().enumerate() {
        valid[i] = d > 0 && d < 10000;
        let dd = (d as f32).clamp(near, far);
        let norm = ((far - dd) / (far - near).max(1.0) * 255.0) as usize;
        out[i] = if valid[i] { lut[norm.min(255)] } else { [0, 0, 0] };
    }
}

/// Gaussian-ish blur + Sobel gradient edges, dilated: a cheap stand-in for Canny.
fn edge_mask(depth: &[u16], near: f32, far: f32) -> Vec<bool> {
    let norm: Vec<f32> = depth.iter().map(|&d| (far - (d as f32).clamp(near, far)) / (far - near).max(1.0) * 255.0).collect();
    let blur = |src: &[f32]| -> Vec<f32> {
        let mut tmp = vec![0.0; W * H];
        let mut out = vec![0.0; W * H];
        let k = [1.0, 4.0, 6.0, 4.0, 1.0];
        for y in 0..H {
            for x in 0..W {
                let mut s = 0.0;
                for (i, w) in k.iter().enumerate() {
                    s += w * src[y * W + (x + i).saturating_sub(2).min(W - 1)];
                }
                tmp[y * W + x] = s / 16.0;
            }
        }
        for y in 0..H {
            for x in 0..W {
                let mut s = 0.0;
                for (i, w) in k.iter().enumerate() {
                    s += w * tmp[((y + i).saturating_sub(2)).min(H - 1) * W + x];
                }
                out[y * W + x] = s / 16.0;
            }
        }
        out
    };
    let b = blur(&norm);
    let mut e = vec![false; W * H];
    for y in 1..H - 1 {
        for x in 1..W - 1 {
            let g = |dy: i32, dx: i32| b[((y as i32 + dy) as usize) * W + (x as i32 + dx) as usize];
            let gx = (g(-1, 1) + 2.0 * g(0, 1) + g(1, 1)) - (g(-1, -1) + 2.0 * g(0, -1) + g(1, -1));
            let gy = (g(1, -1) + 2.0 * g(1, 0) + g(1, 1)) - (g(-1, -1) + 2.0 * g(-1, 0) + g(-1, 1));
            e[y * W + x] = (gx * gx + gy * gy).sqrt() > 60.0;
        }
    }
    let mut d = e.clone();
    for y in 1..H - 1 {
        for x in 1..W - 1 {
            if e[y * W + x] {
                for dy in 0..3 {
                    for dx in 0..3 {
                        d[(y + dy - 1) * W + x + dx - 1] = true;
                    }
                }
            }
        }
    }
    d
}

fn draw_grid(c: &mut Canvas, depth: &[u16], mirror: bool, cols: usize, rows: usize, x0: usize, pw: usize) {
    let ph = c.h;
    for col in 1..cols {
        let x = (x0 + col * pw / cols) as i32;
        c.line_blend(x, 0, x, ph as i32, WHITE, 0.35);
    }
    for r in 1..rows {
        let y = (r * ph / rows) as i32;
        c.line_blend(x0 as i32, y, (x0 + pw) as i32, y, WHITE, 0.35);
    }
    let scale = if cols <= 8 { 0.5 } else { 0.38 };
    for r in 0..rows {
        for col in 0..cols {
            let mut v: Vec<f32> = Vec::new();
            for y in r * H / rows..(r + 1) * H / rows {
                for x in col * W / cols..(col + 1) * W / cols {
                    let sx = if mirror { W - 1 - x } else { x };
                    let d = depth[y * W + sx];
                    if d > 0 {
                        v.push(d as f32);
                    }
                }
            }
            let txt = if v.len() > 20 { format!("{:.2}m", kinect3d::math::median(&mut v) / 1000.0) } else { "--".into() };
            let (tw, th) = (text_width(&txt, scale), (px_for(scale) * 0.72) as i32);
            let cx = (x0 + (2 * col + 1) * pw / (2 * cols)) as i32;
            let cy = ((2 * r + 1) * ph / (2 * rows)) as i32;
            c.rect(cx - tw / 2 - 3, cy - th / 2 - 3, cx + tw / 2 + 3, cy + th / 2 + 3, [0, 0, 0]);
            c.text(&txt, cx - tw / 2, cy + th / 2, scale, WHITE, false);
        }
    }
}

fn side_panel(c: &mut Canvas, x0: usize, title: &str, rows: &[(&str, String)]) {
    c.rect(x0 as i32, 0, (x0 + PANEL_W) as i32, c.h as i32, [28, 28, 28]);
    c.text(title, x0 as i32 + 12, 28, 0.6, bgr(255, 200, 120), false);
    c.line(x0 as i32 + 12, 38, (x0 + PANEL_W) as i32 - 12, 38, bgr(80, 80, 80));
    let mut y = 66;
    for (k, d) in rows {
        if k.is_empty() {
            y += 12;
            continue;
        }
        c.text(k, x0 as i32 + 12, y, 0.55, WHITE, false);
        c.text(d, x0 as i32 + 12, y + 17, 0.42, bgr(170, 170, 170), false);
        y += 40;
    }
}

struct Orbit {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    pz: Option<f32>,
    drag: Option<(f32, f32)>,
}

fn render_cloud(c: &mut Canvas, rgb: &[u8], depth: &[u16], cam: &mut Orbit, step: usize) {
    c.fill([18, 18, 18]);
    let mut pts: Vec<([f32; 3], [u8; 3])> = Vec::new();
    for v in (0..H).step_by(step) {
        for u in (0..W).step_by(step) {
            let d = depth[v * W + u] as f32 / 1000.0;
            if d > 0.3 && d < 8.0 {
                pts.push(([(u as f32 - CX) * d / FX, (v as f32 - CY) * d / FY, d], [rgb[(v * W + u) * 3], rgb[(v * W + u) * 3 + 1], rgb[(v * W + u) * 3 + 2]]));
            }
        }
    }
    if pts.is_empty() {
        return;
    }
    let mut zs: Vec<f32> = pts.iter().map(|p| p.0[2]).collect();
    let med = kinect3d::math::median(&mut zs);
    let pz = cam.pz.map_or(med, |p| 0.9 * p + 0.1 * med);
    cam.pz = Some(pz);
    let (cy, sy, cp, sp) = (cam.yaw.cos(), cam.yaw.sin(), cam.pitch.cos(), cam.pitch.sin());
    let mut zbuf = vec![f32::INFINITY; c.w * c.h];
    for (p, col) in &pts {
        let q = [p[0], p[1], p[2] - pz];
        let a = [cy * q[0] + sy * q[2], q[1], -sy * q[0] + cy * q[2]];
        let b = [a[0], cp * a[1] - sp * a[2], sp * a[1] + cp * a[2]];
        let zc = b[2] + pz * cam.zoom;
        if zc <= 0.1 {
            continue;
        }
        let (sx, sy_) = ((CX + FX * b[0] / zc) as i32, (CY + FY * b[1] / zc) as i32);
        for dy in 0..=step as i32 {
            for dx in 0..=step as i32 {
                let (x, y) = (sx + dx, sy_ + dy);
                if x >= 0 && y >= 0 && (x as usize) < c.w && (y as usize) < c.h {
                    let i = y as usize * c.w + x as usize;
                    if zc < zbuf[i] {
                        zbuf[i] = zc;
                        c.buf[i] = pack(*col);
                    }
                }
            }
        }
    }
}

struct Recorder {
    dir: PathBuf,
    tx: Option<std::sync::mpsc::SyncSender<(usize, Rgb, Depth)>>,
    handle: Option<std::thread::JoinHandle<()>>,
    count: usize,
    times: Vec<f64>,
    t0: Instant,
}

impl Recorder {
    fn new(root: &std::path::Path) -> Recorder {
        let dir = root.join(timestamp("%Y%m%d_%H%M%S"));
        let _ = std::fs::create_dir_all(&dir);
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Rgb, Depth)>(90);
        let d2 = dir.clone();
        let handle = std::thread::spawn(move || {
            while let Ok((i, rgb, depth)) = rx.recv() {
                io::save_rgb_jpg(&d2.join(format!("rgb_{i:05}.jpg")), &rgb);
                io::save_depth_png(&d2.join(format!("depth_{i:05}.png")), &depth);
            }
        });
        Recorder { dir, tx: Some(tx), handle: Some(handle), count: 0, times: vec![], t0: Instant::now() }
    }
    fn add(&mut self, rgb: &Rgb, depth: &Depth) {
        if let Some(tx) = &self.tx {
            if tx.try_send((self.count, rgb.clone(), depth.clone())).is_ok() {
                self.times.push(self.t0.elapsed().as_secs_f64());
                self.count += 1;
            }
        }
    }
    fn stop(mut self) -> (PathBuf, usize) {
        self.tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let _ = std::fs::write(self.dir.join("timestamps.json"), serde_json::to_string(&self.times).unwrap());
        (self.dir, self.count)
    }
}

struct Chars(Arc<Mutex<Vec<char>>>);
impl InputCallback for Chars {
    fn add_char(&mut self, c: u32) {
        if let Some(c) = char::from_u32(c) {
            self.0.lock().unwrap().push(c);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let play = args.iter().position(|a| a == "--play").and_then(|i| args.get(i + 1).cloned());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let src: Arc<dyn Source> = match &play {
        Some(p) => Arc::new(Player::new(p).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1)
        })),
        None => Arc::new(Kinect::new()),
    };
    src.start();
    let (mut alpha, mut near, mut far) = (0.5f32, 500.0f32, 4000.0f32);
    let (mut cmap_i, mut view_i, mut grid_i) = (0usize, 0usize, 2usize);
    let (mut mirror, mut edges) = (false, false);
    let mut luts: Vec<Vec<Col>> = COLORMAPS.iter().map(|n| colormap(n)).collect();
    let mut orbit = Orbit { yaw: 0.0, pitch: 0.0, zoom: 1.0, pz: None, drag: None };
    let mut rec: Option<Recorder> = None;
    let mut last_seq = u64::MAX;
    let (mut fps_t, mut fps_n, mut fps) = (Instant::now(), 0u32, 0.0f32);

    let mut win = Window::new("Kinect RGB + Depth", 640 + 2 * PANEL_W, 480, WindowOptions { resize: true, scale_mode: minifb::ScaleMode::AspectRatioStretch, ..Default::default() }).expect("window");
    win.set_target_fps(30);
    let typed = Arc::new(Mutex::new(Vec::new()));
    win.set_input_callback(Box::new(Chars(typed.clone())));
    let (mut dcol, mut valid) = (vec![[0u8; 3]; W * H], vec![false; W * H]);
    println!("Waiting for frames from Kinect...");
    let mut quit = false;
    while win.is_open() && !quit {
        let chars: Vec<char> = std::mem::take(&mut *typed.lock().unwrap());
        let keys = win.get_keys_pressed(KeyRepeat::Yes);
        if keys.contains(&Key::Escape) {
            break;
        }
        if keys.contains(&Key::Up) {
            src.set_tilt(src.tilt_target() + 5);
        }
        if keys.contains(&Key::Down) {
            src.set_tilt(src.tilt_target() - 5);
        }
        let frames = src.frames();
        let Some((rgb, depth)) = frames else {
            win.update();
            continue;
        };
        for ch in &chars {
            match ch {
                'q' => quit = true,
                '+' | '=' => alpha = (alpha + 0.05).min(1.0),
                '-' | '_' => alpha = (alpha - 0.05).max(0.0),
                '[' => near = (near - 100.0).max(0.0),
                ']' => far = (far + 100.0).min(10000.0),
                'c' => cmap_i = (cmap_i + 1) % COLORMAPS.len(),
                'v' => view_i = (view_i + 1) % VIEWS.len(),
                'g' => grid_i = (grid_i + 1) % GRIDS.len(),
                'e' => edges = !edges,
                'm' => mirror = !mirror,
                '0' => orbit = Orbit { yaw: 0.0, pitch: 0.0, zoom: 1.0, pz: None, drag: None },
                'p' => {
                    let dir = root.join("pointclouds");
                    let _ = std::fs::create_dir_all(&dir);
                    let path = dir.join(format!("kinect_{}.ply", timestamp("%Y%m%d_%H%M%S")));
                    let (mut pts, mut cols) = (Vec::new(), Vec::new());
                    for v in 0..H {
                        for u in 0..W {
                            let d = depth[v * W + u] as f32 / 1000.0;
                            if d > 0.3 && d < 8.0 {
                                pts.push([(u as f32 - CX) * d / FX, -(v as f32 - CY) * d / FY, -d]); // y up, camera looks down -z
                                cols.push([rgb[(v * W + u) * 3], rgb[(v * W + u) * 3 + 1], rgb[(v * W + u) * 3 + 2]]);
                            }
                        }
                    }
                    let _ = io::save_cloud_ply(&path, &pts, &cols);
                    println!("saved {} ({} points)", path.display(), pts.len());
                }
                'r' if play.is_none() => match rec.take() {
                    Some(r) => {
                        let (d, n) = r.stop();
                        println!("recording saved: {} ({n} frames)", d.display());
                    }
                    None => {
                        let r = Recorder::new(&root.join("recordings"));
                        println!("recording to {}", r.dir.display());
                        rec = Some(r);
                    }
                },
                _ => {}
            }
        }
        let seq = src.seq();
        if let (Some(r), true) = (rec.as_mut(), seq != last_seq) {
            r.add(&rgb, &depth);
        }
        last_seq = seq;
        let _ = &mut luts;

        let view = VIEWS[view_i];
        let is3d = view == "3d";
        let pw_ = if view == "side-by-side" { 2 * W } else { W };
        let mut frame = Canvas::new(pw_, H);
        let lut = &luts[cmap_i];
        colorize(&depth, near, far, lut, &mut dcol, &mut valid);
        if edges {
            let em = edge_mask(&depth, near, far);
            for i in 0..W * H {
                valid[i] = em[i];
                if !em[i] {
                    dcol[i] = [0, 0, 0];
                }
            }
        }
        let rgb_at = |i: usize| [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]];
        match view {
            "overlay" => {
                for i in 0..W * H {
                    let b = rgb_at(i);
                    frame.buf[i] = pack(if valid[i] { [0, 1, 2].map(|k| (b[k] as f32 * (1.0 - alpha) + dcol[i][k] as f32 * alpha) as u8) } else { b });
                }
            }
            "side-by-side" => {
                for y in 0..H {
                    for x in 0..W {
                        frame.buf[y * 2 * W + x] = pack(rgb_at(y * W + x));
                        frame.buf[y * 2 * W + W + x] = pack(dcol[y * W + x]);
                    }
                }
            }
            "depth" => (0..W * H).for_each(|i| frame.buf[i] = pack(dcol[i])),
            "3d" => {
                let m = win.get_mouse_pos(MouseMode::Clamp);
                if win.get_mouse_down(MouseButton::Left) {
                    if let Some(p) = m {
                        if let Some(d) = orbit.drag {
                            orbit.yaw += (p.0 - d.0) * 0.01;
                            orbit.pitch = (orbit.pitch - (p.1 - d.1) * 0.01).clamp(-1.5, 1.5);
                        }
                        orbit.drag = Some(p);
                    }
                } else {
                    orbit.drag = None;
                }
                if let Some(w) = win.get_scroll_wheel() {
                    if w.1 != 0.0 {
                        orbit.zoom = (orbit.zoom * if w.1 > 0.0 { 0.9 } else { 1.1 }).clamp(0.15, 4.0);
                    }
                }
                render_cloud(&mut frame, &rgb, &depth, &mut orbit, 2);
            }
            _ => (0..W * H).for_each(|i| frame.buf[i] = pack(rgb_at(i))),
        }
        if mirror && !is3d {
            for y in 0..H {
                frame.buf[y * pw_..(y + 1) * pw_].reverse();
            }
        }
        let dist = {
            let mut v: Vec<f32> = Vec::new();
            for y in H / 2 - 2..H / 2 + 3 {
                for x in W / 2 - 2..W / 2 + 3 {
                    if depth[y * W + x] > 0 {
                        v.push(depth[y * W + x] as f32);
                    }
                }
            }
            if v.is_empty() { "--".to_string() } else { format!("{:.2} m", kinect3d::math::median(&mut v) / 1000.0) }
        };
        let grid = if is3d { None } else { GRIDS[grid_i] };
        if let Some((gc, gr)) = grid {
            let panels = if view == "side-by-side" { 2 } else { 1 };
            for p in 0..panels {
                draw_grid(&mut frame, &depth, mirror, gc, gr, p * W, W);
            }
        } else if view != "side-by-side" && !is3d {
            let (cx, cy) = (frame.w as i32 / 2, frame.h as i32 / 2);
            frame.line(cx - 7, cy, cx + 7, cy, WHITE);
            frame.line(cx, cy - 7, cx, cy + 7, WHITE);
            let (tw, th) = (text_width(&dist, 0.6), (px_for(0.6) * 0.72) as i32);
            frame.rect(cx + 8, cy - 14 - th, cx + 14 + tw, cy - 6, [0, 0, 0]);
            frame.text(&dist, cx + 11, cy - 10, 0.6, WHITE, false);
        }
        fps_n += 1;
        if fps_t.elapsed().as_secs_f32() >= 1.0 {
            fps = fps_n as f32 / fps_t.elapsed().as_secs_f32();
            fps_n = 0;
            fps_t = Instant::now();
        }
        let gtxt = grid.map_or("grid off".to_string(), |(a, b)| format!("grid {a}x{b}"));
        let mut status = format!("{view} | {gtxt} | {} | a={alpha:.2} | {near:.0}-{far:.0}mm | tilt {} | {fps:.0}fps", COLORMAPS[cmap_i], src.tilt_target());
        if play.is_some() {
            status = format!("PLAYBACK | {status}");
        }
        if let Some(r) = &rec {
            status = format!("REC {} | {status}", r.count);
        }
        frame.panel(0, 0, pw_ as i32, 30, 5.0 / 6.0);
        frame.text(&status, 8, 21, 0.6, if rec.is_some() { bgr(80, 80, 255) } else { WHITE }, false);

        let mut out = Canvas::new(pw_ + 2 * PANEL_W, H);
        for y in 0..H {
            out.buf[y * out.w + PANEL_W..y * out.w + PANEL_W + pw_].copy_from_slice(&frame.buf[y * pw_..(y + 1) * pw_]);
        }
        side_panel(&mut out, 0, "VIEW", &[
            ("v", format!("view: {view}")),
            ("g", format!("grid: {}", gtxt.trim_start_matches("grid "))),
            ("c", format!("colormap: {}", COLORMAPS[cmap_i])),
            ("e", format!("edges: {}", if edges { "on" } else { "off" })),
            ("m", format!("mirror: {}", if mirror { "on" } else { "off" })),
            ("drag/wheel", "orbit/zoom in 3d".into()),
            ("0", "reset 3d view".into()),
        ]);
        side_panel(&mut out, PANEL_W + pw_, "ADJUST", &[
            ("+ / -", format!("opacity: {alpha:.2}")),
            ("[ / ]", format!("range: {near:.0}-{far:.0} mm")),
            ("Up / Down", format!("tilt: {} deg", src.tilt_target())),
            ("", String::new()),
            ("s", "save snapshot".into()),
            ("p", "save point cloud".into()),
            ("r", if rec.is_some() { "recording: ON".into() } else { "record".into() }),
            ("q / Esc", "quit".into()),
        ]);
        if chars.contains(&'s') {
            let dir = root.join("snapshots");
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("kinect_{}.png", timestamp("%Y%m%d_%H%M%S")));
            io::save_rgb_png(&path, &frame.buf, frame.w, frame.h);
            println!("saved {}", path.display());
        }
        if win.update_with_buffer(&out.buf, out.w, out.h).is_err() {
            break;
        }
    }
    if let Some(r) = rec {
        let (d, n) = r.stop();
        println!("recording saved: {} ({n} frames)", d.display());
    }
    src.stop();
}
