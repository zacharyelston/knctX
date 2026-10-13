//! camgrid: a live grid of every camera attached to this machine.
//!
//! Keys:  Esc / Q  quit        1..9  show a single camera full-window
//!        0 / G    back to the grid     R  force a full rescan
//!
//! Cameras are detected automatically: /dev is polled every second and tiles
//! are added or removed as devices are plugged in or unplugged. With the
//! `kinect2` feature, Kinect v2 sensors (libfreenect2) are detected too.

mod capture;
#[cfg(feature = "kinect2")]
mod kinect2;

use std::time::{Duration, Instant};

use capture::{Camera, Frame, Watcher};
use fontdue::{Font, FontSettings};
use minifb::{Key, KeyRepeat, ScaleMode, Window, WindowOptions};

const BG: u32 = 0x00141414;
const LABEL_BG: u32 = 0x00000000;
const LABEL_FG: u32 = 0x00F0F0F0;
const STALE_FG: u32 = 0x00FF8040;
const GAP: usize = 4;
const LABEL_H: usize = 22;

struct Canvas {
    w: usize,
    h: usize,
    px: Vec<u32>,
}

impl Canvas {
    fn new(w: usize, h: usize) -> Self { Canvas { w, h, px: vec![BG; w * h] } }
    fn clear(&mut self) { self.px.fill(BG); }

    fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: u32) {
        for yy in y..(y + h).min(self.h) {
            let row = &mut self.px[yy * self.w + x..yy * self.w + (x + w).min(self.w)];
            row.fill(color);
        }
    }

    /// Draw `frame` scaled to fit inside the rect, preserving aspect ratio.
    /// Returns the rect actually covered by the picture.
    fn blit_fit(&mut self, frame: &Frame, x: usize, y: usize, w: usize, h: usize) -> (usize, usize, usize, usize) {
        if frame.width == 0 || frame.height == 0 || w == 0 || h == 0 { return (x, y, w, h); }
        let scale = (w as f32 / frame.width as f32).min(h as f32 / frame.height as f32);
        let dw = ((frame.width as f32 * scale) as usize).max(1);
        let dh = ((frame.height as f32 * scale) as usize).max(1);
        let ox = x + (w - dw) / 2;
        let oy = y + (h - dh) / 2;
        let sx_step = frame.width as f32 / dw as f32;
        let sy_step = frame.height as f32 / dh as f32;
        for dy in 0..dh {
            let sy = ((dy as f32 + 0.5) * sy_step) as usize;
            let src_row = &frame.rgb[sy * frame.width * 3..(sy + 1) * frame.width * 3];
            let dst_row = &mut self.px[(oy + dy) * self.w + ox..(oy + dy) * self.w + ox + dw];
            for (dx, out) in dst_row.iter_mut().enumerate() {
                let sx = ((dx as f32 + 0.5) * sx_step) as usize;
                let p = &src_row[sx * 3..sx * 3 + 3];
                *out = (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32;
            }
        }
        (ox, oy, dw, dh)
    }

    fn text(&mut self, font: &Font, size: f32, mut x: i32, baseline: i32, s: &str, color: u32) {
        for ch in s.chars() {
            let (m, bitmap) = font.rasterize(ch, size);
            let gx = x + m.xmin;
            let gy = baseline - m.height as i32 - m.ymin;
            for (i, &a) in bitmap.iter().enumerate() {
                if a == 0 { continue; }
                let px = gx + (i % m.width) as i32;
                let py = gy + (i / m.width) as i32;
                if px < 0 || py < 0 || px as usize >= self.w || py as usize >= self.h { continue; }
                let idx = py as usize * self.w + px as usize;
                self.px[idx] = blend(self.px[idx], color, a);
            }
            x += m.advance_width.round() as i32;
        }
    }
}

fn blend(dst: u32, src: u32, a: u8) -> u32 {
    let a = a as u32;
    let ch = |shift: u32| {
        let d = (dst >> shift) & 0xFF;
        let s = (src >> shift) & 0xFF;
        (d * (255 - a) + s * a) / 255
    };
    ch(16) << 16 | ch(8) << 8 | ch(0)
}

/// Rows and columns for `n` tiles in a `w`x`h` area, chosen to maximise tile area at 4:3.
fn grid_dims(n: usize, w: usize, h: usize) -> (usize, usize) {
    if n <= 1 { return (1, 1); }
    let mut best = (1, n, 0.0f32);
    for cols in 1..=n {
        let rows = n.div_ceil(cols);
        let tw = w as f32 / cols as f32;
        let th = (h as f32 / rows as f32) - LABEL_H as f32;
        let s = (tw / 4.0).min(th / 3.0);
        if s > best.2 { best = (cols, rows, s); }
    }
    (best.0, best.1)
}

fn load_font() -> Font {
    Font::from_bytes(include_bytes!("../assets/NotoSans-Regular.ttf") as &[u8], FontSettings::default()).expect("embedded font")
}

fn main() {
    let verbose = std::env::args().any(|a| a == "--verbose" || a == "-v");
    let font = load_font();
    let mut watcher = Watcher::default();
    let mut cams: Vec<Camera> = Vec::new();
    watcher.poll(&mut cams);
    print_summary(&cams);
    let mut last_poll = Instant::now();

    let (mut win_w, mut win_h) = (1280usize, 800usize);
    let mut win = Window::new(
        "camgrid",
        win_w,
        win_h,
        WindowOptions { resize: true, scale_mode: ScaleMode::Stretch, ..Default::default() },
    )
    .expect("window");
    win.set_target_fps(60);

    let mut canvas = Canvas::new(win_w, win_h);
    let mut focus: Option<usize> = None;
    let mut last_report = Instant::now();

    while win.is_open() && !win.is_key_down(Key::Escape) && !win.is_key_down(Key::Q) {
        let (ww, wh) = win.get_size();
        if ww != win_w || wh != win_h {
            win_w = ww.max(64);
            win_h = wh.max(64);
            canvas = Canvas::new(win_w, win_h);
        }
        for k in win.get_keys_pressed(KeyRepeat::No) {
            match k {
                Key::Key0 | Key::G => focus = None,
                Key::Key1 => focus = Some(0),
                Key::Key2 => focus = Some(1),
                Key::Key3 => focus = Some(2),
                Key::Key4 => focus = Some(3),
                Key::Key5 => focus = Some(4),
                Key::Key6 => focus = Some(5),
                Key::Key7 => focus = Some(6),
                Key::Key8 => focus = Some(7),
                Key::Key9 => focus = Some(8),
                Key::R => {
                    cams.clear(); // drops and joins capture threads, releasing devices
                    watcher = Watcher::default();
                    watcher.poll(&mut cams);
                    print_summary(&cams);
                    focus = None;
                }
                _ => {}
            }
        }
        if last_poll.elapsed() >= Duration::from_secs(1) {
            last_poll = Instant::now();
            if watcher.poll(&mut cams) { print_summary(&cams); }
        }
        if let Some(f) = focus { if f >= cams.len() { focus = None; } }

        if verbose && last_report.elapsed() >= Duration::from_secs(2) {
            last_report = Instant::now();
            for (i, c) in cams.iter().enumerate() {
                eprintln!("[{}] {:<12} {:>3} fps  {}", i + 1, c.path, c.slot.fps.load(std::sync::atomic::Ordering::Relaxed), c.slot.status());
            }
        }

        canvas.clear();
        if cams.is_empty() {
            canvas.text(&font, 24.0, 20, 50, "No cameras found. Plug one in and it will appear here.", LABEL_FG);
        } else {
            let shown: Vec<usize> = match focus { Some(f) => vec![f], None => (0..cams.len()).collect() };
            let (cols, rows) = grid_dims(shown.len(), win_w, win_h);
            let tile_w = win_w / cols;
            let tile_h = win_h / rows;
            for (slot_i, &ci) in shown.iter().enumerate() {
                let cam = &cams[ci];
                let x = (slot_i % cols) * tile_w + GAP / 2;
                let y = (slot_i / cols) * tile_h + GAP / 2;
                let w = tile_w.saturating_sub(GAP);
                let h = tile_h.saturating_sub(GAP + LABEL_H);

                let stale = cam.slot.since_last_frame() > Duration::from_secs(2);

                // (x, y, w, h) of the picture itself; the label bar hangs directly below it.
                let (x, y, w, h) = match cam.slot.frame() {
                    Some(frame) => canvas.blit_fit(&frame, x, y, w, h),
                    None => {
                        canvas.fill_rect(x, y, w, h, 0x00202020);
                        let msg = fit_text(&font, 16.0, &cam.slot.status(), w as f32 - 20.0);
                        canvas.text(&font, 16.0, x as i32 + 10, (y + h / 2) as i32, &msg, STALE_FG);
                        (x, y, w, h)
                    }
                };
                canvas.fill_rect(x, y + h, w, LABEL_H, LABEL_BG);
                let fps = cam.slot.fps.load(std::sync::atomic::Ordering::Relaxed);
                let baseline = (y + h + LABEL_H - 6) as i32;
                // Right-hand text (fps or status) gets up to 60% of the tile; the label gets the rest.
                let right = if stale { cam.slot.status() } else { format!("{fps} fps") };
                let right = fit_text(&font, 14.0, &right, (w as f32 * 0.6) - 12.0);
                let rw = text_width(&font, 14.0, &right);
                let label = format!("[{}] {}", ci + 1, cam.label());
                let label = fit_text(&font, 14.0, &label, w as f32 - rw - 18.0);
                canvas.text(&font, 14.0, x as i32 + 6, baseline, &label, LABEL_FG);
                canvas.text(&font, 14.0, (x + w) as i32 - rw as i32 - 6, baseline, &right, if stale { STALE_FG } else { LABEL_FG });
            }
        }
        win.update_with_buffer(&canvas.px, win_w, win_h).expect("update");
    }
}

fn text_width(font: &Font, size: f32, s: &str) -> f32 {
    s.chars().map(|c| font.metrics(c, size).advance_width).sum()
}

/// Truncate `s` with an ellipsis so it is no wider than `max_w` pixels.
fn fit_text(font: &Font, size: f32, s: &str, max_w: f32) -> String {
    if text_width(font, size, s) <= max_w { return s.to_string(); }
    let ell = font.metrics('…', size).advance_width;
    let mut out = String::new();
    let mut w = 0.0;
    for c in s.chars() {
        let cw = font.metrics(c, size).advance_width;
        if w + cw + ell > max_w { break; }
        out.push(c);
        w += cw;
    }
    out.push('…');
    out
}

fn print_summary(cams: &[Camera]) {
    if cams.is_empty() {
        eprintln!("no usable cameras found (waiting for one to be plugged in)");
    }
    eprintln!("-- {} camera(s)", cams.len());
    for (i, c) in cams.iter().enumerate() {
        eprintln!("[{}] {}", i + 1, c.label());
    }
}
