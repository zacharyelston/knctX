//! Tiny software canvas: rectangles, lines, anti-aliased text (fontdue), camera blits.
use fontdue::{Font, FontSettings};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

pub type Col = [u8; 3]; // RGB

/// Colour from OpenCV-style (B, G, R) numbers.
pub const fn bgr(b: u8, g: u8, r: u8) -> Col { [r, g, b] }
pub const WHITE: Col = [255, 255, 255];

#[inline]
pub fn pack(c: Col) -> u32 { ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32 }
#[inline]
pub fn unpack(p: u32) -> Col { [(p >> 16) as u8, (p >> 8) as u8, p as u8] }

fn font() -> &'static Font {
    static F: OnceLock<Font> = OnceLock::new();
    F.get_or_init(|| Font::from_bytes(include_bytes!("../assets/NotoSans-Regular.ttf") as &[u8], FontSettings::default()).unwrap())
}

type Glyph = (fontdue::Metrics, Vec<u8>);
thread_local! {
    static CACHE: RefCell<HashMap<(char, u32), std::rc::Rc<Glyph>>> = RefCell::new(HashMap::new());
}

fn glyph(c: char, px: f32) -> std::rc::Rc<Glyph> {
    let key = (c, (px * 4.0) as u32);
    CACHE.with(|m| m.borrow_mut().entry(key).or_insert_with(|| std::rc::Rc::new(font().rasterize(c, px))).clone())
}

/// Pixel size matching an OpenCV Hershey `scale` (roughly the same visual height).
pub fn px_for(scale: f32) -> f32 { scale * 32.0 }

pub fn text_width(s: &str, scale: f32) -> i32 {
    let px = px_for(scale);
    s.chars().map(|c| glyph(c, px).0.advance_width).sum::<f32>().round() as i32
}

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub buf: Vec<u32>,
}

impl Canvas {
    pub fn new(w: usize, h: usize) -> Canvas { Canvas { w, h, buf: vec![0; w * h] } }
    pub fn fill(&mut self, c: Col) { self.buf.iter_mut().for_each(|p| *p = pack(c)); }

    #[inline]
    pub fn put(&mut self, x: i32, y: i32, c: Col) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            self.buf[y as usize * self.w + x as usize] = pack(c);
        }
    }

    #[inline]
    pub fn blend(&mut self, x: i32, y: i32, c: Col, a: f32) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let i = y as usize * self.w + x as usize;
            let o = unpack(self.buf[i]);
            let m = |a0: u8, b0: u8| (a0 as f32 * (1.0 - a) + b0 as f32 * a) as u8;
            self.buf[i] = pack([m(o[0], c[0]), m(o[1], c[1]), m(o[2], c[2])]);
        }
    }

    pub fn rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: Col) {
        let (x0, x1) = (x0.max(0), x1.min(self.w as i32));
        let (y0, y1) = (y0.max(0), y1.min(self.h as i32));
        let p = pack(c);
        for y in y0..y1 {
            for x in x0..x1 {
                self.buf[y as usize * self.w + x as usize] = p;
            }
        }
    }

    /// Darken a region (`alpha` = how much black is mixed in).
    pub fn panel(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, alpha: f32) {
        let k = ((1.0 - alpha) * 256.0) as u32;
        let (x0, x1) = (x0.max(0) as usize, (x1.max(0) as usize).min(self.w));
        for y in y0.max(0) as usize..(y1.max(0) as usize).min(self.h) {
            for p in &mut self.buf[y * self.w + x0..y * self.w + x1.max(x0)] {
                let (r, g, b) = ((*p >> 16) & 255, (*p >> 8) & 255, *p & 255);
                *p = ((r * k >> 8) << 16) | ((g * k >> 8) << 8) | (b * k >> 8);
            }
        }
    }

    pub fn line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: Col) {
        let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
        let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
        let (mut x, mut y, mut err) = (x0, y0, dx + dy);
        loop {
            self.put(x, y, c);
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    pub fn line_blend(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: Col, a: f32) {
        let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
        let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
        let (mut x, mut y, mut err) = (x0, y0, dx + dy);
        loop {
            self.blend(x, y, c, a);
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    pub fn triangle(&mut self, p: [[i32; 2]; 3], c: Col) {
        let (minx, maxx) = (p.iter().map(|a| a[0]).min().unwrap(), p.iter().map(|a| a[0]).max().unwrap());
        let (miny, maxy) = (p.iter().map(|a| a[1]).min().unwrap(), p.iter().map(|a| a[1]).max().unwrap());
        let e = |a: [i32; 2], b: [i32; 2], x: i32, y: i32| (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        for y in miny..=maxy {
            for x in minx..=maxx {
                let (w0, w1, w2) = (e(p[0], p[1], x, y), e(p[1], p[2], x, y), e(p[2], p[0], x, y));
                if (w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0) {
                    self.put(x, y, c);
                }
            }
        }
    }

    /// Draw text with its baseline at `y`. `scale` follows OpenCV Hershey sizing.
    pub fn text(&mut self, s: &str, x: i32, y: i32, scale: f32, c: Col, shadow: bool) {
        if shadow {
            self.text_raw(s, x + 2, y + 2, scale, [0, 0, 0]);
        }
        self.text_raw(s, x, y, scale, c);
    }

    fn text_raw(&mut self, s: &str, x: i32, y: i32, scale: f32, c: Col) {
        let px = px_for(scale);
        let mut pen = x as f32;
        for ch in s.chars() {
            let g = glyph(ch, px);
            let (m, bm) = (&g.0, &g.1);
            let gx = pen.round() as i32 + m.xmin;
            let gy = y - m.height as i32 - m.ymin;
            for r in 0..m.height {
                for q in 0..m.width {
                    let a = bm[r * m.width + q];
                    if a > 0 {
                        self.blend(gx + q as i32, gy + r as i32, c, a as f32 / 255.0);
                    }
                }
            }
            pen += m.advance_width;
        }
    }

    pub fn text_c(&mut self, s: &str, y: i32, scale: f32, c: Col, shadow: bool) {
        let w = text_width(s, scale);
        self.text(s, (self.w as i32 - w) / 2, y, scale, c, shadow);
    }

    /// Blit a 640x480 RGB frame scaled to the whole canvas (bilinear, fixed point), optionally
    /// mirrored, with pixels outside `mask` darkened and the whole picture dimmed.
    pub fn camera(&mut self, rgb: &[u8], mask: Option<&[bool]>, dim: f32, mirror: bool) {
        use crate::source::{H, W};
        let (sx, sy) = (W as f32 / self.w as f32, H as f32 / self.h as f32);
        // per-column source taps: (x0, x1, weight of x1 in 0..256)
        let cols: Vec<(usize, usize, u32)> = (0..self.w)
            .map(|x| {
                let mut fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (W - 1) as f32);
                if mirror {
                    fx = (W - 1) as f32 - fx;
                }
                let x0 = fx as usize;
                (x0, (x0 + 1).min(W - 1), (fx.fract() * 256.0) as u32)
            })
            .collect();
        let k = ((1.0 - dim) * 256.0) as u32;
        let kd = (k as f32 * 0.35) as u32;
        for y in 0..self.h {
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (H - 1) as f32);
            let (y0, wy) = (fy as usize, (fy.fract() * 256.0) as u32);
            let y1 = (y0 + 1).min(H - 1);
            let (r0, r1) = (&rgb[y0 * W * 3..(y0 + 1) * W * 3], &rgb[y1 * W * 3..(y1 + 1) * W * 3]);
            let mrow = mask.map(|m| &m[(fy.round() as usize).min(H - 1) * W..][..W]);
            let out = &mut self.buf[y * self.w..(y + 1) * self.w];
            for (x, o) in out.iter_mut().enumerate() {
                let (x0, x1, wx) = cols[x];
                let mut px = [0u32; 3];
                for ch in 0..3 {
                    let a = (r0[x0 * 3 + ch] as u32 * (256 - wx) + r0[x1 * 3 + ch] as u32 * wx) >> 8;
                    let b = (r1[x0 * 3 + ch] as u32 * (256 - wx) + r1[x1 * 3 + ch] as u32 * wx) >> 8;
                    px[ch] = (a * (256 - wy) + b * wy) >> 8;
                }
                let f = match mrow {
                    Some(m) if !m[if wx >= 128 { x1 } else { x0 }] => kd,
                    _ => k,
                };
                *o = ((px[0] * f >> 8) << 16) | ((px[1] * f >> 8) << 8) | (px[2] * f >> 8);
            }
        }
    }
}
