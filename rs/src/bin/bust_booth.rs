//! Seated bust booth: the subject sits on a swivel chair about 1.1 m from the Kinect and turns
//! themselves slowly once round; head and shoulders are scanned in fine detail (6 mm voxels).
//!
//!   bust_booth            real Kinect
//!   bust_booth --demo     synthetic spinning bust (no sensor needed)
//!   bust_booth --view DIR open a saved scan
//!   bust_booth --autorun NAME   (demo/testing) start by itself after a few seconds
//!
//! Gravity comes from the Kinect's accelerometer, so no floor is needed and the sensor may be tilted.
//! Viewer keys: drag = orbit, wheel = zoom, a = auto-spin, 0 = reset, s = export + screenshot, n = new, Esc = quit.
use kinect3d::gfx::*;
use kinect3d::io;
use kinect3d::math::*;
use kinect3d::render::*;
use kinect3d::scan::capture::*;
use kinect3d::scan::mesh::vertex_normals;
use kinect3d::scan::segment::Segmenter;
use kinect3d::scan::world::*;
use kinect3d::source::*;
use kinect3d::synthetic::{Scene, SyntheticKinect};
use minifb::{InputCallback, Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const CW: usize = 960;
const CH: usize = 720;
const TURNS_NEEDED: f64 = 1.25;
const BUST_H: f32 = 0.62; // metres of subject kept below the top of the head

const ACCENT: Col = bgr(255, 200, 120);
const OK_GREEN: Col = bgr(120, 230, 120);
const WARN: Col = bgr(80, 180, 255);
const TOAST: Col = bgr(120, 220, 255);
const GRAY_T: Col = bgr(210, 210, 210);

/// Distance (m) to the nearest sizeable object in the middle of the view (0.5-2.2 m), or None.
fn bust_distance(depth: &[u16]) -> Option<f32> {
    let mut v: Vec<f32> = Vec::new();
    for r in 100..420 {
        for c in 220..420 {
            let d = depth[r * W + c];
            if d > 500 && d < 2200 {
                v.push(d as f32);
            }
        }
    }
    if v.len() < 5000 {
        return None;
    }
    let z0 = percentile(&mut v.clone(), 8.0);
    let mut blob: Vec<f32> = v.into_iter().filter(|d| (d - z0).abs() < 200.0).collect();
    if blob.len() > 4000 { Some(median(&mut blob) / 1000.0) } else { None }
}

/// First image row holding the subject (depth within 25 cm of `dist`) and the subject's pixel count.
fn subject_rows(depth: &[u16], dist: f32) -> (usize, usize) {
    let target = dist * 1000.0;
    let (mut top, mut area) = (H, 0);
    for r in 0..H {
        let n = (160..480).filter(|&c| {
            let d = depth[r * W + c];
            d > 0 && (d as f32 - target).abs() < 250.0
        }).count();
        if n >= 15 && top == H {
            top = r;
        }
        area += n;
    }
    (top, area)
}

#[derive(PartialEq, Clone, Copy)]
enum State {
    Name,
    Countdown,
    Capture,
    Process,
    View,
}

struct Probe {
    t: f64,
    dist: Option<f32>,
    text: String,
    ok: bool,
    tilt_t: f64,
}

struct Bust {
    verts: Vec<V3>,
    normals: Vec<V3>,
    faces: Vec<[u32; 3]>,
    rgb: Vec<[u8; 3]>,
    focus: V3,
    dist: f32,
}

impl Bust {
    fn new(verts: Vec<V3>, faces: Vec<[u32; 3]>, rgb: Vec<[u8; 3]>) -> Bust {
        let normals = vertex_normals(&verts, &faces);
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for v in &verts {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        let focus = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
        let ext = (hi[1] - lo[1]).max(hi[0] - lo[0]);
        Bust { verts, normals, faces, rgb, focus, dist: 1.5 * ext }
    }
    fn draw(&self, c: &mut Canvas, orbit: &Orbit3D) { render_mesh_at(c, &self.verts, &self.normals, &self.faces, &self.rgb, orbit, self.focus, self.dist, false); }
}

type Built = Result<Bust, String>;

struct Booth {
    src: Arc<dyn Source>,
    root: PathBuf,
    t0: Instant,
    state: State,
    t_state: f64,
    name: String,
    msg: String,
    msg_until: f64,
    orbit: Orbit3D,
    world: World,
    seg: Option<Segmenter>,
    probe: Probe,
    bust: Option<Bust>,
    folder: PathBuf,
    auto: bool,
    prog: Arc<Mutex<(String, f32)>>,
    force_until: f64,
    autotilt: bool,
    raw: Option<RawSaver>,
    cap: Option<Capture>,
    center0: Option<[f32; 2]>,
    y_range: (f32, f32),
    last_seq: u64,
    last_t: f64,
    last_mask: Option<Vec<bool>>,
    visible: bool,
    result: Arc<Mutex<Option<Built>>>,
    last_render: Vec<u32>,
    refresh_t: f64,
}

impl Booth {
    fn new(src: Arc<dyn Source>, root: PathBuf) -> Booth {
        Booth {
            src, root, t0: Instant::now(), state: State::Name, t_state: 0.0, name: String::new(), msg: String::new(), msg_until: 0.0,
            orbit: Orbit3D::new(), world: World::default(), seg: None,
            probe: Probe { t: 0.0, dist: None, text: "Looking for you...".into(), ok: false, tilt_t: 0.0 },
            bust: None, folder: PathBuf::new(), auto: true, prog: Arc::new(Mutex::new((String::new(), 0.0))), force_until: 0.0, autotilt: true,
            raw: None, cap: None, center0: None, y_range: (0.0, 1.0), last_seq: u64::MAX, last_t: 0.0, last_mask: None, visible: true,
            result: Arc::new(Mutex::new(None)), last_render: Vec::new(), refresh_t: 0.0,
        }
    }

    fn now(&self) -> f64 { self.t0.elapsed().as_secs_f64() }
    fn toast(&mut self, s: &str, secs: f64) {
        self.msg = s.to_string();
        self.msg_until = self.now() + secs;
    }

    fn open_saved(&mut self, folder: &Path) -> Result<(), String> {
        let (v, f, c) = io::load_ply(&folder.join("model.ply"))?;
        self.bust = Some(Bust::new(v, f, c));
        self.folder = folder.to_path_buf();
        self.name = folder.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        self.state = State::View;
        Ok(())
    }

    /// Gravity-aligned world from the accelerometer (falls back to a level camera).
    fn refresh_world(&mut self) {
        self.world = self.src.accel().and_then(world_from_accel).unwrap_or_default();
    }

    fn probe(&mut self) {
        let now = self.now();
        let Some((_rgb, depth)) = self.src.frames() else { return };
        if now - self.probe.t < 0.4 {
            return;
        }
        self.probe.t = now;
        self.refresh_world();
        let dist = bust_distance(&depth);
        self.probe.dist = dist;
        let (top, area) = dist.map_or((H, 0), |d| subject_rows(&depth, d));
        // registered depth is blank above row ~45, so a subject starting there is cut off by the sensor
        let head_cut = dist.is_some() && top < 60;
        let head_low = dist.is_some() && top > 170;
        if let (true, Some(_)) = (self.autotilt, dist) {
            if (head_cut || head_low) && now - self.probe.tilt_t > 1.5 {
                let cur = self.src.tilt_target();
                let new = (cur + if head_cut { 3 } else { -3 }).clamp(-20, 20);
                if new != cur {
                    self.src.set_tilt(new);
                    self.probe.tilt_t = now;
                }
            }
        }
        let res = Segmenter::new(self.world).find_bust(&depth, BUST_H);
        let auto = if self.autotilt { " (auto-tilting)" } else { "" };
        let (text, ok) = match dist {
            None => ("Nothing in view - sit down in front of the sensor".to_string(), false),
            Some(d) if d < 0.8 => (format!("{d:.2} m - too close, move back to about 1.0 m"), false),
            Some(d) if d > 1.25 => (format!("{d:.2} m - too far, move closer (about 1.0 m)"), false),
            Some(d) if head_cut => (format!("{d:.2} m - head cut off: tilting sensor up{auto}, or sit lower"), false),
            Some(d) if head_low => (format!("{d:.2} m - too much empty space above you: tilting down{auto}"), false),
            Some(d) if res.is_none() || area < 25000 => (format!("{d:.2} m - good distance, but I can't see head and shoulders yet"), false),
            Some(d) => (format!("Perfect - {d:.2} m away, head and shoulders in view"), true),
        };
        self.probe.text = text;
        self.probe.ok = ok;
    }

    fn distance_meter(&self, c: &mut Canvas, y: i32) {
        let (x0, x1) = (170, CW as i32 - 170);
        let px = |m: f32| (x0 as f32 + (x1 - x0) as f32 * (m.clamp(0.5, 2.0) - 0.5) / 1.5) as i32;
        c.rect(x0, y, x1, y + 10, bgr(70, 70, 70));
        c.rect(px(0.8), y, px(1.2), y + 10, bgr(90, 200, 90));
        for m in [0.5f32, 1.0, 1.5, 2.0] {
            c.line(px(m), y + 10, px(m), y + 16, bgr(200, 200, 200));
            c.text(&format!("{m:.1}m"), px(m) - 14, y + 34, 0.45, bgr(190, 190, 190), false);
        }
        if let Some(d) = self.probe.dist {
            let col = if (0.8..=1.2).contains(&d) { OK_GREEN } else { WARN };
            c.triangle([[px(d), y - 2], [px(d) - 9, y - 16], [px(d) + 9, y - 16]], col);
            c.text(&format!("{d:.2} m"), px(d) - 30, y - 22, 0.6, col, true);
        }
    }

    // ------------------------------------------------------------------ states
    fn find_subject(&mut self) -> bool {
        let Some((_, depth)) = self.src.frames() else { return false };
        self.refresh_world();
        let mut seg = Segmenter::new(self.world);
        if seg.find_bust(&depth, BUST_H).is_none() {
            return false;
        }
        self.y_range = (seg.y_lo, seg.y_hi);
        self.seg = Some(seg);
        true
    }

    fn start_countdown(&mut self) {
        if !self.find_subject() {
            self.toast("I can't see you - sit in front of the sensor and try again", 4.0);
            return;
        }
        self.state = State::Countdown;
        self.t_state = self.now();
    }

    fn start_capture(&mut self) {
        self.folder = new_scan_folder(&self.root, &format!("{} bust", self.name));
        self.raw = Some(RawSaver::new(&self.folder));
        self.cap = Some(Capture::with(self.seg.take().unwrap(), Profile::bust()));
        self.center0 = None;
        self.last_seq = u64::MAX;
        self.last_t = 0.0;
        self.last_mask = None;
        self.visible = true;
        self.state = State::Capture;
        self.t_state = self.now();
    }

    fn sample(&mut self) -> Option<Rgb> {
        let (rgb, depth) = self.src.frames()?;
        let seq = self.src.seq();
        let now = self.now();
        if seq == self.last_seq || now - self.last_t < 0.09 {
            return Some(rgb);
        }
        self.last_seq = seq;
        self.last_t = now;
        let cap = self.cap.as_mut().unwrap();
        let (m, used) = cap.add(&rgb, &depth);
        self.last_mask = Some(m);
        self.visible = used;
        if used {
            self.raw.as_mut().unwrap().add(&rgb, &depth);
            if self.center0.is_none() {
                let p = &cap.caps[0].pts;
                let n = p.len() as f32;
                self.center0 = Some([p.iter().map(|q| q[0]).sum::<f32>() / n, p.iter().map(|q| q[2]).sum::<f32>() / n]);
            }
        }
        Some(rgb)
    }

    fn finish_raw(&mut self, extra: serde_json::Value) {
        if let Some(mut raw) = self.raw.take() {
            let q = self.cap.as_ref().unwrap();
            let mut e = serde_json::json!({"profile": "bust", "y_lo": self.y_range.0, "y_hi": self.y_range.1,
                "online_turns": q.trk.yaw().abs() / std::f64::consts::TAU, "quality": q.trk.quality, "name": self.name});
            if let (Some(o), Some(x)) = (e.as_object_mut(), extra.as_object()) {
                for (k, v) in x {
                    o.insert(k.clone(), v.clone());
                }
            }
            raw.finish(&self.world, self.center0, e);
        }
    }

    fn start_process(&mut self) {
        self.state = State::Process;
        self.t_state = self.now();
        *self.prog.lock().unwrap() = ("Aligning the turn".into(), 0.05);
        let cap = self.cap.take().unwrap();
        let (world, center, folder, name, yr) = (self.world, self.center0.unwrap(), self.folder.clone(), self.name.clone(), self.y_range);
        let (prog, result) = (self.prog.clone(), self.result.clone());
        std::thread::spawn(move || {
            let p2 = prog.clone();
            let built = build_model(&cap.caps, &world, center, &Profile::bust(), yr, &move |s, f| *p2.lock().unwrap() = (s.to_string(), f), &mut |s| println!("{s}"));
            let out: Built = built.map(|mesh| {
                let b = Bust::new(mesh.verts.clone(), mesh.faces.clone(), mesh.rgb.clone());
                let mut thumb = Canvas::new(CW, CH);
                b.draw(&mut thumb, &Orbit3D::new());
                save_scan(&folder, &name, &mesh, None, Some((&thumb.buf, CW, CH)));
                println!("saved scan: {}", folder.display());
                b
            });
            *result.lock().unwrap() = Some(out);
        });
    }

    // ------------------------------------------------------------------ frames
    fn tick(&mut self, inp: &Input) -> Canvas {
        let mut c = match self.state {
            State::Name => self.tick_name(inp),
            State::Countdown => self.tick_countdown(inp),
            State::Capture => self.tick_capture(inp),
            State::Process => self.tick_process(),
            State::View => self.tick_view(inp),
        };
        if self.now() < self.msg_until {
            c.panel(0, CH as i32 - 56, CW as i32, CH as i32, 0.7);
            c.text_c(&self.msg, CH as i32 - 20, 0.7, TOAST, true);
        }
        c
    }

    fn tick_name(&mut self, inp: &Input) -> Canvas {
        let frames = self.src.frames();
        if inp.key(Key::Up) {
            self.src.set_tilt(self.src.tilt_target() + 3);
        }
        if inp.key(Key::Down) {
            self.src.set_tilt(self.src.tilt_target() - 3);
        }
        if inp.key(Key::Enter) || inp.key(Key::NumPadEnter) {
            let now = self.now();
            if self.name.trim().is_empty() {
                self.toast("Please type your name first", 3.0);
            } else if self.probe.ok || now < self.force_until {
                self.start_countdown();
            } else {
                self.force_until = now + 6.0;
                self.toast("Not ready - press ENTER again to scan anyway", 6.0);
            }
        }
        if inp.key(Key::Tab) {
            self.autotilt = !self.autotilt;
            let s = format!("Auto-tilt {}", if self.autotilt { "on" } else { "off" });
            self.toast(&s, 3.0);
        }
        if inp.key(Key::Backspace) {
            self.name.pop();
        }
        for ch in &inp.chars {
            if (' '..='~').contains(ch) && self.name.len() < 24 {
                self.name.push(*ch);
            }
        }
        let mut c = Canvas::new(CW, CH);
        let Some((rgb, _)) = frames else {
            c.fill([25, 25, 25]);
            c.text_c("Waiting for the Kinect...", CH as i32 / 2, 1.0, WHITE, true);
            return c;
        };
        self.probe();
        c.camera(&rgb, None, 0.45, true);
        c.text_c("3D BUST SCAN", 90, 1.8, ACCENT, true);
        c.panel(120, 150, CW as i32 - 120, 300, 0.62);
        c.text_c("What's your name?", 205, 0.9, bgr(220, 220, 220), true);
        let cursor = if (self.now() * 2.0) as i64 % 2 == 0 { "_" } else { " " };
        c.text_c(&format!("{}{cursor}", self.name), 270, 1.5, WHITE, true);
        if self.now() < self.force_until {
            c.panel(120, 310, CW as i32 - 120, 380, 0.8);
            c.text_c("Not ready - press ENTER again to scan anyway", 355, 0.85, WARN, true);
        } else {
            c.text_c("Press ENTER to start", 342, 0.8, bgr(180, 255, 180), true);
        }
        let col = if self.probe.ok { OK_GREEN } else { WARN };
        c.panel(120, CH as i32 - 262, CW as i32 - 120, CH as i32 - 24, 0.62);
        c.text_c(&format!("{}{}", if self.probe.ok { "OK  " } else { "..  " }, self.probe.text), CH as i32 - 224, 0.75, col, true);
        self.distance_meter(&mut c, CH as i32 - 168);
        c.text_c("Sit on the swivel chair about 1.1 m from the sensor, arms relaxed", CH as i32 - 106, 0.6, GRAY_T, false);
        c.text_c("After 3-2-1, turn the chair slowly round - keep your HEAD STRAIGHT with your body, don't look at the camera", CH as i32 - 80, 0.6, GRAY_T, false);
        c.text_c(
            &format!("Tilt {:+} deg ({}, Tab toggles)  Up/Down arrows tilt   Esc quits", self.src.tilt_target(), if self.autotilt { "auto" } else { "manual" }),
            CH as i32 - 50, 0.5, bgr(150, 150, 150), false,
        );
        c
    }

    fn tick_countdown(&mut self, inp: &Input) -> Canvas {
        let frames = self.src.frames();
        let left = 3.0 - (self.now() - self.t_state);
        if inp.key(Key::Escape) {
            self.state = State::Name;
        }
        if self.now() - self.refresh_t > 0.5 {
            self.refresh_t = self.now();
            self.find_subject(); // keep the capture volume centred on where they sit
        }
        let mut c = Canvas::new(CW, CH);
        if let Some((rgb, _)) = &frames {
            c.camera(rgb, None, 0.35, true);
        }
        if left <= 0.0 && self.state == State::Countdown {
            if self.seg.is_some() {
                self.start_capture();
            } else {
                self.state = State::Name;
                self.toast("I lost you - please try again", 4.0);
            }
            return c;
        }
        let n = left.ceil() as i32;
        let frac = if left != left.floor() { left - left.floor() } else { 1.0 } as f32;
        let scale = 7.0 + 2.5 * frac;
        let s = n.to_string();
        let (tw, th) = (text_width(&s, scale), (px_for(scale) * 0.72) as i32);
        c.text(&s, (CW as i32 - tw) / 2, (CH as i32 + th) / 2, scale, WHITE, true);
        c.text_c(&format!("Get ready, {}!", self.name), 120, 1.2, TOAST, true);
        c.text_c("Turn slowly, one full turn - head straight, eyes forward", CH as i32 - 60, 0.8, WHITE, true);
        c
    }

    fn tick_capture(&mut self, inp: &Input) -> Canvas {
        let rgb = self.sample();
        let (turns, quality, ncap) = {
            let cap = self.cap.as_ref().unwrap();
            (cap.trk.yaw().abs() / std::f64::consts::TAU, cap.trk.quality, cap.caps.len())
        };
        let elapsed = self.now() - self.t_state;
        if inp.key(Key::Escape) {
            self.finish_raw(serde_json::json!({"result": "cancelled"}));
            self.cap = None;
            self.state = State::Name;
            self.toast("Scan cancelled", 3.0);
        } else if (inp.key(Key::Space) && turns >= 1.0) || turns >= TURNS_NEEDED * 0.985 || ncap >= 360 || elapsed > 75.0 {
            if ncap >= 40 {
                self.finish_raw(serde_json::json!({}));
                self.start_process();
            } else {
                self.finish_raw(serde_json::json!({"result": "too few frames"}));
                self.cap = None;
                self.state = State::Name;
                self.toast("That didn't capture enough - please try again", 5.0);
            }
        }
        let mut c = Canvas::new(CW, CH);
        let Some(rgb) = rgb else { return c };
        c.camera(&rgb, self.last_mask.as_deref(), 0.15, true);
        c.text_c("TURN SLOWLY!", 70, 1.4, ACCENT, true);
        c.text_c("head straight with your body - don't track the camera", 100, 0.6, GRAY_T, true);
        let (x0, x1, y) = (160, CW as i32 - 160, CH as i32 - 70);
        c.panel(x0 - 14, y - 52, x1 + 14, y + 24, 0.7);
        c.rect(x0, y - 12, x1, y + 6, bgr(70, 70, 70));
        c.rect(x0, y - 12, x0 + ((x1 - x0) as f64 * (turns / TURNS_NEEDED).min(1.0)) as i32, y + 6, OK_GREEN);
        let xx = x0 + ((x1 - x0) as f64 / TURNS_NEEDED) as i32;
        c.line(xx, y - 12, xx, y + 6, bgr(200, 200, 200));
        c.text_c(&format!("Turn {:.2} of 1 (+ overlap)", turns.min(TURNS_NEEDED)), y - 22, 0.8, WHITE, false);
        if !self.visible {
            c.text_c("I can't see you - stay in the frame!", 130, 0.9, bgr(80, 80, 255), true);
        } else if quality < 0.6 {
            c.text_c("Tracking slipping - turn a bit slower", 130, 0.9, WARN, true);
        }
        c
    }

    fn tick_process(&mut self) -> Canvas {
        let finished = self.result.lock().unwrap().take();
        if let Some(res) = finished {
            match res {
                Ok(b) => {
                    self.bust = Some(b);
                    self.orbit.reset();
                    self.state = State::View;
                }
                Err(e) => {
                    eprintln!("scan failed: {e}");
                    self.state = State::Name;
                    self.toast(&format!("Scan failed: {e}"), 6.0);
                }
            }
        }
        let (label, frac) = self.prog.lock().unwrap().clone();
        let mut c = Canvas::new(CW, CH);
        match self.src.frames() {
            Some((rgb, _)) => c.camera(&rgb, None, 0.7, true),
            None => c.fill([25, 25, 25]),
        }
        c.text_c("Building your 3D bust...", CH as i32 / 2 - 60, 1.3, ACCENT, true);
        c.text_c(&label, CH as i32 / 2 - 10, 0.8, WHITE, true);
        let (x0, x1, y) = (200, CW as i32 - 200, CH as i32 / 2 + 40);
        c.rect(x0, y, x1, y + 16, bgr(70, 70, 70));
        c.rect(x0, y, x0 + ((x1 - x0) as f32 * frac.clamp(0.0, 1.0)) as i32, y + 16, OK_GREEN);
        c
    }

    fn tick_view(&mut self, inp: &Input) -> Canvas {
        for ch in &inp.chars {
            match ch {
                'a' => self.auto = !self.auto,
                '0' => self.orbit.reset(),
                's' => self.export(),
                'n' => {
                    self.name.clear();
                    self.bust = None;
                    self.state = State::Name;
                    return self.tick_name(&Input::default());
                }
                _ => {}
            }
        }
        if self.auto && self.orbit.drag.is_none() {
            self.orbit.yaw += 0.01;
        }
        let b = self.bust.as_ref().unwrap();
        let mut c = Canvas::new(CW, CH);
        b.draw(&mut c, &self.orbit);
        self.last_render = c.buf.clone();
        c.text(&self.name, 24, 44, 1.1, WHITE, true);
        c.text(&format!("{} vertices", b.verts.len()), 24, 76, 0.6, bgr(180, 180, 180), true);
        c.text("drag: orbit   wheel: zoom   a: auto-spin   0: reset", 24, CH as i32 - 60, 0.55, bgr(160, 160, 160), false);
        c.text("s: export + screenshot   n: new scan   Esc: quit", 24, CH as i32 - 32, 0.55, bgr(160, 160, 160), false);
        c
    }

    fn export(&mut self) {
        let Some(b) = &self.bust else { return };
        let base = self.folder.join(format!("view_{}", timestamp("%H%M%S")));
        let _ = io::save_ply(&base.with_extension("ply"), &b.verts, &b.faces, &b.rgb);
        let _ = io::save_obj(&base.with_extension("obj"), &b.verts, &b.faces, &b.rgb);
        io::save_rgb_png(&base.with_extension("png"), &self.last_render, CW, CH);
        let s = format!("Saved to {}", self.folder.display());
        self.toast(&s, 3.0);
    }
}

#[derive(Default)]
struct Input {
    chars: Vec<char>,
    keys: Vec<Key>,
}
impl Input {
    fn key(&self, k: Key) -> bool { self.keys.contains(&k) }
}

struct Chars(Arc<Mutex<Vec<char>>>);
impl InputCallback for Chars {
    fn add_char(&mut self, uni_char: u32) {
        if let Some(c) = char::from_u32(uni_char) {
            self.0.lock().unwrap().push(c);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let val = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1).cloned());
    if flag("-h") || flag("--help") {
        println!("bust_booth [--demo] [--view DIR] [--out DIR] [--autorun NAME]");
        return;
    }
    let out = val("--out").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../scans"));
    let view = val("--view");
    let src: Arc<dyn Source> = if view.is_some() {
        Arc::new(NoCam::new())
    } else if flag("--demo") {
        Arc::new(SyntheticKinect::with_scene(Scene::bust(), 12.0, 30.0, 2.0))
    } else {
        Arc::new(Kinect::new())
    };
    src.start();
    let mut booth = Booth::new(src.clone(), out);
    if let Some(v) = &view {
        if let Err(e) = booth.open_saved(Path::new(v)) {
            eprintln!("cannot open {v}: {e}");
            std::process::exit(1);
        }
    }
    let mut win = Window::new("3D Bust Scan", CW, CH, WindowOptions { resize: true, scale_mode: minifb::ScaleMode::AspectRatioStretch, ..Default::default() }).expect("window");
    win.set_target_fps(30);
    let autorun = val("--autorun");
    let started = Instant::now();
    let mut autorun_done = false;
    let typed = Arc::new(Mutex::new(Vec::new()));
    win.set_input_callback(Box::new(Chars(typed.clone())));
    while win.is_open() {
        let inp = Input { chars: std::mem::take(&mut *typed.lock().unwrap()), keys: win.get_keys_pressed(KeyRepeat::Yes) };
        if inp.key(Key::Escape) && !matches!(booth.state, State::Capture) {
            break;
        }
        let pos = win.get_mouse_pos(MouseMode::Clamp);
        let wheel = win.get_scroll_wheel().map_or(0.0, |w| w.1);
        let viewing = booth.state == State::View;
        booth.orbit.mouse(pos, win.get_mouse_down(MouseButton::Left) && viewing, if viewing { wheel } else { 0.0 });
        if let Some(n) = &autorun {
            if !autorun_done && booth.state == State::Name && started.elapsed().as_secs_f32() > 3.0 {
                autorun_done = true;
                booth.name = n.clone();
                booth.start_countdown();
            }
        }
        let canvas = booth.tick(&inp);
        if win.update_with_buffer(&canvas.buf, CW, CH).is_err() {
            break;
        }
    }
    src.stop();
}
