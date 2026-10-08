//! 3D photo booth: a visitor types their name, spins twice in front of the Kinect and gets a
//! posable 3D figure of themselves.
//!
//!   scan_booth            real Kinect
//!   scan_booth --demo     synthetic spinning person (no sensor needed)
//!   scan_booth --view DIR open a saved scan
//!
//! Flow: NAME -> 3,2,1 -> spin capture (2 turns, tracked live) -> processing -> figure viewer.
//! Viewer keys: 1-6 poses, drag = orbit, wheel = zoom, a = auto-spin, 0 = reset view,
//!              s = export current pose (+ screenshot), n = new scan, Esc = quit.
use kinect3d::gfx::*;
use kinect3d::io;
use kinect3d::render::*;
use kinect3d::scan::capture::*;
use kinect3d::scan::rig::*;
use kinect3d::scan::segment::Segmenter;
use kinect3d::scan::world::*;
use kinect3d::source::*;
use kinect3d::synthetic::SyntheticKinect;
use minifb::{InputCallback, Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const CW: usize = 960;
const CH: usize = 720;
const TURNS_NEEDED: f64 = 2.0;

// colours (the numbers are OpenCV-style BGR, as in the original UI)
const ACCENT: Col = bgr(255, 200, 120);
const OK_GREEN: Col = bgr(120, 230, 120);
const WARN: Col = bgr(80, 180, 255);
const TOAST: Col = bgr(120, 220, 255);
const GRAY_T: Col = bgr(210, 210, 210);

/// Floor-independent check of whether the person runs off the bottom/top of the image.
/// Returns (feet_cut, head_cut).
fn framing(depth: &[u16], dist: f32) -> (bool, bool) {
    let target = dist * 1000.0;
    let band = |r: usize, c: usize| {
        let z = depth[r * W + c];
        z > 0 && (z as f32 - target).abs() < 350.0
    };
    let (mut cmin, mut cmax, mut n_cols) = (W, 0, 0);
    for c in 0..W {
        let n = (200..320).filter(|&r| band(r, c)).count();
        if n > 40 {
            cmin = cmin.min(c);
            cmax = cmax.max(c);
            n_cols += 1;
        }
    }
    if n_cols < 15 {
        return (false, false);
    }
    let (mut feet, mut head) = (false, false);
    for c in cmin..=cmax {
        let (z_mid, z_bot) = (depth[300 * W + c] as i32, depth[(H - 2) * W + c] as i32);
        if (H - 4..H).any(|r| band(r, c)) && z_mid > 0 && z_bot > 0 && (z_mid - z_bot).abs() < 120 && (300..476).filter(|&r| band(r, c)).count() as f32 / 176.0 > 0.7 {
            feet = true;
        }
        if (0..4).any(|r| band(r, c)) && (5..200).filter(|&r| band(r, c)).count() as f32 / 195.0 > 0.7 {
            head = true;
        }
    }
    (feet, head)
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
    floor_t: f64,
    last_res: Option<([f32; 2], f32)>,
    last_res_t: f64,
}

type Built = Result<(Figure, ()), String>;

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
    fig: Option<Figure>,
    folder: PathBuf,
    pose_i: usize,
    cur: Params,
    auto: bool,
    t_view: f64,
    prog: Arc<Mutex<(String, f32)>>,
    force_until: f64,
    autotilt: bool,
    raw: Option<RawSaver>,
    cap: Option<Capture>,
    center0: Option<[f32; 2]>,
    last_seq: u64,
    last_t: f64,
    last_mask: Option<Vec<bool>>,
    visible: bool,
    result: Arc<Mutex<Option<Built>>>,
    last_render: Vec<u32>,
}

impl Booth {
    fn new(src: Arc<dyn Source>, root: PathBuf) -> Booth {
        Booth {
            src,
            root,
            t0: Instant::now(),
            state: State::Name,
            t_state: 0.0,
            name: String::new(),
            msg: String::new(),
            msg_until: 0.0,
            orbit: Orbit3D::new(),
            world: World::default(),
            seg: None,
            probe: Probe { t: 0.0, dist: None, text: "Looking for you...".into(), ok: false, tilt_t: 0.0, floor_t: 0.0, last_res: None, last_res_t: 0.0 },
            fig: None,
            folder: PathBuf::new(),
            pose_i: 0,
            cur: [[0.0; 3]; 16],
            auto: false,
            t_view: 0.0,
            prog: Arc::new(Mutex::new((String::new(), 0.0))),
            force_until: 0.0,
            autotilt: true,
            raw: None,
            cap: None,
            center0: None,
            last_seq: u64::MAX,
            last_t: 0.0,
            last_mask: None,
            visible: true,
            result: Arc::new(Mutex::new(None)),
            last_render: Vec::new(),
        }
    }

    fn now(&self) -> f64 { self.t0.elapsed().as_secs_f64() }
    fn toast(&mut self, s: &str, secs: f64) {
        self.msg = s.to_string();
        self.msg_until = self.now() + secs;
    }

    fn open_saved(&mut self, folder: &Path) -> Result<(), String> {
        let (v, f, c) = io::load_ply(&folder.join("model.ply"))?;
        self.fig = Some(Figure::new(v, f, c));
        self.folder = folder.to_path_buf();
        self.name = folder.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        self.state = State::View;
        self.t_view = self.now();
        Ok(())
    }

    /// Live feedback while the visitor is typing their name / getting into place.
    fn probe(&mut self) {
        let now = self.now();
        let Some((_rgb, depth)) = self.src.frames() else { return };
        if now - self.probe.t < 0.5 {
            return;
        }
        self.probe.t = now;
        let dist = nearest_distance(&depth);
        self.probe.dist = dist;
        let (mut feet_cut, mut head_cut) = (false, false);
        if let Some(d) = dist {
            if d > 1.5 && d < 5.0 {
                (feet_cut, head_cut) = framing(&depth, d);
            }
        }
        // auto-framing: nudge the sensor tilt (rate limited: the motor dislikes rapid moves)
        if let (true, Some(d)) = (self.autotilt, dist) {
            if d > 2.2 && d < 4.6 && now - self.probe.tilt_t > 2.5 && !self.world.floor_found {
                let step = if feet_cut && !head_cut { -4 } else if head_cut && !feet_cut { 4 } else { 0 };
                let cur = self.src.tilt_target();
                let new = (cur + step).clamp(-15, 10);
                if step != 0 && new != cur {
                    self.src.set_tilt(new);
                    self.probe.tilt_t = now;
                }
            }
        }
        // keep re-fitting the floor, but never trust a fit while the feet are out of view
        if feet_cut {
            self.world = World::default();
        } else if now - self.probe.floor_t > 2.0 {
            self.probe.floor_t = now;
            match fit_floor(&depth, 0) {
                Some(w) => self.world = w,
                None => {
                    if dist.map_or(false, |d| d < 1.8) {
                        self.world = World::default();
                    }
                }
            }
        }
        let res = Segmenter::new(self.world).find_person(&depth);
        let res = match res {
            Some(r) => {
                self.probe.last_res = Some(r);
                self.probe.last_res_t = now;
                Some(r)
            }
            None if now - self.probe.last_res_t < 1.5 => self.probe.last_res, // brief dropouts shouldn't flicker the status
            None => None,
        };
        let auto = if self.autotilt { " (auto-tilting)" } else { "" };
        let (text, ok) = match dist {
            None => ("Nothing in view - step in front of the sensor".to_string(), false),
            Some(d) if d < 2.4 => (format!("{d:.1} m - too close, step back to about 3 m"), false),
            Some(d) if d > 4.0 => (format!("{d:.1} m - too far, come closer (about 3 m)"), false),
            Some(d) if feet_cut && head_cut => (format!("{d:.1} m - head and feet cut off: step back"), false),
            Some(d) if feet_cut => (format!("{d:.1} m - feet cut off: tilting sensor down{auto}, or step back"), false),
            Some(d) if head_cut => (format!("{d:.1} m - head cut off: tilting sensor up{auto}, or step back"), false),
            Some(d) => match res {
                None => (format!("{d:.1} m - good distance, but I can't see a full person yet"), false),
                Some((_, h)) if h < 1.3 || !self.world.floor_found => {
                    (format!("{d:.1} m - can't see the floor at your feet: tilt down a little or step back"), false)
                }
                Some((_, h)) => (format!("Perfect - {h:.2} m tall, {d:.1} m away"), true),
            },
        };
        self.probe.text = text;
        self.probe.ok = ok;
    }

    fn distance_meter(&self, c: &mut Canvas, y: i32) {
        let (x0, x1) = (170, CW as i32 - 170);
        let px = |m: f32| (x0 as f32 + (x1 - x0) as f32 * (m.clamp(1.0, 5.0) - 1.0) / 4.0) as i32;
        c.rect(x0, y, x1, y + 10, bgr(70, 70, 70));
        c.rect(px(2.6), y, px(3.6), y + 10, bgr(90, 200, 90));
        for m in 1..=5 {
            c.line(px(m as f32), y + 10, px(m as f32), y + 16, bgr(200, 200, 200));
            c.text(&format!("{m}m"), px(m as f32) - 12, y + 34, 0.45, bgr(190, 190, 190), false);
        }
        if let Some(d) = self.probe.dist {
            let col = if (2.6..=3.6).contains(&d) { OK_GREEN } else { WARN };
            c.triangle([[px(d), y - 2], [px(d) - 9, y - 16], [px(d) + 9, y - 16]], col);
            c.text(&format!("{d:.1} m"), px(d) - 26, y - 22, 0.6, col, true);
        }
    }

    // ------------------------------------------------------------------ states
    fn start_countdown(&mut self) {
        let Some((_, depth)) = self.src.frames() else {
            self.toast("Camera not ready yet", 3.0);
            return;
        };
        if let Some(w) = fit_floor(&depth, 0) {
            self.world = w;
        }
        let mut seg = Segmenter::new(self.world);
        if seg.find_person(&depth).is_none() {
            self.toast("I can't see you - step into view and try again", 4.0);
            return;
        }
        self.seg = Some(seg);
        self.state = State::Countdown;
        self.t_state = self.now();
    }

    fn start_capture(&mut self) {
        self.folder = new_scan_folder(&self.root, &self.name);
        self.raw = Some(RawSaver::new(&self.folder));
        self.cap = Some(Capture::new(self.seg.take().unwrap()));
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
            let mut e = serde_json::json!({"online_turns": q.trk.yaw().abs() / std::f64::consts::TAU, "quality": q.trk.quality, "name": self.name});
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
        *self.prog.lock().unwrap() = ("Aligning the two turns".into(), 0.05);
        let cap = self.cap.take().unwrap();
        let (world, center, folder, name) = (self.world, self.center0.unwrap(), self.folder.clone(), self.name.clone());
        let (prog, result) = (self.prog.clone(), self.result.clone());
        std::thread::spawn(move || {
            let p2 = prog.clone();
            let built = build_model(&cap.caps, &world, center, &move |s, f| *p2.lock().unwrap() = (s.to_string(), f), &mut |s| println!("{s}"));
            let out: Built = match built {
                Err(e) => Err(e),
                Ok(mesh) => {
                    *prog.lock().unwrap() = ("Rigging the skeleton".into(), 0.93);
                    let fig = Figure::new(mesh.verts.clone(), mesh.faces.clone(), mesh.rgb.clone());
                    let mut thumb = Canvas::new(CW, CH);
                    render_mesh(&mut thumb, &fig.rest, &fig.rest_normals, &fig.faces, &fig.rgb, &Orbit3D::new(), fig.top);
                    save_scan(&folder, &name, &mesh, Some(&fig), Some((&thumb.buf, CW, CH)));
                    println!("saved scan: {}", folder.display());
                    Ok((fig, ()))
                }
            };
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
            self.src.set_tilt(self.src.tilt_target() + 5);
        }
        if inp.key(Key::Down) {
            self.src.set_tilt(self.src.tilt_target() - 5);
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
        c.text_c("3D BODY SCAN", 90, 1.8, ACCENT, true);
        c.panel(120, 180, CW as i32 - 120, 330, 0.62);
        c.text_c("What's your name?", 235, 0.9, bgr(220, 220, 220), true);
        let cursor = if (self.now() * 2.0) as i64 % 2 == 0 { "_" } else { " " };
        c.text_c(&format!("{}{cursor}", self.name), 300, 1.5, WHITE, true);
        if self.now() < self.force_until {
            c.panel(120, 340, CW as i32 - 120, 410, 0.8);
            c.text_c("Not ready - press ENTER again to scan anyway", 385, 0.85, WARN, true);
        } else {
            c.text_c("Press ENTER to start", 372, 0.8, bgr(180, 255, 180), true);
        }
        let col = if self.probe.ok { OK_GREEN } else { WARN };
        c.panel(120, CH as i32 - 262, CW as i32 - 120, CH as i32 - 24, 0.62);
        c.text_c(&format!("{}{}", if self.probe.ok { "OK  " } else { "..  " }, self.probe.text), CH as i32 - 224, 0.75, col, true);
        self.distance_meter(&mut c, CH as i32 - 168);
        c.text_c("Stand ~3 m back, arms slightly away from your body (A-pose)", CH as i32 - 106, 0.6, GRAY_T, false);
        c.text_c("After 3-2-1, turn slowly on the spot - TWO full turns (about 10 s each)", CH as i32 - 80, 0.6, GRAY_T, false);
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
        // keep tracking where they stand, so the capture volume is centred on them
        if let (Some((_, depth)), Some(seg)) = (&frames, self.seg.as_mut()) {
            if ((left * 4.0) as i64) % 2 == 0 {
                seg.find_person(depth);
            }
        }
        let mut c = Canvas::new(CW, CH);
        if let Some((rgb, _)) = &frames {
            c.camera(rgb, None, 0.35, true);
        }
        if left <= 0.0 && self.state == State::Countdown {
            self.start_capture();
            return c;
        }
        let n = left.ceil() as i32;
        let frac = if left != left.floor() { left - left.floor() } else { 1.0 } as f32;
        let scale = 7.0 + 2.5 * frac;
        let s = n.to_string();
        let (tw, th) = (text_width(&s, scale), (px_for(scale) * 0.72) as i32);
        c.text(&s, (CW as i32 - tw) / 2, (CH as i32 + th) / 2, scale, WHITE, true);
        c.text_c(&format!("Get ready, {}!", self.name), 120, 1.2, TOAST, true);
        c.text_c("Then turn slowly on the spot - two full turns", CH as i32 - 60, 0.8, WHITE, true);
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
        c.text_c("SPIN SLOWLY!", 70, 1.4, ACCENT, true);
        let (x0, x1, y) = (160, CW as i32 - 160, CH as i32 - 70);
        c.panel(x0 - 14, y - 52, x1 + 14, y + 24, 0.7);
        c.rect(x0, y - 12, x1, y + 6, bgr(70, 70, 70));
        c.rect(x0, y - 12, x0 + ((x1 - x0) as f64 * (turns / TURNS_NEEDED).min(1.0)) as i32, y + 6, OK_GREEN);
        for t in [0.5, 1.0, 1.5] {
            let xx = x0 + ((x1 - x0) as f64 * t / TURNS_NEEDED) as i32;
            c.line(xx, y - 12, xx, y + 6, bgr(200, 200, 200));
        }
        c.text_c(&format!("Turn {:.1} of {}", turns.min(TURNS_NEEDED), TURNS_NEEDED as i32), y - 22, 0.8, WHITE, false);
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
                Ok((fig, _)) => {
                    self.fig = Some(fig);
                    self.cur = [[0.0; 3]; 16];
                    self.pose_i = 0;
                    self.t_view = self.now();
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
        c.text_c("Building your 3D model...", CH as i32 / 2 - 60, 1.3, ACCENT, true);
        c.text_c(&label, CH as i32 / 2 - 10, 0.8, WHITE, true);
        let (x0, x1, y) = (200, CW as i32 - 200, CH as i32 / 2 + 40);
        c.rect(x0, y, x1, y + 16, bgr(70, 70, 70));
        c.rect(x0, y, x0 + ((x1 - x0) as f32 * frac.clamp(0.0, 1.0)) as i32, y + 16, OK_GREEN);
        c
    }

    fn tick_view(&mut self, inp: &Input) -> Canvas {
        for ch in &inp.chars {
            match ch {
                '1'..='6' => self.pose_i = (*ch as u8 - b'1') as usize,
                'a' => self.auto = !self.auto,
                '0' => self.orbit.reset(),
                's' => self.export(),
                'n' => {
                    self.name.clear();
                    self.fig = None;
                    self.state = State::Name;
                    return self.tick_name(&Input::default());
                }
                _ => {}
            }
        }
        let t = (self.now() - self.t_view) as f32;
        let fig = self.fig.as_ref().unwrap();
        let target = fig.pose_params(POSES[self.pose_i], t);
        for j in 0..16 {
            for k in 0..3 {
                self.cur[j][k] += (target[j][k] - self.cur[j][k]) * 0.3; // ease toward the target pose
            }
        }
        if self.auto {
            self.orbit.yaw += 0.02;
        }
        let (verts, normals) = fig.posed(&self.cur);
        let mut c = Canvas::new(CW, CH);
        render_mesh(&mut c, &verts, &normals, &fig.faces, &fig.rgb, &self.orbit, fig.top);
        self.last_render = c.buf.clone();
        c.text(&self.name, 24, 44, 1.1, WHITE, true);
        c.text(&format!("{:.2} m tall", fig.top), 24, 76, 0.6, bgr(180, 180, 180), true);
        for (i, nme) in POSES.iter().enumerate() {
            let on = i == self.pose_i;
            c.text(&format!("{}  {}", i + 1, nme), CW as i32 - 220, 50 + 38 * i as i32, 0.8, if on { OK_GREEN } else { bgr(200, 200, 200) }, true);
        }
        c.text("drag: orbit   wheel: zoom   a: auto-spin   0: reset", 24, CH as i32 - 60, 0.55, bgr(160, 160, 160), false);
        c.text("s: export pose + screenshot   n: new scan   Esc: quit", 24, CH as i32 - 32, 0.55, bgr(160, 160, 160), false);
        c
    }

    fn export(&mut self) {
        let Some(fig) = &self.fig else { return };
        let (verts, _) = fig.posed(&self.cur);
        let tag = POSES[self.pose_i].to_lowercase().replace(' ', "_");
        let stamp = timestamp("%H%M%S");
        let base = self.folder.join(format!("pose_{tag}_{stamp}"));
        let _ = io::save_ply(&base.with_extension("ply"), &verts, &fig.faces, &fig.rgb);
        let _ = io::save_obj(&base.with_extension("obj"), &verts, &fig.faces, &fig.rgb);
        io::save_rgb_png(&base.with_extension("png"), &self.last_render, CW, CH);
        let s = format!("Saved {tag} to {}", self.folder.display());
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
        println!("scan_booth [--demo] [--view DIR] [--out DIR] [--autorun NAME]");
        return;
    }
    let out = val("--out").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../scans"));
    let view = val("--view");
    let src: Arc<dyn Source> = if view.is_some() {
        Arc::new(NoCam::new())
    } else if flag("--demo") {
        Arc::new(SyntheticKinect::new(9.0, 30.0, 4.0))
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
    let mut win = Window::new("3D Body Scan", CW, CH, WindowOptions { resize: true, scale_mode: minifb::ScaleMode::AspectRatioStretch, ..Default::default() }).expect("window");
    win.set_target_fps(30);
    let autorun = val("--autorun"); // demo/testing: type this name and start by itself after a few seconds
    let started = Instant::now();
    let mut autorun_done = false;
    let typed = Arc::new(Mutex::new(Vec::new()));
    win.set_input_callback(Box::new(Chars(typed.clone())));
    while win.is_open() {
        let inp = Input { chars: std::mem::take(&mut *typed.lock().unwrap()), keys: win.get_keys_pressed(KeyRepeat::Yes) };
        if inp.key(Key::Escape) && matches!(booth.state, State::Name | State::View) {
            break;
        }
        let pos = win.get_mouse_pos(MouseMode::Clamp);
        let wheel = win.get_scroll_wheel().map_or(0.0, |w| w.1);
        booth.orbit.mouse(pos.map(|p| p), win.get_mouse_down(MouseButton::Left) && booth.state == State::View, if booth.state == State::View { wheel } else { 0.0 });
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
