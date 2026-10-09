//! KnctXplore: one menu for everything Kinect - live viewer, both scan booths, the scan library,
//! recordings and the self-test. Each tool runs as its own window (the sensor can only be opened
//! by one program at a time); this launcher never touches the Kinect itself.
use kinect3d::gfx::*;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const W: usize = 1120;
const H: usize = 720;

const BG: Col = [18, 20, 26];
const CARD: Col = [34, 38, 48];
const CARD_HI: Col = [52, 60, 78];
const ACCENT: Col = [255, 170, 60];
const TXT: Col = [235, 238, 245];
const DIM: Col = [140, 148, 165];
const GOOD: Col = [90, 210, 120];
const BAD: Col = [235, 95, 95];

#[derive(Clone, Copy, PartialEq)]
enum Screen {
    Home,
    Library,
    Recordings,
    SelfTest,
}

#[derive(Clone, Copy)]
enum Tool {
    Viewer,
    Body,
    Bust,
    Library,
    Recordings,
    SelfTest,
}

struct Tile {
    key: char,
    title: &'static str,
    blurb: [&'static str; 2],
    tool: Tool,
    accent: Col,
}

const TILES: [Tile; 6] = [
    Tile { key: '1', title: "Live Viewer", blurb: ["RGB + depth overlay, 3D point cloud,", "recording, snapshots"], tool: Tool::Viewer, accent: [90, 170, 255] },
    Tile { key: '2', title: "Body Booth", blurb: ["Stand and spin: full-body 3D scan", "with a posable rigged figure"], tool: Tool::Body, accent: [120, 220, 140] },
    Tile { key: '3', title: "Bust Booth", blurb: ["Sit in a chair and turn: close-up", "head-and-shoulders scan"], tool: Tool::Bust, accent: [255, 170, 60] },
    Tile { key: '4', title: "Scan Library", blurb: ["Open, view and rebuild", "your saved scans"], tool: Tool::Library, accent: [200, 140, 255] },
    Tile { key: '5', title: "Recordings", blurb: ["Play back recorded RGB + depth", "sequences in the viewer"], tool: Tool::Recordings, accent: [255, 120, 150] },
    Tile { key: '6', title: "Self-test", blurb: ["Run the whole scan pipeline on a", "synthetic person - no sensor needed"], tool: Tool::SelfTest, accent: [120, 215, 215] },
];

struct ScanEntry {
    dir: PathBuf,
    name: String,
    bust: bool,
    created: String,
    verts: u64,
    has_raw: bool,
}

struct RecEntry {
    dir: PathBuf,
    name: String,
    frames: usize,
}

struct Thumb {
    w: usize,
    h: usize,
    px: Vec<u32>,
}

struct Running {
    child: Child,
    label: String,
    started: Instant,
    /// refresh the scan list when this finishes
    refresh: bool,
}

struct App {
    screen: Screen,
    demo: bool,
    sel: usize,
    scroll: usize,
    scans: Vec<ScanEntry>,
    recs: Vec<RecEntry>,
    thumbs: HashMap<PathBuf, Option<Thumb>>,
    running: Option<Running>,
    toast: Option<(String, Instant)>,
    kinect: bool,
    kinect_checked: Instant,
    selftest_log: Arc<Mutex<Vec<String>>>,
    selftest_done: Arc<Mutex<bool>>,
    root: PathBuf,
    bin_dir: PathBuf,
}

fn kinect_present() -> bool {
    let Ok(rd) = std::fs::read_dir("/sys/bus/usb/devices") else { return false };
    rd.filter_map(|e| e.ok()).any(|e| {
        let rd = |f: &str| std::fs::read_to_string(e.path().join(f)).unwrap_or_default().trim().to_string();
        rd("idVendor") == "045e" && rd("idProduct") == "02ae"
    })
}

fn load_scans(dir: &Path) -> Vec<ScanEntry> {
    let mut v: Vec<ScanEntry> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.path().join("model.ply").exists() || e.path().join("raw").exists()).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .map(|e| {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let meta: serde_json::Value = std::fs::read_to_string(path.join("meta.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
            let bust = name.contains("_bust_") || std::fs::read_to_string(path.join("raw/capture.json")).map_or(false, |t| t.contains("\"profile\":\"bust\""));
            ScanEntry {
                created: meta["created"].as_str().unwrap_or("").to_string(),
                verts: meta["vertices"].as_u64().unwrap_or(0),
                has_raw: path.join("raw/capture.json").exists(),
                bust,
                name,
                dir: path,
            }
        })
        .collect();
    v.sort_by(|a, b| b.name.rsplit('_').take(2).collect::<Vec<_>>().cmp(&a.name.rsplit('_').take(2).collect::<Vec<_>>()));
    v
}

fn load_recs(dir: &Path) -> Vec<RecEntry> {
    let mut v: Vec<RecEntry> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .map(|e| {
            let frames = std::fs::read_dir(e.path()).map_or(0, |rd| rd.filter_map(|f| f.ok()).filter(|f| f.file_name().to_string_lossy().starts_with("depth_")).count());
            RecEntry { name: e.file_name().to_string_lossy().to_string(), dir: e.path(), frames }
        })
        .filter(|r| r.frames > 0)
        .collect();
    v.sort_by(|a, b| b.name.cmp(&a.name));
    v
}

fn load_thumb(dir: &Path, bust: bool) -> Option<Thumb> {
    let img = image::open(dir.join("preview.png")).ok()?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    // bust previews are a 4-view sheet: use the first view
    let cw = if w >= 2 * h { w / 4 } else { w };
    let _ = bust;
    let (tw, th) = (150usize, 100usize);
    let scale = (tw as f32 / cw as f32).min(th as f32 / h as f32);
    let (ow, oh) = ((cw as f32 * scale) as usize, (h as f32 * scale) as usize);
    let mut px = vec![pack(CARD); tw * th];
    for y in 0..oh {
        for x in 0..ow {
            let (sx, sy) = ((x as f32 / scale) as u32, (y as f32 / scale) as u32);
            let p = img.get_pixel(sx.min(cw as u32 - 1), sy.min(h as u32 - 1));
            px[(y + (th - oh) / 2) * tw + x + (tw - ow) / 2] = pack([p[0], p[1], p[2]]);
        }
    }
    Some(Thumb { w: tw, h: th, px })
}

impl App {
    fn new() -> App {
        let root = std::env::var("KNCT_ROOT").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."));
        let bin_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())).unwrap_or_default();
        let mut a = App {
            screen: Screen::Home,
            demo: false,
            sel: 0,
            scroll: 0,
            scans: vec![],
            recs: vec![],
            thumbs: HashMap::new(),
            running: None,
            toast: None,
            kinect: kinect_present(),
            kinect_checked: Instant::now(),
            selftest_log: Arc::new(Mutex::new(vec![])),
            selftest_done: Arc::new(Mutex::new(true)),
            root,
            bin_dir,
        };
        a.refresh();
        a
    }

    fn refresh(&mut self) {
        self.scans = load_scans(&self.root.join("scans"));
        self.recs = load_recs(&self.root.join("recordings"));
        self.thumbs.clear();
    }

    fn say(&mut self, s: &str) { self.toast = Some((s.to_string(), Instant::now())); }

    fn launch(&mut self, bin: &str, args: &[String], label: &str, refresh: bool) {
        if self.running.is_some() {
            return;
        }
        let exe = self.bin_dir.join(bin);
        match Command::new(&exe).args(args).current_dir(&self.root).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
            Ok(child) => self.running = Some(Running { child, label: label.to_string(), started: Instant::now(), refresh }),
            Err(e) => self.say(&format!("Can't start {bin}: {e}")),
        }
    }

    fn open_tool(&mut self, t: Tool) {
        let demo: Vec<String> = if self.demo { vec!["--demo".into()] } else { vec![] };
        match t {
            Tool::Viewer => self.launch("kinect_overlay", &[], "Live Viewer", false),
            Tool::Body => self.launch("scan_booth", &demo, "Body Booth", true),
            Tool::Bust => self.launch("bust_booth", &demo, "Bust Booth", true),
            Tool::Library => {
                self.screen = Screen::Library;
                self.sel = 0;
                self.scroll = 0;
            }
            Tool::Recordings => {
                self.screen = Screen::Recordings;
                self.sel = 0;
                self.scroll = 0;
            }
            Tool::SelfTest => self.run_selftest(),
        }
    }

    fn run_selftest(&mut self) {
        if !*self.selftest_done.lock().unwrap() {
            self.screen = Screen::SelfTest;
            return;
        }
        self.screen = Screen::SelfTest;
        self.selftest_log.lock().unwrap().clear();
        *self.selftest_done.lock().unwrap() = false;
        let (log, done) = (self.selftest_log.clone(), self.selftest_done.clone());
        let exe = self.bin_dir.join("selftest");
        std::thread::spawn(move || {
            match Command::new(&exe).stdout(Stdio::piped()).stderr(Stdio::null()).spawn() {
                Ok(mut c) => {
                    if let Some(out) = c.stdout.take() {
                        for line in std::io::BufReader::new(out).lines().map_while(Result::ok) {
                            log.lock().unwrap().push(line);
                        }
                    }
                    let ok = c.wait().map_or(false, |s| s.success());
                    log.lock().unwrap().push(if ok { "-- finished OK --".into() } else { "-- FAILED --".into() });
                }
                Err(e) => log.lock().unwrap().push(format!("cannot start selftest: {e}")),
            }
            *done.lock().unwrap() = true;
        });
    }

    fn open_scan(&mut self, i: usize) {
        let Some(s) = self.scans.get(i) else { return };
        let bin = if s.bust { "bust_booth" } else { "scan_booth" };
        let args = vec!["--view".to_string(), s.dir.to_string_lossy().to_string()];
        let label = format!("Viewing {}", s.name);
        self.launch(bin, &args, &label, false);
    }

    fn rescan(&mut self, i: usize) {
        let Some(s) = self.scans.get(i) else { return };
        if !s.has_raw {
            self.say("This scan has no raw frames to rebuild");
            return;
        }
        let args = vec![s.dir.to_string_lossy().to_string()];
        let label = format!("Rebuilding {}", s.name);
        self.launch("rescan", &args, &label, true);
    }

    fn play_rec(&mut self, i: usize) {
        let Some(r) = self.recs.get(i) else { return };
        let args = vec!["--play".to_string(), r.dir.to_string_lossy().to_string()];
        let label = format!("Playing {}", r.name);
        self.launch("kinect_overlay", &args, &label, false);
    }

    fn poll(&mut self) {
        if let Some(r) = &mut self.running {
            if let Ok(Some(_)) = r.child.try_wait() {
                let refresh = r.refresh;
                self.running = None;
                if refresh {
                    self.refresh();
                }
            }
        }
        if self.kinect_checked.elapsed() > Duration::from_secs(2) {
            self.kinect = kinect_present();
            self.kinect_checked = Instant::now();
        }
        if let Some((_, t)) = &self.toast {
            if t.elapsed() > Duration::from_secs(4) {
                self.toast = None;
            }
        }
    }
}

// ---------- layout ----------
const TILE_W: i32 = 330;
const TILE_H: i32 = 190;
const GAP: i32 = 24;

fn tile_rect(i: usize) -> (i32, i32, i32, i32) {
    let (cols, rows) = (3, 2);
    let total_w = cols * TILE_W + (cols - 1) * GAP;
    let x0 = (W as i32 - total_w) / 2;
    let y0 = 190;
    let _ = rows;
    let (c, r) = ((i % 3) as i32, (i / 3) as i32);
    let x = x0 + c * (TILE_W + GAP);
    let y = y0 + r * (TILE_H + GAP);
    (x, y, x + TILE_W, y + TILE_H)
}

const ROW_H: i32 = 112;
const LIST_TOP: i32 = 120;
const LIST_X0: i32 = 60;
const LIST_X1: i32 = W as i32 - 60;

fn visible_rows() -> usize { ((H as i32 - LIST_TOP - 70) / ROW_H) as usize }

fn inside(p: (f32, f32), r: (i32, i32, i32, i32)) -> bool { p.0 >= r.0 as f32 && p.0 < r.2 as f32 && p.1 >= r.1 as f32 && p.1 < r.3 as f32 }

fn box_rounded(c: &mut Canvas, r: (i32, i32, i32, i32), col: Col) { c.rect(r.0, r.1, r.2, r.3, col) }

fn blit_thumb(c: &mut Canvas, t: &Thumb, x: i32, y: i32) {
    for yy in 0..t.h {
        for xx in 0..t.w {
            let (px, py) = (x + xx as i32, y + yy as i32);
            if px >= 0 && py >= 0 && (px as usize) < c.w && (py as usize) < c.h {
                c.buf[py as usize * c.w + px as usize] = t.px[yy * t.w + xx];
            }
        }
    }
}

fn header(c: &mut Canvas, title: &str, sub: &str) {
    c.rect(0, 0, W as i32, 86, [24, 27, 35]);
    c.rect(0, 86, W as i32, 88, ACCENT);
    c.text(title, 60, 56, 1.3, TXT, false);
    c.text(sub, 60 + text_width(title, 1.3) + 24, 56, 0.6, DIM, false);
}

fn draw_home(app: &App, c: &mut Canvas, mouse: (f32, f32)) {
    c.fill(BG);
    // title
    c.text("Knct", 60, 100, 2.0, TXT, false);
    c.text("Xplore", 60 + text_width("Knct", 2.0), 100, 2.0, ACCENT, false);
    c.text("Xbox 360 Kinect on Linux - scan yourself in 3D", 64, 138, 0.65, DIM, false);
    // sensor chip
    let (txt, col) = if app.kinect { ("Kinect connected", GOOD) } else { ("No Kinect found", BAD) };
    let w = text_width(txt, 0.6) + 50;
    let x1 = W as i32 - 60;
    c.rect(x1 - w, 70, x1, 108, CARD);
    c.rect(x1 - w + 14, 84, x1 - w + 28, 98, col);
    c.text(txt, x1 - w + 40, 100, 0.6, TXT, false);

    for (i, t) in TILES.iter().enumerate() {
        let r = tile_rect(i);
        let hot = inside(mouse, r) || app.sel == i;
        box_rounded(c, r, if hot { CARD_HI } else { CARD });
        c.rect(r.0, r.1, r.0 + 8, r.3, t.accent);
        if hot {
            c.rect(r.0 + 8, r.3 - 3, r.2, r.3, t.accent);
        }
        c.text(&t.key.to_string(), r.2 - 40, r.1 + 40, 0.9, [90, 98, 118], false);
        c.text(t.title, r.0 + 30, r.1 + 70, 1.0, TXT, false);
        c.text(t.blurb[0], r.0 + 30, r.1 + 112, 0.5, DIM, false);
        c.text(t.blurb[1], r.0 + 30, r.1 + 138, 0.5, DIM, false);
        let n = match t.tool {
            Tool::Library => Some(format!("{} scan{}", app.scans.len(), if app.scans.len() == 1 { "" } else { "s" })),
            Tool::Recordings => Some(format!("{} recording{}", app.recs.len(), if app.recs.len() == 1 { "" } else { "s" })),
            Tool::Body | Tool::Bust if app.demo => Some("demo mode".to_string()),
            _ => None,
        };
        if let Some(n) = n {
            c.text(&n, r.0 + 30, r.1 + 172, 0.5, t.accent, false);
        }
    }
    // footer
    let y = H as i32 - 40;
    let demo_r = (60, y - 28, 60 + 420, y + 12);
    let hot = inside(mouse, demo_r);
    c.rect(demo_r.0, demo_r.1, demo_r.2, demo_r.3, if hot { CARD_HI } else { CARD });
    c.rect(demo_r.0 + 12, y - 16, demo_r.0 + 32, y + 4, if app.demo { ACCENT } else { [60, 66, 84] });
    c.text("D   Demo mode: booths use a synthetic person", demo_r.0 + 46, y, 0.5, TXT, false);
    c.text("Arrows/1-6/click: choose    Enter: open    Esc: quit", W as i32 - 60 - text_width("Arrows/1-6/click: choose    Enter: open    Esc: quit", 0.5), y, 0.5, DIM, false);
}

fn draw_list_scans(app: &mut App, c: &mut Canvas, mouse: (f32, f32)) {
    c.fill(BG);
    header(c, "Scan Library", &format!("{} scans  -  scans/ folder", app.scans.len()));
    if app.scans.is_empty() {
        c.text_c("No scans yet - make one in a booth", 360, 0.8, DIM, false);
    }
    let rows = visible_rows();
    for k in 0..rows {
        let i = app.scroll + k;
        if i >= app.scans.len() {
            break;
        }
        let y = LIST_TOP + k as i32 * ROW_H;
        let r = (LIST_X0, y, LIST_X1, y + ROW_H - 8);
        let hot = i == app.sel || inside(mouse, r);
        box_rounded(c, r, if hot { CARD_HI } else { CARD });
        let (dir, bust) = (app.scans[i].dir.clone(), app.scans[i].bust);
        let th = app.thumbs.entry(dir.clone()).or_insert_with(|| load_thumb(&dir, bust));
        if let Some(t) = th {
            blit_thumb(c, t, r.0 + 4, r.1 + 2);
        }
        let s = &app.scans[i];
        let accent = if s.bust { ACCENT } else { [120, 220, 140] };
        c.text(&s.name, r.0 + 175, r.1 + 36, 0.8, TXT, false);
        c.rect(r.0 + 175, r.1 + 52, r.0 + 175 + 70, r.1 + 76, accent);
        c.text(if s.bust { "BUST" } else { "BODY" }, r.0 + 188, r.1 + 71, 0.45, [20, 20, 20], false);
        let info = format!("{}   {} vertices{}", s.created, s.verts, if s.has_raw { "   raw frames kept" } else { "" });
        c.text(&info, r.0 + 262, r.1 + 72, 0.5, DIM, false);
        if hot {
            let t = "Enter: view   R: rebuild";
            c.text(t, r.2 - 20 - text_width(t, 0.5), r.1 + 36, 0.5, accent, false);
        }
    }
    footer(c, "Up/Down: choose    Enter: view    R: rebuild from raw frames    Esc: back");
}

fn draw_list_recs(app: &App, c: &mut Canvas, mouse: (f32, f32)) {
    c.fill(BG);
    header(c, "Recordings", &format!("{} recordings  -  recordings/ folder", app.recs.len()));
    if app.recs.is_empty() {
        c.text_c("No recordings yet - press R in the Live Viewer to record", 360, 0.8, DIM, false);
    }
    let rows = visible_rows() + 2;
    for k in 0..rows {
        let i = app.scroll + k;
        if i >= app.recs.len() {
            break;
        }
        let y = LIST_TOP + k as i32 * 72;
        let r = (LIST_X0, y, LIST_X1, y + 64);
        let hot = i == app.sel || inside(mouse, r);
        box_rounded(c, r, if hot { CARD_HI } else { CARD });
        c.rect(r.0, r.1, r.0 + 8, r.3, [255, 120, 150]);
        c.text(&app.recs[i].name, r.0 + 30, r.1 + 40, 0.75, TXT, false);
        c.text(&format!("{} frames  (~{:.0} s)", app.recs[i].frames, app.recs[i].frames as f32 / 30.0), r.0 + 360, r.1 + 40, 0.55, DIM, false);
        if hot {
            let t = "Enter: play";
            c.text(t, r.2 - 20 - text_width(t, 0.5), r.1 + 40, 0.5, [255, 120, 150], false);
        }
    }
    footer(c, "Up/Down: choose    Enter: play in the viewer    Esc: back");
}

fn draw_selftest(app: &App, c: &mut Canvas) {
    c.fill(BG);
    let done = *app.selftest_done.lock().unwrap();
    header(c, "Self-test", if done { "finished" } else { "running..." });
    let log = app.selftest_log.lock().unwrap();
    let max = ((H as i32 - LIST_TOP - 70) / 26) as usize;
    let start = log.len().saturating_sub(max);
    for (k, l) in log[start..].iter().enumerate() {
        let col = if l.contains("FAILED") { BAD } else if l.contains("OK") { GOOD } else { TXT };
        c.text(l, 60, LIST_TOP + 20 + k as i32 * 26, 0.5, col, false);
    }
    footer(c, "Esc: back");
}

fn footer(c: &mut Canvas, s: &str) {
    c.text(s, 60, H as i32 - 30, 0.5, DIM, false);
}

fn draw_running(app: &App, c: &mut Canvas) {
    c.panel(0, 0, W as i32, H as i32, 0.55);
    let Some(r) = &app.running else { return };
    c.rect(W as i32 / 2 - 330, H as i32 / 2 - 90, W as i32 / 2 + 330, H as i32 / 2 + 90, CARD);
    c.rect(W as i32 / 2 - 330, H as i32 / 2 - 90, W as i32 / 2 + 330, H as i32 / 2 - 84, ACCENT);
    c.text_c(&r.label, H as i32 / 2 - 20, 0.9, TXT, false);
    let dots = ".".repeat(1 + (r.started.elapsed().as_millis() / 400 % 3) as usize);
    c.text_c(&format!("Running{dots}  close its window to come back here"), H as i32 / 2 + 40, 0.5, DIM, false);
}

fn draw_toast(app: &App, c: &mut Canvas) {
    if let Some((s, _)) = &app.toast {
        let w = text_width(s, 0.55) + 50;
        let x0 = (W as i32 - w) / 2;
        c.rect(x0, H as i32 - 90, x0 + w, H as i32 - 50, [60, 40, 40]);
        c.text(s, x0 + 25, H as i32 - 62, 0.55, TXT, false);
    }
}

fn render(app: &mut App, c: &mut Canvas, mouse: (f32, f32)) {
    match app.screen {
        Screen::Home => draw_home(app, c, mouse),
        Screen::Library => draw_list_scans(app, c, mouse),
        Screen::Recordings => draw_list_recs(app, c, mouse),
        Screen::SelfTest => draw_selftest(app, c),
    }
    if app.running.is_some() {
        draw_running(app, c);
    }
    draw_toast(app, c);
}

fn list_len(app: &App) -> usize {
    match app.screen {
        Screen::Library => app.scans.len(),
        Screen::Recordings => app.recs.len(),
        _ => 0,
    }
}

fn row_at(app: &App, mouse: (f32, f32)) -> Option<usize> {
    let (top, h, rows) = match app.screen {
        Screen::Library => (LIST_TOP, ROW_H, visible_rows()),
        Screen::Recordings => (LIST_TOP, 72, visible_rows() + 2),
        _ => return None,
    };
    for k in 0..rows {
        let y = top + k as i32 * h;
        if inside(mouse, (LIST_X0, y, LIST_X1, y + h - 8)) && app.scroll + k < list_len(app) {
            return Some(app.scroll + k);
        }
    }
    None
}

fn activate(app: &mut App, i: usize) {
    match app.screen {
        Screen::Library => app.open_scan(i),
        Screen::Recordings => app.play_rec(i),
        _ => {}
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut app = App::new();
    // headless screenshot of a screen, for checking the layout: knctxplore --shot home|library|recordings out.png
    if let Some(i) = args.iter().position(|a| a == "--shot") {
        let which = args.get(i + 1).map(|s| s.as_str()).unwrap_or("home");
        app.screen = match which {
            "library" => Screen::Library,
            "recordings" => Screen::Recordings,
            _ => Screen::Home,
        };
        let mut c = Canvas::new(W, H);
        render(&mut app, &mut c, (-1.0, -1.0));
        kinect3d::io::save_rgb_png(Path::new(args.get(i + 2).map(|s| s.as_str()).unwrap_or("/tmp/knctxplore.png")), &c.buf, W, H);
        return;
    }
    let mut win = Window::new("KnctXplore", W, H, WindowOptions { resize: true, scale_mode: minifb::ScaleMode::AspectRatioStretch, ..Default::default() }).expect("window");
    win.set_target_fps(30);
    let mut canvas = Canvas::new(W, H);
    let mut was_down = false;
    let mut quiet_until = Instant::now();
    while win.is_open() {
        app.poll();
        let mouse = win.get_mouse_pos(MouseMode::Clamp).unwrap_or((-1.0, -1.0));
        let down = win.get_mouse_down(MouseButton::Left);
        let clicked = down && !was_down;
        was_down = down;
        // no key auto-repeat for launching: a key-up lost to the tool's window would otherwise look
        // like a held key and re-launch the tool every time it closes
        let mut keys = win.get_keys_pressed(KeyRepeat::No);
        if !win.is_active() {
            keys.clear();
        }
        let busy = app.running.is_some();
        if busy {
            quiet_until = Instant::now() + Duration::from_millis(600);
        }
        let quiet = Instant::now() < quiet_until;
        if quiet {
            keys.clear();
        }
        let clicked = clicked && !quiet && win.is_active();
        if !busy {
            let wheel = win.get_scroll_wheel().map_or(0.0, |w| w.1);
            match app.screen {
                Screen::Home => {
                    if keys.contains(&Key::Escape) {
                        break;
                    }
                    for k in &keys {
                        let n = match k {
                            Key::Key1 => Some(0),
                            Key::Key2 => Some(1),
                            Key::Key3 => Some(2),
                            Key::Key4 => Some(3),
                            Key::Key5 => Some(4),
                            Key::Key6 => Some(5),
                            _ => None,
                        };
                        if let Some(n) = n {
                            app.sel = n;
                            app.open_tool(TILES[n].tool);
                        }
                    }
                    if keys.contains(&Key::D) {
                        app.demo = !app.demo;
                    }
                    if keys.contains(&Key::Right) {
                        app.sel = (app.sel + 1).min(5);
                    }
                    if keys.contains(&Key::Left) {
                        app.sel = app.sel.saturating_sub(1);
                    }
                    if keys.contains(&Key::Down) {
                        app.sel = (app.sel + 3).min(5);
                    }
                    if keys.contains(&Key::Up) {
                        app.sel = app.sel.saturating_sub(3);
                    }
                    if keys.contains(&Key::Enter) {
                        app.open_tool(TILES[app.sel].tool);
                    }
                    if clicked {
                        if let Some(i) = (0..TILES.len()).find(|&i| inside(mouse, tile_rect(i))) {
                            app.sel = i;
                            app.open_tool(TILES[i].tool);
                        } else if inside(mouse, (60, H as i32 - 68, 480, H as i32 - 28)) {
                            app.demo = !app.demo;
                        }
                    }
                }
                Screen::Library | Screen::Recordings => {
                    let n = list_len(&app);
                    let rows = if app.screen == Screen::Library { visible_rows() } else { visible_rows() + 2 };
                    if keys.contains(&Key::Escape) {
                        app.sel = if app.screen == Screen::Library { 3 } else { 4 };
                        app.screen = Screen::Home;
                    }
                    if keys.contains(&Key::Down) && app.sel + 1 < n {
                        app.sel += 1;
                    }
                    if keys.contains(&Key::Up) {
                        app.sel = app.sel.saturating_sub(1);
                    }
                    if wheel != 0.0 {
                        let d = if wheel > 0.0 { -1i64 } else { 1 };
                        app.scroll = (app.scroll as i64 + d).clamp(0, (n as i64 - rows as i64).max(0)) as usize;
                    }
                    if app.sel < app.scroll {
                        app.scroll = app.sel;
                    }
                    if app.sel >= app.scroll + rows {
                        app.scroll = app.sel + 1 - rows;
                    }
                    if keys.contains(&Key::Enter) && n > 0 {
                        { let s = app.sel; activate(&mut app, s); }
                    }
                    if keys.contains(&Key::R) && app.screen == Screen::Library && n > 0 {
                        { let s = app.sel; app.rescan(s); }
                    }
                    if clicked {
                        if let Some(i) = row_at(&app, mouse) {
                            app.sel = i;
                            activate(&mut app, i);
                        }
                    }
                }
                Screen::SelfTest => {
                    if keys.contains(&Key::Escape) {
                        app.screen = Screen::Home;
                        app.sel = 5;
                    }
                }
            }
        }
        render(&mut app, &mut canvas, mouse);
        if win.update_with_buffer(&canvas.buf, W, H).is_err() {
            break;
        }
    }
    if let Some(r) = &mut app.running {
        let _ = r.child.kill();
    }
}
