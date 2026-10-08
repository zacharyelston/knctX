//! Debug: dump the segmentation mask of one raw frame of a scan: maskdump <scan> <frame> <out.png>
use kinect3d::scan::segment::Segmenter;
use kinect3d::scan::world::World;
use kinect3d::source::{H, W};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let raw = std::path::Path::new(&a[1]).join("raw");
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(raw.join("capture.json")).unwrap()).unwrap();
    let f = |v: &serde_json::Value| v.as_f64().unwrap() as f32;
    let world = World { r: [0, 1, 2].map(|i| [0, 1, 2].map(|j| f(&cfg["world_R"][i][j]))), t: [0, 1, 2].map(|i| f(&cfg["world_t"][i])), floor_found: true };
    let mut seg = Segmenter::new(world);
    if cfg["profile"] == "bust" {
        seg.bust_defaults();
        seg.y_lo = f(&cfg["y_lo"]);
        seg.y_hi = f(&cfg["y_hi"]);
    }
    seg.center = Some([f(&cfg["center"][0]), f(&cfg["center"][1])]);
    let i: usize = a[2].parse().unwrap();
    let (rgb, d) = kinect3d::io::load_frame(&raw, &format!("rgb_{i:05}.jpg"), &format!("depth_{i:05}.png")).unwrap();
    let (m, _) = seg.mask(&d);
    let mut buf = vec![0u32; W * H];
    for p in 0..W * H {
        let c = [rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2]];
        let k = if m[p] { 1.0 } else { 0.25 };
        buf[p] = kinect3d::gfx::pack([(c[0] as f32 * k) as u8, (c[1] as f32 * k) as u8, (c[2] as f32 * k) as u8]);
    }
    kinect3d::io::save_rgb_png(std::path::Path::new(&a[3]), &buf, W, H);
    println!("mask px {}  y gate {:.2}..{:.2}", m.iter().filter(|b| **b).count(), seg.y_lo, seg.y_hi);
}
