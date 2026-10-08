//! Rebuild a model from a saved raw capture: rescan <scan folder> [--out DIR] [--refloor]
use kinect3d::gfx::Canvas;
use kinect3d::render::*;
use kinect3d::scan::capture::*;
use kinect3d::scan::rig::Figure;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut folder = None;
    let mut out = None;
    let mut refloor = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out = args.get(i).cloned();
            }
            "--refloor" => refloor = true,
            a => folder = Some(a.to_string()),
        }
        i += 1;
    }
    let Some(folder) = folder else {
        eprintln!("usage: rescan <scan folder> [--out DIR] [--refloor]");
        std::process::exit(2);
    };
    let t = Instant::now();
    match rescan(Path::new(&folder), out.as_deref().map(Path::new), &mut |s| println!("{s}"), refloor) {
        Ok((out, _stats, mesh)) => {
            let fig = Figure::new(mesh.verts.clone(), mesh.faces.clone(), mesh.rgb.clone());
            let mut sheet = Canvas::new(4 * 480, 540);
            for (n, yaw) in [0.3f32, 0.3 + 1.57, 0.3 + 3.14, 0.3 + 4.71].iter().enumerate() {
                let mut c = Canvas::new(480, 540);
                let mut o = Orbit3D::new();
                o.yaw = *yaw;
                render_mesh(&mut c, &fig.rest, &fig.rest_normals, &fig.faces, &fig.rgb, &o, fig.top);
                for y in 0..540 {
                    sheet.buf[y * sheet.w + n * 480..y * sheet.w + n * 480 + 480].copy_from_slice(&c.buf[y * 480..(y + 1) * 480]);
                }
            }
            kinect3d::io::save_rgb_png(&out.join("preview.png"), &sheet.buf, sheet.w, sheet.h);
            println!("model: {} verts, {} faces, {:.2} m tall -> {} ({:?})", mesh.verts.len(), mesh.faces.len(), fig.top, out.display(), t.elapsed());
        }
        Err(e) => {
            eprintln!("rescan failed: {e}");
            std::process::exit(1);
        }
    }
}
