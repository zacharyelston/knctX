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
            let bust = std::fs::read_to_string(Path::new(&folder).join("raw/capture.json")).map_or(false, |t| t.contains("\"profile\":\"bust\""));
            let fig = if bust { None } else { Some(Figure::new(mesh.verts.clone(), mesh.faces.clone(), mesh.rgb.clone())) };
            let mut sheet = Canvas::new(4 * 480, 540);
            for (n, yaw) in [0.3f32, 0.3 + 1.57, 0.3 + 3.14, 0.3 + 4.71].iter().enumerate() {
                let mut c = Canvas::new(480, 540);
                let mut o = Orbit3D::new();
                o.yaw = *yaw;
                match &fig {
                    Some(f) => render_mesh(&mut c, &f.rest, &f.rest_normals, &f.faces, &f.rgb, &o, f.top),
                    None => {
                        let nrm = kinect3d::scan::mesh::vertex_normals(&mesh.verts, &mesh.faces);
                        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
                        for v in &mesh.verts {
                            for k in 0..3 {
                                lo[k] = lo[k].min(v[k]);
                                hi[k] = hi[k].max(v[k]);
                            }
                        }
                        let focus = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
                        let ext = (hi[1] - lo[1]).max(hi[0] - lo[0]);
                        render_mesh_at(&mut c, &mesh.verts, &nrm, &mesh.faces, &mesh.rgb, &o, focus, 1.5 * ext, false)
                    }
                }
                for y in 0..540 {
                    sheet.buf[y * sheet.w + n * 480..y * sheet.w + n * 480 + 480].copy_from_slice(&c.buf[y * 480..(y + 1) * 480]);
                }
            }
            kinect3d::io::save_rgb_png(&out.join("preview.png"), &sheet.buf, sheet.w, sheet.h);
            println!("model: {} verts, {} faces, {:.2} m tall -> {} ({:?})", mesh.verts.len(), mesh.faces.len(), fig.as_ref().map_or(0.0, |f| f.top), out.display(), t.elapsed());
        }
        Err(e) => {
            eprintln!("rescan failed: {e}");
            std::process::exit(1);
        }
    }
}
