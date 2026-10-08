//! Headless end-to-end check on the synthetic person: floor -> track -> fuse -> mesh -> rig.
use kinect3d::math::*;
use kinect3d::scan::capture::*;
use kinect3d::scan::rig::*;
use kinect3d::scan::segment::Segmenter;
use kinect3d::scan::world::*;
use kinect3d::synthetic::*;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let body = make_body(1);
    let bg = {
        let k = SyntheticKinect::new(9.0, 30.0, 4.0);
        let _ = k;
    };
    let _ = bg;
    let k = SyntheticKinect::new(9.0, 30.0, 4.0);
    let _ = k;
    let sc = Scene::body();
    let mut rng = Rng::new(3);
    let bgd = kinect3d::synthetic::background_pub(&sc);
    let (rgb0, d0) = render(&sc,&body, &bgd, 0.0, 4.0, &mut rng);
    let t = Instant::now();
    let world = fit_floor(&d0, 0).expect("floor");
    println!("floor: cam height {:.2} m ({:?})", world.t[1], t.elapsed());
    let mut seg = Segmenter::new(world);
    println!("person: {:?}", seg.find_person(&d0));
    let mut cap = Capture::new(seg);
    let n = 200;
    let (mut tr, mut tc) = (0.0, 0.0);
    let t = Instant::now();
    for i in 0..n {
        let theta = std::f32::consts::TAU * 2.0 * i as f32 / n as f32;
        let t1 = Instant::now();
        let (rgb, d) = render(&sc,&body, &bgd, theta, 4.0, &mut rng);
        tr += t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        cap.add(&Arc::new(rgb), &d);
        tc += t2.elapsed().as_secs_f64();
    }
    println!("tracked {} frames in {:?}: {:.2} turns, quality {:.2}", cap.caps.len(), t.elapsed(), cap.trk.yaw().abs() / std::f64::consts::TAU, cap.trk.quality);
    println!("render {tr:.2}s capture.add {tc:.2}s");
    let _ = rgb0;
    let c0 = {
        let p = &cap.caps[0].pts;
        let nn = p.len() as f32;
        [p.iter().map(|q| q[0]).sum::<f32>() / nn, p.iter().map(|q| q[2]).sum::<f32>() / nn]
    };
    let t = Instant::now();
    let mesh = build_model(&cap.caps, &world, c0, &Profile::body(), (0.0, 2.1), &|_, _| {}, &mut |s| println!("{s}")).expect("model");
    println!("mesh: {} verts {} faces in {:?}", mesh.verts.len(), mesh.faces.len(), t.elapsed());
    let t = Instant::now();
    let fig = Figure::new(mesh.verts.clone(), mesh.faces.clone(), mesh.rgb.clone());
    println!("rig: height {:.3} m ({:?})", fig.top, t.elapsed());
    let t = Instant::now();
    let p = fig.pose_params("T-pose", 0.0);
    let (v, _) = fig.posed(&p);
    println!("posed {} verts in {:?}", v.len(), t.elapsed());
    std::fs::create_dir_all("/tmp/k3d").ok();
    kinect3d::io::save_ply(std::path::Path::new("/tmp/k3d/test.ply"), &mesh.verts, &mesh.faces, &mesh.rgb).unwrap();
}
