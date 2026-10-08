//! Compare the Kinect accelerometer with the floor-derived up vector (axis/sign check).
use kinect3d::scan::world::*;
use kinect3d::source::*;
fn main() {
    let k = Kinect::new();
    k.start();
    for t in [0, 10, -10] {
        k.set_tilt(t);
        std::thread::sleep(std::time::Duration::from_secs(5));
        let a = k.accel().unwrap();
        let (_, d) = k.frames().unwrap();
        let w = fit_floor(&d, 0);
        println!("tilt {t:+}: accel {:.2?}  floor up(cam) {:?}  accel-world up {:?}", a, w.map(|w| w.r[1]), world_from_accel(a).map(|w| w.r[1]));
    }
    k.stop();
}
