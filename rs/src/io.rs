//! Image / file helpers: raw frame load & save, PLY / OBJ writers.
use crate::math::V3;
use crate::source::{H, W};
use std::io::Write;
use std::path::Path;

pub fn load_frame(dir: &Path, rgb_name: &str, depth_name: &str) -> Option<(Vec<u8>, Vec<u16>)> {
    let rgb = image::open(dir.join(rgb_name)).ok()?.to_rgb8();
    let depth = image::open(dir.join(depth_name)).ok()?.to_luma16();
    if rgb.width() as usize != W || rgb.height() as usize != H || depth.width() as usize != W || depth.height() as usize != H {
        return None;
    }
    Some((rgb.into_raw(), depth.into_raw()))
}

pub fn save_rgb_jpg(path: &Path, rgb: &[u8]) {
    if let Ok(f) = std::fs::File::create(path) {
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(f), 92);
        let _ = enc.encode(rgb, W as u32, H as u32, image::ExtendedColorType::Rgb8);
    }
}

pub fn save_depth_png(path: &Path, depth: &[u16]) {
    if let Some(img) = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(W as u32, H as u32, depth.to_vec()) {
        let _ = image::DynamicImage::ImageLuma16(img).save(path);
    }
}

pub fn save_rgb_png(path: &Path, buf: &[u32], w: usize, h: usize) {
    let mut raw = Vec::with_capacity(w * h * 3);
    for p in buf {
        raw.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, *p as u8]);
    }
    let _ = image::save_buffer(path, &raw, w as u32, h as u32, image::ExtendedColorType::Rgb8);
}

/// Binary little-endian PLY with vertex colours and triangle faces.
pub fn save_ply(path: &Path, verts: &[V3], faces: &[[u32; 3]], rgb: &[[u8; 3]]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n", verts.len(), faces.len())?;
    for (v, c) in verts.iter().zip(rgb) {
        for k in v {
            f.write_all(&k.to_le_bytes())?;
        }
        f.write_all(c)?;
    }
    for t in faces {
        f.write_all(&[3u8])?;
        for k in t {
            f.write_all(&(*k as i32).to_le_bytes())?;
        }
    }
    Ok(())
}

/// Colour point cloud PLY (no faces).
pub fn save_cloud_ply(path: &Path, pts: &[V3], rgb: &[[u8; 3]]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n", pts.len())?;
    for (v, c) in pts.iter().zip(rgb) {
        for k in v {
            f.write_all(&k.to_le_bytes())?;
        }
        f.write_all(c)?;
    }
    Ok(())
}

pub fn load_ply(path: &Path) -> Result<(Vec<V3>, Vec<[u32; 3]>, Vec<[u8; 3]>), String> {
    let raw = std::fs::read(path).map_err(|e| e.to_string())?;
    let key = b"end_header\n";
    let end = raw.windows(key.len()).position(|w| w == key).ok_or("bad ply")? + key.len();
    let head = String::from_utf8_lossy(&raw[..end]).to_string();
    let count = |name: &str| -> usize {
        head.lines().find(|l| l.starts_with(&format!("element {name}"))).and_then(|l| l.split_whitespace().nth(2)).and_then(|s| s.parse().ok()).unwrap_or(0)
    };
    let (nv, nf) = (count("vertex"), count("face"));
    let mut verts = Vec::with_capacity(nv);
    let mut cols = Vec::with_capacity(nv);
    let mut o = end;
    for _ in 0..nv {
        let g = |i: usize| f32::from_le_bytes(raw[o + i * 4..o + i * 4 + 4].try_into().unwrap());
        verts.push([g(0), g(1), g(2)]);
        cols.push([raw[o + 12], raw[o + 13], raw[o + 14]]);
        o += 15;
    }
    let mut faces = Vec::with_capacity(nf);
    for _ in 0..nf {
        o += 1;
        let g = |i: usize| i32::from_le_bytes(raw[o + i * 4..o + i * 4 + 4].try_into().unwrap()) as u32;
        faces.push([g(0), g(1), g(2)]);
        o += 12;
    }
    Ok((verts, faces, cols))
}

/// OBJ with per-vertex colours (the `v x y z r g b` extension Blender reads).
pub fn save_obj(path: &Path, verts: &[V3], faces: &[[u32; 3]], rgb: &[[u8; 3]]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for (p, c) in verts.iter().zip(rgb) {
        writeln!(f, "v {:.5} {:.5} {:.5} {:.4} {:.4} {:.4}", p[0], p[1], p[2], c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0)?;
    }
    for t in faces {
        writeln!(f, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1)?;
    }
    Ok(())
}
