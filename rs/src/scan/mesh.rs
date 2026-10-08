//! Surface extraction from the fused TSDF (naive surface nets), clean-up, smoothing and colouring.
use super::fuse::Volume;
use crate::math::*;

pub struct Mesh {
    pub verts: Vec<V3>,
    pub faces: Vec<[u32; 3]>,
    pub rgb: Vec<[u8; 3]>,
}

fn find(p: &mut Vec<u32>, mut x: u32) -> u32 {
    while p[x as usize] != x {
        p[x as usize] = p[p[x as usize] as usize];
        x = p[x as usize];
    }
    x
}

pub fn vertex_normals(verts: &[V3], faces: &[[u32; 3]]) -> Vec<V3> {
    let mut vn = vec![[0.0f32; 3]; verts.len()];
    for f in faces {
        let (a, b, c) = (verts[f[0] as usize], verts[f[1] as usize], verts[f[2] as usize]);
        let n = cross(sub(b, a), sub(c, a));
        for &i in f {
            vn[i as usize] = add(vn[i as usize], n);
        }
    }
    vn.into_iter().map(normalize).collect()
}

pub fn extract_mesh(vol: &Volume, min_obs: f32, smooth_iters: usize, min_extent: f32) -> Result<Mesh, String> {
    let [nx, ny, nz] = vol.dims;
    let idx = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    if vol.wt.iter().filter(|w| **w >= min_obs).count() < 1000 {
        return Err("not enough data to build a model".into());
    }
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    let cidx = |i: usize, j: usize, k: usize| (i * cy + j) * cz + k;
    let mut cell_vert = vec![-1i32; cx * cy * cz];
    let mut verts: Vec<V3> = Vec::new();
    const CORNER: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
    const EDGES: [[usize; 2]; 12] = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
    for i in 0..cx {
        for j in 0..cy {
            for k in 0..cz {
                let mut val = [0.0f32; 8];
                let mut ok = true;
                let mut neg = 0;
                for (c, o) in CORNER.iter().enumerate() {
                    let id = idx(i + o[0], j + o[1], k + o[2]);
                    if vol.wt[id] < min_obs {
                        ok = false;
                        break;
                    }
                    val[c] = vol.tsdf[id];
                    if val[c] < 0.0 {
                        neg += 1;
                    }
                }
                if !ok || neg == 0 || neg == 8 {
                    continue;
                }
                let (mut sum, mut cnt) = ([0.0f32; 3], 0.0f32);
                for e in EDGES {
                    let (a, b) = (val[e[0]], val[e[1]]);
                    if (a < 0.0) != (b < 0.0) {
                        let t = a / (a - b);
                        for ax in 0..3 {
                            let (pa, pb) = (CORNER[e[0]][ax] as f32, CORNER[e[1]][ax] as f32);
                            sum[ax] += pa + t * (pb - pa);
                        }
                        cnt += 1.0;
                    }
                }
                let p = [i as f32 + sum[0] / cnt, j as f32 + sum[1] / cnt, k as f32 + sum[2] / cnt];
                cell_vert[cidx(i, j, k)] = verts.len() as i32;
                verts.push([vol.origin[0] + p[0] * vol.voxel, vol.origin[1] + p[1] * vol.voxel, vol.origin[2] + p[2] * vol.voxel]);
            }
        }
    }
    // quads across every grid edge with a sign change
    let mut faces: Vec<[u32; 3]> = Vec::new();
    let cv = |i: usize, j: usize, k: usize| cell_vert[cidx(i, j, k)];
    let mut quad = |q: [i32; 4], flip: bool| {
        if q.iter().any(|&x| x < 0) {
            return;
        }
        let q = if flip { [q[3], q[2], q[1], q[0]] } else { q };
        faces.push([q[0] as u32, q[1] as u32, q[2] as u32]);
        faces.push([q[0] as u32, q[2] as u32, q[3] as u32]);
    };
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let a = idx(i, j, k);
                if vol.wt[a] < min_obs {
                    continue;
                }
                let ain = vol.tsdf[a] < 0.0;
                if i + 1 < nx && j >= 1 && k >= 1 && j < ny - 1 + 0 && k < nz - 1 + 0 {
                    let b = idx(i + 1, j, k);
                    if vol.wt[b] >= min_obs && (vol.tsdf[b] < 0.0) != ain {
                        quad([cv(i, j - 1, k - 1), cv(i, j, k - 1), cv(i, j, k), cv(i, j - 1, k)], !ain);
                    }
                }
                if j + 1 < ny && k >= 1 && i >= 1 && k < nz - 1 && i < nx - 1 {
                    let b = idx(i, j + 1, k);
                    if vol.wt[b] >= min_obs && (vol.tsdf[b] < 0.0) != ain {
                        quad([cv(i - 1, j, k - 1), cv(i - 1, j, k), cv(i, j, k), cv(i, j, k - 1)], !ain);
                    }
                }
                if k + 1 < nz && i >= 1 && j >= 1 && i < nx - 1 && j < ny - 1 {
                    let b = idx(i, j, k + 1);
                    if vol.wt[b] >= min_obs && (vol.tsdf[b] < 0.0) != ain {
                        quad([cv(i - 1, j - 1, k), cv(i, j - 1, k), cv(i, j, k), cv(i - 1, j, k)], !ain);
                    }
                }
            }
        }
    }
    if faces.is_empty() {
        return Err("not enough data to build a model".into());
    }
    // keep sizeable components that stand up (drop floaters and flat floor scraps)
    let mut parent: Vec<u32> = (0..verts.len() as u32).collect();
    for f in &faces {
        for e in [(f[0], f[1]), (f[1], f[2])] {
            let (ra, rb) = (find(&mut parent, e.0), find(&mut parent, e.1));
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }
    }
    let nv = verts.len();
    let mut size = vec![0usize; nv];
    let mut ymin = vec![f32::INFINITY; nv];
    let mut ymax = vec![f32::NEG_INFINITY; nv];
    let mut root = vec![0u32; nv];
    for v in 0..nv {
        let r = find(&mut parent, v as u32);
        root[v] = r;
        size[r as usize] += 1;
        ymin[r as usize] = ymin[r as usize].min(verts[v][1]);
        ymax[r as usize] = ymax[r as usize].max(verts[v][1]);
    }
    let smax = *size.iter().max().unwrap();
    let mut good: Vec<bool> = (0..nv).map(|r| size[r] > 0 && size[r] as f32 >= 0.05 * smax as f32 && ymax[r] - ymin[r] > min_extent).collect();
    if !good.iter().any(|g| *g) {
        good = (0..nv).map(|r| size[r] == smax).collect();
    }
    let mut remap = vec![u32::MAX; nv];
    let mut kept: Vec<V3> = Vec::new();
    for v in 0..nv {
        if good[root[v] as usize] {
            remap[v] = kept.len() as u32;
            kept.push(verts[v]);
        }
    }
    let faces: Vec<[u32; 3]> = faces
        .iter()
        .filter(|f| f.iter().all(|&i| remap[i as usize] != u32::MAX))
        .map(|f| [remap[f[0] as usize], remap[f[1] as usize], remap[f[2] as usize]])
        .collect();
    let mut v = kept;
    // Taubin smoothing (no shrinkage)
    let mut nbr: Vec<Vec<u32>> = vec![Vec::new(); v.len()];
    for f in &faces {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            nbr[a as usize].push(b);
            nbr[b as usize].push(a);
        }
    }
    for l in nbr.iter_mut() {
        l.sort_unstable();
        l.dedup();
    }
    for it in 0..smooth_iters {
        let lam = if it % 2 == 0 { 0.5 } else { -0.53 };
        let next: Vec<V3> = (0..v.len())
            .map(|i| {
                if nbr[i].is_empty() {
                    return v[i];
                }
                let mut m = [0.0f32; 3];
                for &n in &nbr[i] {
                    m = add(m, v[n as usize]);
                }
                let m = scale(m, 1.0 / nbr[i].len() as f32);
                add(v[i], scale(sub(m, v[i]), lam))
            })
            .collect();
        v = next;
    }
    // colours from the nearest observed voxel
    let rgb: Vec<[u8; 3]> = v.iter().map(|p| nearest_colour(vol, *p)).collect();
    Ok(Mesh { verts: v, faces, rgb })
}

fn nearest_colour(vol: &Volume, p: V3) -> [u8; 3] {
    let [nx, ny, nz] = vol.dims;
    let ij = [0, 1, 2].map(|a| (((p[a] - vol.origin[a]) / vol.voxel).round() as i64).clamp(0, vol.dims[a] as i64 - 1));
    let at = |i: i64, j: i64, k: i64| (i as usize * ny + j as usize) * nz + k as usize;
    let _ = nx;
    for r in 0i64..=12 {
        let mut best: Option<(i64, usize)> = None;
        for di in -r..=r {
            for dj in -r..=r {
                for dk in -r..=r {
                    if di.abs().max(dj.abs()).max(dk.abs()) != r {
                        continue;
                    }
                    let (i, j, k) = (ij[0] + di, ij[1] + dj, ij[2] + dk);
                    if i < 0 || j < 0 || k < 0 || i >= vol.dims[0] as i64 || j >= vol.dims[1] as i64 || k >= vol.dims[2] as i64 {
                        continue;
                    }
                    let id = at(i, j, k);
                    if vol.cwt[id] >= 1.0 {
                        let d2 = di * di + dj * dj + dk * dk;
                        if best.map_or(true, |b| d2 < b.0) {
                            best = Some((d2, id));
                        }
                    }
                }
            }
        }
        if let Some((_, id)) = best {
            let c = vol.col[id];
            return [c[0].clamp(0.0, 255.0) as u8, c[1].clamp(0.0, 255.0) as u8, c[2].clamp(0.0, 255.0) as u8];
        }
    }
    [128, 128, 128]
}
