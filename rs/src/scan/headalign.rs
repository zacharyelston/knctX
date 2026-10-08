//! The head can move (nod, tilt, turn) relative to the body while the person spins, so a pose
//! from body tracking blurs it. Group the frames into angular chunks, each rigid over its short
//! window, and align the chunks to one another with full 6-DOF rigid ICP on head points.
use super::track::Grid;
use crate::math::*;

fn jacobi4(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [[0.0f64; 4]; 4];
    for i in 0..4 {
        v[i][i] = 1.0;
    }
    for _ in 0..40 {
        let mut off = 0.0;
        for i in 0..4 {
            for j in i + 1..4 {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1e-18 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-30 {
                    continue;
                }
                let th = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..4 {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut best = 0;
    for i in 1..4 {
        if a[i][i] > a[best][best] {
            best = i;
        }
    }
    [v[0][best], v[1][best], v[2][best], v[3][best]]
}

/// Horn's closed form: rotation R and translation t with tgt ~ R src + t.
fn rigid_fit(src: &[V3], tgt: &[V3]) -> (M3, V3) {
    let n = src.len() as f64;
    let mut pc = [0.0f64; 3];
    let mut qc = [0.0f64; 3];
    for (p, q) in src.iter().zip(tgt) {
        for k in 0..3 {
            pc[k] += p[k] as f64 / n;
            qc[k] += q[k] as f64 / n;
        }
    }
    let mut h = [[0.0f64; 3]; 3];
    for (p, q) in src.iter().zip(tgt) {
        for a in 0..3 {
            for b in 0..3 {
                h[a][b] += (p[a] as f64 - pc[a]) * (q[b] as f64 - qc[b]);
            }
        }
    }
    let (sxx, sxy, sxz, syx, syy, syz, szx, szy, szz) = (h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2]);
    let nm = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let q = jacobi4(nm);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let r = [
        [w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z],
    ];
    let t = [0, 1, 2].map(|a| (qc[a] - (r[a][0] * pc[0] + r[a][1] * pc[1] + r[a][2] * pc[2])) as f32);
    (r.map(|row| row.map(|v| v as f32)), t)
}

fn rot_angle(r: &M3) -> f32 { (((r[0][0] + r[1][1] + r[2][2]) - 1.0) / 2.0).clamp(-1.0, 1.0).acos() }

fn apply(a: &(M3, V3), p: V3) -> V3 { add(mat_vec(&a.0, p), a.1) }

/// `chunk_pts[c]`: head points of chunk c in model space. Returns one rigid correction per chunk.
pub fn align_chunks(chunk_pts_in: &[Vec<V3>], max_d: f32, passes: usize, log: &mut dyn FnMut(String)) -> Vec<(M3, V3)> {
    // work about the head centre so rotations don't drag the head metres away
    let (mut c0, mut cnt) = ([0.0f32; 3], 0.0f32);
    for p in chunk_pts_in.iter().flatten() {
        c0 = add(c0, *p);
        cnt += 1.0;
    }
    let c0 = scale(c0, 1.0 / cnt.max(1.0));
    let chunk_pts: Vec<Vec<V3>> = chunk_pts_in.iter().map(|v| v.iter().map(|p| sub(*p, c0)).collect()).collect();
    let nc = chunk_pts.len();
    let ident: (M3, V3) = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0; 3]);
    let mut acc = vec![ident; nc];
    if nc < 3 {
        return acc;
    }
    for pass in 0..passes {
        let (mut pool, mut ids) = (Vec::new(), Vec::new());
        for (c, pts) in chunk_pts.iter().enumerate() {
            for p in pts {
                pool.push(apply(&acc[c], *p));
                ids.push(c as u16);
            }
        }
        let grid = Grid::new(&pool, max_d).with_ids(ids);
        let mut next = acc.clone();
        let (mut rms_sum, mut moved) = (0.0f64, 0.0f64);
        for c in 0..nc {
            let mut cur = acc[c];
            let mut rms = 0.0f64;
            for _ in 0..6 {
                let (mut s, mut t, mut d2): (Vec<V3>, Vec<V3>, Vec<f32>) = (vec![], vec![], vec![]);
                for p in &chunk_pts[c] {
                    let q = apply(&cur, *p);
                    if let Some((j, d)) = grid.nearest_excl(q, max_d, (c as u16, c as u16)) {
                        s.push(q);
                        t.push(pool[j]);
                        d2.push(d);
                    }
                }
                if s.len() < 300 {
                    break;
                }
                let cut = percentile(&mut d2.clone(), 80.0);
                let (s2, t2): (Vec<V3>, Vec<V3>) = s.iter().zip(&t).zip(&d2).filter(|(_, d)| **d <= cut).map(|((a, b), _)| (*a, *b)).unzip();
                rms = (s2.iter().zip(&t2).map(|(a, b)| norm(sub(*a, *b)) as f64).map(|d| d * d).sum::<f64>() / s2.len() as f64).sqrt();
                let (r, tr) = rigid_fit(&s2, &t2);
                cur = (mat_mul(&r, &cur.0), add(mat_vec(&r, cur.1), tr));
            }
            // keep corrections small and sane: the body pose is already close
            let (ang, sh) = (rot_angle(&cur.0), norm(cur.1));
            if ang < 8f32.to_radians() && sh < 0.04 {
                next[c] = cur;
                moved += (ang as f64).to_degrees();
            }
            rms_sum += rms;
        }
        acc = next;
        log(format!("head align pass {}: chunk rms {:.1} mm, mean rotation {:.2} deg", pass + 1, rms_sum / nc as f64 * 1000.0, moved / nc as f64));
    }
    acc.into_iter().map(|(r, t)| (r, add(t, sub(c0, mat_vec(&r, c0))))).collect()
}
