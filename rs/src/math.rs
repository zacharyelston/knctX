//! Small vector / matrix helpers (f32 for point data, f64 for poses).
pub type V3 = [f32; 3];
pub type M3 = [[f32; 3]; 3];

pub fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn add(a: V3, b: V3) -> V3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
pub fn scale(a: V3, s: f32) -> V3 { [a[0] * s, a[1] * s, a[2] * s] }
pub fn dot(a: V3, b: V3) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm(a: V3) -> f32 { dot(a, a).sqrt() }
pub fn normalize(a: V3) -> V3 { scale(a, 1.0 / norm(a).max(1e-9)) }
pub fn mat_vec(m: &M3, v: V3) -> V3 { [dot(m[0], v), dot(m[1], v), dot(m[2], v)] }
pub fn mat_t_vec(m: &M3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}
pub fn mat_mul(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

/// Rotation matrix Rz @ Ry @ Rx from degrees.
pub fn euler(rx: f32, ry: f32, rz: f32) -> M3 {
    let (rx, ry, rz) = (rx.to_radians(), ry.to_radians(), rz.to_radians());
    let (cx, sx, cy, sy, cz, sz) = (rx.cos(), rx.sin(), ry.cos(), ry.sin(), rz.cos(), rz.sin());
    let rxm = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let rym = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rzm = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    mat_mul(&rzm, &mat_mul(&rym, &rxm))
}

/// Eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix (Jacobi iteration).
pub fn smallest_eigvec(c: [[f64; 3]; 3]) -> V3 {
    let mut a = c;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..30 {
        let (mut p, mut q, mut big) = (0, 1, 0.0f64);
        for i in 0..3 {
            for j in i + 1..3 {
                if a[i][j].abs() > big {
                    big = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if big < 1e-14 {
            break;
        }
        let th = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (c_, s_) = (th.cos(), th.sin());
        for k in 0..3 {
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = c_ * akp - s_ * akq;
            a[k][q] = s_ * akp + c_ * akq;
        }
        for k in 0..3 {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = c_ * apk - s_ * aqk;
            a[q][k] = s_ * apk + c_ * aqk;
        }
        for k in 0..3 {
            let (vkp, vkq) = (v[k][p], v[k][q]);
            v[k][p] = c_ * vkp - s_ * vkq;
            v[k][q] = s_ * vkp + c_ * vkq;
        }
    }
    let mut m = 0;
    for i in 1..3 {
        if a[i][i] < a[m][m] {
            m = i;
        }
    }
    [v[0][m] as f32, v[1][m] as f32, v[2][m] as f32]
}

/// Normal (smallest-variance direction) and mean of a point set.
pub fn plane_fit(pts: &[V3]) -> (V3, V3) {
    let n = pts.len().max(1) as f64;
    let mut m = [0.0f64; 3];
    for p in pts {
        for k in 0..3 {
            m[k] += p[k] as f64;
        }
    }
    for k in 0..3 {
        m[k] /= n;
    }
    let mut c = [[0.0f64; 3]; 3];
    for p in pts {
        let d = [p[0] as f64 - m[0], p[1] as f64 - m[1], p[2] as f64 - m[2]];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] += d[i] * d[j];
            }
        }
    }
    (smallest_eigvec(c), [m[0] as f32, m[1] as f32, m[2] as f32])
}

/// Tiny xorshift PRNG (no external crate needed).
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1) }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn f32(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 }
    pub fn below(&mut self, n: usize) -> usize { (self.next_u64() % n as u64) as usize }
    pub fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.f32().max(1e-7), self.f32());
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

pub fn percentile(v: &mut [f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = (p / 100.0) * (v.len() - 1) as f32;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f32)
}

pub fn median(v: &mut [f32]) -> f32 { percentile(v, 50.0) }
