"""Fake Kinect: a procedural person spinning on the spot, with floor and back wall.

Same interface as kinect_overlay.Kinect (start/stop/frames/seq/tilt_target), so the scan
pipeline and booth UI can be tested without a person or a sensor.
"""
import threading
import time

import numpy as np

from bodyscan import CX, CY, FX, FY, H, W, _RAYX, _RAYY

CAM_H = 1.0          # camera height above the floor (m)
PITCH = np.radians(-2.5)  # camera tilted slightly down
PERSON_Z = -3.0      # person position in scene coords (camera looks toward -z)
WALL_Z = -4.8


def _sphere(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _ellipsoid(rng, c, r, n, color):
    return np.asarray(c) + _sphere(rng, n) * np.asarray(r), np.tile(color, (n, 1))


def _capsule(rng, a, b, ra, rb, n, color):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = rng.random(n)[:, None]
    axis = (b - a) / np.linalg.norm(b - a)
    u = _sphere(rng, n)
    u -= (u @ axis)[:, None] * axis
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    r = ra + (rb - ra) * t
    return a + t * (b - a) + u * r, np.tile(color, (n, 1))


def make_body(seed=1):
    """Surface samples (N,3) in metres, person centred on x=z=0, floor y=0, facing +z."""
    rng = np.random.default_rng(seed)
    skin, shirt, pants, shoe = (225, 175, 150), (200, 70, 60), (50, 60, 80), (40, 40, 40)  # RGB
    parts = [
        _ellipsoid(rng, (0, 1.18, 0), (0.17, 0.27, 0.10), 9000, shirt),
        _ellipsoid(rng, (0, 0.92, 0), (0.16, 0.12, 0.10), 4000, pants),
        _ellipsoid(rng, (0, 1.63, 0), (0.095, 0.115, 0.10), 4000, skin),
        _ellipsoid(rng, (0, 1.62, 0.115), (0.02, 0.03, 0.03), 400, skin),  # nose: breaks symmetry
        _capsule(rng, (0, 1.45, 0), (0, 1.52, 0), 0.05, 0.05, 600, skin),
        _capsule(rng, (-0.14, 1.41, 0), (0.14, 1.41, 0), 0.06, 0.06, 1500, shirt),  # shoulders
    ]
    for s in (1, -1):
        parts += [
            _capsule(rng, (s * 0.09, 0.90, 0), (s * 0.10, 0.50, 0.01), 0.075, 0.055, 4500, pants),
            _capsule(rng, (s * 0.10, 0.50, 0.01), (s * 0.10, 0.08, 0), 0.055, 0.045, 4000, pants),
            _ellipsoid(rng, (s * 0.10, 0.04, 0.05), (0.045, 0.04, 0.11), 1500, shoe),
            _capsule(rng, (s * 0.20, 1.40, 0), (s * 0.35, 1.14, 0.02), 0.05, 0.04, 2500, shirt),
            _capsule(rng, (s * 0.35, 1.14, 0.02), (s * 0.48, 0.915, 0.04), 0.04, 0.032, 2500, skin),
            _ellipsoid(rng, (s * 0.495, 0.88, 0.045), (0.03, 0.05, 0.02), 500, skin),
        ]
    pts = np.concatenate([p for p, _ in parts]).astype(np.float32)
    col = np.concatenate([c for _, c in parts]).astype(np.float32)
    col += rng.normal(0, 8, col.shape)  # texture noise
    # a stripe pattern on the torso so colour varies a bit
    stripe = (np.sin(pts[:, 1] * 40) > 0.6) & (np.abs(pts[:, 0]) < 0.2) & (pts[:, 1] > 1.0) & (pts[:, 1] < 1.45)
    col[stripe] = (230, 230, 230)
    return pts, np.clip(col, 0, 255).astype(np.uint8)


def _background():
    """Static floor + wall depth (mm) and RGB, in the pitched camera."""
    cg, sg = np.cos(PITCH), np.sin(PITCH)
    # unit ray in pitched camera coords (rx, ry, 1) -> level camera coords
    y_c = _RAYY * cg - 1.0 * sg
    z_c = _RAYY * sg + 1.0 * cg
    x_c = _RAYX
    dy_s = -y_c            # scene-space y per unit camera z'
    dz_s = -z_c
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dy_s < -1e-6, -CAM_H / dy_s, np.inf)   # z' at the floor
        t_wall = np.where(dz_s < -1e-6, (-WALL_Z) / (-dz_s), np.inf)
    t = np.minimum(t_floor, t_wall)
    t = np.where(np.isfinite(t) & (t < 8), t, 0)
    depth = (t * 1000).astype(np.uint16)
    on_floor = t_floor <= t_wall
    xs = x_c * t
    zs = dz_s * t
    tile = ((np.floor(xs * 2) + np.floor(zs * 2)) % 2) * 30
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[on_floor] = (110 + tile[on_floor, None]).clip(0, 255).astype(np.uint8)
    rgb[~on_floor] = (170, 175, 165)
    return depth, rgb


class SyntheticKinect:
    def __init__(self, seconds_per_turn=8.0, fps=30.0, noise_mm=4.0, seed=3):
        self.body, self.body_col = make_body()
        self.bg_depth, self.bg_rgb = _background()
        self.spt, self.fps, self.noise = seconds_per_turn, fps, noise_mm
        self.rng = np.random.default_rng(seed)
        self.rgb = self.depth = None
        self.seq = 0
        self.tilt_target = 0
        self.running = True
        self.t0 = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.t0 = time.time()
        self.thread.start()

    def render(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        p = self.body.copy()
        p[:, 0], p[:, 2] = c * self.body[:, 0] + s * self.body[:, 2], -s * self.body[:, 0] + c * self.body[:, 2]
        p[:, 2] += PERSON_Z
        # scene -> level camera -> pitched camera
        xl, yl, zl = p[:, 0], -(p[:, 1] - CAM_H), -p[:, 2]
        cg, sg = np.cos(PITCH), np.sin(PITCH)
        xc, yc, zc = xl, yl * cg + zl * sg, -yl * sg + zl * cg
        u = np.rint(FX * xc / zc + CX).astype(np.int32)
        v = np.rint(FY * yc / zc + CY).astype(np.int32)
        order = np.argsort(-zc)  # far first -> near wins
        depth, rgb = self.bg_depth.copy(), self.bg_rgb.copy()
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                uu, vv = u[order] + dx - 1, v[order] + dy - 1
                ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                z = zc[order][ok]
                nearer = (depth[vv[ok], uu[ok]] == 0) | (z * 1000 < depth[vv[ok], uu[ok]])
                idx = np.nonzero(ok)[0][nearer]
                depth[vv[idx], uu[idx]] = (z[nearer] * 1000).astype(np.uint16)
                rgb[vv[idx], uu[idx]] = self.body_col[order][idx]
        if self.noise:
            depth = (depth.astype(np.float32) + self.rng.normal(0, self.noise, depth.shape)).clip(0, 65000).astype(np.uint16)
            depth[self.bg_depth == 0] = 0
        return rgb, depth

    def _run(self):
        while self.running:
            tic = time.time()
            theta = 2 * np.pi * (tic - self.t0) / self.spt
            rgb, depth = self.render(theta)
            with self.lock:
                self.rgb, self.depth = rgb, depth
                self.seq += 1
            time.sleep(max(0, 1 / self.fps - (time.time() - tic)))

    def frames(self):
        with self.lock:
            return self.rgb, self.depth

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)
