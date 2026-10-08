"""Turntable body scanning with a fixed Kinect.

Pipeline:  segment person -> track their spin (yaw ICP) -> close the loop ->
           fuse depth into a TSDF volume -> marching-cubes mesh -> auto-rig -> pose.

World frame used everywhere here: x right, y up (floor = 0), z toward the camera,
so a person facing the camera faces +z.  A "pose2d" is (phi, tx, tz): a rotation
about the vertical axis plus a translation in the floor plane.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

# Approximate Kinect v1 intrinsics, 640x480 registered depth (same as kinect_overlay.py).
FX = FY = 580.0
CX, CY = 319.5, 239.5
H, W = 480, 640

_V, _U = np.mgrid[0:H, 0:W]
_RAYX = ((_U - CX) / FX).astype(np.float32)
_RAYY = ((_V - CY) / FY).astype(np.float32)


# --------------------------------------------------------------------------- world / floor

def cam_points(depth_mm):
    """HxWx3 camera-space points in metres (x right, y down, z forward)."""
    z = depth_mm.astype(np.float32) / 1000
    return np.stack([_RAYX * z, _RAYY * z, z], -1)


class World:
    """Rigid camera->world transform: p_w = R @ p_c + t (gravity-aligned via the floor plane)."""

    def __init__(self, R=None, t=None, floor_found=False):
        self.R = np.diag([1, -1, -1]).astype(np.float32) if R is None else R.astype(np.float32)
        self.t = np.array([0, 1.0, 0], np.float32) if t is None else t.astype(np.float32)
        self.floor_found = floor_found

    def to_world(self, pc):
        return pc @ self.R.T + self.t

    def to_camera(self, pw):
        return (pw - self.t) @ self.R


def fit_floor(depth_mm, seed=0):
    """Find the floor: the *lowest* strong horizontal surface. Returns World or None.

    RANSAC finds the dominant roughly-horizontal plane (it may be a table), which fixes the
    up direction.  All points are then binned by height along that direction and the lowest
    well-supported level is taken as the floor, so desks and tables above it are ignored.
    """
    pts = cam_points(depth_mm)[::4, ::4].reshape(-1, 3)
    pts = pts[(pts[:, 2] > 1.0) & (pts[:, 2] < 5)]
    if len(pts) < 2000:
        return None
    rng = np.random.default_rng(seed)
    up = np.array([0, -1, 0], np.float32)
    best = None
    for _ in range(300):
        s = pts[rng.integers(0, len(pts), 3)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if n @ up < 0:
            n = -n
        if n @ up < 0.85:  # must be roughly horizontal
            continue
        inl = np.abs(pts @ n - float(n @ s[0])) < 0.03
        cnt = int(inl.sum())
        if best is None or cnt > best[0]:
            best = (cnt, inl)
    if best is None or best[0] < 0.03 * len(pts):
        return None
    P = pts[best[1]]
    mean = P.mean(0)
    n = np.linalg.svd(P - mean, full_matrices=False)[2][-1]
    if n @ up < 0:
        n = -n
    # lowest well-supported horizontal level along n (heights are n.p, floor is the most negative)
    h = pts @ n
    edges = np.arange(h.min(), h.max() + 0.02, 0.02)
    hist, _ = np.histogram(h, edges)
    thr = max(150, 0.012 * len(pts))
    floor_c = None
    for i in range(len(hist)):
        c = 0.5 * (edges[i] + edges[i + 1])
        if hist[i] >= thr and hist[i] >= hist[max(i - 1, 0)] and hist[i] >= hist[min(i + 1, len(hist) - 1)]:
            if -2.5 < c < -0.4:
                floor_c = c
                break
    if floor_c is None:
        return None
    # refine the plane on that level, allowing a small tilt correction
    inl = np.abs(h - floor_c) < 0.04
    if inl.sum() < 300:
        return None
    P = pts[inl]
    mean = P.mean(0)
    n2 = np.linalg.svd(P - mean, full_matrices=False)[2][-1]
    if n2 @ up < 0:
        n2 = -n2
    if n2 @ n > 0.97:
        n = n2
    c = float(n @ mean)
    if not -2.5 < c < -0.4:
        return None
    ex = np.array([1, 0, 0], np.float32)
    xw = ex - (ex @ n) * n
    xw /= np.linalg.norm(xw)
    zw = np.cross(xw, n)
    return World(np.stack([xw, n, zw]), np.array([0, -c, 0]), floor_found=True)


# --------------------------------------------------------------------------- segmentation

def nearest_distance(depth):
    """Distance (m) to the nearest sizeable object in the middle of the view, or None."""
    d = depth[:, 120:520]
    v = d[d > 500]
    if v.size < 4000:
        return None
    z0 = np.percentile(v, 8)
    blob = d[(d > 0) & (np.abs(d.astype(np.int32) - z0) < 250)]
    return float(np.median(blob)) / 1000 if blob.size > 3000 else None


class Segmenter:
    """Finds the nearest person-sized object and masks it (floor and background removed)."""

    def __init__(self, world):
        self.world = world
        self.center = None  # (x, z) in world metres
        self.radius = 0.95

    def world_points(self, depth):
        return self.world.to_world(cam_points(depth))

    def find_person(self, depth):
        """Return (center_xz, height_m) of the nearest large upright object, or None."""
        pw = self.world_points(depth)[::2, ::2].reshape(-1, 3)
        zc = depth[::2, ::2].reshape(-1).astype(np.float32) / 1000
        sel = (zc > 1.0) & (zc < 4.5) & (pw[:, 1] > 0.25) & (pw[:, 1] < 2.1)
        if sel.sum() < 1200:
            return None
        pw, zc = pw[sel], zc[sel]
        # nearest 10 cm window of camera-distance holding enough points
        for lo in np.arange(1.0, 4.4, 0.05):
            win = (zc >= lo) & (zc < lo + 0.15)
            if win.sum() >= 450:
                near = (zc >= lo - 0.05) & (zc < lo + 0.6)
                c = np.median(pw[near][:, [0, 2]], axis=0)
                around = (np.abs(pw[:, 0] - c[0]) < 0.6) & (np.abs(pw[:, 2] - c[1]) < 0.6)
                height = float(np.percentile(pw[around][:, 1], 99)) if around.sum() else 0.0
                if height < 1.0:
                    continue  # chair, shelf, ...
                self.center = c.astype(np.float32)
                return self.center, height
        return None

    def mask(self, depth):
        """Return (mask HxW bool, world points HxWx3) for the tracked person.

        Pixels are gated by height above the floor and distance from the person, then cut
        along depth discontinuities so chairs/desks behind them don't join the person.
        """
        pw = self.world_points(depth)
        m = (depth > 0) & (pw[..., 1] > 0.08) & (pw[..., 1] < 2.3)
        if self.center is not None:
            m &= (np.abs(pw[..., 0] - self.center[0]) < self.radius) & (np.abs(pw[..., 2] - self.center[1]) < self.radius)
        d32 = depth.astype(np.int32)
        cut = np.zeros(m.shape, bool)
        jx = np.abs(np.diff(d32, axis=1)) > 50
        jy = np.abs(np.diff(d32, axis=0)) > 50
        cut[:, 1:] |= jx
        cut[:, :-1] |= jx
        cut[1:, :] |= jy
        cut[:-1, :] |= jy
        core = cv2.erode((m & ~cut).astype(np.uint8), np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        keep = np.zeros(m.shape, bool)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            big = int(np.argmax(areas))
            keep = lab == (1 + big)
            cb = pw[keep][::7][:, [0, 2]].mean(0)
            for i in np.argsort(-areas)[1:6]:  # arms/hands that got cut off from the torso
                if areas[i] < 0.03 * areas[big]:
                    break
                comp = lab == (1 + i)
                if np.linalg.norm(pw[comp][::5][:, [0, 2]].mean(0) - cb) < 0.5:
                    keep |= comp
        if keep.sum() > 500:
            c = pw[keep][::7][:, [0, 2]].mean(0)
            self.center = c if self.center is None else (0.8 * self.center + 0.2 * c)
        return keep, pw


# --------------------------------------------------------------------------- 2d rigid poses

def rot2(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]], np.float64)


def pose_apply(p, pts):
    xz = pts[:, [0, 2]] @ rot2(p[0]).T + p[1:]
    return np.column_stack([xz[:, 0], pts[:, 1], xz[:, 1]])


def pose_compose(a, b):
    """a after b."""
    return np.array([a[0] + b[0], *(rot2(a[0]) @ b[1:] + a[1:])])


def pose_inverse(a):
    return np.array([-a[0], *(-rot2(-a[0]) @ a[1:])])


def icp_yaw(src, tgt, tree, init, iters=25, max_d=0.10):
    """Align src to tgt with a yaw rotation + floor translation. Returns (pose, rms, inlier_frac)."""
    pose = np.array(init, np.float64)
    keep = None
    d = None
    for _ in range(iters):
        moved = pose_apply(pose, src)
        d, idx = tree.query(moved)
        keep = d < max_d
        if keep.sum() < 100:
            return pose, 1.0, 0.0
        keep &= d <= np.percentile(d[keep], 85)
        p, q = moved[keep][:, [0, 2]], tgt[idx[keep]][:, [0, 2]]
        pc, qc = p.mean(0), q.mean(0)
        a, b = p - pc, q - qc
        dphi = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
        dt = qc - rot2(dphi) @ pc
        pose = pose_compose(np.array([dphi, *dt]), pose)
        if abs(dphi) < 1e-4 and np.linalg.norm(dt) < 1e-4:
            break
    return pose, float(np.sqrt(np.mean(d[keep] ** 2))), float(keep.mean())


class Tracker:
    """Keyframe-based yaw tracking of a spinning person.

    Aligning every frame to the previous one fails at small steps: a few degrees per frame
    moves the body a centimetre, below the sensor's depth quantisation, and ICP then biases
    toward "hardly moved".  Instead each frame is aligned to the last *keyframe*, which is only
    replaced after ~KEY_DEG of rotation, so every alignment sees a motion well above the noise.
    """

    KEY_DEG = 12.0

    def __init__(self):
        self.key = None            # keyframe points
        self.tree = None
        self.key_pose = np.zeros(3)   # model <- keyframe
        self.pose = np.zeros(3)       # model <- current frame
        self.vel = 0.0                # yaw rad/frame, smoothed
        self.last_phi = 0.0
        self.quality = 1.0

    @property
    def yaw(self):
        return float(self.pose[0])

    def add(self, pts):
        if self.key is None:
            self._set_key(pts, np.zeros(3))
            return self.pose.copy()
        c = self.key[:, [0, 2]].mean(0)
        # constant-velocity guess: rotate about the body centre (not the camera) so the
        # translation guess is small
        guess_phi = self.last_phi - self.key_pose[0] + self.vel
        init = np.array([guess_phi, *(c - rot2(guess_phi) @ c)])
        rel, rms, frac = icp_yaw(pts, self.key, self.tree, init)
        ok = rms < 0.04 and frac > 0.35 and abs(rel[0]) < np.radians(60)
        self.quality = 0.85 * self.quality + 0.15 * (1.0 if ok else 0.0)
        if not ok:
            rel = init  # coast on the velocity estimate until tracking recovers
        self.pose = pose_compose(self.key_pose, rel)
        self.vel = 0.6 * self.vel + 0.4 * (self.pose[0] - self.last_phi) if ok else self.vel
        self.last_phi = float(self.pose[0])
        if ok and abs(rel[0]) > np.radians(self.KEY_DEG):
            self._set_key(pts, self.pose)
        return self.pose.copy()

    def _set_key(self, pts, pose):
        self.key, self.tree, self.key_pose = pts, cKDTree(pts), np.array(pose, float)
        self.pose = np.array(pose, float)
        self.last_phi = float(pose[0])


def close_loops(pts_list, poses, log=None):
    """Distribute accumulated drift so each full revolution lands back on frame 0.

    The frame that completes revolution j is found by geometry, not by the (biased) yaw
    estimate: candidates near j*360 degrees are ICP-matched against frame 0 from several
    starting angles and the best fit wins.  Drift is then spread linearly between anchors.
    pts_list: per-frame tracking points (world coords); poses: Kx3 model<-frame poses.
    """
    poses = np.array(poses, np.float64)
    K = len(poses)
    sign = 1.0 if poses[-1, 0] >= 0 else -1.0
    yaw = sign * poses[:, 0]
    revs = int(np.floor((yaw[-1] + np.radians(40)) / (2 * np.pi)))
    if revs < 1 or K < 8:
        return poses
    tree0 = cKDTree(pts_list[0])
    anchors, corr = [0], [np.zeros(3)]
    for j in range(1, revs + 1):
        cands = [k for k in range(anchors[-1] + 1, K) if abs(yaw[k] - j * 2 * np.pi) < np.radians(40)]
        best = None
        for k in cands[::2] or cands:
            for off in np.radians([0, -20, 20, -35, 35]):
                init = poses[k].copy()
                init[0] = (init[0] + off + np.pi) % (2 * np.pi) - np.pi
                new, rms, frac = icp_yaw(pts_list[k], pts_list[0], tree0, init, iters=30, max_d=0.12)
                score = rms / max(frac, 1e-3)
                if frac > 0.5 and (best is None or score < best[0]):
                    best = (score, k, new, rms, frac)
        if best is None:
            if log:
                log(f"loop {j}: no match found")
            continue
        _, k, new, rms, frac = best
        e = pose_compose(new, pose_inverse(poses[k]))      # correction in the model frame
        e[0] = (e[0] + np.pi) % (2 * np.pi) - np.pi
        c0 = pts_list[0][:, [0, 2]].mean(0)  # judge the correction where the person stands
        shift = np.linalg.norm(rot2(e[0]) @ c0 + e[1:] - c0)
        if rms > 0.04 or abs(e[0]) > np.radians(60) or shift > 0.25:
            if log:
                log(f"loop {j}: rejected (rms {rms:.3f}, yaw {np.degrees(e[0]):+.1f})")
            continue
        if log:
            log(f"loop {j}: frame {k}, drift {np.degrees(e[0]):+.1f} deg, {shift * 100:.1f} cm, rms {rms * 100:.1f} cm")
        anchors.append(k)
        corr.append(e)
    out = poses.copy()
    for k in range(K):
        i = int(np.searchsorted(anchors, k, side="left"))
        if i >= len(anchors):
            e = corr[-1]
        elif i == 0 or anchors[i] == k:
            e = corr[i]
        else:
            w = (k - anchors[i - 1]) / (anchors[i] - anchors[i - 1])
            e = corr[i - 1] + w * (corr[i] - corr[i - 1])
        out[k] = pose_compose(e, poses[k])
    return out


class Capture:
    """Turns raw (rgb, depth) frames into tracked, masked captures ready for fusion."""

    def __init__(self, world, seg):
        self.world, self.seg = world, seg
        self.trk = Tracker()
        self.caps = []
        self.mask_px = []

    def add(self, rgb, depth):
        """Returns (mask, used). `used` is False when the person wasn't usable in this frame."""
        m, pw = self.seg.mask(depth)
        if int(m.sum()) < 3000:
            return m, False
        p = pw[m]
        p = p[(p[:, 1] > 0.15) & (p[:, 1] < 1.9)]  # no floor / overhead: static points bias the spin tracker
        if len(p) < 1500:
            return m, False
        pts = p[:: max(1, len(p) // 3000)].astype(np.float32)
        pose = self.trk.add(pts)
        self.caps.append(dict(depth=np.where(m, depth, 0).astype(np.uint16), rgb=rgb, pts=pts, pose=pose))
        self.mask_px.append(int(m.sum()))
        return m, True


class RawSaver:
    """Writes every captured frame (unmasked rgb jpg + 16-bit depth png) so scans can be re-processed."""

    def __init__(self, folder):
        import queue
        import threading
        self.dir = Path(folder) / "raw"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.q = queue.Queue(maxsize=400)
        self.n = 0
        self.times = []
        self.t0 = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add(self, rgb, depth):
        try:
            self.q.put_nowait((self.n, rgb, depth))
        except Exception:  # queue full: drop rather than stall the capture
            return
        self.times.append(time.time() - self.t0)
        self.n += 1

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            i, rgb, depth = item
            cv2.imwrite(str(self.dir / f"rgb_{i:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(self.dir / f"depth_{i:05d}.png"), depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def finish(self, world, center=None, extra=None):
        self.q.put(None)
        self.thread.join()
        info = dict(frames=self.n, times=self.times, world_R=world.R.tolist(), world_t=world.t.tolist(),
                    floor_found=bool(world.floor_found),
                    center=None if center is None else [float(x) for x in center],
                    intrinsics=dict(fx=FX, fy=FY, cx=CX, cy=CY))
        info.update(extra or {})
        (self.dir / "capture.json").write_text(json.dumps(info))


def rescan(folder, out=None, log=print, refloor=False):
    """Rebuild a model from a saved raw capture (`<scan>/raw`). Returns (out_folder, stats)."""
    folder = Path(folder)
    raw = folder / "raw"
    cfg = json.loads((raw / "capture.json").read_text())
    n = cfg["frames"]
    load = lambda i: (cv2.cvtColor(cv2.imread(str(raw / f"rgb_{i:05d}.jpg")), cv2.COLOR_BGR2RGB),  # noqa: E731
                      cv2.imread(str(raw / f"depth_{i:05d}.png"), cv2.IMREAD_UNCHANGED))
    rgb0, d0 = load(0)
    world = World(np.array(cfg["world_R"]), np.array(cfg["world_t"]), cfg["floor_found"])
    if refloor:
        w2 = fit_floor(d0)
        log(f"refit floor: {'camera height %.2f m' % w2.t[1] if w2 else 'FAILED, keeping the saved one'}")
        world = w2 or world
    seg = Segmenter(world)
    if cfg.get("center") and not refloor:
        seg.center = np.array(cfg["center"], np.float32)
    elif seg.find_person(d0) is None:
        raise RuntimeError("no person found in the first raw frame")
    cap = Capture(world, seg)
    for i in range(n):
        rgb, depth = load(i)
        cap.add(rgb, depth)
    stats = dict(frames=n, used=len(cap.caps), turns=abs(cap.trk.yaw) / (2 * np.pi), quality=cap.trk.quality,
                 mean_mask_px=float(np.mean(cap.mask_px)) if cap.mask_px else 0)
    log(f"rescan: {stats}")
    center = cap.caps[0]["pts"][:, [0, 2]].mean(0)
    verts, faces, rgb = build_model(cap.caps, world, center, log=log)
    out = Path(out) if out else folder / f"rescan_{time.strftime('%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    save_ply(out / "model.ply", verts, faces, rgb)
    save_obj(out / "model.obj", verts, faces, rgb)
    return out, stats, cap


# --------------------------------------------------------------------------- TSDF fusion

def fuse(frames, poses, world, center, voxel=0.015, trunc=0.06, radius=0.95, height=2.1,
         progress=None):
    """Integrate masked depth frames into a TSDF volume in the frame-0 world coordinates.

    frames: list of (depth_mm uint16 masked, rgb uint8). Returns dict with volume arrays.
    """
    xs = center[0] + np.arange(-radius, radius, voxel, dtype=np.float32) + voxel / 2
    ys = np.arange(0.0, height, voxel, dtype=np.float32) + voxel / 2
    zs = center[1] + np.arange(-radius, radius, voxel, dtype=np.float32) + voxel / 2
    dims = (len(xs), len(ys), len(zs))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vx, vy, vz = X.ravel(), Y.ravel(), Z.ravel()
    N = len(vx)
    tsdf = np.ones(N, np.float32)
    wt = np.zeros(N, np.float32)
    col = np.zeros((N, 3), np.float32)
    R, t = world.R, world.t
    for k, (depth, rgb) in enumerate(frames):
        phi, tx, tz = poses[k]
        c, s = np.cos(phi), np.sin(phi)
        dx, dz = vx - tx, vz - tz
        pw = np.stack([c * dx + s * dz, vy, -s * dx + c * dz], 1)  # model -> frame-k world
        pc = (pw - t) @ R                                           # -> frame-k camera
        z = pc[:, 2]
        zs_ = np.maximum(z, 1e-3)
        u = np.rint(FX * pc[:, 0] / zs_ + CX).astype(np.int32)
        v = np.rint(FY * pc[:, 1] / zs_ + CY).astype(np.int32)
        ok = (z > 0.3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.nonzero(ok)[0]
        d = depth[v[idx], u[idx]].astype(np.float32) / 1000
        good = d > 0
        idx, d = idx[good], d[good]
        sdf = d - z[idx]
        upd = sdf > -trunc
        idx, sdf = idx[upd], sdf[upd]
        if len(idx) == 0:
            continue
        sv = np.clip(sdf / trunc, -1, 1)
        w0 = wt[idx]
        tsdf[idx] = (tsdf[idx] * w0 + sv) / (w0 + 1)
        col[idx] = (col[idx] * w0[:, None] + rgb[v[idx], u[idx]]) / (w0[:, None] + 1)
        wt[idx] = w0 + 1
        if progress:
            progress(k + 1, len(frames))
    origin = np.array([xs[0], ys[0], zs[0]], np.float32)
    return dict(tsdf=tsdf.reshape(dims), wt=wt.reshape(dims), col=col.reshape(*dims, 3),
                origin=origin, voxel=voxel)


# --------------------------------------------------------------------------- meshing

def _adjacency(nv, faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    return A


def vertex_normals(verts, faces):
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)


def extract_mesh(vol, min_obs=2, smooth_iters=8):
    """Marching cubes on the fused TSDF -> (verts, faces, vertex colours uint8 RGB, normals)."""
    from skimage import measure

    mask = vol["wt"] >= min_obs
    if mask.sum() < 1000:
        raise RuntimeError("not enough data to build a model")
    v, f, _, _ = measure.marching_cubes(vol["tsdf"], level=0.0, spacing=(vol["voxel"],) * 3, mask=mask)
    v = (v + vol["origin"]).astype(np.float32)
    f = f.astype(np.int32)
    # keep the largest connected component
    A = _adjacency(len(v), f)
    n, lab = csgraph.connected_components(A, directed=False)
    sizes = np.bincount(lab)
    ymin = np.full(n, np.inf)
    ymax = np.full(n, -np.inf)
    np.minimum.at(ymin, lab, v[:, 1])
    np.maximum.at(ymax, lab, v[:, 1])
    good = (sizes >= 0.05 * sizes.max()) & ((ymax - ymin) > 0.3)  # drop floaters and flat floor scraps
    if not good.any():
        good = sizes == sizes.max()
    keep = np.isin(lab, np.nonzero(good)[0])
    remap = -np.ones(len(v), np.int32)
    remap[keep] = np.arange(keep.sum())
    f = remap[f]
    f = f[(f >= 0).all(1)]
    v = v[keep]
    # Taubin smoothing (no shrinkage)
    A = _adjacency(len(v), f)
    deg = np.asarray(A.sum(1)).ravel()
    L = sparse.diags(1 / np.maximum(deg, 1)) @ A
    for i in range(smooth_iters):
        lam = 0.5 if i % 2 == 0 else -0.53
        v = v + lam * (L @ v - v)
    n = vertex_normals(v, f)
    # colours from the nearest voxel
    from scipy import ndimage

    ij = np.clip(np.rint((v - vol["origin"]) / vol["voxel"]).astype(int), 0, np.array(vol["wt"].shape) - 1)
    # vertices can sit next to voxels that never got colour: borrow from the nearest observed voxel
    near = ndimage.distance_transform_edt(vol["wt"] < 1, return_distances=False, return_indices=True)
    ni = near[:, ij[:, 0], ij[:, 1], ij[:, 2]]
    rgb = np.clip(vol["col"][ni[0], ni[1], ni[2]], 0, 255).astype(np.uint8)
    return v, f, rgb, n


def save_ply(path, verts, faces, rgb):
    """Binary little-endian PLY with vertex colours and triangle faces."""
    vrec = np.empty(len(verts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vrec["x"], vrec["y"], vrec["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    vrec["r"], vrec["g"], vrec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    frec = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    frec["n"], frec["i"] = 3, faces
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vrec)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(frec)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(vrec.tobytes())
        fh.write(frec.tobytes())


def load_ply(path):
    """Read a PLY written by save_ply. Returns (verts, faces, rgb)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode().split("\n")
    nv = int([l for l in head if l.startswith("element vertex")][0].split()[2])
    nf = int([l for l in head if l.startswith("element face")][0].split()[2])
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    v = np.frombuffer(raw, vdt, nv, end)
    f = np.frombuffer(raw, fdt, nf, end + nv * vdt.itemsize)
    verts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rgb = np.stack([v["r"], v["g"], v["b"]], 1)
    return verts, f["i"].astype(np.int32), rgb


def save_obj(path, verts, faces, rgb):
    """OBJ with per-vertex colours (the `v x y z r g b` extension Blender reads)."""
    with open(path, "w") as fh:
        for p, c in zip(verts, rgb / 255.0):
            fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        for t in faces + 1:
            fh.write(f"f {t[0]} {t[1]} {t[2]}\n")


# --------------------------------------------------------------------------- rig + pose

JOINTS = ["pelvis", "spine", "neck", "head",
          "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist",
          "l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"]
PARENT = [-1, 0, 1, 2, 1, 4, 5, 1, 7, 8, 0, 10, 11, 0, 13, 14]
JI = {n: i for i, n in enumerate(JOINTS)}
# +x side is "l_" (the person's left when facing the camera)
SIDE = {"l": 1.0, "r": -1.0}


def estimate_rig(verts):
    """Place a 16-joint skeleton from the mesh (assumes an upright A-pose-ish scan facing +z)."""
    top = float(verts[:, 1].max())
    Hh = top
    mid = verts[(verts[:, 1] > 0.5 * Hh) & (verts[:, 1] < 0.8 * Hh)]
    cx, cz = np.median(mid[:, 0]), np.median(mid[:, 2])
    J = np.zeros((16, 3), np.float32)

    def put(name, x, y, z):
        J[JI[name]] = (x, y, z)

    put("pelvis", cx, 0.53 * Hh, cz)
    put("spine", cx, 0.68 * Hh, cz)
    put("neck", cx, 0.855 * Hh, cz)
    put("head", cx, 0.89 * Hh, cz)
    for side, s in SIDE.items():
        # legs: hip from proportions, knee/ankle from slab centroids on that side
        put(f"{side}_hip", cx + s * 0.055 * Hh, 0.53 * Hh, cz)
        for nm, lo, hi in (("knee", 0.25, 0.33), ("ankle", 0.04, 0.10)):
            sel = verts[(verts[:, 1] > lo * Hh) & (verts[:, 1] < hi * Hh) & (s * (verts[:, 0] - cx) > 0)]
            if len(sel) > 30:
                put(f"{side}_{nm}", sel[:, 0].mean(), (lo + hi) / 2 * Hh, sel[:, 2].mean())
            else:
                put(f"{side}_{nm}", cx + s * 0.055 * Hh, (0.285 if nm == "knee" else 0.045) * Hh, cz)
        # arms: find the lateral-most points beyond the torso, then place elbow/wrist along them
        sh = np.array([cx + s * 0.12 * Hh, 0.81 * Hh, cz], np.float32)
        arm = verts[(verts[:, 1] > 0.3 * Hh) & (verts[:, 1] < 0.86 * Hh) & (s * (verts[:, 0] - cx) > 0.13 * Hh)]
        tip = None
        if len(arm) > 200:
            d = np.linalg.norm(arm - sh, axis=1)
            far = arm[d >= np.percentile(d, 97)].mean(0)
            if np.linalg.norm(far - sh) > 0.25 * Hh:
                tip = far
        if tip is None:  # arms hang at ~25 degrees from the body
            tip = sh + 0.44 * Hh * np.array([s * np.sin(np.radians(25)), -np.cos(np.radians(25)), 0], np.float32)
        put(f"{side}_shoulder", *sh)
        put(f"{side}_elbow", *(sh + 0.42 * (tip - sh)))
        put(f"{side}_wrist", *(sh + 0.755 * (tip - sh)))
    return J, top


def bone_segments(J, top):
    """(start, end) point per joint; each joint drives the bone from itself to its child/tip."""
    B = np.zeros((16, 2, 3), np.float32)
    P = lambda n: J[JI[n]]  # noqa: E731
    B[JI["pelvis"]] = P("pelvis"), P("spine")
    B[JI["spine"]] = P("spine"), P("neck")
    B[JI["neck"]] = P("neck"), P("head")
    B[JI["head"]] = P("head"), np.array([P("head")[0], top, P("head")[2]], np.float32)
    for side in "lr":
        sh, el, wr = P(f"{side}_shoulder"), P(f"{side}_elbow"), P(f"{side}_wrist")
        B[JI[f"{side}_shoulder"]] = sh, el
        B[JI[f"{side}_elbow"]] = el, wr
        B[JI[f"{side}_wrist"]] = wr, wr + 0.45 * (wr - el)
        hp, kn, an = P(f"{side}_hip"), P(f"{side}_knee"), P(f"{side}_ankle")
        B[JI[f"{side}_hip"]] = hp, kn
        B[JI[f"{side}_knee"]] = kn, an
        B[JI[f"{side}_ankle"]] = an, an + np.array([0, -0.02, 0.10], np.float32) * top
    return B


def skin_weights(verts, J, top, k=4):
    """Inverse-distance (^4) skinning weights to the nearest k bones. Returns (idx Vxk, w Vxk)."""
    B = bone_segments(J, top)
    a, b = B[:, 0], B[:, 1]
    ab = b - a
    t = ((verts[:, None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)
    t = np.clip(t, 0, 1)
    closest = a + t[..., None] * ab
    d = np.linalg.norm(verts[:, None, :] - closest, axis=-1)  # V x 16
    w = 1.0 / (d + 0.015) ** 4
    # arms must not claim the chest: damp arm bones for vertices between the shoulders
    cx = J[JI["spine"]][0]
    chest = (np.abs(verts[:, 0] - cx) < 0.10 * top) & (verts[:, 1] > 0.55 * top)
    for n in ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"):
        w[chest, JI[n]] *= 0.02
    idx = np.argsort(-w, axis=1)[:, :k]
    wk = np.take_along_axis(w, idx, 1)
    wk /= wk.sum(1, keepdims=True)
    return idx.astype(np.int32), wk.astype(np.float32)


def euler(rx, ry, rz):
    """Rotation matrix Rz @ Ry @ Rx from degrees."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx_, sx_, cy_, sy_, cz_, sz_ = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx_, -sx_], [0, sx_, cx_]])
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    Rz = np.array([[cz_, -sz_, 0], [sz_, cz_, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Figure:
    """A scanned mesh with a skeleton, ready to be posed with linear blend skinning."""

    def __init__(self, verts, faces, rgb):
        self.rest = verts.astype(np.float32)
        self.faces = faces
        self.rgb = rgb
        self.J, self.top = estimate_rig(self.rest)
        self.idx, self.w = skin_weights(self.rest, self.J, self.top)
        self.rest_normals = vertex_normals(self.rest, faces)
        # rest abduction angle of each upper arm (0 = straight down, 90 = horizontal), degrees
        self.arm_rest = {}
        for side, s in SIDE.items():
            v = self.J[JI[f"{side}_elbow"]] - self.J[JI[f"{side}_shoulder"]]
            self.arm_rest[side] = float(np.degrees(np.arctan2(s * v[0], -v[1])))

    def pose_params(self, name, t=0.0):
        """Joint -> (rx, ry, rz) degrees for a named pose at time t."""
        p = {}
        arm = lambda side, deg: p.__setitem__(f"{side}_shoulder", (0, 0, SIDE[side] * (deg - self.arm_rest[side])))  # noqa: E731
        w = 2 * np.pi * t
        if name == "T-pose":
            arm("l", 90), arm("r", 90)
        elif name == "Hands up":
            arm("l", 165), arm("r", 165)
        elif name == "Wave":
            arm("l", 100)
            p["l_elbow"] = (0, 0, 55 + 28 * np.sin(2 * w))
            arm("r", 15)
        elif name == "Walk":
            a = np.sin(w)
            p["l_hip"], p["r_hip"] = (-28 * a, 0, 0), (28 * a, 0, 0)
            p["l_knee"], p["r_knee"] = (max(0, 40 * np.cos(w)), 0, 0), (max(0, -40 * np.cos(w)), 0, 0)
            p["l_shoulder"], p["r_shoulder"] = (25 * a, 0, 0), (-25 * a, 0, 0)
            p["l_elbow"], p["r_elbow"] = (-25 * (1 - a), 0, 0), (-25 * (1 + a), 0, 0)
            p["spine"] = (0, 8 * a, 0)
        elif name == "Dance":
            a = np.sin(2 * w)
            arm("l", 120 + 35 * a), arm("r", 120 - 35 * a)
            p["l_elbow"], p["r_elbow"] = (0, 0, 25 * a), (0, 0, 25 * a)
            p["pelvis"] = (0, 18 * np.sin(w), 0)
            p["spine"] = (0, -12 * np.sin(w), 0)
            p["head"] = (0, 0, 10 * np.sin(2 * w))
            p["l_knee"], p["r_knee"] = (max(0, 20 * a), 0, 0), (max(0, -20 * a), 0, 0)
        return p

    def posed(self, params):
        """Apply joint rotations (FK) + linear blend skinning. Returns (verts, normals)."""
        n = len(JOINTS)
        Rw = [None] * n
        Pw = [None] * n
        for j in range(n):
            Rl = euler(*params.get(JOINTS[j], (0, 0, 0)))
            par = PARENT[j]
            if par < 0:
                Rw[j] = Rl
                Pw[j] = self.J[j].astype(np.float64)
            else:
                Rw[j] = Rw[par] @ Rl
                Pw[j] = Pw[par] + Rw[par] @ (self.J[j] - self.J[par])
        Rs = np.stack(Rw).astype(np.float32)                         # 16x3x3
        ts = np.stack([Pw[j] - Rw[j] @ self.J[j] for j in range(n)]).astype(np.float32)
        out = np.zeros_like(self.rest)
        nrm = np.zeros_like(self.rest)
        for c in range(self.idx.shape[1]):
            ji, wc = self.idx[:, c], self.w[:, c:c + 1]
            R = Rs[ji]
            out += wc * (np.einsum("vij,vj->vi", R, self.rest) + ts[ji])
            nrm += wc * np.einsum("vij,vj->vi", R, self.rest_normals)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        return out, nrm

    def meta(self):
        return dict(height_m=round(self.top, 3), joints={n: [round(float(x), 4) for x in self.J[i]]
                                                           for i, n in enumerate(JOINTS)})


POSES = ["Rest", "T-pose", "Hands up", "Wave", "Walk", "Dance"]


# --------------------------------------------------------------------------- mesh rendering

class Orbit3D:
    """Mouse orbit camera around a figure."""

    def __init__(self):
        self.reset()
        self.drag = None

    def reset(self):
        self.yaw, self.pitch, self.zoom = 0.3, 0.1, 1.0

    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self.yaw += (x - self.drag[0]) * 0.01
            self.pitch = float(np.clip(self.pitch + (y - self.drag[1]) * 0.01, -1.3, 1.3))
            self.drag = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom = float(np.clip(self.zoom * (0.9 if cv2.getMouseWheelDelta(flags) > 0 else 1.1), 0.5, 2.5))


def render_mesh(verts, normals, faces, vcol_bgr, orbit, size=(720, 960), focus=None, top=1.75):
    """Software triangle rasteriser (painter's algorithm, back-face culled, headlight shading)."""
    Hc, Wc = size
    img = np.empty((Hc, Wc, 3), np.float32)
    img[:] = np.linspace(46, 20, Hc, dtype=np.float32)[:, None, None]
    f = 0.95 * Hc
    if focus is None:
        focus = np.array([verts[:, 0].mean(), 0.5 * top, verts[:, 2].mean()], np.float32)
    cy_, sy_, cp, sp = np.cos(orbit.yaw), np.sin(orbit.yaw), np.cos(orbit.pitch), np.sin(orbit.pitch)
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    Rv = Rx @ Ry
    dist = 3.3 * top / 1.75 / orbit.zoom

    def project(P):
        q = (P - focus) @ Rv.T
        depth = dist - q[:, 2]
        d = np.maximum(depth, 0.05)
        return Wc / 2 + f * q[:, 0] / d, Hc / 2 - f * (q[:, 1] - 0.0) / d, depth

    # floor ring for context
    ang = np.linspace(0, 2 * np.pi, 72)
    ring = np.stack([focus[0] + 0.7 * np.cos(ang), np.zeros_like(ang), focus[2] + 0.7 * np.sin(ang)], 1).astype(np.float32)
    rx_, ry_, _ = project(ring)
    cv2.polylines(img, [np.stack([rx_, ry_], 1).astype(np.int32)], True, (90, 90, 90), 1, cv2.LINE_AA)

    sx, sy, sz = project(verts)
    nv = normals @ Rv.T
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    fn = nv[a] + nv[b] + nv[c]
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    facing = fn[:, 2]                       # toward the viewer (+z in view space)
    keep = facing > 0.02
    a, b, c, facing = a[keep], b[keep], c[keep], facing[keep]
    zm = (sz[a] + sz[b] + sz[c]) / 3
    order = np.argsort(-zm)
    a, b, c, facing = a[order], b[order], c[order], facing[order]
    col = (vcol_bgr[a].astype(np.float32) + vcol_bgr[b] + vcol_bgr[c]) / 3
    col *= (0.35 + 0.65 * facing)[:, None]
    x0, x1, x2, y0, y1, y2 = sx[a], sx[b], sx[c], sy[a], sy[b], sy[c]
    minx, maxx = np.floor(np.minimum(np.minimum(x0, x1), x2)), np.ceil(np.maximum(np.maximum(x0, x1), x2))
    miny, maxy = np.floor(np.minimum(np.minimum(y0, y1), y2)), np.ceil(np.maximum(np.maximum(y0, y1), y2))
    vis = (maxx >= 0) & (minx < Wc) & (maxy >= 0) & (miny < Hc)
    bw, bh = (maxx - minx + 1), (maxy - miny + 1)
    vis &= (bw <= 14) & (bh <= 14)
    sel = np.nonzero(vis)[0]
    if len(sel):
        x0, x1, x2, y0, y1, y2 = (v[sel] for v in (x0, x1, x2, y0, y1, y2))
        minx, miny, col = minx[sel], miny[sel], col[sel]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        den = np.where(np.abs(den) < 1e-6, 1e-6, den)
        nb = int(min(14, max(bw[sel].max(), bh[sel].max())))
        for dy in range(nb):
            for dx in range(nb):
                px, py = minx + dx + 0.5, miny + dy + 0.5
                l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                m = (l0 >= -0.05) & (l1 >= -0.05) & (1 - l0 - l1 >= -0.05)
                ix, iy = (minx + dx).astype(np.int32), (miny + dy).astype(np.int32)
                m &= (ix >= 0) & (ix < Wc) & (iy >= 0) & (iy < Hc)
                img[iy[m], ix[m]] = col[m]
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- full pipeline

class ScanResult:
    def __init__(self, verts, faces, rgb):
        self.verts, self.faces, self.rgb = verts, faces, rgb


def build_model(captures, world, center, progress=None, log=print):
    """captures: list of dict(depth, rgb, pts, pose). Returns (verts, faces, rgb)."""
    poses = close_loops([c["pts"] for c in captures], [c["pose"] for c in captures], log=log)
    step = max(1, int(np.ceil(len(captures) / 110)))
    use = list(range(0, len(captures), step))
    frames = [(captures[i]["depth"], captures[i]["rgb"]) for i in use]
    vol = fuse(frames, poses[use], world, center,
               progress=(lambda k, n: progress("Fusing depth", 0.1 + 0.7 * k / n)) if progress else None)
    if progress:
        progress("Building mesh", 0.85)
    verts, faces, rgb, _ = extract_mesh(vol)
    return verts, faces, rgb


def safe_name(name):
    keep = "".join(ch for ch in name.strip() if ch.isalnum() or ch in " _-")[:30].strip()
    return keep or "visitor"


def new_scan_folder(root, name):
    folder = Path(root) / f"{safe_name(name).replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True)
    return folder


def save_scan(folder, name, verts, faces, rgb, fig=None, thumb=None):
    folder = Path(folder)
    save_ply(folder / "model.ply", verts, faces, rgb)
    save_obj(folder / "model.obj", verts, faces, rgb)
    meta = dict(name=name, created=time.strftime("%Y-%m-%d %H:%M:%S"), vertices=len(verts), faces=len(faces))
    if fig is not None:
        meta.update(fig.meta())
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    if thumb is not None:
        cv2.imwrite(str(folder / "preview.png"), thumb)
    return folder
