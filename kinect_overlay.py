#!/usr/bin/env python3
"""Xbox 360 Kinect viewer: RGB scene with a colorized depth map overlaid.

Uses libfreenect's registered depth (depth pixels aligned to the RGB camera, in mm),
so the overlay lines up with the scene.

Keys:
  + / -      overlay opacity
  [ / ]      near / far depth limit (mm) used for colouring
  c          cycle colormap
  v          cycle view: overlay / side-by-side / depth only / rgb only
  e          toggle edge-only overlay (depth contours on the RGB image)
  g          cycle measurement grid (off, 3x3, 5x4, 8x6, 12x9)
  m          mirror horizontally
  Up / Down  tilt motor +/- 5 degrees
  s          save snapshot to ./snapshots
  p          save the current scene as a .ply point cloud to ./pointclouds
  r          start / stop recording RGB + depth to ./recordings/<timestamp>
  0          reset the 3d view
  q / Esc    quit

In the 3d view: drag with the mouse to orbit, scroll to zoom.
Replay a recording with:  kinect_overlay.py --play recordings/<timestamp>
"""
import argparse
import os
import sys
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import freenect
import numpy as np

COLORMAPS = [
    ("TURBO", cv2.COLORMAP_TURBO),
    ("JET", cv2.COLORMAP_JET),
    ("INFERNO", cv2.COLORMAP_INFERNO),
    ("VIRIDIS", cv2.COLORMAP_VIRIDIS),
    ("BONE", cv2.COLORMAP_BONE),
]
GRIDS = [None, (3, 3), (5, 4), (8, 6), (12, 9)]  # (cols, rows)
VIEWS = ["overlay", "side-by-side", "depth", "rgb", "3d"]

# Approximate Kinect v1 intrinsics for 640x480 registered depth.
FX = FY = 580.0
CX, CY = 319.5, 239.5


class Kinect:
    """Runs the libfreenect event loop in a thread and keeps the latest frames."""

    def __init__(self):
        self.rgb = None
        self.depth = None
        self.lock = threading.Lock()
        self.tilt_target = 0
        self._tilt_sent = None
        self.running = True
        self._restart = False
        self._last_ok = time.time()
        self._t_rgb = self._t_depth = 0.0
        self.seq = 0  # increments with every depth frame
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _on_video(self, dev, data, ts):
        with self.lock:
            self.rgb = data.copy()
            self._t_rgb = time.time()
            self._mark_ok()

    def _on_depth(self, dev, data, ts):
        with self.lock:
            self.depth = data.copy()
            self.seq += 1
            self._t_depth = time.time()
            self._mark_ok()

    def _mark_ok(self):
        # healthy only while both streams have delivered within the last second
        if abs(self._t_rgb - self._t_depth) < 1 and time.time() - self._t_rgb < 1:
            self._last_ok = time.time()

    def _on_body(self, dev, ctx):
        if not self.running or self._restart:
            raise freenect.Kill
        if self.tilt_target != self._tilt_sent:
            freenect.set_tilt_degs(dev, self.tilt_target)
            self._tilt_sent = self.tilt_target
        # Kinect streams sometimes fail to come up on USB 2; restart if either stalls.
        if time.time() - self._last_ok > 3:
            self._restart = True

    def _run(self):
        while self.running:
            self._restart = False
            self._last_ok = time.time()
            self.rgb = self.depth = None
            ctx = freenect.init()
            dev = freenect.open_device(ctx, 0)
            if not dev:
                time.sleep(1)
                continue
            freenect.set_video_mode(dev, freenect.RESOLUTION_MEDIUM, freenect.VIDEO_RGB)
            freenect.set_depth_mode(dev, freenect.RESOLUTION_MEDIUM, freenect.DEPTH_REGISTERED)
            freenect.runloop(dev=dev, depth=self._on_depth, video=self._on_video, body=self._on_body)
            freenect.close_device(dev)
            freenect.shutdown(ctx)
            if self._restart:
                print("Kinect stream stalled, restarting...")

    def frames(self):
        with self.lock:
            return self.rgb, self.depth

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)


class Player:
    """Replays a recording made with the `r` key; same interface as Kinect."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.names = sorted(p.stem[len("depth_"):] for p in self.folder.glob("depth_*.png"))
        if not self.names:
            raise SystemExit(f"no frames found in {self.folder}")
        meta = self.folder / "timestamps.json"
        self.times = json.loads(meta.read_text()) if meta.exists() else [i / 30 for i in range(len(self.names))]
        self.rgb = self.depth = None
        self.seq = 0
        self.tilt_target = 0
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while self.running:
            t0 = time.time()
            for i, name in enumerate(self.names):
                wait = self.times[i] - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
                if not self.running:
                    return
                bgr = cv2.imread(str(self.folder / f"rgb_{name}.jpg"))
                depth = cv2.imread(str(self.folder / f"depth_{name}.png"), cv2.IMREAD_UNCHANGED)
                if bgr is None or depth is None:
                    continue
                with self.lock:
                    self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    self.depth = depth
                    self.seq += 1

    def frames(self):
        with self.lock:
            return self.rgb, self.depth

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)


class Recorder:
    """Writes frames (rgb jpg + 16-bit depth png) from a background thread."""

    def __init__(self, root):
        self.dir = Path(root) / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True)
        self.q = queue.Queue(maxsize=90)
        self.count = 0
        self.times = []
        self.t0 = time.time()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def add(self, rgb, depth):
        try:
            self.q.put_nowait((self.count, rgb, depth))
        except queue.Full:
            return  # disk can't keep up; drop the frame
        self.times.append(time.time() - self.t0)
        self.count += 1

    def _write(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            i, rgb, depth = item
            cv2.imwrite(str(self.dir / f"rgb_{i:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(self.dir / f"depth_{i:05d}.png"), depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def stop(self):
        self.q.put(None)
        self.thread.join()
        (self.dir / "timestamps.json").write_text(json.dumps(self.times))
        return self.dir, self.count


class Orbit:
    """Mouse-driven orbit camera for the 3d view."""

    def __init__(self):
        self.reset()
        self.drag = None

    def reset(self):
        self.yaw = self.pitch = 0.0
        self.zoom = 1.0
        self.pz = None  # smoothed pivot depth (m)

    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self.yaw += (x - self.drag[0]) * 0.01
            self.pitch = float(np.clip(self.pitch - (y - self.drag[1]) * 0.01, -1.5, 1.5))
            self.drag = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom = float(np.clip(self.zoom * (0.9 if cv2.getMouseWheelDelta(flags) > 0 else 1.1), 0.15, 4))


def depth_to_points(depth, bgr, step=1):
    """Back-project registered depth (mm) to metres. Returns (xyz Nx3, bgr Nx3)."""
    h, w = depth.shape
    d = depth[::step, ::step].astype(np.float32) / 1000
    v, u = np.mgrid[0:h:step, 0:w:step]
    m = (d > 0.3) & (d < 8)
    z = d[m]
    pts = np.stack([(u[m] - CX) * z / FX, (v[m] - CY) * z / FY, z], 1)
    return pts, bgr[::step, ::step][m]


def render_cloud(bgr, depth, cam, step=2):
    """Software-render the point cloud from the orbit camera (painter's algorithm)."""
    h, w = depth.shape
    out = np.full((h, w, 3), 18, np.uint8)
    pts, col = depth_to_points(depth, bgr, step)
    if len(pts) == 0:
        return out
    med = float(np.median(pts[:, 2]))
    cam.pz = med if cam.pz is None else 0.9 * cam.pz + 0.1 * med
    pivot = np.array([0, 0, cam.pz], np.float32)
    cyaw, syaw, cp, sp = np.cos(cam.yaw), np.sin(cam.yaw), np.cos(cam.pitch), np.sin(cam.pitch)
    ry = np.array([[cyaw, 0, syaw], [0, 1, 0], [-syaw, 0, cyaw]], np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    p = (pts - pivot) @ (rx @ ry).T
    zc = p[:, 2] + cam.pz * cam.zoom
    keep = zc > 0.1
    p, zc, col = p[keep], zc[keep], col[keep]
    order = np.argsort(-zc)  # far first, so near points overwrite
    p, zc, col = p[order], zc[order], col[order]
    sx = (CX + FX * p[:, 0] / zc).astype(np.int32)
    sy = (CY + FY * p[:, 1] / zc).astype(np.int32)
    for dx in range(step + 1):
        for dy in range(step + 1):
            x, y = sx + dx, sy + dy
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            out[y[ok], x[ok]] = col[ok]
    return out


def save_ply(path, depth, rgb):
    """Write a binary little-endian PLY (x, y, z in metres + RGB colour)."""
    pts, rgbc = depth_to_points(depth, rgb, 1)  # rgb in, rgb out (the helper just indexes colours)
    rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], -pts[:, 1], -pts[:, 2]  # y up, camera looks down -z
    rec["r"], rec["g"], rec["b"] = rgbc[:, 0], rgbc[:, 1], rgbc[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(rec)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())
    return len(rec)


def colorize(depth_mm, near, far, cmap):
    """Map depth (mm) to a BGR image; invalid (0) pixels return mask False."""
    valid = (depth_mm > 0) & (depth_mm < 10000)
    d = np.clip(depth_mm.astype(np.float32), near, far)
    norm = ((far - d) / max(far - near, 1) * 255).astype(np.uint8)  # near = hot
    color = cv2.applyColorMap(norm, cmap)
    color[~valid] = 0
    return color, valid


def draw_grid(out, depth, cols, rows, x0, pw):
    """Draw a cols x rows measurement grid on the panel out[:, x0:x0+pw].

    Each cell is labelled with the median valid distance in metres.
    """
    ph = out.shape[0]
    dh, dw = depth.shape
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, (0.5 if cols <= 8 else 0.38)
    lines = out.copy()
    for c in range(1, cols):
        x = x0 + c * pw // cols
        cv2.line(lines, (x, 0), (x, ph), (255, 255, 255), 1)
    for r in range(1, rows):
        y = r * ph // rows
        cv2.line(lines, (x0, y), (x0 + pw, y), (255, 255, 255), 1)
    out[:] = cv2.addWeighted(out, 0.65, lines, 0.35, 0)
    for r in range(rows):
        for c in range(cols):
            cell = depth[r * dh // rows : (r + 1) * dh // rows, c * dw // cols : (c + 1) * dw // cols]
            cell = cell[cell > 0]
            txt = f"{np.median(cell) / 1000:.2f}m" if cell.size > 20 else "--"
            (tw, th), _ = cv2.getTextSize(txt, font, scale, 1)
            cx = x0 + (2 * c + 1) * pw // (2 * cols)
            cy = (2 * r + 1) * ph // (2 * rows)
            cv2.rectangle(out, (cx - tw // 2 - 3, cy - th // 2 - 3), (cx + tw // 2 + 3, cy + th // 2 + 3), (0, 0, 0), -1)
            cv2.putText(out, txt, (cx - tw // 2, cy + th // 2), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


PANEL_W = 200


def side_panel(height, title, rows):
    """Dark panel listing (key, description) rows; a row with key None is a section gap."""
    p = np.full((height, PANEL_W, 3), 28, np.uint8)
    cv2.putText(p, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 1, cv2.LINE_AA)
    cv2.line(p, (12, 38), (PANEL_W - 12, 38), (80, 80, 80), 1)
    y = 66
    for key, desc in rows:
        if key is None:
            y += 12
            continue
        cv2.putText(p, key, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(p, desc, (12, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        y += 40
    return p


def main():
    ap = argparse.ArgumentParser(description="Kinect RGB + depth viewer")
    ap.add_argument("--play", metavar="DIR", help="replay a recording instead of using the Kinect")
    args = ap.parse_args()

    kin = Player(args.play) if args.play else Kinect()
    kin.start()
    orbit = Orbit()
    rec = None
    last_seq = -1
    root = Path(__file__).parent

    alpha, near, far = 0.5, 500, 4000
    cmap_i, view_i = 0, 0
    mirror = edges = False
    grid_i = 2
    snap_dir = Path(__file__).parent / "snapshots"
    win = "Kinect RGB + Depth"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, orbit.on_mouse)
    fps_t, fps_n, fps = time.time(), 0, 0.0

    print("Waiting for frames from Kinect...")
    while True:
        rgb, depth = kin.frames()
        if rgb is None or depth is None:
            if cv2.waitKey(30) in (27, ord("q")):
                break
            continue

        seq = kin.seq
        if rec and seq != last_seq:
            rec.add(rgb, depth)
        last_seq = seq

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dcolor, valid = colorize(depth, near, far, COLORMAPS[cmap_i][1])

        if edges:
            norm = np.clip((far - np.clip(depth, near, far)) / max(far - near, 1) * 255, 0, 255).astype(np.uint8)
            e = cv2.Canny(cv2.GaussianBlur(norm, (5, 5), 0), 20, 60)
            dcolor = cv2.bitwise_and(dcolor, dcolor, mask=cv2.dilate(e, None))
            valid = e > 0

        view = VIEWS[view_i]
        if view == "overlay":
            out = bgr.copy()
            blended = cv2.addWeighted(bgr, 1 - alpha, dcolor, alpha, 0)
            out[valid] = blended[valid]
        elif view == "side-by-side":
            out = np.hstack([bgr, dcolor])
        elif view == "depth":
            out = dcolor
        elif view == "3d":
            out = render_cloud(bgr, depth, orbit)
        else:
            out = bgr
        is3d = view == "3d"

        if mirror and not is3d:
            out = out[:, ::-1].copy()

        # Centre-pixel distance readout
        h, w = depth.shape
        cd = depth[h // 2 - 2 : h // 2 + 3, w // 2 - 2 : w // 2 + 3]
        cd = cd[cd > 0]
        dist = f"{np.median(cd) / 1000:.2f} m" if cd.size else "--"
        grid_cfg = GRIDS[grid_i]
        grid = None if is3d else grid_cfg
        if grid:
            gd = depth[:, ::-1] if mirror else depth
            panels = 2 if view == "side-by-side" else 1
            pw = out.shape[1] // panels
            for p in range(panels):
                draw_grid(out, gd, grid[0], grid[1], p * pw, pw)
        elif view not in ("side-by-side", "3d"):
            cx, cy = out.shape[1] // 2, out.shape[0] // 2
            cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
            (tw, th), _ = cv2.getTextSize(dist, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (cx + 8, cy - 14 - th), (cx + 14 + tw, cy - 6), (0, 0, 0), -1)
            cv2.putText(out, dist, (cx + 11, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        fps_n += 1
        if time.time() - fps_t >= 1:
            fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
        gtxt = f"grid {grid_cfg[0]}x{grid_cfg[1]}" if grid_cfg else "grid off"
        status = (f"{view} | {gtxt} | {COLORMAPS[cmap_i][0]} | a={alpha:.2f} | "
                  f"{near}-{far}mm | tilt {kin.tilt_target} | {fps:.0f}fps")
        if args.play:
            status = "PLAYBACK | " + status
        if rec:
            status = f"REC {rec.count} | " + status
        bar_h = 30
        out[:bar_h] = out[:bar_h] // 6  # darken a strip so the text is always readable
        cv2.putText(out, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (80, 80, 255) if rec else (255, 255, 255), 1, cv2.LINE_AA)

        frame = out
        left = side_panel(frame.shape[0], "VIEW", [
            ("v", f"view: {view}"),
            ("g", f"grid: {gtxt[5:]}"),
            ("c", f"colormap: {COLORMAPS[cmap_i][0]}"),
            ("e", f"edges: {'on' if edges else 'off'}"),
            ("m", f"mirror: {'on' if mirror else 'off'}"),
            ("drag/wheel", "orbit/zoom in 3d"),
            ("0", "reset 3d view"),
        ])
        right = side_panel(frame.shape[0], "ADJUST", [
            ("+ / -", f"opacity: {alpha:.2f}"),
            ("[ / ]", f"range: {near}-{far} mm"),
            ("Up / Down", f"tilt: {kin.tilt_target} deg"),
            (None, ""),
            ("s", "save snapshot"),
            ("p", "save point cloud"),
            ("r", "recording: ON" if rec else "record"),
            ("q / Esc", "quit"),
        ])
        cv2.imshow(win, np.hstack([left, frame, right]))
        k = cv2.waitKeyEx(1)
        if k == -1:
            continue
        ch = k & 0xFF
        if ch in (27, ord("q")):
            break
        elif ch in (ord("+"), ord("=")):
            alpha = min(1.0, alpha + 0.05)
        elif ch in (ord("-"), ord("_")):
            alpha = max(0.0, alpha - 0.05)
        elif ch == ord("["):
            near = max(0, near - 100)
        elif ch == ord("]"):
            far = min(10000, far + 100)
        elif ch == ord("c"):
            cmap_i = (cmap_i + 1) % len(COLORMAPS)
        elif ch == ord("v"):
            view_i = (view_i + 1) % len(VIEWS)
        elif ch == ord("g"):
            grid_i = (grid_i + 1) % len(GRIDS)
        elif ch == ord("e"):
            edges = not edges
        elif ch == ord("m"):
            mirror = not mirror
        elif ch == ord("0"):
            orbit.reset()
        elif ch == ord("p"):
            out_dir = root / "pointclouds"
            out_dir.mkdir(exist_ok=True)
            path = out_dir / f"kinect_{time.strftime('%Y%m%d_%H%M%S')}.ply"
            print("saved", path, f"({save_ply(path, depth, rgb)} points)")
        elif ch == ord("r") and not args.play:
            if rec:
                folder, n = rec.stop()
                print(f"recording saved: {folder} ({n} frames)")
                rec = None
            else:
                rec = Recorder(root / "recordings")
                print("recording to", rec.dir)
        elif ch == ord("s"):
            snap_dir.mkdir(exist_ok=True)
            p = snap_dir / f"kinect_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(p), frame)
            print("saved", p)
        elif k in (65362, 2490368, 82):  # Up
            kin.tilt_target = min(30, kin.tilt_target + 5)
        elif k in (65364, 2621440, 84):  # Down
            kin.tilt_target = max(-30, kin.tilt_target - 5)

    if rec:
        print("recording saved:", *rec.stop())
    kin.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    sys.stdout.flush()
    os._exit(0)  # skip Qt's atexit teardown, which segfaults with the fcitx5 input plugin


if __name__ == "__main__":
    main()
