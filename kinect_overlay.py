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
  q / Esc    quit
"""
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
VIEWS = ["overlay", "side-by-side", "depth", "rgb"]


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
    kin = Kinect()
    kin.start()

    alpha, near, far = 0.5, 500, 4000
    cmap_i, view_i = 0, 0
    mirror = edges = False
    grid_i = 2
    snap_dir = Path(__file__).parent / "snapshots"
    win = "Kinect RGB + Depth"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fps_t, fps_n, fps = time.time(), 0, 0.0

    print("Waiting for frames from Kinect...")
    while True:
        rgb, depth = kin.frames()
        if rgb is None or depth is None:
            if cv2.waitKey(30) in (27, ord("q")):
                break
            continue

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
        else:
            out = bgr

        if mirror:
            out = out[:, ::-1].copy()

        # Centre-pixel distance readout
        h, w = depth.shape
        cd = depth[h // 2 - 2 : h // 2 + 3, w // 2 - 2 : w // 2 + 3]
        cd = cd[cd > 0]
        dist = f"{np.median(cd) / 1000:.2f} m" if cd.size else "--"
        grid = GRIDS[grid_i]
        if grid:
            gd = depth[:, ::-1] if mirror else depth
            panels = 2 if view == "side-by-side" else 1
            pw = out.shape[1] // panels
            for p in range(panels):
                draw_grid(out, gd, grid[0], grid[1], p * pw, pw)
        elif view != "side-by-side":
            cx, cy = out.shape[1] // 2, out.shape[0] // 2
            cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
            (tw, th), _ = cv2.getTextSize(dist, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (cx + 8, cy - 14 - th), (cx + 14 + tw, cy - 6), (0, 0, 0), -1)
            cv2.putText(out, dist, (cx + 11, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        fps_n += 1
        if time.time() - fps_t >= 1:
            fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
        gtxt = f"grid {grid[0]}x{grid[1]}" if grid else "grid off"
        status = (f"{view} | {gtxt} | {COLORMAPS[cmap_i][0]} | a={alpha:.2f} | "
                  f"{near}-{far}mm | tilt {kin.tilt_target} | {fps:.0f}fps")
        bar_h = 30
        out[:bar_h] = out[:bar_h] // 6  # darken a strip so the text is always readable
        cv2.putText(out, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        frame = out
        left = side_panel(frame.shape[0], "VIEW", [
            ("v", f"view: {view}"),
            ("g", f"grid: {gtxt[5:]}"),
            ("c", f"colormap: {COLORMAPS[cmap_i][0]}"),
            ("e", f"edges: {'on' if edges else 'off'}"),
            ("m", f"mirror: {'on' if mirror else 'off'}"),
        ])
        right = side_panel(frame.shape[0], "ADJUST", [
            ("+ / -", f"opacity: {alpha:.2f}"),
            ("[ / ]", f"range: {near}-{far} mm"),
            ("Up / Down", f"tilt: {kin.tilt_target} deg"),
            (None, ""),
            ("s", "save snapshot"),
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
        elif ch == ord("s"):
            snap_dir.mkdir(exist_ok=True)
            p = snap_dir / f"kinect_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(p), frame)
            print("saved", p)
        elif k in (65362, 2490368, 82):  # Up
            kin.tilt_target = min(30, kin.tilt_target + 5)
        elif k in (65364, 2621440, 84):  # Down
            kin.tilt_target = max(-30, kin.tilt_target - 5)

    kin.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
