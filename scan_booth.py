#!/usr/bin/env python3
"""3D photo booth: a visitor types their name, spins twice in front of the Kinect,
and gets a posable 3D figure of themselves.

  python3 scan_booth.py            # real Kinect
  python3 scan_booth.py --demo     # synthetic spinning person (no sensor needed)
  python3 scan_booth.py --view scans/<folder>   # open a saved scan

Flow: NAME -> 3,2,1 -> spin capture (2 turns, tracked live) -> processing -> figure viewer.
Viewer keys: 1-6 poses, drag = orbit, wheel = zoom, a = auto-spin, 0 = reset view,
             s = export current pose (+ screenshot), n = new scan, F11 = fullscreen, Esc = quit.
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import bodyscan as bs

CW, CH = 960, 720           # canvas size
FONT = cv2.FONT_HERSHEY_DUPLEX
TURNS_NEEDED = 2.0
KEY_UP = (65362, 2490368)
KEY_DOWN = (65364, 2621440)
KEY_ENTER = (13, 10, 65293, 65421, 16777220, 16777221, 3)  # incl. Qt Key_Return / Key_Enter / keypad
KEY_BACK = (8, 65288, 127, 65535, 16777219)
KEY_F11 = (65480,)


def text(img, s, org, scale=0.8, color=(255, 255, 255), thick=1, shadow=True):
    if shadow:
        cv2.putText(img, s, (org[0] + 2, org[1] + 2), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_c(img, s, y, scale=0.8, color=(255, 255, 255), thick=1, shadow=True):
    (w, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    text(img, s, ((img.shape[1] - w) // 2, y), scale, color, thick, shadow)


def panel(img, x0, y0, x1, y1, alpha=0.62):
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def camera_canvas(rgb, mask=None, dim=0.0, mirror=True):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if mask is not None:
        bgr = np.where(mask[..., None], bgr, (bgr * 0.35).astype(np.uint8))
    if mirror:
        bgr = bgr[:, ::-1]
    big = cv2.resize(np.ascontiguousarray(bgr), (CW, CH), interpolation=cv2.INTER_LINEAR)
    if dim:
        big = (big * (1 - dim)).astype(np.uint8)
    return big


class Booth:
    def __init__(self, src, root, view_dir=None):
        self.src = src
        self.root = Path(root)
        self.state = "NAME"
        self.name = ""
        self.msg = ""
        self.msg_until = 0.0
        self.orbit = bs.Orbit3D()
        self.world = bs.World()
        self.seg = None
        self.probe = dict(t=0.0, text="Looking for you...", ok=False)
        self.fig = None
        self.folder = None
        self.pose_i = 0
        self.cur = {}
        self.auto = False
        self.t_view = time.time()
        self.prog = ("", 0.0)
        self.error = ""
        self.fullscreen = False
        self.force_until = 0.0
        self.raw = None
        if view_dir:
            self._open_saved(Path(view_dir))

    # ------------------------------------------------------------------ helpers
    def toast(self, s, secs=3.0):
        self.msg, self.msg_until = s, time.time() + secs

    def _open_saved(self, folder):
        verts, faces, rgb = bs.load_ply(folder / "model.ply")
        self.fig = bs.Figure(verts, faces, rgb)
        self.folder = folder
        self.name = folder.name
        self.state = "VIEW"
        self.t_view = time.time()

    def _probe(self):
        """Live feedback while the visitor is typing their name / getting into place."""
        now = time.time()
        rgb, depth = self.src.frames()
        if depth is None or now - self.probe["t"] < 0.5:
            return
        self.probe["t"] = now
        dist = bs.nearest_distance(depth)
        self.probe["dist"] = dist
        # keep re-fitting the floor (a person at the keyboard gives a bad fit at first)
        if now - self.probe.get("floor_t", 0) > 2.0:
            self.probe["floor_t"] = now
            w = bs.fit_floor(depth)
            if w is not None:
                self.world = w
            elif dist is not None and dist < 1.8:
                self.world = bs.World()  # too close to see the floor: forget any stale fit
        seg = bs.Segmenter(self.world)
        res = seg.find_person(depth)
        if res is not None:
            self.probe["last_res"], self.probe["last_res_t"] = res, now
        elif now - self.probe.get("last_res_t", 0) < 1.5:
            res = self.probe["last_res"]  # brief dropouts (e.g. side-on pose) shouldn't flicker the status
        if dist is None:
            self.probe.update(text="Nothing in view - step in front of the sensor", ok=False)
        elif dist < 2.4:
            self.probe.update(text=f"{dist:.1f} m - too close, step back to about 3 m", ok=False)
        elif dist > 4.0:
            self.probe.update(text=f"{dist:.1f} m - too far, come closer (about 3 m)", ok=False)
        elif res is None:
            self.probe.update(text=f"{dist:.1f} m - good distance, but I can't see a full person yet", ok=False)
        else:
            m, pw = seg.mask(depth)
            ymin = float(np.percentile(pw[m][:, 1], 1)) if m.any() else 9.0
            if m[-4:].any() or ymin > 0.35:
                self.probe.update(text=f"{dist:.1f} m - feet cut off: tilt the sensor down (Down arrow) or step back", ok=False)
            elif m[:4].any():
                self.probe.update(text=f"{dist:.1f} m - head cut off: tilt the sensor up (Up arrow) or step back", ok=False)
            elif res[1] < 1.3:
                self.probe.update(text=f"{dist:.1f} m - I only see {res[1]:.1f} m of you: step back or tilt so your whole body is in view", ok=False)
            elif not self.world.floor_found:
                self.probe.update(text=f"{dist:.1f} m - can't see the floor, tilt the sensor down a little", ok=False)
            else:
                self.probe.update(text=f"Perfect - {res[1]:.2f} m tall, {dist:.1f} m away", ok=True)

    def _distance_meter(self, c, y):
        """Horizontal 1-5 m scale with the ideal zone in green and a marker at the current distance."""
        x0, x1 = 170, CW - 170
        px = lambda m: int(x0 + (x1 - x0) * (np.clip(m, 1.0, 5.0) - 1.0) / 4.0)  # noqa: E731
        cv2.rectangle(c, (x0, y), (x1, y + 10), (70, 70, 70), -1)
        cv2.rectangle(c, (px(2.6), y), (px(3.6), y + 10), (90, 200, 90), -1)
        for m in (1, 2, 3, 4, 5):
            cv2.line(c, (px(m), y + 10), (px(m), y + 16), (200, 200, 200), 1)
            text(c, f"{m}m", (px(m) - 12, y + 34), 0.45, (190, 190, 190), 1, False)
        d = self.probe.get("dist")
        if d is not None:
            ok = 2.6 <= d <= 3.6
            col = (120, 230, 120) if ok else (80, 180, 255)
            cv2.fillPoly(c, [np.array([[px(d), y - 2], [px(d) - 9, y - 16], [px(d) + 9, y - 16]])], col)
            text(c, f"{d:.1f} m", (px(d) - 26, y - 22), 0.6, col, 1)

    # ------------------------------------------------------------------ states
    def _start_countdown(self):
        rgb, depth = self.src.frames()
        if depth is None:
            self.toast("Camera not ready yet")
            return
        w = bs.fit_floor(depth)
        self.world = w if w is not None else self.world
        self.seg = bs.Segmenter(self.world)
        if self.seg.find_person(depth) is None:
            self.toast("I can't see you - step into view and try again", 4)
            return
        self.state, self.t_state = "COUNTDOWN", time.time()

    def _start_capture(self):
        self.folder = bs.new_scan_folder(self.root, self.name)
        self.raw = bs.RawSaver(self.folder)
        self.cap = bs.Capture(self.world, self.seg)
        self.center0 = None
        self.last_seq, self.last_t = -1, 0.0
        self.last_mask = None
        self.visible = True
        self.state, self.t_state = "CAPTURE", time.time()

    def _sample(self):
        rgb, depth = self.src.frames()
        seq = self.src.seq
        if rgb is None or seq == self.last_seq or time.time() - self.last_t < 0.09:
            return rgb
        self.last_seq, self.last_t = seq, time.time()
        m, used = self.cap.add(rgb, depth)
        self.last_mask = m
        self.visible = used
        if used:
            self.raw.add(rgb, depth)
            if self.center0 is None:
                self.center0 = self.cap.caps[0]["pts"][:, [0, 2]].mean(0)
        return rgb

    def _finish_raw(self, extra=None):
        if self.raw is not None:
            q = self.cap
            self.raw.finish(self.world, self.center0, dict(online_turns=abs(q.trk.yaw) / (2 * np.pi),
                                                           quality=q.trk.quality, name=self.name, **(extra or {})))
            self.raw = None

    def _start_process(self):
        self.state, self.t_state = "PROCESS", time.time()
        self.prog = ("Aligning the two turns", 0.05)
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        try:
            caps = self.cap.caps
            center = self.center0
            verts, faces, rgb = bs.build_model(caps, self.world, center,
                                               progress=lambda s, f: setattr(self, "prog", (s, f)), log=print)
            self.prog = ("Rigging the skeleton", 0.93)
            fig = bs.Figure(verts, faces, rgb)
            thumb = bs.render_mesh(verts, fig.rest_normals, faces, rgb[:, ::-1].copy(), bs.Orbit3D(),
                                   size=(CH, CW), top=fig.top)
            bs.save_scan(self.folder, self.name, verts, faces, rgb, fig, thumb)
            print("saved scan:", self.folder)
            self.fig = fig
            self.cur, self.pose_i, self.t_view = {}, 0, time.time()
            self.orbit.reset()
            self.cap = None
            self.state = "VIEW"
        except Exception as e:  # noqa: BLE001 - show any failure to the operator
            import traceback
            traceback.print_exc()
            self.error = str(e)
            self.cap = None
            self.state = "NAME"
            self.toast(f"Scan failed: {e}", 6)

    # ------------------------------------------------------------------ frame
    def tick(self, key):
        if key != -1:
            print(f"key {key} (state {self.state})", flush=True)
        handler = getattr(self, f"_tick_{self.state.lower()}")
        canvas = handler(key)
        if time.time() < self.msg_until:
            panel(canvas, 0, CH - 56, CW, CH, 0.7)
            text_c(canvas, self.msg, CH - 20, 0.7, (120, 220, 255))
        return canvas

    def _tick_name(self, key):
        rgb, depth = self.src.frames()
        if key in KEY_UP:
            self.src.tilt_target = min(30, self.src.tilt_target + 5)
        elif key in KEY_DOWN:
            self.src.tilt_target = max(-30, self.src.tilt_target - 5)
        elif key in KEY_ENTER:
            if not self.name.strip():
                self.toast("Please type your name first")
            elif self.probe["ok"] or time.time() < self.force_until:
                self._start_countdown()
            else:
                self.force_until = time.time() + 6
                self.toast("Not ready - press ENTER again to scan anyway", 6)
        elif key in KEY_BACK:
            self.name = self.name[:-1]
        elif 32 <= key < 127 and len(self.name) < 24:
            self.name += chr(key)
        if rgb is None:
            c = np.full((CH, CW, 3), 25, np.uint8)
            text_c(c, "Waiting for the Kinect...", CH // 2, 1.0)
            return c
        self._probe()
        c = camera_canvas(rgb, dim=0.45)
        text_c(c, "3D BODY SCAN", 90, 1.8, (120, 200, 255), 2)
        panel(c, 120, 180, CW - 120, 330)
        text_c(c, "What's your name?", 235, 0.9, (220, 220, 220))
        cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
        text_c(c, self.name + cursor, 300, 1.5, (255, 255, 255), 2)
        if time.time() < self.force_until:
            panel(c, 120, 340, CW - 120, 410, 0.8)
            text_c(c, "Not ready - press ENTER again to scan anyway", 385, 0.85, (80, 180, 255))
        else:
            text_c(c, "Press ENTER to start", 372, 0.8, (180, 255, 180))
        col = (120, 230, 120) if self.probe["ok"] else (80, 180, 255)
        panel(c, 120, CH - 262, CW - 120, CH - 24)
        text_c(c, ("OK  " if self.probe["ok"] else "..  ") + self.probe["text"], CH - 224, 0.75, col)
        self._distance_meter(c, CH - 168)
        text_c(c, "Stand ~3 m back, arms slightly away from your body (A-pose)", CH - 106, 0.6, (210, 210, 210), 1, False)
        text_c(c, "After 3-2-1, turn slowly on the spot - TWO full turns (about 10 s each)", CH - 80, 0.6, (210, 210, 210), 1, False)
        text_c(c, f"Up/Down arrows tilt the sensor ({self.src.tilt_target} deg)   Esc quits", CH - 50, 0.5, (150, 150, 150), 1, False)
        return c

    def _tick_countdown(self, key):
        rgb, depth = self.src.frames()
        left = 3.0 - (time.time() - self.t_state)
        if key == 27:
            self.state = "NAME"
        # keep tracking where they stand, so the capture volume is centred on them
        if depth is not None and int(left * 4) % 2 == 0:
            self.seg.find_person(depth)
        c = camera_canvas(rgb, dim=0.35)
        if left <= 0:
            self._start_capture()
            return c
        n = int(np.ceil(left))
        frac = left - int(left) if left != int(left) else 1.0
        scale = 7 + 2.5 * frac
        (tw, th), _ = cv2.getTextSize(str(n), FONT, scale, 12)
        text(c, str(n), ((CW - tw) // 2, (CH + th) // 2), scale, (255, 255, 255), 12)
        text_c(c, f"Get ready, {self.name}!", 120, 1.2, (120, 220, 255), 2)
        text_c(c, "Then turn slowly on the spot - two full turns", CH - 60, 0.8)
        return c

    def _tick_capture(self, key):
        rgb = self._sample()
        turns = abs(self.cap.trk.yaw) / (2 * np.pi)
        elapsed = time.time() - self.t_state
        ncap = len(self.cap.caps)
        if key == 27:
            self._finish_raw(dict(result="cancelled"))
            self.state = "NAME"
            self.toast("Scan cancelled")
        elif (key == 32 and turns >= 1.0) or turns >= TURNS_NEEDED * 0.985 or ncap >= 360 or elapsed > 75:
            if ncap >= 40:
                self._finish_raw()
                self._start_process()
            else:
                self._finish_raw(dict(result="too few frames"))
                self.state = "NAME"
                self.toast("That didn't capture enough - please try again", 5)
        if rgb is None:
            return np.zeros((CH, CW, 3), np.uint8)
        c = camera_canvas(rgb, self.last_mask, dim=0.15)
        text_c(c, "SPIN SLOWLY!", 70, 1.4, (120, 200, 255), 2)
        # progress bar
        x0, x1, y = 160, CW - 160, CH - 70
        panel(c, x0 - 14, y - 52, x1 + 14, y + 24, 0.7)
        cv2.rectangle(c, (x0, y - 12), (x1, y + 6), (70, 70, 70), -1)
        cv2.rectangle(c, (x0, y - 12), (x0 + int((x1 - x0) * min(turns / TURNS_NEEDED, 1)), y + 6), (120, 230, 120), -1)
        for t in (0.5, 1.0, 1.5):
            xx = x0 + int((x1 - x0) * t / TURNS_NEEDED)
            cv2.line(c, (xx, y - 12), (xx, y + 6), (200, 200, 200), 1)
        text_c(c, f"Turn {min(turns, TURNS_NEEDED):.1f} of {TURNS_NEEDED:.0f}", y - 22, 0.8, shadow=False)
        if not self.visible:
            text_c(c, "I can't see you - stay in the frame!", 130, 0.9, (80, 80, 255))
        elif self.cap.trk.quality < 0.6:
            text_c(c, "Tracking slipping - turn a bit slower", 130, 0.9, (80, 180, 255))
        return c

    def _tick_process(self, key):
        rgb, _ = self.src.frames()
        label, frac = self.prog
        c = camera_canvas(rgb, dim=0.7) if rgb is not None else np.full((CH, CW, 3), 25, np.uint8)
        text_c(c, "Building your 3D model...", CH // 2 - 60, 1.3, (120, 200, 255), 2)
        text_c(c, f"{label}", CH // 2 - 10, 0.8)
        x0, x1, y = 200, CW - 200, CH // 2 + 40
        cv2.rectangle(c, (x0, y), (x1, y + 16), (70, 70, 70), -1)
        cv2.rectangle(c, (x0, y), (x0 + int((x1 - x0) * np.clip(frac, 0, 1)), y + 16), (120, 230, 120), -1)
        return c

    def _tick_view(self, key):
        fig = self.fig
        if 49 <= key < 49 + len(bs.POSES):
            self.pose_i = key - 49
        elif key == ord("a"):
            self.auto = not self.auto
        elif key == ord("0"):
            self.orbit.reset()
        elif key == ord("s"):
            self._export()
        elif key == ord("n"):
            self.name, self.state, self.fig = "", "NAME", None
            return self._tick_name(-1)
        t = time.time() - self.t_view
        target = fig.pose_params(bs.POSES[self.pose_i], t)
        for j in set(self.cur) | set(target):  # ease toward the target pose
            a = np.array(self.cur.get(j, (0, 0, 0)), float)
            b = np.array(target.get(j, (0, 0, 0)), float)
            self.cur[j] = tuple(a + (b - a) * 0.3)
        if self.auto:
            self.orbit.yaw += 0.02
        verts, normals = fig.posed(self.cur)
        c = bs.render_mesh(verts, normals, fig.faces, fig.rgb[:, ::-1].copy(), self.orbit, size=(CH, CW), top=fig.top)
        self._last_render = c
        text(c, self.name, (24, 44), 1.1, (255, 255, 255), 2)
        text(c, f"{fig.top:.2f} m tall", (24, 76), 0.6, (180, 180, 180), 1)
        for i, nme in enumerate(bs.POSES):
            on = i == self.pose_i
            text(c, f"{i + 1}  {nme}", (CW - 220, 50 + 38 * i), 0.8, (120, 230, 120) if on else (200, 200, 200), 2 if on else 1)
        text(c, "drag: orbit   wheel: zoom   a: auto-spin   0: reset", (24, CH - 60), 0.55, (160, 160, 160), 1, False)
        text(c, "s: export pose + screenshot   n: new scan   Esc: quit", (24, CH - 32), 0.55, (160, 160, 160), 1, False)
        return c

    def _export(self):
        fig = self.fig
        verts, _ = fig.posed(self.cur)
        tag = bs.POSES[self.pose_i].lower().replace(" ", "_")
        stamp = time.strftime("%H%M%S")
        bs.save_ply(self.folder / f"pose_{tag}_{stamp}.ply", verts, fig.faces, fig.rgb)
        bs.save_obj(self.folder / f"pose_{tag}_{stamp}.obj", verts, fig.faces, fig.rgb)
        cv2.imwrite(str(self.folder / f"pose_{tag}_{stamp}.png"), self._last_render)
        self.toast(f"Saved {tag} to {self.folder}")


def main():
    ap = argparse.ArgumentParser(description="Kinect 3D body-scan booth")
    ap.add_argument("--demo", action="store_true", help="use a synthetic spinning person instead of the Kinect")
    ap.add_argument("--view", metavar="DIR", help="open a saved scan folder")
    ap.add_argument("--out", default=str(Path(__file__).parent / "scans"), help="where scans are saved")
    args = ap.parse_args()

    if args.view:
        class _NoCam:  # viewer-only: no sensor needed
            seq, tilt_target = 0, 0
            def frames(self): return None, None
            def start(self): pass
            def stop(self): pass
        src = _NoCam()
    elif args.demo:
        from synthetic import SyntheticKinect
        src = SyntheticKinect(seconds_per_turn=9.0)
    else:
        from kinect_overlay import Kinect
        src = Kinect()
    src.start()
    booth = Booth(src, args.out, args.view)

    win = "3D Body Scan"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, CW, CH)
    cv2.setMouseCallback(win, booth.orbit.on_mouse)
    while True:
        k = cv2.waitKeyEx(1)
        if k == 27 and booth.state in ("NAME", "VIEW"):
            break
        if k in KEY_F11:
            booth.fullscreen = not booth.fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if booth.fullscreen else cv2.WINDOW_NORMAL)
            k = -1
        cv2.imshow(win, booth.tick(k))
    src.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    sys.stdout.flush()
    os._exit(0)  # skip Qt's atexit teardown, which segfaults with the fcitx5 input plugin


if __name__ == "__main__":
    main()
