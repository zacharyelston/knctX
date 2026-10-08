# kinect-depth-overlay

Live viewer for the Xbox 360 Kinect (model 1414/1473) on Linux. Shows the RGB scene with a
colorized depth map overlaid, plus a configurable grid of distance measurements.

Uses libfreenect's registered depth, so depth pixels are aligned to the color camera.

## Requirements

- `libfreenect` with Python 3 bindings (Arch: AUR `libfreenect`)
- `python-opencv`, `python-numpy`, `python-scipy`, `python-scikit-image` (3D scanning)
- Optional: `blender` to open/rig/render the scans
- Kinect udev rules (installed by the AUR package) so no root is needed

## Run

```
python3 kinect_overlay.py
```

## Controls

| Key | Action |
|---|---|
| `v` | Cycle view: overlay / side-by-side / depth / rgb |
| `g` | Cycle measurement grid: off, 3x3, 5x4, 8x6, 12x9 |
| `c` | Cycle colormap |
| `e` | Edge-only depth overlay |
| `m` | Mirror |
| `+` / `-` | Overlay opacity |
| `[` / `]` | Near / far depth limits (mm) |
| `Up` / `Down` | Tilt motor |
| `s` | Save snapshot to `snapshots/` |
| `q` / `Esc` | Quit |

## Notes

The Kinect's streams sometimes fail to start on USB 2 hubs (`Invalid magic` /
`Lost too many packets` messages). The app detects a stalled stream and restarts the device.

## 3D body-scan booth

```
python3 scan_booth.py            # real Kinect
python3 scan_booth.py --demo     # synthetic spinning person, no sensor needed
python3 scan_booth.py --view scans/<folder>   # reopen a saved scan
```

A visitor types their name, gets a 3-2-1 countdown, then turns slowly on the spot **twice**.
The spin is tracked live (yaw ICP on the segmented person), drift is closed using the second
turn, depth is fused into a TSDF volume, meshed with marching cubes, auto-rigged with a
16-joint skeleton and shown in a viewer with poses (Rest, T-pose, Hands up, Wave, Walk, Dance).

Tips for good scans: stand about 3 m from the sensor with the whole body (feet to fingertips)
in view, arms slightly away from the body (A-pose), turn slowly (about 10 s per turn), wear
non-reflective clothing, and keep other people out of the area behind you.

Each scan is saved in `scans/<name>_<time>/` (with every raw frame in `raw/`, so
`python3 rescan.py scans/<name>_<time>` can rebuild and debug it without spinning again): `model.ply` / `model.obj` (vertex colours,
floor at y=0, y up; opens directly in Blender), `meta.json` (height, joint positions) and
`preview.png`. In the viewer, `s` exports the current pose as PLY + OBJ + PNG.

Files: `bodyscan.py` (pipeline), `scan_booth.py` (UI), `synthetic.py` (fake sensor for tests).
