# kinect-depth-overlay

Live viewer for the Xbox 360 Kinect (model 1414/1473) on Linux. Shows the RGB scene with a
colorized depth map overlaid, plus a configurable grid of distance measurements.

Uses libfreenect's registered depth, so depth pixels are aligned to the color camera.

## Requirements

- `libfreenect` with Python 3 bindings (Arch: AUR `libfreenect`)
- `python-opencv`, `python-numpy`
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
