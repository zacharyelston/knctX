# kinect-depth-overlay

Live viewer for the Xbox 360 Kinect (model 1414/1473) on Linux. Shows the RGB scene with a
colorized depth map overlaid, plus a configurable grid of distance measurements.

Uses libfreenect's registered depth, so depth pixels are aligned to the color camera.

## KnctXplore (the app)

One menu for everything: live viewer, body booth, bust booth, scan library (view / rebuild any saved
scan), recordings playback and the self-test. Each tool opens in its own window; close it to return.

```bash
cd rs && ./install.sh   # builds, adds "KnctXplore" to the app menu and a `knctxplore` command
knctxplore              # or launch it from the menu
```

Keys: arrows / 1-6 / click to choose, Enter to open, `D` toggles demo mode (booths use a synthetic
person instead of the Kinect), Esc goes back. In the Scan Library, `R` rebuilds the selected scan from
its raw frames. `knctxplore --shot home out.png` renders a screen headlessly.

## roommap (3D map of the room from both Kinects)

`rs/src/bin/roommap.rs` fuses the Kinect v1 (libfreenect, registered depth) and the Kinect v2
(libfreenect2, registered XYZ+RGB) into one coloured voxel map (3 cm) and renders it with an
orbit camera. The v2's pose relative to the v1 is found automatically with ICP on first start
(both sensors must see some of the same scene) and saved to `rs/roommap_pose.json`; it can also
be nudged by hand and refined again with `I`. libfreenect2 delivers mirror images, so the v2
cloud is flipped to match the v1 (`Y` toggles this). `J` (or `--tint`) colours the clouds per
sensor, blue v1 / orange v2, to judge the alignment. ICP matches brightness as well as
position in its fine rounds so flat walls cannot slide. `E` exports the map as PLY to `pointclouds/`.

```bash
cd rs && cargo run --release --bin roommap     # press H for the key list
```

Both sensors are user-space USB drivers and together need more than the kernel's default
16 MB usbfs transfer budget (`libusb` then fails with LIBUSB_ERROR_IO). This machine has
`/etc/tmpfiles.d/usbfs-memory.conf` raising `usbfs_memory_mb` to 128. camgrid and roommap
cannot run at the same time: each needs exclusive access to the sensors.

## camgrid (all cameras at once)

`camgrid/` is a separate Rust app that shows every camera on the machine in one window:
V4L2 webcams, the Kinect v1's RGB camera, and (with `--features kinect2`) the Kinect v2's
color and depth streams through libfreenect2. Cameras are detected as they are plugged in.

```bash
cd camgrid && cargo run --release --features kinect2
```

See `camgrid/README.md` for keys, the Kinect v2 setup (patched libfreenect2 PKGBUILD in
`camgrid/contrib/`), and the USB bandwidth notes.

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

## Rust version (recommended)

The whole thing is also ported to Rust in `rs/` - same behaviour, a fraction of the CPU
(the Python booth idles at ~9 cores on the name screen, the Rust one at a quarter of one core,
and a rescan that took 40 s takes 7 s).

```
cd rs
cargo build --release
./target/release/scan_booth               # real Kinect (or --demo, --view DIR, --autorun NAME)
./target/release/bust_booth              # seated close-up bust booth (--demo, --view DIR)
./target/release/rescan ../scans/<folder> # rebuild a saved scan (--refloor, --out DIR)
./target/release/kinect_overlay           # RGB + depth viewer (or --play ../recordings/<ts>)
./target/release/selftest                 # headless end-to-end check on the synthetic person
```

The **bust booth** is for detail: the subject sits on a swivel chair about 1.1 m from the sensor and
turns themselves once round (plus a little overlap). Head and shoulders are fused at 6 mm voxels
(the full-body booth uses 15 mm). No floor is needed - the Kinect's accelerometer supplies gravity,
so the sensor may be tilted (it auto-tilts to keep the head in frame). Rescans work the same way
(`rescan` reads the profile saved in `raw/capture.json`).

Needs only `libfreenect` (linked directly via FFI) - no OpenCV or Python. Notes: F11 fullscreen
is not available (resize the window instead); the mesher is surface-nets rather than marching
cubes, so vertex counts differ slightly from the Python models.

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
