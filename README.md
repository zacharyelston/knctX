# camgrid

Live grid view of every V4L2 camera attached to the machine, in Rust.

    cargo run --release            # open the grid
    cargo run --release -- -v      # also print per-camera fps/status every 2 s

Keys: `1`..`9` focus one camera, `0`/`G` back to the grid, `R` force a full rescan, `Esc`/`Q` quit.

## Behaviour

- **Auto-detect.** `/dev` is polled once a second. A camera that is plugged in
  appears as a new tile within about a second; one that is unplugged disappears.
  Other streams keep running untouched. Metadata-only `/dev/videoN` nodes are
  skipped and remembered so they are not re-probed.
- **Self-healing.** If a camera can't start streaming (no USB bandwidth, held by
  another program, transient I/O error) its tile stays in the grid showing the
  reason, and the capture thread retries every 2 s until it works or the device
  is removed.
- **Capture.** One thread per camera, mmap streaming at the size closest to
  640x480, preferring MJPG, then YUYV, UYVY, GREY. MJPEG is decoded with
  zune-jpeg (standard Huffman tables are injected if a camera omits them).
- **Display.** A minifb window; tiles are aspect-fit with device, name, format
  and live fps underneath.

## Kinect v2 (Xbox One)

The Kinect v2 has no kernel V4L2 driver; it is driven in user space by
libfreenect2. Build with the `kinect2` feature to get two extra tiles per
sensor: the 1920x1080 color camera and the 512x424 depth map (red = 0.5 m,
blue = 4.5 m, black = no reading).

    # Arch: the AUR libfreenect2 package no longer configures under CMake 4;
    # contrib/libfreenect2/PKGBUILD is a patched copy that does.
    (cd contrib/libfreenect2 && makepkg -si)
    cargo run --release --features kinect2

Requirements: a USB 3 port (the sensor streams ~300 Mbit/s), and udev access
to 045e:02c4 / 02d8 / 02d9 for your user (the package installs
65-kinect2.rules; replug the sensor after installing).

Depth processing uses OpenCL when an ICD is registered in /etc/OpenCL/vendors,
else OpenGL, else CPU; `CAMGRID_K2_PIPELINE=cpu|gl|cl` forces one. JPEG
decoding uses TurboJPEG: libfreenect2's VA-API decoder is disabled because it
silently produces no color frames on machines without a working VA driver
(NVIDIA). Set `CAMGRID_K2_VAAPI=1` to let it try VA-API anyway.

## USB bandwidth note

A high-speed (480 Mbit/s) USB bus has a fixed isochronous budget shared by every
device on that controller. The Xbox Kinect camera reserves two 1920-byte
isochronous endpoints permanently while plugged in, and many webcams need a
2048-byte endpoint to stream at all. Two such devices on one controller fail
with "No space left on device" / "Not enough bandwidth" in `dmesg`. The fix is
to move one of them to a port on a different controller; the tile status tells
you when this is happening.
