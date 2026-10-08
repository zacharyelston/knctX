#!/usr/bin/env python3
"""Rebuild a 3D model from the raw frames saved with a scan (scans/<name>/raw).

  python3 rescan.py scans/<folder> [--out DIR]

Re-runs segmentation, spin tracking, loop closure, fusion and meshing on the saved frames, so
the pipeline can be tuned without anyone having to spin again. Writes model.ply/obj plus
preview.png (the figure from 4 angles) and debug_masks.png (what was segmented as "the person").
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import bodyscan as bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", help="scan folder containing raw/")
    ap.add_argument("--out", help="output folder (default: <scan>/rescan_<time>)")
    args = ap.parse_args()

    out, stats, cap = bs.rescan(args.scan, args.out)
    verts, faces, rgb = bs.load_ply(out / "model.ply")
    fig = bs.Figure(verts, faces, rgb)
    (out / "meta.json").write_text(json.dumps(dict(stats=stats, **fig.meta()), indent=2))
    print(f"model: {len(verts)} verts, {fig.top:.2f} m tall -> {out}")

    o = bs.Orbit3D()
    tiles = []
    for yaw in (0.0, 1.57, 3.14, 4.71):
        o.yaw, o.pitch = yaw, 0.05
        tiles.append(bs.render_mesh(verts, fig.rest_normals, faces, rgb[:, ::-1].copy(), o, size=(480, 360), top=fig.top))
    cv2.imwrite(str(out / "preview.png"), np.hstack(tiles))

    # what the segmenter kept, across the capture
    raw = Path(args.scan) / "raw"
    n = json.loads((raw / "capture.json").read_text())["frames"]
    tiles = []
    seg = cap.seg
    for i in np.linspace(0, n - 1, 8).astype(int):
        im = cv2.cvtColor(cv2.imread(str(raw / f"rgb_{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
        d = cv2.imread(str(raw / f"depth_{i:05d}.png"), cv2.IMREAD_UNCHANGED)
        m, _ = seg.mask(d)
        vis = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        vis[~m] = (vis[~m] * 0.25).astype(np.uint8)
        tiles.append(cv2.resize(vis, (320, 240)))
    cv2.imwrite(str(out / "debug_masks.png"), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    yaws = np.degrees([c["pose"][0] for c in cap.caps])
    print("tracked yaw (deg) every ~10%:", np.round(yaws[:: max(1, len(yaws) // 10)]).astype(int).tolist())


if __name__ == "__main__":
    main()
