"""
inspect_fire_warm_channels_SAFE.py

Isaac Sim 5.1 / Warp 1.8.2

Read-only diagnostic for the warmed Fire capture.

Loads ONE NanoVDB at a time to minimize GPU/native memory pressure and reports:
- positive voxel count
- min/max/percentiles
- positive world bounds

Inputs:
    /scratch/apagnoux/flow_native_fire_warm_v2/fire_fuel.nvdb
    /scratch/apagnoux/flow_native_fire_warm_v2/fire_burn.nvdb
    /scratch/apagnoux/flow_native_fire_warm_v2/fire_temperature.nvdb

Output:
    /scratch/apagnoux/flow_native_fire_warm_v2/channel_stats.txt

No timeline changes. No OmniGraph changes. No projection.
"""

from pathlib import Path
import gc
import traceback

import numpy as np
import warp as wp


ROOT = Path("/scratch/apagnoux/flow_native_fire_warm_v2")
OUT = ROOT / "channel_stats.txt"

FILES = [
    ("fuel", ROOT / "fire_fuel.nvdb"),
    ("burn", ROOT / "fire_burn.nvdb"),
    ("temperature", ROOT / "fire_temperature.nvdb"),
]

DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"

LOG = []


def log(*parts):
    text = " ".join(str(x) for x in parts)
    print(text)
    LOG.append(text)
    OUT.write_text("\n".join(LOG) + "\n")


@wp.kernel
def lookup_values(
    volume_id: wp.uint64,
    coords: wp.array2d(dtype=wp.int32),
    values: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    values[tid] = wp.volume_lookup_f(
        volume_id,
        coords[tid, 0],
        coords[tid, 1],
        coords[tid, 2],
    )


def inspect_one(label, path):
    log("")
    log("=" * 72)
    log(label.upper())
    log("=" * 72)

    raw_np = np.fromfile(path, dtype=np.uint8)
    raw_wp = wp.array(raw_np, dtype=wp.uint8, device=DEVICE)
    volume = wp.Volume(raw_wp, copy=False)

    info = volume.get_grid_info()
    voxel = float(volume.get_voxel_size()[0])
    coords = volume.get_voxels()

    n = int(coords.shape[0])

    values = wp.empty(
        n,
        dtype=wp.float32,
        device=DEVICE,
    )

    wp.launch(
        lookup_values,
        dim=n,
        inputs=[volume.id, coords, values],
        device=DEVICE,
    )
    wp.synchronize()

    coords_np = coords.numpy().astype(np.int32)
    vals = values.numpy().reshape(-1).astype(np.float32)

    finite = np.isfinite(vals)
    vals = np.where(finite, vals, 0.0)

    positive = vals > 0.0
    pcount = int(np.count_nonzero(positive))

    log("file:", path)
    log("bytes:", path.stat().st_size)
    log("grid:", info.name, info.type_str)
    log("grid index/count:", info.grid_index, "/", info.grid_count)
    log("voxel size [m]:", voxel)
    log("allocated voxels:", n)
    log("value min/max/mean:", float(vals.min()), float(vals.max()), float(vals.mean()))
    log("positive voxels:", pcount)

    if pcount:
        pv = vals[positive]
        pc = coords_np[positive]

        log(
            "positive percentiles [1,10,25,50,75,90,95,99]:",
            np.percentile(
                pv,
                [1, 10, 25, 50, 75, 90, 95, 99],
            ).tolist(),
        )

        wmin = pc.min(axis=0).astype(np.float64) * voxel
        wmax = pc.max(axis=0).astype(np.float64) * voxel

        log("positive bounds [m]:", wmin.tolist(), "->", wmax.tolist())

    # Release references before loading the next grid.
    del values
    del coords
    del volume
    del raw_wp
    del raw_np
    gc.collect()
    wp.synchronize()


def main():
    wp.init()

    log("FIRE WARM CHANNEL INSPECTION — SAFE")
    log("device:", DEVICE)

    for label, path in FILES:
        if not path.exists():
            log("MISSING:", path)
            continue

        inspect_one(label, path)

    log("")
    log("DONE")
    log("report:", OUT)


try:
    main()
except Exception:
    log("")
    log("========== EXCEPTION ==========")
    for line in traceback.format_exc().splitlines():
        log(line)
    raise
