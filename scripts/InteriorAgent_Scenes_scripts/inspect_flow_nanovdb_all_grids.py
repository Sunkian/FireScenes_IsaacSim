"""
inspect_flow_nanovdb_all_grids.py

Isaac Sim 5.1 / Warp 1.8.2

The native Flow temperature buffers report grid_count=2.
Our previous projector only sampled grid_index=0.

This script inspects BOTH linked NanoVDB grids for Fire and DenseSmoke:
- grid index / count
- grid name / type
- index-to-world transform + translation
- voxel size
- allocated voxel/tile counts
- positive value statistics
- positive-value world-space bounds

Run INSIDE Isaac Sim Script Editor.

Output:
    /scratch/apagnoux/flow_nvdb_all_grids/report.txt
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import warp as wp


FIRE_PATH = Path(
    "/scratch/apagnoux/flow_native_readback/fire_temperature.nvdb"
)

SMOKE_PATH = Path(
    "/scratch/apagnoux/flow_native_readback_smoke/smoke_temperature.nvdb"
)

OUT = Path("/scratch/apagnoux/flow_nvdb_all_grids")
OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "report.txt"

DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"

LOG = []


def log(*parts):
    text = " ".join(str(x) for x in parts)
    print(text)
    LOG.append(text)
    REPORT.write_text("\n".join(LOG) + "\n")


@wp.kernel
def lookup_values_nx3(
    volume_id: wp.uint64,
    coords: wp.array2d(dtype=wp.int32),
    values: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    x = coords[tid, 0]
    y = coords[tid, 1]
    z = coords[tid, 2]

    values[tid] = wp.volume_lookup_f(
        volume_id,
        x,
        y,
        z,
    )


@wp.kernel
def coords_to_world_nx3(
    volume_id: wp.uint64,
    coords: wp.array2d(dtype=wp.int32),
    world: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x = coords[tid, 0]
    y = coords[tid, 1]
    z = coords[tid, 2]

    world[tid] = wp.volume_index_to_world(
        volume_id,
        wp.vec3(
            float(x),
            float(y),
            float(z),
        ),
    )


def load_first_grid(path: Path):
    raw_np = np.fromfile(path, dtype=np.uint8)
    raw_wp = wp.array(
        raw_np,
        dtype=wp.uint8,
        device=DEVICE,
    )

    volume = wp.Volume(
        raw_wp,
        copy=False,
    )

    return volume, raw_wp


def inspect_grid(label: str, grid_no: int, volume):
    info = volume.get_grid_info()

    log("")
    log(
        f"--- {label.upper()} GRID {grid_no} "
        f"(grid_index={info.grid_index}, grid_count={info.grid_count}) ---"
    )

    log("name:", info.name)
    log("type:", info.type_str)
    log("Warp dtype:", volume.dtype)
    log("size_in_bytes:", info.size_in_bytes)
    log("translation:", info.translation)
    log("transform_matrix:", info.transform_matrix)

    try:
        log("voxel_size:", volume.get_voxel_size())
    except Exception as exc:
        log("voxel_size ERROR:", repr(exc))

    try:
        log("tile_count:", int(volume.get_tile_count()))
    except Exception as exc:
        log("tile_count ERROR:", repr(exc))

    try:
        voxel_count = int(volume.get_voxel_count())
    except Exception as exc:
        log("voxel_count ERROR:", repr(exc))
        return

    log("allocated_voxel_count:", voxel_count)

    if voxel_count <= 0:
        return

    coords = volume.get_voxels()

    log("coords dtype:", coords.dtype)
    log("coords shape:", coords.shape)
    log("coords ndim:", coords.ndim)

    # We can only use the float lookup kernel on float grids.
    if volume.dtype != wp.float32:
        log(
            "Skipping value statistics: this grid is not float32."
        )
        return

    if coords.ndim != 2 or coords.shape[1] != 3:
        log(
            "Skipping values: expected N x 3 int32 coordinates."
        )
        return

    n = int(coords.shape[0])

    values = wp.empty(
        n,
        dtype=wp.float32,
        device=DEVICE,
    )

    world = wp.empty(
        n,
        dtype=wp.vec3,
        device=DEVICE,
    )

    wp.launch(
        lookup_values_nx3,
        dim=n,
        inputs=[
            volume.id,
            coords,
            values,
        ],
        device=DEVICE,
    )

    wp.launch(
        coords_to_world_nx3,
        dim=n,
        inputs=[
            volume.id,
            coords,
            world,
        ],
        device=DEVICE,
    )

    wp.synchronize()

    values_np = values.numpy().reshape(-1)
    world_np = world.numpy().reshape(-1, 3)

    finite = np.isfinite(values_np)

    if not np.any(finite):
        log("No finite values.")
        return

    v = values_np[finite]

    log(
        "value min/max/mean:",
        float(v.min()),
        float(v.max()),
        float(v.mean()),
    )

    positive = values_np > 0.0
    positive_count = int(
        np.count_nonzero(positive)
    )

    log(
        "positive voxels:",
        positive_count,
        "/",
        int(values_np.size),
    )

    if positive_count:
        pv = values_np[positive]

        log(
            "positive percentiles [1,10,25,50,75,90,95,99]:",
            np.percentile(
                pv,
                [1, 10, 25, 50, 75, 90, 95, 99],
            ).tolist(),
        )

        pw = world_np[positive]

        log(
            "positive world bounds min:",
            np.min(
                pw,
                axis=0,
            ).tolist(),
        )

        log(
            "positive world bounds max:",
            np.max(
                pw,
                axis=0,
            ).tolist(),
        )

        log(
            "positive world extent:",
            (
                np.max(
                    pw,
                    axis=0,
                )
                - np.min(
                    pw,
                    axis=0,
                )
            ).tolist(),
        )


def inspect_buffer(label: str, path: Path):
    log("")
    log("=" * 72)
    log(label.upper(), "BUFFER")
    log("=" * 72)

    log("path:", path)
    log("bytes:", path.stat().st_size)

    volume, backing = load_first_grid(path)

    first_info = volume.get_grid_info()
    grid_count = int(first_info.grid_count)

    log(
        "reported grid_count:",
        grid_count,
    )

    current = volume
    keep_alive = [
        backing,
        current,
    ]

    for grid_no in range(grid_count):
        inspect_grid(
            label,
            grid_no,
            current,
        )

        if grid_no >= grid_count - 1:
            break

        try:
            nxt = current.load_next_grid()
        except Exception as exc:
            log("")
            log(
                "load_next_grid() ERROR after grid",
                grid_no,
                ":",
                repr(exc),
            )
            break

        if nxt is None:
            log(
                "load_next_grid() returned None after grid",
                grid_no,
            )
            break

        current = nxt
        keep_alive.append(current)

    return keep_alive


def main():
    wp.init()

    log("FLOW NANOVDB ALL-GRID INSPECTION")
    log(
        "Warp version:",
        getattr(wp, "__version__", "<unknown>"),
    )
    log("device:", DEVICE)

    fire_refs = inspect_buffer(
        "fire",
        FIRE_PATH,
    )

    smoke_refs = inspect_buffer(
        "smoke",
        SMOKE_PATH,
    )

    _ = (
        fire_refs,
        smoke_refs,
    )

    log("")
    log("=" * 72)
    log("DONE")
    log("=" * 72)
    log(
        "Use this report to determine which linked grid "
        "contains the actual Flow field to project."
    )
    log("report:", REPORT)


try:
    main()

except Exception:
    log("")
    log("========== EXCEPTION ==========")

    for line in traceback.format_exc().splitlines():
        log(line)

    raise
