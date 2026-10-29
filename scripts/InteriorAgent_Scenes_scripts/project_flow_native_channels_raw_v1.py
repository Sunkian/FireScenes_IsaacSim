"""
project_flow_native_channels_raw_v1.py

Isaac Sim 5.1 / Warp 1.8.2

Purpose
-------
Validate the NEW native Flow channels before building the final Lepton model.

Inputs:
    fire_burn.nvdb
    fire_temperature.nvdb
    smoke_density.nvdb
    smoke_temperature.nvdb

Outputs raw Lepton-view projections:
    fire_burn_max.npy/.png
    fire_temperature_max.npy/.png
    smoke_density_max.npy/.png
    smoke_density_tau.npy/.png
    smoke_temperature_max.npy/.png
    report.txt

IMPORTANT
---------
This script does NOT threshold masks yet and does NOT apply the Lepton
radiometric/noise model yet. It is only to verify that:
    burn  -> flame-shaped projection
    smoke -> plume-shaped projection

Run INSIDE Isaac Sim Script Editor.
"""

from __future__ import annotations

import math
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import warp as wp

import omni.usd
from pxr import Usd, UsdGeom, Gf


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path("/scratch/apagnoux/flow_native_channels_v1")

FIRE_BURN_NVDB = ROOT / "fire_burn.nvdb"
FIRE_TEMP_NVDB = ROOT / "fire_temperature.nvdb"
SMOKE_DENSITY_NVDB = ROOT / "smoke_density.nvdb"
SMOKE_TEMP_NVDB = ROOT / "smoke_temperature.nvdb"

THERMAL_CAMERA = (
    "/Root/World/Meshes/SensorRig/"
    "ThermalCamera/FLIR_Lepton_3_1R"
)

OUT = Path("/scratch/apagnoux/flow_native_projection_raw_v1")
OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "report.txt"

DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"

WIDTH = 160
HEIGHT = 120

STEP_SCALE = 0.5
MAX_TRACE_DISTANCE_M = 12.0


# =============================================================================
# LOGGING
# =============================================================================

LOG = []


def log(*parts):
    text = " ".join(str(x) for x in parts)
    print(text)
    LOG.append(text)
    REPORT.write_text("\n".join(LOG) + "\n")


# =============================================================================
# NANOVDB READ
# =============================================================================

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


def load_field(label: str, path: Path):
    log("")
    log("=" * 72)
    log("LOAD", label.upper())
    log("=" * 72)

    if not path.exists():
        raise FileNotFoundError(path)

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

    info = volume.get_grid_info()

    if volume.dtype != wp.float32:
        raise RuntimeError(
            f"{label}: expected float32 NanoVDB, got {volume.dtype}"
        )

    voxel_size = float(
        volume.get_voxel_size()[0]
    )

    coords = volume.get_voxels()

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(
            f"{label}: expected N x 3 voxel coordinates, got {coords.shape}"
        )

    n = int(coords.shape[0])

    values = wp.empty(
        n,
        dtype=wp.float32,
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

    wp.synchronize()

    coords_np = coords.numpy().astype(np.int32)
    values_np = values.numpy().reshape(-1).astype(np.float32)

    finite = np.isfinite(values_np)
    values_np = np.where(finite, values_np, 0.0)

    positive = values_np > 0.0

    if not np.any(positive):
        log("WARNING:", label, "has no positive voxels.")

        min_idx = coords_np.min(axis=0)
        max_idx = coords_np.max(axis=0)

        pos_values = np.empty(0, dtype=np.float32)
        pos_coords = coords_np

    else:
        pos_values = values_np[positive]
        pos_coords = coords_np[positive]

        min_idx = pos_coords.min(axis=0)
        max_idx = pos_coords.max(axis=0)

    shape = (
        max_idx - min_idx + 1
    ).astype(np.int32)

    dense = np.zeros(
        tuple(shape.tolist()),
        dtype=np.float32,
    )

    if np.any(positive):
        local = pos_coords - min_idx
        dense[
            local[:, 0],
            local[:, 1],
            local[:, 2],
        ] = pos_values

    world_min = min_idx.astype(np.float64) * voxel_size
    world_max = max_idx.astype(np.float64) * voxel_size

    log("File:", path)
    log("Grid name:", info.name)
    log("Grid type:", info.type_str)
    log("Grid index/count:", info.grid_index, "/", info.grid_count)
    log("Voxel size [m]:", voxel_size)
    log("Allocated voxels:", n)

    if pos_values.size:
        log("Positive voxels:", int(pos_values.size))
        log(
            "Positive min/max:",
            float(pos_values.min()),
            float(pos_values.max()),
        )
        log(
            "Positive percentiles [1,10,25,50,75,90,95,99]:",
            np.percentile(
                pos_values,
                [1, 10, 25, 50, 75, 90, 95, 99],
            ).tolist(),
        )
    else:
        log("Positive voxels: 0")

    log("Positive/native bounds [m]:", world_min.tolist(), "->", world_max.tolist())

    return {
        "label": label,
        "volume": volume,
        "backing": raw_wp,
        "voxel_size": voxel_size,
        "dense": dense,
        "min_idx": min_idx,
        "max_idx": max_idx,
        "world_min": world_min,
        "world_max": world_max,
        "center_world": 0.5 * (world_min + world_max),
    }


# =============================================================================
# CAMERA
# =============================================================================

def get_camera(stage):
    prim = stage.GetPrimAtPath(THERMAL_CAMERA)

    if not prim.IsValid():
        raise RuntimeError(
            f"Thermal camera not found: {THERMAL_CAMERA}"
        )

    cam = UsdGeom.Camera(prim)

    camera_to_world = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    tr = camera_to_world.ExtractTranslation()

    origin = np.array(
        [tr[0], tr[1], tr[2]],
        dtype=np.float64,
    )

    focal_mm = float(cam.GetFocalLengthAttr().Get())
    h_ap_mm = float(cam.GetHorizontalApertureAttr().Get())
    v_ap_mm = float(cam.GetVerticalApertureAttr().Get())

    fx = focal_mm / h_ap_mm * WIDTH
    fy = focal_mm / v_ap_mm * HEIGHT

    cx = (WIDTH - 1.0) * 0.5
    cy = (HEIGHT - 1.0) * 0.5

    fwd = camera_to_world.TransformDir(
        Gf.Vec3d(0.0, 0.0, -1.0)
    )

    forward = np.array(
        [fwd[0], fwd[1], fwd[2]],
        dtype=np.float64,
    )

    forward /= np.linalg.norm(forward)

    hfov = 2.0 * math.degrees(
        math.atan(
            h_ap_mm / (2.0 * focal_mm)
        )
    )

    vfov = 2.0 * math.degrees(
        math.atan(
            v_ap_mm / (2.0 * focal_mm)
        )
    )

    log("")
    log("=" * 72)
    log("CAMERA")
    log("=" * 72)
    log("Origin [m]:", origin.tolist())
    log("Forward:", forward.tolist())
    log("fx/fy:", fx, fy)
    log("HFOV/VFOV [deg]:", hfov, vfov)

    return {
        "origin": origin,
        "camera_to_world": camera_to_world,
        "forward": forward,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def camera_ray_world(cam, px, py):
    x = (px - cam["cx"]) / cam["fx"]
    y = -(py - cam["cy"]) / cam["fy"]

    local = Gf.Vec3d(
        float(x),
        float(y),
        -1.0,
    )

    local.Normalize()

    d = cam["camera_to_world"].TransformDir(local)

    out = np.array(
        [d[0], d[1], d[2]],
        dtype=np.float64,
    )

    out /= np.linalg.norm(out)

    return out


# =============================================================================
# RAY MARCH
# =============================================================================

def ray_box_intersection(
    origin,
    direction,
    bmin,
    bmax,
):
    t_near = 0.0
    t_far = MAX_TRACE_DISTANCE_M

    for axis in range(3):
        d = direction[axis]

        if abs(d) < 1e-12:
            if (
                origin[axis] < bmin[axis]
                or origin[axis] > bmax[axis]
            ):
                return None
            continue

        inv_d = 1.0 / d

        t0 = (
            bmin[axis] - origin[axis]
        ) * inv_d

        t1 = (
            bmax[axis] - origin[axis]
        ) * inv_d

        if t0 > t1:
            t0, t1 = t1, t0

        t_near = max(t_near, t0)
        t_far = min(t_far, t1)

        if t_far < t_near:
            return None

    return (
        max(t_near, 0.0),
        t_far,
    )


def sample_nearest(field, p_world):
    idx = np.rint(
        p_world / field["voxel_size"]
    ).astype(np.int32)

    loc = idx - field["min_idx"]

    dense = field["dense"]

    if (
        loc[0] < 0
        or loc[0] >= dense.shape[0]
        or loc[1] < 0
        or loc[1] >= dense.shape[1]
        or loc[2] < 0
        or loc[2] >= dense.shape[2]
    ):
        return 0.0

    return float(
        dense[
            loc[0],
            loc[1],
            loc[2],
        ]
    )


def march_field(
    field,
    origin,
    direction,
    integrate=False,
):
    pad = 0.5 * field["voxel_size"]

    hit = ray_box_intersection(
        origin,
        direction,
        field["world_min"] - pad,
        field["world_max"] + pad,
    )

    if hit is None:
        return 0.0, 0.0

    t0, t1 = hit

    dt = field["voxel_size"] * STEP_SCALE

    max_value = 0.0
    integral = 0.0

    t = t0

    while t <= t1:
        p = origin + t * direction

        v = sample_nearest(
            field,
            p,
        )

        if v > max_value:
            max_value = v

        if integrate and v > 0.0:
            integral += v * dt

        t += dt

    return max_value, integral


def project_all(
    cam,
    fire_burn,
    fire_temp,
    smoke_density,
    smoke_temp,
):
    fire_burn_max = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32,
    )

    fire_temp_max = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32,
    )

    smoke_density_max = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32,
    )

    smoke_density_tau = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32,
    )

    smoke_temp_max = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32,
    )

    origin = cam["origin"]

    for py in range(HEIGHT):
        if py % 20 == 0:
            log("Projecting row", py, "/", HEIGHT)

        for px in range(WIDTH):
            d = camera_ray_world(
                cam,
                px,
                py,
            )

            burn_max, _ = march_field(
                fire_burn,
                origin,
                d,
                integrate=False,
            )

            ft_max, _ = march_field(
                fire_temp,
                origin,
                d,
                integrate=False,
            )

            sd_max, sd_tau = march_field(
                smoke_density,
                origin,
                d,
                integrate=True,
            )

            st_max, _ = march_field(
                smoke_temp,
                origin,
                d,
                integrate=False,
            )

            fire_burn_max[py, px] = burn_max
            fire_temp_max[py, px] = ft_max
            smoke_density_max[py, px] = sd_max
            smoke_density_tau[py, px] = sd_tau
            smoke_temp_max[py, px] = st_max

    return (
        fire_burn_max,
        fire_temp_max,
        smoke_density_max,
        smoke_density_tau,
        smoke_temp_max,
    )


# =============================================================================
# SAVE
# =============================================================================

def save_norm_png(path, arr):
    arr = arr.astype(np.float32)

    vmax = float(np.max(arr))

    if vmax <= 0.0:
        img = np.zeros_like(
            arr,
            dtype=np.uint8,
        )
    else:
        img = np.round(
            np.clip(
                arr / vmax,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8)

    Image.fromarray(
        img,
        mode="L",
    ).save(path)


def report_projection(name, arr):
    log("")
    log(name)
    log("  min/max:", float(arr.min()), float(arr.max()))
    log("  nonzero pixels:", int(np.count_nonzero(arr)))

    positive = arr[arr > 0.0]

    if positive.size:
        log(
            "  positive percentiles [1,10,25,50,75,90,95,99]:",
            np.percentile(
                positive,
                [1, 10, 25, 50, 75, 90, 95, 99],
            ).tolist(),
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    wp.init()

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError(
            "No USD stage is open."
        )

    log("FLOW NATIVE CHANNEL RAW PROJECTION V1")

    fire_burn = load_field(
        "fire burn",
        FIRE_BURN_NVDB,
    )

    fire_temp = load_field(
        "fire temperature",
        FIRE_TEMP_NVDB,
    )

    smoke_density = load_field(
        "smoke density",
        SMOKE_DENSITY_NVDB,
    )

    smoke_temp = load_field(
        "smoke temperature",
        SMOKE_TEMP_NVDB,
    )

    cam = get_camera(stage)

    log("")
    log("=" * 72)
    log("GEOMETRY CHECK")
    log("=" * 72)

    for field in (
        fire_burn,
        fire_temp,
        smoke_density,
        smoke_temp,
    ):
        to_center = (
            field["center_world"]
            - cam["origin"]
        )

        dist = float(
            np.linalg.norm(to_center)
        )

        if dist > 0.0:
            unit = to_center / dist

            dot = float(
                np.dot(
                    cam["forward"],
                    unit,
                )
            )

            angle = math.degrees(
                math.acos(
                    np.clip(
                        dot,
                        -1.0,
                        1.0,
                    )
                )
            )
        else:
            angle = 0.0

        log(
            field["label"],
            "| center:",
            field["center_world"].tolist(),
            "| distance [m]:",
            dist,
            "| angle [deg]:",
            angle,
        )

    log("")
    log("=" * 72)
    log("PROJECT")
    log("=" * 72)

    (
        fire_burn_max,
        fire_temp_max,
        smoke_density_max,
        smoke_density_tau,
        smoke_temp_max,
    ) = project_all(
        cam,
        fire_burn,
        fire_temp,
        smoke_density,
        smoke_temp,
    )

    arrays = {
        "fire_burn_max": fire_burn_max,
        "fire_temperature_max": fire_temp_max,
        "smoke_density_max": smoke_density_max,
        "smoke_density_tau": smoke_density_tau,
        "smoke_temperature_max": smoke_temp_max,
    }

    for name, arr in arrays.items():
        np.save(
            OUT / f"{name}.npy",
            arr,
        )

        save_norm_png(
            OUT / f"{name}.png",
            arr,
        )

    log("")
    log("=" * 72)
    log("PROJECTION SUMMARY")
    log("=" * 72)

    for name, arr in arrays.items():
        report_projection(name, arr)

    log("")
    log("Saved to:", OUT)


try:
    main()

except Exception:
    log("")
    log("========== EXCEPTION ==========")

    for line in traceback.format_exc().splitlines():
        log(line)

    raise
