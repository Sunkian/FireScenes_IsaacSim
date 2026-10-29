"""
project_flow_to_lepton_v3_native_world.py

Isaac Sim 5.1 / Warp 1.8.2

Corrects V2 by treating the native Flow NanoVDB transform as authoritative.

Why:
The FlowSimulate prim inherits a 0.01 USD scale from the imported scene:
    local-to-world scale = 0.01

But the NanoVDB itself already reports:
    voxel size = 0.05 m

Applying the FlowSimulate USD transform again shrinks a ~2 m Flow volume to
~2 cm, which is wrong. Therefore this version:

- keeps the corrected USD camera-ray transform from V2;
- DOES NOT apply the FlowSimulate prim transform to NanoVDB coordinates;
- ray-marches directly in NanoVDB/world coordinates.

Run INSIDE Isaac Sim Script Editor.

Outputs:
    /scratch/apagnoux/flow_lepton_projection_v3/
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

FIRE_NVDB = Path(
    "/scratch/apagnoux/flow_native_readback/fire_temperature.nvdb"
)

SMOKE_NVDB = Path(
    "/scratch/apagnoux/flow_native_readback_smoke/smoke_temperature.nvdb"
)

THERMAL_CAMERA = (
    "/Root/World/Meshes/SensorRig/"
    "ThermalCamera/FLIR_Lepton_3_1R"
)

OUT = Path("/scratch/apagnoux/flow_lepton_projection_v3")
OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "report.txt"

DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"

WIDTH = 160
HEIGHT = 120

STEP_SCALE = 0.5
MAX_TRACE_DISTANCE_M = 12.0

# Initial thresholds from measured Flow distributions
FIRE_MASK_THRESHOLD = 0.30

SMOKE_SAMPLE_THRESHOLD = 0.02

# Lower than V2 because the integrated tau depends on physical path length.
# We will tune it from the output distribution if needed.
SMOKE_TAU_MASK_THRESHOLD = 0.01

AMBIENT_C = 22.0

FIRE_TEMP_MIN_C = 450.0
FIRE_TEMP_MAX_C = 850.0

SMOKE_TEMP_MIN_C = 35.0
SMOKE_TEMP_MAX_C = 110.0

SMOKE_ALPHA_SCALE = 1.2

DISPLAY_MIN_C = 20.0
DISPLAY_MAX_C = 80.0

BASE_SURFACE_CANDIDATES = [
    Path("/scratch/apagnoux/lepton_thermal_v2_1/surface_temperature_c.npy"),
    Path("/scratch/apagnoux/lepton_thermal_v2/surface_temperature_c.npy"),
]


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
# NANOVDB
# =============================================================================

@wp.kernel
def read_values_nx3(
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

    raw_np = np.fromfile(
        path,
        dtype=np.uint8,
    )

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

    voxel_size = float(
        volume.get_voxel_size()[0]
    )

    coords = volume.get_voxels()

    n = int(coords.shape[0])

    values = wp.empty(
        n,
        dtype=wp.float32,
        device=DEVICE,
    )

    wp.launch(
        read_values_nx3,
        dim=n,
        inputs=[
            volume.id,
            coords,
            values,
        ],
        device=DEVICE,
    )

    wp.synchronize()

    coords_np = coords.numpy().astype(
        np.int32
    )

    values_np = values.numpy().reshape(
        -1
    ).astype(
        np.float32
    )

    positive = values_np > 0.0

    if not np.any(positive):
        raise RuntimeError(
            f"{label}: no positive voxels."
        )

    pos_coords = coords_np[positive]
    pos_values = values_np[positive]

    min_idx = pos_coords.min(axis=0)
    max_idx = pos_coords.max(axis=0)

    shape = (
        max_idx - min_idx + 1
    ).astype(
        np.int32
    )

    dense = np.zeros(
        tuple(shape.tolist()),
        dtype=np.float32,
    )

    local = (
        pos_coords - min_idx
    )

    dense[
        local[:, 0],
        local[:, 1],
        local[:, 2],
    ] = pos_values

    # IMPORTANT:
    # Native NanoVDB transform is authoritative.
    # The grid transform already says 0.05 m per voxel with zero translation.
    world_min = (
        min_idx.astype(np.float64)
        * voxel_size
    )

    world_max = (
        max_idx.astype(np.float64)
        * voxel_size
    )

    log("NanoVDB:", path)
    log("Grid name:", info.name)
    log("Grid transform:", info.transform_matrix)
    log("Voxel size [m]:", voxel_size)
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
    log("Index min/max:", min_idx.tolist(), max_idx.tolist())
    log("Native world bounds [m]:", world_min.tolist(), "->", world_max.tolist())

    return {
        "volume": volume,
        "backing": raw_wp,
        "voxel_size": voxel_size,
        "dense": dense,
        "min_idx": min_idx,
        "max_idx": max_idx,
        "world_min": world_min,
        "world_max": world_max,
        "center_world": 0.5 * (world_min + world_max),
        "value_max": float(pos_values.max()),
    }


# =============================================================================
# CAMERA
# =============================================================================

def get_camera(stage):
    prim = stage.GetPrimAtPath(
        THERMAL_CAMERA
    )

    if not prim.IsValid():
        raise RuntimeError(
            f"Thermal camera not found: {THERMAL_CAMERA}"
        )

    cam = UsdGeom.Camera(
        prim
    )

    camera_to_world = (
        UsdGeom.Xformable(
            prim
        ).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
    )

    tr = camera_to_world.ExtractTranslation()

    origin = np.array(
        [tr[0], tr[1], tr[2]],
        dtype=np.float64,
    )

    focal_mm = float(
        cam.GetFocalLengthAttr().Get()
    )

    h_ap_mm = float(
        cam.GetHorizontalApertureAttr().Get()
    )

    v_ap_mm = float(
        cam.GetVerticalApertureAttr().Get()
    )

    fx = focal_mm / h_ap_mm * WIDTH
    fy = focal_mm / v_ap_mm * HEIGHT

    cx = (WIDTH - 1.0) * 0.5
    cy = (HEIGHT - 1.0) * 0.5

    forward_gf = camera_to_world.TransformDir(
        Gf.Vec3d(
            0.0,
            0.0,
            -1.0,
        )
    )

    forward = np.array(
        [
            forward_gf[0],
            forward_gf[1],
            forward_gf[2],
        ],
        dtype=np.float64,
    )

    forward /= np.linalg.norm(
        forward
    )

    hfov = 2.0 * math.degrees(
        math.atan(
            h_ap_mm
            / (2.0 * focal_mm)
        )
    )

    vfov = 2.0 * math.degrees(
        math.atan(
            v_ap_mm
            / (2.0 * focal_mm)
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


def camera_ray_world(
    cam,
    px,
    py,
):
    x = (
        px - cam["cx"]
    ) / cam["fx"]

    y = -(
        py - cam["cy"]
    ) / cam["fy"]

    local = Gf.Vec3d(
        float(x),
        float(y),
        -1.0,
    )

    local.Normalize()

    d = cam[
        "camera_to_world"
    ].TransformDir(
        local
    )

    out = np.array(
        [d[0], d[1], d[2]],
        dtype=np.float64,
    )

    out /= np.linalg.norm(
        out
    )

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

        if abs(
            direction[axis]
        ) < 1e-12:

            if (
                origin[axis] < bmin[axis]
                or origin[axis] > bmax[axis]
            ):
                return None

            continue

        inv_d = 1.0 / direction[
            axis
        ]

        t0 = (
            bmin[axis]
            - origin[axis]
        ) * inv_d

        t1 = (
            bmax[axis]
            - origin[axis]
        ) * inv_d

        if t0 > t1:
            t0, t1 = t1, t0

        t_near = max(
            t_near,
            t0,
        )

        t_far = min(
            t_far,
            t1,
        )

        if t_far < t_near:
            return None

    return (
        max(t_near, 0.0),
        t_far,
    )


def sample_dense_nearest(
    field,
    point_world,
):
    idx = np.rint(
        point_world
        / field["voxel_size"]
    ).astype(
        np.int32
    )

    loc = (
        idx - field["min_idx"]
    )

    dense = field[
        "dense"
    ]

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
    pad = (
        field["voxel_size"]
        * 0.5
    )

    bmin = (
        field["world_min"]
        - pad
    )

    bmax = (
        field["world_max"]
        + pad
    )

    hit = ray_box_intersection(
        origin,
        direction,
        bmin,
        bmax,
    )

    if hit is None:
        return 0.0, 0.0

    t0, t1 = hit

    dt = (
        field["voxel_size"]
        * STEP_SCALE
    )

    max_value = 0.0
    integral = 0.0

    t = t0

    while t <= t1:

        p = (
            origin
            + t * direction
        )

        value = sample_dense_nearest(
            field,
            p,
        )

        if value > max_value:
            max_value = value

        if (
            integrate
            and value > SMOKE_SAMPLE_THRESHOLD
        ):
            integral += value * dt

        t += dt

    return (
        max_value,
        integral,
    )


def project(
    cam,
    fire,
    smoke,
):
    fire_max = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32,
    )

    smoke_max = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32,
    )

    smoke_tau = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32,
    )

    origin = cam[
        "origin"
    ]

    for py in range(
        HEIGHT
    ):

        if py % 20 == 0:
            log(
                "Projecting row",
                py,
                "/",
                HEIGHT,
            )

        for px in range(
            WIDTH
        ):

            d = camera_ray_world(
                cam,
                px,
                py,
            )

            fv, _ = march_field(
                fire,
                origin,
                d,
                integrate=False,
            )

            sv, tau = march_field(
                smoke,
                origin,
                d,
                integrate=True,
            )

            fire_max[
                py,
                px,
            ] = fv

            smoke_max[
                py,
                px,
            ] = sv

            smoke_tau[
                py,
                px,
            ] = tau

    return (
        fire_max,
        smoke_max,
        smoke_tau,
    )


# =============================================================================
# THERMAL
# =============================================================================

def load_base_surface():
    for path in BASE_SURFACE_CANDIDATES:

        if not path.exists():
            continue

        arr = np.load(
            path
        ).astype(
            np.float32
        )

        if arr.shape == (
            HEIGHT,
            WIDTH,
        ):
            log(
                "Using base surface:",
                path,
            )
            return arr

    log(
        "No matching base surface found; using ambient."
    )

    return np.full(
        (HEIGHT, WIDTH),
        AMBIENT_C,
        dtype=np.float32,
    )


def build_outputs(
    fire,
    smoke,
    fire_max,
    smoke_max,
    smoke_tau,
    base_surface,
):
    flame_mask = (
        fire_max
        >= FIRE_MASK_THRESHOLD
    )

    smoke_mask = (
        smoke_tau
        >= SMOKE_TAU_MASK_THRESHOLD
    )

    fire_norm = np.clip(
        fire_max
        / max(
            fire["value_max"],
            1e-8,
        ),
        0.0,
        1.0,
    )

    smoke_norm = np.clip(
        smoke_max
        / max(
            smoke["value_max"],
            1e-8,
        ),
        0.0,
        1.0,
    )

    fire_temp = (
        FIRE_TEMP_MIN_C
        + fire_norm
        * (
            FIRE_TEMP_MAX_C
            - FIRE_TEMP_MIN_C
        )
    ).astype(
        np.float32
    )

    smoke_temp = (
        SMOKE_TEMP_MIN_C
        + smoke_norm
        * (
            SMOKE_TEMP_MAX_C
            - SMOKE_TEMP_MIN_C
        )
    ).astype(
        np.float32
    )

    smoke_alpha = (
        1.0
        - np.exp(
            -SMOKE_ALPHA_SCALE
            * smoke_tau
        )
    ).astype(
        np.float32
    )

    smoke_alpha = np.clip(
        smoke_alpha,
        0.0,
        1.0,
    )

    composite = base_surface.copy()

    composite = (
        (1.0 - smoke_alpha)
        * composite
        + smoke_alpha
        * smoke_temp
    )

    composite[
        flame_mask
    ] = np.maximum(
        composite[
            flame_mask
        ],
        fire_temp[
            flame_mask
        ],
    )

    return (
        flame_mask.astype(
            np.uint8
        ),
        smoke_mask.astype(
            np.uint8
        ),
        fire_temp,
        smoke_temp,
        smoke_alpha,
        composite,
    )


def save_gray(
    path,
    arr,
    vmin,
    vmax,
):
    x = (
        arr.astype(
            np.float32
        )
        - vmin
    ) / max(
        vmax - vmin,
        1e-8,
    )

    img = np.round(
        np.clip(
            x,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    Image.fromarray(
        img,
        mode="L",
    ).save(
        path
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

    log(
        "FLOW -> LEPTON PROJECTION V3 (NATIVE NANOVDB WORLD FRAME)"
    )

    fire = load_field(
        "fire",
        FIRE_NVDB,
    )

    smoke = load_field(
        "smoke",
        SMOKE_NVDB,
    )

    cam = get_camera(
        stage
    )

    # ---------------------------------------------------------------------
    # Geometry sanity
    # ---------------------------------------------------------------------

    log("")
    log("=" * 72)
    log("GEOMETRY SANITY CHECK")
    log("=" * 72)

    for label, field in [
        ("fire", fire),
        ("smoke", smoke),
    ]:

        to_center = (
            field["center_world"]
            - cam["origin"]
        )

        distance = float(
            np.linalg.norm(
                to_center
            )
        )

        unit = (
            to_center
            / max(
                distance,
                1e-12,
            )
        )

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

        log(
            label,
            "center =",
            field["center_world"].tolist(),
            "| distance [m] =",
            distance,
            "| angle from optical axis [deg] =",
            angle,
        )

    log("")
    log("=" * 72)
    log("RAY MARCH")
    log("=" * 72)

    fire_max, smoke_max, smoke_tau = (
        project(
            cam,
            fire,
            smoke,
        )
    )

    base_surface = load_base_surface()

    (
        flame_mask,
        smoke_mask,
        fire_temp,
        smoke_temp,
        smoke_alpha,
        composite,
    ) = build_outputs(
        fire,
        smoke,
        fire_max,
        smoke_max,
        smoke_tau,
        base_surface,
    )

    # Arrays
    np.save(
        OUT / "fire_max_value.npy",
        fire_max,
    )

    np.save(
        OUT / "smoke_max_value.npy",
        smoke_max,
    )

    np.save(
        OUT / "smoke_optical_depth.npy",
        smoke_tau,
    )

    np.save(
        OUT / "flame_mask.npy",
        flame_mask,
    )

    np.save(
        OUT / "smoke_mask.npy",
        smoke_mask,
    )

    np.save(
        OUT / "fire_temperature_c.npy",
        fire_temp,
    )

    np.save(
        OUT / "smoke_temperature_c.npy",
        smoke_temp,
    )

    np.save(
        OUT / "smoke_alpha.npy",
        smoke_alpha,
    )

    np.save(
        OUT / "thermal_composite_c.npy",
        composite,
    )

    # PNGs
    save_gray(
        OUT / "fire_max_value.png",
        fire_max,
        0.0,
        max(
            float(
                fire_max.max()
            ),
            1e-8,
        ),
    )

    save_gray(
        OUT / "smoke_max_value.png",
        smoke_max,
        0.0,
        max(
            float(
                smoke_max.max()
            ),
            1e-8,
        ),
    )

    save_gray(
        OUT / "smoke_optical_depth.png",
        smoke_tau,
        0.0,
        max(
            float(
                smoke_tau.max()
            ),
            1e-8,
        ),
    )

    save_gray(
        OUT / "flame_mask.png",
        flame_mask * 255.0,
        0.0,
        255.0,
    )

    save_gray(
        OUT / "smoke_mask.png",
        smoke_mask * 255.0,
        0.0,
        255.0,
    )

    save_gray(
        OUT / "thermal_preview.png",
        composite,
        DISPLAY_MIN_C,
        DISPLAY_MAX_C,
    )

    # ---------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------

    log("")
    log("=" * 72)
    log("OUTPUT SUMMARY")
    log("=" * 72)

    log(
        "fire_max min/max:",
        float(
            fire_max.min()
        ),
        float(
            fire_max.max()
        ),
    )

    log(
        "smoke_max min/max:",
        float(
            smoke_max.min()
        ),
        float(
            smoke_max.max()
        ),
    )

    log(
        "smoke_tau min/max:",
        float(
            smoke_tau.min()
        ),
        float(
            smoke_tau.max()
        ),
    )

    nonzero_tau = smoke_tau[
        smoke_tau > 0.0
    ]

    if nonzero_tau.size:
        log(
            "smoke_tau positive percentiles "
            "[1,10,25,50,75,90,95,99]:",
            np.percentile(
                nonzero_tau,
                [1, 10, 25, 50, 75, 90, 95, 99],
            ).tolist(),
        )

    log(
        "flame pixels:",
        int(
            flame_mask.sum()
        ),
    )

    log(
        "smoke pixels:",
        int(
            smoke_mask.sum()
        ),
    )

    log(
        "thermal min/max [C]:",
        float(
            composite.min()
        ),
        float(
            composite.max()
        ),
    )

    log(
        "Saved to:",
        OUT,
    )


try:
    main()

except Exception:
    log("")
    log("========== EXCEPTION ==========")

    for line in traceback.format_exc().splitlines():
        log(line)

    raise
