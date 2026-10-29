"""
record_dual_pov_semantics_with_flow_masks_v2.py

Isaac Sim 5.1

Purpose
-------
Single recording script for your scene.

It records:
1) Gemini RGB camera POV
2) Thermal camera POV

And saves for each frame:
- Gemini RGB
- Gemini depth (.npy + better preview .png)
- Gemini solid-object semantics from Replicator
- Gemini flame mask from native Flow burn
- Gemini smoke mask from native Flow smoke density
- Gemini combined semantics (+ labels)

- Thermal POV RGB
- Thermal solid-object semantics
- Thermal flame mask from native Flow burn
- Thermal smoke mask from native Flow smoke density
- Thermal combined semantics (+ labels)

- per-frame metadata

Notes
-----
- Fire/smoke growth is controlled inside this script.
- Depth data is kept as distance_to_camera, but the preview is improved.
- Flow flame/smoke masks are NOT taken from Replicator semantics.
  They are projected from native NanoVDB readback.
- This script does NOT yet generate the final radiometric Lepton image.
- Timeline is PAUSED at the end.
- Final fire and smoke source radii are 1.0 m.

Output
------
/scratch/apagnoux/record_dual_pov_semantics_v2/sequence_001/
"""

from __future__ import annotations

import asyncio
import json
import math
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

import omni.graph.core as og
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom

import warp as wp


# =============================================================================
# PATHS
# =============================================================================

GEMINI_RGB_CAMERA = "/Root/World/Meshes/SensorRig/Gemini335L/camera_rgb/Camera_rgb"
THERMAL_CAMERA = "/Root/World/Meshes/SensorRig/ThermalCamera/FLIR_Lepton_3_1R"

FIRE_EMITTER = "/Root/World/Meshes/hazards/Fire/flowEmitterSphere"
SMOKE_EMITTER = "/Root/World/Meshes/hazards/DenseSmoke/flowEmitterSphere"

FIRE_VEL_ADV = "/Root/World/Meshes/hazards/Fire/flowSimulate/advection/velocity"
SMOKE_VEL_ADV = "/Root/World/Meshes/hazards/DenseSmoke/flowSimulate/advection/velocity"
SMOKE_DENSITY_ADV = "/Root/World/Meshes/hazards/DenseSmoke/flowSimulate/advection/smoke"

FIRE_EXPORT = "/Root/World/Meshes/hazards/Fire/flowSimulate/nanoVdbExport"
SMOKE_EXPORT = "/Root/World/Meshes/hazards/DenseSmoke/flowSimulate/nanoVdbExport"

READBACK_GRAPH_PATH = "/Root/World/FlowRecordReadbackGraph"
READBACK_NODE_NAME = "NanoVdbReadback"


# =============================================================================
# OUTPUT
# =============================================================================

ROOT_OUT = Path("/scratch/apagnoux/record_dual_pov_semantics_v2")
SEQ_DIR = ROOT_OUT / "sequence_001"

DIRS = {
    "gemini_rgb": SEQ_DIR / "gemini_rgb",
    "gemini_depth": SEQ_DIR / "gemini_depth",
    "gemini_semantic_usd": SEQ_DIR / "gemini_semantic_usd",
    "gemini_flame_mask": SEQ_DIR / "gemini_flame_mask",
    "gemini_smoke_mask": SEQ_DIR / "gemini_smoke_mask",
    "gemini_semantic_combined": SEQ_DIR / "gemini_semantic_combined",

    "thermal_rgb": SEQ_DIR / "thermal_rgb",
    "thermal_semantic_usd": SEQ_DIR / "thermal_semantic_usd",
    "thermal_flame_mask": SEQ_DIR / "thermal_flame_mask",
    "thermal_smoke_mask": SEQ_DIR / "thermal_smoke_mask",
    "thermal_semantic_combined": SEQ_DIR / "thermal_semantic_combined",

    "metadata": SEQ_DIR / "metadata",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# RECORD CONFIG
# =============================================================================

DURATION_S = 10.0
CAPTURE_FPS = 8.6
FLOW_HZ = 60.0
RT_SUBFRAMES = 2

TOTAL_FRAMES = int(round(DURATION_S * CAPTURE_FPS))
SIM_STEPS_PER_CAPTURE = max(1, int(round(FLOW_HZ / CAPTURE_FPS)))

# Two extra small Flow steps per frame are used to fetch native burn and smoke.
FLOW_READBACK_STEPS = 1

GEMINI_RESOLUTION = (1280, 720)
THERMAL_RESOLUTION = (160, 120)

# To keep the projection tractable, Flow masks are projected at lower resolution
# for Gemini and then upsampled with nearest-neighbor to Gemini image size.
GEMINI_MASK_RES = (640, 360)
THERMAL_MASK_RES = (160, 120)

MAX_TRACE_DISTANCE_M = 12.0
STEP_SCALE = 0.5

# Thresholds chosen from the diagnostics you already ran.
FIRE_BURN_THRESHOLD = 0.020
SMOKE_TAU_THRESHOLD = 0.050

# Semantic IDs reserved for Flow classes in the combined maps.
FLAME_LABEL_ID = 250
SMOKE_LABEL_ID = 251


# =============================================================================
# FIRE/SMOKE GROWTH CONFIG
# =============================================================================

FIRE_RADIUS_START_M = 0.10
FIRE_RADIUS_END_M = 1.00

SMOKE_RADIUS_START_M = 0.15
SMOKE_RADIUS_END_M = 1.00

FIRE_FUEL_START = 0.15
FIRE_FUEL_END = 0.80

FIRE_TEMP_START = 0.35
FIRE_TEMP_END = 2.00

SMOKE_START = 0.15
SMOKE_END = 1.00

SMOKE_TEMP_START = 0.10
SMOKE_TEMP_END = 0.50

FIRE_UP_START = 0.15
FIRE_UP_END = 0.85

SMOKE_UP_START = 0.25
SMOKE_UP_END = 1.00

LATERAL_VELOCITY_MAX = 0.18


# =============================================================================
# GLOBAL LOG
# =============================================================================

REPORT = SEQ_DIR / "record_report.txt"
LOG = []


def log(*parts):
    text = " ".join(str(x) for x in parts)
    print(text)
    LOG.append(text)
    REPORT.write_text("\n".join(LOG) + "\n")


# =============================================================================
# BASIC HELPERS
# =============================================================================

def require_prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {path}")
    return prim


def attr(prim, name):
    a = prim.GetAttribute(name)
    if not a.IsValid():
        raise RuntimeError(f"Missing attribute '{name}' on {prim.GetPath()}")
    return a


def optional_attr(prim, name):
    a = prim.GetAttribute(name)
    return a if a.IsValid() else None


def lerp(a, b, x):
    return a + (b - a) * x


def smoothstep01(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def growth_curve(t):
    if t <= 2.0:
        q = smoothstep01(t / 2.0)
        return lerp(0.0, 0.18, q)

    if t <= 5.0:
        q = smoothstep01((t - 2.0) / 3.0)
        return lerp(0.18, 0.50, q)

    if t <= 8.0:
        q = smoothstep01((t - 5.0) / 3.0)
        return lerp(0.50, 0.82, q)

    q = smoothstep01((t - 8.0) / 2.0)
    return lerp(0.82, 1.00, q)


async def settle(n=8):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


# =============================================================================
# SAVE HELPERS
# =============================================================================

def save_rgb(path: Path, arr):
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise RuntimeError(f"Unexpected RGB shape: {arr.shape}")
    Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB").save(path)


def save_depth(npy_path: Path, png_path: Path, arr):
    arr = np.asarray(arr).astype(np.float32)
    np.save(npy_path, arr)

    finite = np.isfinite(arr)
    vis = np.zeros(arr.shape, dtype=np.uint8)

    # Better indoor preview: percentile-clipped and inverted
    if np.any(finite):
        d = arr[finite]
        near = float(np.percentile(d, 2))
        far = float(np.percentile(d, 98))

        if far > near:
            norm = np.zeros(arr.shape, dtype=np.float32)
            norm[finite] = (arr[finite] - near) / (far - near)
            norm = np.clip(norm, 0.0, 1.0)
            norm = 1.0 - norm
            vis[finite] = np.round(norm[finite] * 255.0).astype(np.uint8)

    Image.fromarray(vis, mode="L").save(png_path)


def _extract_semantic_data(output):
    if isinstance(output, dict):
        data = output.get("data", None)
        info = output.get("info", {})
        if data is None:
            for _, v in output.items():
                if isinstance(v, np.ndarray):
                    return np.asarray(v), info
            raise RuntimeError(f"Could not find semantic data in keys: {list(output.keys())}")
        return np.asarray(data), info

    return np.asarray(output), {}


def semantic_colorize(ids_2d):
    ids_2d = ids_2d.astype(np.int32)
    max_id = int(np.max(ids_2d)) if ids_2d.size else 0
    lut = np.zeros((max(max_id + 1, 1), 3), dtype=np.uint8)
    for i in range(lut.shape[0]):
        lut[i] = np.array([
            (37 * i) % 255,
            (67 * i) % 255,
            (97 * i) % 255,
        ], dtype=np.uint8)
    return lut[np.clip(ids_2d, 0, lut.shape[0] - 1)]


def save_semantic_array(prefix: Path, ids, info_dict):
    ids = np.asarray(ids)
    if ids.ndim == 3 and ids.shape[2] == 1:
        ids = ids[:, :, 0]
    ids = ids.astype(np.int32)

    np.save(prefix.with_suffix(".npy"), ids)
    vis = semantic_colorize(ids)
    Image.fromarray(vis, mode="RGB").save(prefix.with_suffix(".png"))

    with open(prefix.with_name(prefix.name + "_labels.json"), "w") as f:
        json.dump(info_dict if info_dict is not None else {}, f, indent=2)

    return ids


def save_mask(prefix: Path, mask):
    mask = np.asarray(mask).astype(bool)
    np.save(prefix.with_suffix(".npy"), mask.astype(np.uint8))
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(prefix.with_suffix(".png"))


# =============================================================================
# FLOW READBACK SETUP
# =============================================================================

def configure_export(export_prim, *, readback_enabled, burn_enabled=False, smoke_enabled=False):
    attr(export_prim, "enabled").Set(True)
    attr(export_prim, "readbackEnabled").Set(readback_enabled)
    attr(export_prim, "readbackOpenVDB").Set(False)

    for name, enabled in {
        "fuelEnabled": False,
        "burnEnabled": burn_enabled,
        "smokeEnabled": smoke_enabled,
        "temperatureEnabled": False,
        "velocityEnabled": False,
        "divergenceEnabled": False,
    }.items():
        a = optional_attr(export_prim, name)
        if a:
            a.Set(bool(enabled))


def ensure_readback_node(stage):
    node_path = f"{READBACK_GRAPH_PATH}/{READBACK_NODE_NAME}"

    if not stage.GetPrimAtPath(node_path).IsValid():
        log("Creating native Flow readback node:", node_path)
        og.Controller.edit(
            {"graph_path": READBACK_GRAPH_PATH, "evaluator_name": "push"},
            {og.Controller.Keys.CREATE_NODES: [(READBACK_NODE_NAME, "omni.flowusd.FlowNanoVdbReadback")]},
        )

    node = og.Controller.node(node_path)
    if node is None or not node.is_valid():
        raise RuntimeError(f"Invalid OmniGraph readback node: {node_path}")

    burn_attr = node.get_attribute("outputs:burn")
    smoke_attr = node.get_attribute("outputs:smoke")

    if burn_attr is None or not burn_attr.is_valid():
        raise RuntimeError("Missing outputs:burn on readback node")
    if smoke_attr is None or not smoke_attr.is_valid():
        raise RuntimeError("Missing outputs:smoke on readback node")

    return burn_attr, smoke_attr


def read_words(attr_obj):
    value = og.Controller.get(attr_obj)
    if value is None:
        return np.empty(0, dtype=np.uint32)
    return np.asarray(value, dtype=np.uint32).reshape(-1).copy()


async def fetch_fire_burn(rep_orchestrator, fire_export, smoke_export, burn_attr):
    configure_export(fire_export, readback_enabled=True, burn_enabled=True, smoke_enabled=False)
    configure_export(smoke_export, readback_enabled=False, burn_enabled=False, smoke_enabled=False)
    await settle(2)

    for _ in range(FLOW_READBACK_STEPS):
        await rep_orchestrator.step_async(
            delta_time=1.0 / FLOW_HZ,
            rt_subframes=RT_SUBFRAMES,
            pause_timeline=False,
        )
    await settle(2)
    return read_words(burn_attr)


async def fetch_smoke_density(rep_orchestrator, fire_export, smoke_export, smoke_attr):
    configure_export(fire_export, readback_enabled=False, burn_enabled=False, smoke_enabled=False)
    configure_export(smoke_export, readback_enabled=True, burn_enabled=False, smoke_enabled=True)
    await settle(2)

    for _ in range(FLOW_READBACK_STEPS):
        await rep_orchestrator.step_async(
            delta_time=1.0 / FLOW_HZ,
            rt_subframes=RT_SUBFRAMES,
            pause_timeline=False,
        )
    await settle(2)
    return read_words(smoke_attr)


# =============================================================================
# WARP / NANOVDB LOADING
# =============================================================================

DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"


@wp.kernel
def lookup_values_kernel(
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


def load_field_from_words(label, words):
    if words.size == 0:
        return None

    # Words are uint32. View them as raw bytes for NanoVDB.
    raw_np = words.astype("<u4", copy=False).view(np.uint8).reshape(-1)
    raw_wp = wp.array(raw_np, dtype=wp.uint8, device=DEVICE)

    volume = wp.Volume(raw_wp, copy=False)
    voxel_size = float(volume.get_voxel_size()[0])

    coords = volume.get_voxels()
    n = int(coords.shape[0])

    values = wp.empty(n, dtype=wp.float32, device=DEVICE)

    wp.launch(
        lookup_values_kernel,
        dim=n,
        inputs=[volume.id, coords, values],
        device=DEVICE,
    )
    wp.synchronize()

    coords_np = coords.numpy().astype(np.int32)
    vals = values.numpy().reshape(-1).astype(np.float32)
    vals = np.where(np.isfinite(vals), vals, 0.0)

    pos = vals > 0.0
    if not np.any(pos):
        return None

    pos_coords = coords_np[pos]
    pos_vals = vals[pos]

    min_idx = pos_coords.min(axis=0)
    max_idx = pos_coords.max(axis=0)

    shape = (max_idx - min_idx + 1).astype(np.int32)
    dense = np.zeros(tuple(shape.tolist()), dtype=np.float32)

    local = pos_coords - min_idx
    dense[local[:, 0], local[:, 1], local[:, 2]] = pos_vals

    world_min = min_idx.astype(np.float64) * voxel_size
    world_max = max_idx.astype(np.float64) * voxel_size

    return {
        "label": label,
        "raw_wp": raw_wp,   # keep alive
        "volume": volume,   # keep alive
        "voxel_size": voxel_size,
        "dense": dense,
        "min_idx": min_idx,
        "world_min": world_min,
        "world_max": world_max,
    }


# =============================================================================
# CAMERA / PROJECTION
# =============================================================================

def get_camera(stage, path, width, height):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing camera: {path}")

    cam = UsdGeom.Camera(prim)
    c2w = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    tr = c2w.ExtractTranslation()

    origin = np.array([tr[0], tr[1], tr[2]], dtype=np.float64)

    focal_mm = float(cam.GetFocalLengthAttr().Get())
    h_ap_mm = float(cam.GetHorizontalApertureAttr().Get())
    v_ap_mm = float(cam.GetVerticalApertureAttr().Get())

    fx = focal_mm / h_ap_mm * width
    fy = focal_mm / v_ap_mm * height
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5

    return {
        "origin": origin,
        "camera_to_world": c2w,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "path": path,
    }


def camera_ray_world(cam, px, py):
    x = (px - cam["cx"]) / cam["fx"]
    y = -(py - cam["cy"]) / cam["fy"]

    local = Gf.Vec3d(float(x), float(y), -1.0)
    local.Normalize()

    d = cam["camera_to_world"].TransformDir(local)
    out = np.array([d[0], d[1], d[2]], dtype=np.float64)
    out /= np.linalg.norm(out)
    return out


def ray_box_intersection(origin, direction, bmin, bmax):
    t_near = 0.0
    t_far = MAX_TRACE_DISTANCE_M

    for axis in range(3):
        d = direction[axis]

        if abs(d) < 1e-12:
            if origin[axis] < bmin[axis] or origin[axis] > bmax[axis]:
                return None
            continue

        inv_d = 1.0 / d
        t0 = (bmin[axis] - origin[axis]) * inv_d
        t1 = (bmax[axis] - origin[axis]) * inv_d

        if t0 > t1:
            t0, t1 = t1, t0

        t_near = max(t_near, t0)
        t_far = min(t_far, t1)

        if t_far < t_near:
            return None

    return max(t_near, 0.0), t_far


def sample_nearest(field, p_world):
    idx = np.rint(p_world / field["voxel_size"]).astype(np.int32)
    loc = idx - field["min_idx"]
    dense = field["dense"]

    if (
        loc[0] < 0 or loc[0] >= dense.shape[0] or
        loc[1] < 0 or loc[1] >= dense.shape[1] or
        loc[2] < 0 or loc[2] >= dense.shape[2]
    ):
        return 0.0

    return float(dense[loc[0], loc[1], loc[2]])


def march_field(field, origin, direction, integrate=False):
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

    vmax = 0.0
    integral = 0.0

    t = t0
    while t <= t1:
        p = origin + t * direction
        v = sample_nearest(field, p)

        if v > vmax:
            vmax = v

        if integrate and v > 0.0:
            integral += v * dt

        t += dt

    return vmax, integral


def project_fire_mask(field, cam, threshold):
    if field is None:
        return np.zeros((cam["height"], cam["width"]), dtype=bool), np.zeros((cam["height"], cam["width"]), dtype=np.float32)

    out = np.zeros((cam["height"], cam["width"]), dtype=np.float32)
    origin = cam["origin"]

    for py in range(cam["height"]):
        for px in range(cam["width"]):
            d = camera_ray_world(cam, px, py)
            vmax, _ = march_field(field, origin, d, integrate=False)
            out[py, px] = vmax

    return out >= threshold, out


def project_smoke_mask(field, cam, tau_threshold):
    if field is None:
        return np.zeros((cam["height"], cam["width"]), dtype=bool), np.zeros((cam["height"], cam["width"]), dtype=np.float32)

    tau = np.zeros((cam["height"], cam["width"]), dtype=np.float32)
    origin = cam["origin"]

    for py in range(cam["height"]):
        for px in range(cam["width"]):
            d = camera_ray_world(cam, px, py)
            _, integral = march_field(field, origin, d, integrate=True)
            tau[py, px] = integral

    return tau >= tau_threshold, tau


def resize_mask(mask, out_size):
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    img = img.resize(out_size, resample=Image.NEAREST)
    return (np.asarray(img) > 127)


# =============================================================================
# SEMANTIC COMBINATION
# =============================================================================

def combine_semantics(solid_ids, solid_info, flame_mask, smoke_mask):
    combined = solid_ids.astype(np.int32).copy()

    # smoke first, then flame overrides smoke where both happen
    combined[smoke_mask] = SMOKE_LABEL_ID
    combined[flame_mask] = FLAME_LABEL_ID

    out_info = {
        "base_semantics": solid_info if solid_info is not None else {},
        "added_flow_labels": {
            str(FLAME_LABEL_ID): "flame",
            str(SMOKE_LABEL_ID): "smoke",
        }
    }

    return combined, out_info


# =============================================================================
# FIRE/SMOKE GROWTH CONTROL
# =============================================================================

def apply_growth(fire, smoke, t):
    g = growth_curve(t)

    fire_radius = lerp(FIRE_RADIUS_START_M, FIRE_RADIUS_END_M, g)
    fire_fuel = lerp(FIRE_FUEL_START, FIRE_FUEL_END, g)
    fire_temp = lerp(FIRE_TEMP_START, FIRE_TEMP_END, g)
    fire_up = lerp(FIRE_UP_START, FIRE_UP_END, g)

    lateral_phase = smoothstep01((t - 5.0) / 3.0)
    fire_lat = LATERAL_VELOCITY_MAX * 0.45 * lateral_phase * math.sin(1.7 * t)

    attr(fire, "radius").Set(float(fire_radius))
    attr(fire, "fuel").Set(float(fire_fuel))
    attr(fire, "temperature").Set(float(fire_temp))
    attr(fire, "velocity").Set(Gf.Vec3f(float(fire_lat), 0.0, float(fire_up)))
    attr(fire, "velocityIsWorldSpace").Set(True)

    smoke_radius = lerp(SMOKE_RADIUS_START_M, SMOKE_RADIUS_END_M, g)
    smoke_value = lerp(SMOKE_START, SMOKE_END, g)
    smoke_temp = lerp(SMOKE_TEMP_START, SMOKE_TEMP_END, g)
    smoke_up = lerp(SMOKE_UP_START, SMOKE_UP_END, g)

    smoke_lat_x = LATERAL_VELOCITY_MAX * lateral_phase * math.sin(0.9 * t)
    smoke_lat_y = LATERAL_VELOCITY_MAX * lateral_phase * math.cos(0.7 * t)

    attr(smoke, "radius").Set(float(smoke_radius))
    attr(smoke, "smoke").Set(float(smoke_value))
    attr(smoke, "temperature").Set(float(smoke_temp))
    attr(smoke, "velocity").Set(Gf.Vec3f(float(smoke_lat_x), float(smoke_lat_y), float(smoke_up)))
    attr(smoke, "velocityIsWorldSpace").Set(True)

    return {
        "growth": g,
        "fire_radius_m": fire_radius,
        "fire_fuel": fire_fuel,
        "fire_temperature": fire_temp,
        "smoke_radius_m": smoke_radius,
        "smoke_density": smoke_value,
        "smoke_temperature": smoke_temp,
    }


# =============================================================================
# MAIN
# =============================================================================

async def main():
    wp.init()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")

    fire = require_prim(stage, FIRE_EMITTER)
    smoke = require_prim(stage, SMOKE_EMITTER)
    fire_vel_adv = require_prim(stage, FIRE_VEL_ADV)
    smoke_vel_adv = require_prim(stage, SMOKE_VEL_ADV)
    smoke_density_adv = require_prim(stage, SMOKE_DENSITY_ADV)

    fire_export = require_prim(stage, FIRE_EXPORT)
    smoke_export = require_prim(stage, SMOKE_EXPORT)

    burn_attr, smoke_attr = ensure_readback_node(stage)

    timeline = omni.timeline.get_timeline_interface()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    try:
        timeline.pause()
    except Exception:
        pass
    await settle(10)

    # Reduced dissipation for smoke accumulation / ceiling spread
    damp = optional_attr(smoke_density_adv, "damping")
    if damp:
        damp.Set(0.05)
    fade = optional_attr(smoke_density_adv, "fade")
    if fade:
        fade.Set(0.08)

    for vel_adv in (fire_vel_adv, smoke_vel_adv):
        d = optional_attr(vel_adv, "damping")
        if d:
            d.Set(0.01)
        f = optional_attr(vel_adv, "fade")
        if f:
            f.Set(0.15)

    attr(fire, "enabled").Set(False)
    attr(smoke, "enabled").Set(False)

    attr(fire, "radius").Set(FIRE_RADIUS_START_M)
    attr(fire, "radiusIsWorldSpace").Set(True)
    attr(fire, "fuel").Set(FIRE_FUEL_START)
    attr(fire, "temperature").Set(FIRE_TEMP_START)
    attr(fire, "velocity").Set(Gf.Vec3f(0.0, 0.0, FIRE_UP_START))
    attr(fire, "velocityIsWorldSpace").Set(True)

    attr(smoke, "radius").Set(SMOKE_RADIUS_START_M)
    attr(smoke, "radiusIsWorldSpace").Set(True)
    attr(smoke, "smoke").Set(SMOKE_START)
    attr(smoke, "temperature").Set(SMOKE_TEMP_START)
    attr(smoke, "velocity").Set(Gf.Vec3f(0.0, 0.0, SMOKE_UP_START))
    attr(smoke, "velocityIsWorldSpace").Set(True)

    fuel_attr = optional_attr(smoke, "fuel")
    if fuel_attr:
        fuel_attr.Set(0.0)
    couple_fuel = optional_attr(smoke, "coupleRateFuel")
    if couple_fuel:
        couple_fuel.Set(0.0)

    # disable flow readback initially
    configure_export(fire_export, readback_enabled=False, burn_enabled=False, smoke_enabled=False)
    configure_export(smoke_export, readback_enabled=False, burn_enabled=False, smoke_enabled=False)

    await settle(10)

    # -------------------------------------------------------------------------
    # Render products + annotators
    # -------------------------------------------------------------------------
    gemini_rp = rep.create.render_product(GEMINI_RGB_CAMERA, GEMINI_RESOLUTION)
    thermal_rp = rep.create.render_product(THERMAL_CAMERA, THERMAL_RESOLUTION)

    gemini_rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    gemini_depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    gemini_sem_ann = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")

    thermal_rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    thermal_sem_ann = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")

    gemini_rgb_ann.attach(gemini_rp)
    gemini_depth_ann.attach(gemini_rp)
    gemini_sem_ann.attach(gemini_rp)

    thermal_rgb_ann.attach(thermal_rp)
    thermal_sem_ann.attach(thermal_rp)

    await settle(10)

    # Cameras for flow-mask projection
    gem_cam_mask = get_camera(stage, GEMINI_RGB_CAMERA, GEMINI_MASK_RES[0], GEMINI_MASK_RES[1])
    therm_cam_mask = get_camera(stage, THERMAL_CAMERA, THERMAL_MASK_RES[0], THERMAL_MASK_RES[1])

    # session info
    session_info = {
        "duration_s": DURATION_S,
        "capture_fps": CAPTURE_FPS,
        "total_frames": TOTAL_FRAMES,
        "sim_steps_per_capture": SIM_STEPS_PER_CAPTURE,
        "flow_readback_steps_per_mask": FLOW_READBACK_STEPS,
        "flow_hz": FLOW_HZ,
        "gemini_camera_path": GEMINI_RGB_CAMERA,
        "thermal_camera_path": THERMAL_CAMERA,
        "gemini_resolution": GEMINI_RESOLUTION,
        "thermal_resolution": THERMAL_RESOLUTION,
        "gemini_mask_projection_resolution": GEMINI_MASK_RES,
        "thermal_mask_projection_resolution": THERMAL_MASK_RES,
        "flow_thresholds": {
            "fire_burn_threshold": FIRE_BURN_THRESHOLD,
            "smoke_tau_threshold": SMOKE_TAU_THRESHOLD,
        },
        "combined_label_ids": {
            str(FLAME_LABEL_ID): "flame",
            str(SMOKE_LABEL_ID): "smoke",
        },
        "notes": [
            "Replicator semantics are used for solid scene objects.",
            "Flame and smoke masks are produced from native Flow readback.",
            "Gemini depth is saved as metric .npy plus a percentile-clipped preview.",
        ],
    }

    with open(SEQ_DIR / "session_info.json", "w") as f:
        json.dump(session_info, f, indent=2)

    # -------------------------------------------------------------------------
    # Ignite + record
    # -------------------------------------------------------------------------
    attr(fire, "enabled").Set(True)
    attr(smoke, "enabled").Set(True)

    try:
        timeline.play()
    except Exception:
        pass

    log(f"Recording {TOTAL_FRAMES} frames to {SEQ_DIR}")

    for frame_idx in range(TOTAL_FRAMES):
        t = frame_idx / CAPTURE_FPS
        growth_meta = apply_growth(fire, smoke, t)

        # Main simulation advance for this frame
        for _ in range(SIM_STEPS_PER_CAPTURE):
            await rep.orchestrator.step_async(
                delta_time=1.0 / FLOW_HZ,
                rt_subframes=RT_SUBFRAMES,
                pause_timeline=False,
            )

        await settle(1)

        # Capture regular camera outputs
        gemini_rgb = np.asarray(gemini_rgb_ann.get_data())
        gemini_depth = np.asarray(gemini_depth_ann.get_data())
        gemini_sem_out = gemini_sem_ann.get_data()

        thermal_rgb = np.asarray(thermal_rgb_ann.get_data())
        thermal_sem_out = thermal_sem_ann.get_data()

        gemini_solid_ids, gemini_solid_info = _extract_semantic_data(gemini_sem_out)
        thermal_solid_ids, thermal_solid_info = _extract_semantic_data(thermal_sem_out)

        # Fetch native Flow burn and smoke density
        fire_burn_words = await fetch_fire_burn(rep.orchestrator, fire_export, smoke_export, burn_attr)
        smoke_density_words = await fetch_smoke_density(rep.orchestrator, fire_export, smoke_export, smoke_attr)

        # Disable readback between frames
        configure_export(fire_export, readback_enabled=False, burn_enabled=False, smoke_enabled=False)
        configure_export(smoke_export, readback_enabled=False, burn_enabled=False, smoke_enabled=False)
        await settle(1)

        fire_field = load_field_from_words("fire_burn", fire_burn_words)
        smoke_field = load_field_from_words("smoke_density", smoke_density_words)

        # Project flow masks into both viewpoints
        gem_flame_low, gem_burn_map = project_fire_mask(fire_field, gem_cam_mask, FIRE_BURN_THRESHOLD)
        gem_smoke_low, gem_smoke_tau = project_smoke_mask(smoke_field, gem_cam_mask, SMOKE_TAU_THRESHOLD)

        therm_flame_mask, therm_burn_map = project_fire_mask(fire_field, therm_cam_mask, FIRE_BURN_THRESHOLD)
        therm_smoke_mask, therm_smoke_tau = project_smoke_mask(smoke_field, therm_cam_mask, SMOKE_TAU_THRESHOLD)

        gem_flame_mask = resize_mask(gem_flame_low, GEMINI_RESOLUTION)
        gem_smoke_mask = resize_mask(gem_smoke_low, GEMINI_RESOLUTION)

        # Ensure solid semantics are 2D
        if gemini_solid_ids.ndim == 3 and gemini_solid_ids.shape[2] == 1:
            gemini_solid_ids = gemini_solid_ids[:, :, 0]
        if thermal_solid_ids.ndim == 3 and thermal_solid_ids.shape[2] == 1:
            thermal_solid_ids = thermal_solid_ids[:, :, 0]

        gem_combined, gem_labels = combine_semantics(gemini_solid_ids, gemini_solid_info, gem_flame_mask, gem_smoke_mask)
        therm_combined, therm_labels = combine_semantics(thermal_solid_ids, thermal_solid_info, therm_flame_mask, therm_smoke_mask)

        stem = f"{frame_idx:06d}"

        # Save Gemini outputs
        save_rgb(DIRS["gemini_rgb"] / f"{stem}.png", gemini_rgb)
        save_depth(
            DIRS["gemini_depth"] / f"{stem}.npy",
            DIRS["gemini_depth"] / f"{stem}.png",
            gemini_depth,
        )

        save_semantic_array(DIRS["gemini_semantic_usd"] / stem, gemini_solid_ids, gemini_solid_info)
        save_mask(DIRS["gemini_flame_mask"] / stem, gem_flame_mask)
        save_mask(DIRS["gemini_smoke_mask"] / stem, gem_smoke_mask)
        save_semantic_array(DIRS["gemini_semantic_combined"] / stem, gem_combined, gem_labels)

        # Save Thermal outputs
        save_rgb(DIRS["thermal_rgb"] / f"{stem}.png", thermal_rgb)
        save_semantic_array(DIRS["thermal_semantic_usd"] / stem, thermal_solid_ids, thermal_solid_info)
        save_mask(DIRS["thermal_flame_mask"] / stem, therm_flame_mask)
        save_mask(DIRS["thermal_smoke_mask"] / stem, therm_smoke_mask)
        save_semantic_array(DIRS["thermal_semantic_combined"] / stem, therm_combined, therm_labels)

        # Optional: save raw low-res float maps for debugging if useful
        np.save(DIRS["metadata"] / f"{stem}_gemini_burn_map.npy", gem_burn_map)
        np.save(DIRS["metadata"] / f"{stem}_gemini_smoke_tau.npy", gem_smoke_tau)
        np.save(DIRS["metadata"] / f"{stem}_thermal_burn_map.npy", therm_burn_map)
        np.save(DIRS["metadata"] / f"{stem}_thermal_smoke_tau.npy", therm_smoke_tau)

        frame_meta = {
            "frame_idx": frame_idx,
            "time_s": t,
            "growth": growth_meta,
            "gemini_flame_pixels": int(np.sum(gem_flame_mask)),
            "gemini_smoke_pixels": int(np.sum(gem_smoke_mask)),
            "thermal_flame_pixels": int(np.sum(therm_flame_mask)),
            "thermal_smoke_pixels": int(np.sum(therm_smoke_mask)),
        }

        with open(DIRS["metadata"] / f"{stem}.json", "w") as f:
            json.dump(frame_meta, f, indent=2)

        if frame_idx % 5 == 0:
            log(
                f"frame {frame_idx:04d}/{TOTAL_FRAMES} | "
                f"t={t:0.2f}s | "
                f"flame px G/T = {int(np.sum(gem_flame_mask))}/{int(np.sum(therm_flame_mask))} | "
                f"smoke px G/T = {int(np.sum(gem_smoke_mask))}/{int(np.sum(therm_smoke_mask))}"
            )

    # -------------------------------------------------------------------------
    # End
    # -------------------------------------------------------------------------
    try:
        timeline.pause()
    except Exception:
        pass

    configure_export(fire_export, readback_enabled=False, burn_enabled=False, smoke_enabled=False)
    configure_export(smoke_export, readback_enabled=False, burn_enabled=False, smoke_enabled=False)

    await settle(10)

    log("DONE.")
    log("Saved sequence to:", SEQ_DIR)


def _done_callback(task):
    try:
        task.result()
    except asyncio.CancelledError:
        log("Recording task cancelled.")
    except Exception:
        log("========== EXCEPTION ==========")
        for line in traceback.format_exc().splitlines():
            log(line)
        raise


_task = asyncio.ensure_future(main())
_task.add_done_callback(_done_callback)

print("Dual POV + Flow-mask recording scheduled.")
