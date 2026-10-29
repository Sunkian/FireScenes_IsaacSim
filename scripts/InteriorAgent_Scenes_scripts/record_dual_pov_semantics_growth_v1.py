"""
record_dual_pov_semantics_growth_v1.py

Isaac Sim 5.1

What it does
------------
Records a short fire-growth sequence from BOTH camera points of view:
1) Gemini RGB camera POV
2) Thermal camera POV

For each captured frame it saves:
- Gemini RGB image
- Gemini depth
- Gemini semantic segmentation
- Thermal camera POV image
- Thermal semantic segmentation
- per-frame metadata
- semantic label info JSON (when provided by the annotator)

This script also drives the progressive fire/smoke growth during recording,
so you only need ONE script for the actual dataset capture.

IMPORTANT
---------
- This records camera POV + semantic masks.
- It does NOT yet compute the final radiometric Lepton image in this script.
  It records the thermal camera viewpoint and segmentation from that viewpoint.
- Growth is driven by SIMULATION TIME / capture step, not wall-clock time.
- The timeline is PAUSED at the end.
- Final emitter radius reaches 1.0 m for both fire and smoke.

Output
------
/scratch/apagnoux/record_dual_pov_semantics_v1/
    sequence_001/
        gemini_rgb/
        gemini_depth/
        gemini_semantic/
        thermal_rgb/
        thermal_semantic/
        metadata/
        session_info.json
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Gf


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


# =============================================================================
# OUTPUT
# =============================================================================

ROOT_OUT = Path("/scratch/apagnoux/record_dual_pov_semantics_v1")
SEQ_DIR = ROOT_OUT / "sequence_001"

DIR_GEMINI_RGB = SEQ_DIR / "gemini_rgb"
DIR_GEMINI_DEPTH = SEQ_DIR / "gemini_depth"
DIR_GEMINI_SEM = SEQ_DIR / "gemini_semantic"

DIR_THERMAL_RGB = SEQ_DIR / "thermal_rgb"
DIR_THERMAL_SEM = SEQ_DIR / "thermal_semantic"

DIR_METADATA = SEQ_DIR / "metadata"

for d in [
    DIR_GEMINI_RGB, DIR_GEMINI_DEPTH, DIR_GEMINI_SEM,
    DIR_THERMAL_RGB, DIR_THERMAL_SEM, DIR_METADATA
]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# RECORDING CONFIG
# =============================================================================

DURATION_S = 10.0
CAPTURE_FPS = 8.6
FLOW_HZ = 60.0
RT_SUBFRAMES = 2

TOTAL_FRAMES = int(round(DURATION_S * CAPTURE_FPS))
SIM_STEPS_PER_CAPTURE = max(1, int(round(FLOW_HZ / CAPTURE_FPS)))

# POV resolutions
GEMINI_RESOLUTION = (1280, 720)
THERMAL_RESOLUTION = (160, 120)


# =============================================================================
# GROWTH CONFIG
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
# HELPERS
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
    """
    Piecewise source growth over the recording duration:
      0-2s   : ignition
      2-5s   : development
      5-8s   : stronger vertical growth
      8-10s  : developed source
    """
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


def save_rgb(path: Path, arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3].astype(np.uint8)
    else:
        raise RuntimeError(f"Unexpected RGB shape: {arr.shape}")
    Image.fromarray(rgb, mode="RGB").save(path)


def save_depth(path_npy: Path, path_png: Path, arr):
    arr = np.asarray(arr).astype(np.float32)
    np.save(path_npy, arr)

    finite = np.isfinite(arr)
    if not np.any(finite):
        vis = np.zeros(arr.shape, dtype=np.uint8)
    else:
        valid = arr[finite]
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if vmax <= vmin:
            vis = np.zeros(arr.shape, dtype=np.uint8)
        else:
            norm = np.zeros(arr.shape, dtype=np.float32)
            norm[finite] = (arr[finite] - vmin) / (vmax - vmin)
            vis = np.round(np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)

    Image.fromarray(vis, mode="L").save(path_png)


def _extract_semantic_data(output):
    """
    Handles multiple possible annotator return formats.
    Returns:
      ids_array, info_dict
    """
    if isinstance(output, dict):
        data = output.get("data", None)
        info = output.get("info", {})
        if data is None:
            # Some Isaac versions return the array directly under another key.
            for k, v in output.items():
                if isinstance(v, np.ndarray):
                    return np.asarray(v), info
            raise RuntimeError(f"Could not find semantic 'data' in output keys: {list(output.keys())}")
        return np.asarray(data), info

    return np.asarray(output), {}


def save_semantics(prefix: Path, output):
    ids, info = _extract_semantic_data(output)

    # Save raw ids as npy.
    np.save(prefix.with_suffix(".npy"), ids)

    # Create a simple visualization.
    arr = ids
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    arr = arr.astype(np.int32)

    max_id = int(np.max(arr)) if arr.size else 0
    lut = np.zeros((max(max_id + 1, 1), 3), dtype=np.uint8)

    # deterministic pseudo-colors
    for i in range(lut.shape[0]):
        lut[i] = np.array([
            (37 * i) % 255,
            (67 * i) % 255,
            (97 * i) % 255,
        ], dtype=np.uint8)

    vis = lut[np.clip(arr, 0, lut.shape[0] - 1)]
    Image.fromarray(vis, mode="RGB").save(prefix.with_suffix(".png"))

    # Save info/labels if available.
    info_path = prefix.with_name(prefix.name + "_labels.json")
    with open(info_path, "w") as f:
        json.dump(info if info is not None else {}, f, indent=2)


async def settle(n=8):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


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
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")

    # Scene prims
    fire = require_prim(stage, FIRE_EMITTER)
    smoke = require_prim(stage, SMOKE_EMITTER)
    fire_vel_adv = require_prim(stage, FIRE_VEL_ADV)
    smoke_vel_adv = require_prim(stage, SMOKE_VEL_ADV)
    smoke_density_adv = require_prim(stage, SMOKE_DENSITY_ADV)

    timeline = omni.timeline.get_timeline_interface()

    # -------------------------------------------------------------------------
    # Safe initial setup
    # -------------------------------------------------------------------------
    try:
        timeline.pause()
    except Exception:
        pass

    await settle(10)

    # Reduced dissipation so the plume persists and spreads under the ceiling.
    damp = optional_attr(smoke_density_adv, "damping")
    if damp:
        damp.Set(0.05)

    fade = optional_attr(smoke_density_adv, "fade")
    if fade:
        fade.Set(0.08)

    for vel_adv in (fire_vel_adv, smoke_vel_adv):
        damp = optional_attr(vel_adv, "damping")
        if damp:
            damp.Set(0.01)
        fade = optional_attr(vel_adv, "fade")
        if fade:
            fade.Set(0.15)

    # Start small and disabled.
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

    await settle(10)

    # -------------------------------------------------------------------------
    # Render products and annotators
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

    # -------------------------------------------------------------------------
    # Session metadata
    # -------------------------------------------------------------------------
    session_info = {
        "duration_s": DURATION_S,
        "capture_fps": CAPTURE_FPS,
        "total_frames": TOTAL_FRAMES,
        "sim_steps_per_capture": SIM_STEPS_PER_CAPTURE,
        "flow_hz": FLOW_HZ,
        "gemini_camera_path": GEMINI_RGB_CAMERA,
        "thermal_camera_path": THERMAL_CAMERA,
        "gemini_resolution": GEMINI_RESOLUTION,
        "thermal_resolution": THERMAL_RESOLUTION,
        "notes": [
            "This sequence records RGB/depth/semantics from Gemini POV.",
            "It also records camera-view RGB/semantics from the thermal camera POV.",
            "This script does not yet synthesize radiometric Lepton temperature."
        ]
    }

    with open(SEQ_DIR / "session_info.json", "w") as f:
        json.dump(session_info, f, indent=2)

    # -------------------------------------------------------------------------
    # Ignition + recording
    # -------------------------------------------------------------------------
    attr(fire, "enabled").Set(True)
    attr(smoke, "enabled").Set(True)

    try:
        timeline.play()
    except Exception:
        pass

    print(f"Recording {TOTAL_FRAMES} frames to {SEQ_DIR}")

    for frame_idx in range(TOTAL_FRAMES):
        t = frame_idx / CAPTURE_FPS

        growth_meta = apply_growth(fire, smoke, t)

        # Advance the simulation between captures.
        for _ in range(SIM_STEPS_PER_CAPTURE):
            await rep.orchestrator.step_async(
                delta_time=1.0 / FLOW_HZ,
                rt_subframes=RT_SUBFRAMES,
                pause_timeline=False,
            )

        await omni.kit.app.get_app().next_update_async()

        # Read annotators
        gemini_rgb = np.asarray(gemini_rgb_ann.get_data())
        gemini_depth = np.asarray(gemini_depth_ann.get_data())
        gemini_sem = gemini_sem_ann.get_data()

        thermal_rgb = np.asarray(thermal_rgb_ann.get_data())
        thermal_sem = thermal_sem_ann.get_data()

        stem = f"{frame_idx:06d}"

        # Save Gemini POV
        save_rgb(DIR_GEMINI_RGB / f"{stem}.png", gemini_rgb)
        save_depth(
            DIR_GEMINI_DEPTH / f"{stem}.npy",
            DIR_GEMINI_DEPTH / f"{stem}.png",
            gemini_depth
        )
        save_semantics(DIR_GEMINI_SEM / stem, gemini_sem)

        # Save Thermal POV
        save_rgb(DIR_THERMAL_RGB / f"{stem}.png", thermal_rgb)
        save_semantics(DIR_THERMAL_SEM / stem, thermal_sem)

        # Save frame metadata
        frame_meta = {
            "frame_idx": frame_idx,
            "time_s": t,
            "growth": growth_meta,
            "sim_steps_per_capture": SIM_STEPS_PER_CAPTURE,
        }

        with open(DIR_METADATA / f"{stem}.json", "w") as f:
            json.dump(frame_meta, f, indent=2)

        if frame_idx % 5 == 0:
            print(
                f"frame {frame_idx:04d}/{TOTAL_FRAMES} | "
                f"t={t:0.2f}s | "
                f"fire_r={growth_meta['fire_radius_m']:.2f} | "
                f"smoke_r={growth_meta['smoke_radius_m']:.2f}"
            )

    # -------------------------------------------------------------------------
    # End
    # -------------------------------------------------------------------------
    timeline.pause()
    await settle(10)

    print("DONE.")
    print("Saved sequence to:", SEQ_DIR)


def done_callback(task):
    try:
        task.result()
    except asyncio.CancelledError:
        print("Recording task cancelled.")
    except Exception:
        import traceback
        traceback.print_exc()


_task = asyncio.ensure_future(main())
_task.add_done_callback(done_callback)

print("Dual POV recording scheduled.")
