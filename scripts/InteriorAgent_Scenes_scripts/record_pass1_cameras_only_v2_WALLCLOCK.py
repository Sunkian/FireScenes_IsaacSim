"""
record_pass1_cameras_only_v2_WALLCLOCK.py

Isaac Sim 5.1

PASS 1 ONLY: camera data + Replicator semantics.
NO native Flow readback.
NO FlowNanoVdbReadback graph.
NO NanoVDB disk cache.
NO rep.orchestrator.step_async().
NO per-frame pause/resume.

This intentionally returns to the simple architecture that worked before.

Before recording, it explicitly disables any leftover NanoVDB readback from
previous V3-V9 attempts, while leaving the visible Flow simulation untouched.

Per frame:
- Gemini RGB
- Gemini depth (.npy + PNG preview)
- Gemini semantic IDs (.npy + PNG preview)
- Thermal POV RGB
- Thermal semantic IDs (.npy + PNG preview)
- metadata

Output:
    /scratch/apagnoux/pass1_cameras_only_v2_wallclock/sequence_001/
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import traceback
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

GEMINI_RGB_CAMERA = (
    "/Root/World/Meshes/SensorRig/Gemini335L/"
    "camera_rgb/Camera_rgb"
)

THERMAL_CAMERA = (
    "/Root/World/Meshes/SensorRig/"
    "ThermalCamera/FLIR_Lepton_3_1R"
)

FIRE_EMITTER = "/Root/World/Meshes/hazards/Fire/flowEmitterSphere"
SMOKE_EMITTER = "/Root/World/Meshes/hazards/DenseSmoke/flowEmitterSphere"

FIRE_VEL_ADV = (
    "/Root/World/Meshes/hazards/Fire/"
    "flowSimulate/advection/velocity"
)

SMOKE_VEL_ADV = (
    "/Root/World/Meshes/hazards/DenseSmoke/"
    "flowSimulate/advection/velocity"
)

SMOKE_DENSITY_ADV = (
    "/Root/World/Meshes/hazards/DenseSmoke/"
    "flowSimulate/advection/smoke"
)

FIRE_EXPORT = (
    "/Root/World/Meshes/hazards/Fire/"
    "flowSimulate/nanoVdbExport"
)

SMOKE_EXPORT = (
    "/Root/World/Meshes/hazards/DenseSmoke/"
    "flowSimulate/nanoVdbExport"
)


# =============================================================================
# OUTPUT
# =============================================================================

ROOT = Path("/scratch/apagnoux/pass1_cameras_only_v2_wallclock")
SEQ = ROOT / "sequence_001"

DIR = {
    "gemini_rgb": SEQ / "gemini_rgb",
    "gemini_depth": SEQ / "gemini_depth",
    "gemini_sem": SEQ / "gemini_semantic",
    "thermal_rgb": SEQ / "thermal_rgb",
    "thermal_sem": SEQ / "thermal_semantic",
    "metadata": SEQ / "metadata",
}

for d in DIR.values():
    d.mkdir(parents=True, exist_ok=True)

REPORT = SEQ / "report.txt"


# =============================================================================
# RECORDING CONFIG
# =============================================================================

DURATION_S = 10.0
CAPTURE_FPS = 6.0
TOTAL_FRAMES = int(round(DURATION_S * CAPTURE_FPS))

GEMINI_RESOLUTION = (640, 360)
THERMAL_RESOLUTION = (160, 120)

UPDATE_TIMEOUT_S = 8.0
CAMERA_RETRIES = 8


# =============================================================================
# FIRE/SMOKE GROWTH
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

LOG = []


def log(*parts):
    text = " ".join(str(x) for x in parts)
    print(text)
    LOG.append(text)
    REPORT.write_text("\n".join(LOG) + "\n")


def require_prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {path}")
    return prim


def attr(prim, name):
    a = prim.GetAttribute(name)
    if not a.IsValid():
        raise RuntimeError(
            f"Missing attribute '{name}' on {prim.GetPath()}"
        )
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
        return lerp(0.0, 0.18, smoothstep01(t / 2.0))
    if t <= 5.0:
        return lerp(0.18, 0.50, smoothstep01((t - 2.0) / 3.0))
    if t <= 8.0:
        return lerp(0.50, 0.82, smoothstep01((t - 5.0) / 3.0))
    return lerp(0.82, 1.00, smoothstep01((t - 8.0) / 2.0))


def apply_growth(fire, smoke, t):
    g = growth_curve(t)

    fire_radius = lerp(FIRE_RADIUS_START_M, FIRE_RADIUS_END_M, g)
    fire_fuel = lerp(FIRE_FUEL_START, FIRE_FUEL_END, g)
    fire_temp = lerp(FIRE_TEMP_START, FIRE_TEMP_END, g)
    fire_up = lerp(FIRE_UP_START, FIRE_UP_END, g)

    lateral = smoothstep01((t - 5.0) / 3.0)
    fire_vx = (
        LATERAL_VELOCITY_MAX
        * 0.45
        * lateral
        * math.sin(1.7 * t)
    )

    attr(fire, "radius").Set(float(fire_radius))
    attr(fire, "fuel").Set(float(fire_fuel))
    attr(fire, "temperature").Set(float(fire_temp))
    attr(fire, "velocity").Set(
        Gf.Vec3f(float(fire_vx), 0.0, float(fire_up))
    )
    attr(fire, "velocityIsWorldSpace").Set(True)

    smoke_radius = lerp(SMOKE_RADIUS_START_M, SMOKE_RADIUS_END_M, g)
    smoke_value = lerp(SMOKE_START, SMOKE_END, g)
    smoke_temp = lerp(SMOKE_TEMP_START, SMOKE_TEMP_END, g)
    smoke_up = lerp(SMOKE_UP_START, SMOKE_UP_END, g)

    smoke_vx = (
        LATERAL_VELOCITY_MAX
        * lateral
        * math.sin(0.9 * t)
    )

    smoke_vy = (
        LATERAL_VELOCITY_MAX
        * lateral
        * math.cos(0.7 * t)
    )

    attr(smoke, "radius").Set(float(smoke_radius))
    attr(smoke, "smoke").Set(float(smoke_value))
    attr(smoke, "temperature").Set(float(smoke_temp))
    attr(smoke, "velocity").Set(
        Gf.Vec3f(
            float(smoke_vx),
            float(smoke_vy),
            float(smoke_up),
        )
    )
    attr(smoke, "velocityIsWorldSpace").Set(True)

    return {
        "growth": float(g),
        "fire_radius_m": float(fire_radius),
        "fire_fuel": float(fire_fuel),
        "fire_temperature": float(fire_temp),
        "smoke_radius_m": float(smoke_radius),
        "smoke_density": float(smoke_value),
        "smoke_temperature": float(smoke_temp),
    }


async def safe_update(label):
    try:
        await asyncio.wait_for(
            omni.kit.app.get_app().next_update_async(),
            timeout=UPDATE_TIMEOUT_S,
        )
        return True
    except asyncio.TimeoutError:
        log("TIMEOUT:", label)
        return False


async def safe_updates(n, label):
    for i in range(n):
        if not await safe_update(f"{label} {i+1}/{n}"):
            return False
    return True


def disable_all_native_readback(stage):
    """
    Undo the only Flow-export state that can keep V3-V9 background export alive.
    This does NOT disable the Flow simulation or visible rendering.
    """
    for path in (FIRE_EXPORT, SMOKE_EXPORT):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue

        a = prim.GetAttribute("readbackEnabled")
        if a.IsValid():
            a.Set(False)

        clear_stop = prim.GetAttribute("readbackClearOnStop")
        if clear_stop.IsValid():
            clear_stop.Set(False)


# =============================================================================
# SAVE HELPERS
# =============================================================================

def save_rgb(path, arr):
    a = np.asarray(arr)

    if a.ndim != 3 or a.shape[2] < 3 or a.size == 0:
        raise RuntimeError(f"Unexpected RGB shape: {a.shape}")

    Image.fromarray(
        a[:, :, :3].astype(np.uint8)
    ).save(path)


def save_depth(npy_path, png_path, arr):
    d = np.asarray(arr).astype(np.float32)

    if d.size == 0:
        raise RuntimeError("Empty depth frame")

    np.save(npy_path, d)

    valid = np.isfinite(d) & (d > 0.0)
    vis = np.zeros(d.shape, dtype=np.uint8)

    if np.any(valid):
        vals = d[valid]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))

        if hi <= lo:
            lo = float(vals.min())
            hi = float(vals.max())

        if hi > lo:
            norm = 1.0 - np.clip(
                (d - lo) / (hi - lo),
                0.0,
                1.0,
            )
            vis[valid] = np.round(
                norm[valid] * 255.0
            ).astype(np.uint8)

    Image.fromarray(vis).save(png_path)


def extract_semantic(output):
    if isinstance(output, dict):
        data = output.get("data", None)
        info = output.get("info", {})

        if data is not None:
            return np.asarray(data), info

        for _, value in output.items():
            if isinstance(value, np.ndarray):
                return np.asarray(value), info

        raise RuntimeError(
            "Semantic annotator returned no data array"
        )

    return np.asarray(output), {}


def semantic_preview(ids):
    ids = np.asarray(ids).astype(np.int32)

    if ids.ndim == 3 and ids.shape[2] == 1:
        ids = ids[:, :, 0]

    max_id = int(ids.max()) if ids.size else 0

    lut = np.zeros(
        (max(max_id + 1, 1), 3),
        dtype=np.uint8,
    )

    for i in range(lut.shape[0]):
        lut[i] = [
            (37 * i) % 255,
            (67 * i) % 255,
            (97 * i) % 255,
        ]

    return lut[
        np.clip(
            ids,
            0,
            lut.shape[0] - 1,
        )
    ]


def save_semantic(prefix, ids):
    ids = np.asarray(ids)

    if ids.ndim == 3 and ids.shape[2] == 1:
        ids = ids[:, :, 0]

    if ids.size == 0:
        raise RuntimeError("Empty semantic frame")

    ids = ids.astype(np.int32)

    np.save(
        prefix.with_suffix(".npy"),
        ids,
    )

    Image.fromarray(
        semantic_preview(ids)
    ).save(
        prefix.with_suffix(".png")
    )


def safe_json(path, obj):
    with open(path, "w") as f:
        json.dump(
            obj,
            f,
            indent=2,
            default=str,
        )


async def read_camera_bundle(
    frame_idx,
    gemini_rgb_ann,
    gemini_depth_ann,
    gemini_sem_ann,
    thermal_rgb_ann,
    thermal_sem_ann,
):
    last_shapes = None

    for attempt in range(CAMERA_RETRIES):
        if not await safe_update(
            f"camera refresh frame {frame_idx} attempt {attempt+1}"
        ):
            continue

        gemini_rgb = np.array(
            gemini_rgb_ann.get_data(),
            copy=True,
        )

        gemini_depth = np.array(
            gemini_depth_ann.get_data(),
            copy=True,
        )

        thermal_rgb = np.array(
            thermal_rgb_ann.get_data(),
            copy=True,
        )

        try:
            gemini_ids, gemini_info = extract_semantic(
                gemini_sem_ann.get_data()
            )
            gemini_ids = np.array(
                gemini_ids,
                copy=True,
            )
        except Exception:
            gemini_ids = np.empty(0, dtype=np.int32)
            gemini_info = {}

        try:
            thermal_ids, thermal_info = extract_semantic(
                thermal_sem_ann.get_data()
            )
            thermal_ids = np.array(
                thermal_ids,
                copy=True,
            )
        except Exception:
            thermal_ids = np.empty(0, dtype=np.int32)
            thermal_info = {}

        last_shapes = {
            "gemini_rgb": tuple(gemini_rgb.shape),
            "gemini_depth": tuple(gemini_depth.shape),
            "gemini_sem": tuple(gemini_ids.shape),
            "thermal_rgb": tuple(thermal_rgb.shape),
            "thermal_sem": tuple(thermal_ids.shape),
        }

        valid = (
            gemini_rgb.ndim == 3
            and gemini_rgb.size > 0
            and gemini_depth.ndim >= 2
            and gemini_depth.size > 0
            and gemini_ids.ndim >= 2
            and gemini_ids.size > 0
            and thermal_rgb.ndim == 3
            and thermal_rgb.size > 0
            and thermal_ids.ndim >= 2
            and thermal_ids.size > 0
        )

        if valid:
            return {
                "gemini_rgb": gemini_rgb,
                "gemini_depth": gemini_depth,
                "gemini_ids": gemini_ids,
                "gemini_info": gemini_info,
                "thermal_rgb": thermal_rgb,
                "thermal_ids": thermal_ids,
                "thermal_info": thermal_info,
            }

        log(
            f"frame {frame_idx:04d}: camera not ready "
            f"{attempt+1}/{CAMERA_RETRIES} {last_shapes}"
        )

    raise RuntimeError(
        f"Camera annotators invalid for frame {frame_idx}. "
        f"Last shapes: {last_shapes}"
    )


# =============================================================================
# MAIN
# =============================================================================

async def main():
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No USD stage open")

    timeline = omni.timeline.get_timeline_interface()

    fire = require_prim(stage, FIRE_EMITTER)
    smoke = require_prim(stage, SMOKE_EMITTER)

    fire_vel_adv = require_prim(stage, FIRE_VEL_ADV)
    smoke_vel_adv = require_prim(stage, SMOKE_VEL_ADV)
    smoke_density_adv = require_prim(stage, SMOKE_DENSITY_ADV)

    try:
        timeline.pause()
    except Exception:
        pass

    # Critical cleanup from V3-V9.
    disable_all_native_readback(stage)

    if not await safe_updates(
        8,
        "initial cleanup settle",
    ):
        raise RuntimeError(
            "Initial cleanup/update timed out"
        )

    # Preserve the visual smoke settings that worked.
    d = optional_attr(
        smoke_density_adv,
        "damping",
    )
    if d:
        d.Set(0.05)

    f = optional_attr(
        smoke_density_adv,
        "fade",
    )
    if f:
        f.Set(0.08)

    for vel_adv in (
        fire_vel_adv,
        smoke_vel_adv,
    ):
        d = optional_attr(
            vel_adv,
            "damping",
        )
        if d:
            d.Set(0.01)

        f = optional_attr(
            vel_adv,
            "fade",
        )
        if f:
            f.Set(0.15)

    smoke_fuel = optional_attr(
        smoke,
        "fuel",
    )
    if smoke_fuel:
        smoke_fuel.Set(0.0)

    smoke_couple_fuel = optional_attr(
        smoke,
        "coupleRateFuel",
    )
    if smoke_couple_fuel:
        smoke_couple_fuel.Set(0.0)

    # Small initial ignition.
    attr(fire, "enabled").Set(True)
    attr(smoke, "enabled").Set(True)

    attr(fire, "radius").Set(
        FIRE_RADIUS_START_M
    )
    attr(smoke, "radius").Set(
        SMOKE_RADIUS_START_M
    )

    attr(fire, "radiusIsWorldSpace").Set(True)
    attr(smoke, "radiusIsWorldSpace").Set(True)

    # Render products
    gemini_rp = rep.create.render_product(
        GEMINI_RGB_CAMERA,
        GEMINI_RESOLUTION,
    )

    thermal_rp = rep.create.render_product(
        THERMAL_CAMERA,
        THERMAL_RESOLUTION,
    )

    gemini_rgb_ann = rep.AnnotatorRegistry.get_annotator(
        "rgb"
    )

    gemini_depth_ann = rep.AnnotatorRegistry.get_annotator(
        "distance_to_camera"
    )

    gemini_sem_ann = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation"
    )

    thermal_rgb_ann = rep.AnnotatorRegistry.get_annotator(
        "rgb"
    )

    thermal_sem_ann = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation"
    )

    gemini_rgb_ann.attach(gemini_rp)
    gemini_depth_ann.attach(gemini_rp)
    gemini_sem_ann.attach(gemini_rp)

    thermal_rgb_ann.attach(thermal_rp)
    thermal_sem_ann.attach(thermal_rp)

    if not await safe_updates(
        12,
        "annotator settle",
    ):
        raise RuntimeError(
            "Annotator settle timed out"
        )

    timeline.play()

    if not await safe_updates(
        2,
        "timeline start",
    ):
        raise RuntimeError(
            "Timeline failed to start"
        )

    # IMPORTANT:
    # Do NOT use timeline.get_current_time() to decide when the next frame
    # should be captured. In this Script Editor async context, Kit updates can
    # continue while the timeline time remains unchanged, which caused every
    # previous version to loop forever after frame 0.
    start_wall = time.monotonic()

    safe_json(
        SEQ / "session_info.json",
        {
            "pass": 1,
            "content": "camera products only",
            "duration_s": DURATION_S,
            "capture_fps": CAPTURE_FPS,
            "total_frames": TOTAL_FRAMES,
            "native_flow_readback": False,
            "flow_disk_cache": False,
            "rep_orchestrator_step_async": False,
            "capture_clock": "time.monotonic wall clock",
            "timeline_time_used_for_scheduling": False,
            "gemini_resolution": GEMINI_RESOLUTION,
            "thermal_resolution": THERMAL_RESOLUTION,
        },
    )

    log(
        "PASS 1 CAMERA-ONLY WALLCLOCK START:",
        TOTAL_FRAMES,
        "frames",
    )

    labels_written = False

    for frame_idx in range(TOTAL_FRAMES):
        target_elapsed = (
            frame_idx / CAPTURE_FPS
        )

        # Wait for the requested WALL-CLOCK capture time while continuously
        # yielding Kit updates so Flow/rendering keeps evolving.
        while True:
            wall_elapsed = (
                time.monotonic()
                - start_wall
            )

            apply_growth(
                fire,
                smoke,
                wall_elapsed,
            )

            if wall_elapsed >= target_elapsed:
                break

            if not await safe_update(
                f"wait for wall-clock frame {frame_idx}"
            ):
                raise RuntimeError(
                    f"Timed out before frame {frame_idx}"
                )

        camera = await read_camera_bundle(
            frame_idx,
            gemini_rgb_ann,
            gemini_depth_ann,
            gemini_sem_ann,
            thermal_rgb_ann,
            thermal_sem_ann,
        )

        # The camera helper itself performs a render update, so timestamp after
        # the successful camera read.
        elapsed = (
            time.monotonic()
            - start_wall
        )

        timeline_time = float(
            timeline.get_current_time()
        )

        growth = apply_growth(
            fire,
            smoke,
            elapsed,
        )

        stem = f"{frame_idx:06d}"

        save_rgb(
            DIR["gemini_rgb"] / f"{stem}.png",
            camera["gemini_rgb"],
        )

        save_depth(
            DIR["gemini_depth"] / f"{stem}.npy",
            DIR["gemini_depth"] / f"{stem}.png",
            camera["gemini_depth"],
        )

        save_semantic(
            DIR["gemini_sem"] / stem,
            camera["gemini_ids"],
        )

        save_rgb(
            DIR["thermal_rgb"] / f"{stem}.png",
            camera["thermal_rgb"],
        )

        save_semantic(
            DIR["thermal_sem"] / stem,
            camera["thermal_ids"],
        )

        if not labels_written:
            safe_json(
                SEQ / "gemini_semantic_labels.json",
                camera["gemini_info"],
            )

            safe_json(
                SEQ / "thermal_semantic_labels.json",
                camera["thermal_info"],
            )

            labels_written = True

        safe_json(
            DIR["metadata"] / f"{stem}.json",
            {
                "frame_idx": frame_idx,
                "timeline_time_s_observed": timeline_time,
                "wall_elapsed_s": elapsed,
                "target_wall_elapsed_s": target_elapsed,
                "growth": growth,
            },
        )

        log(
            f"frame {frame_idx:04d}/{TOTAL_FRAMES}: "
            f"SAVED wall={elapsed:0.3f}s "
            f"timeline={timeline_time:0.3f}s"
        )

    timeline.pause()

    await safe_updates(
        4,
        "final settle",
    )

    log("PASS 1 WALLCLOCK DONE.")
    log("Output:", SEQ)


def done_callback(task):
    try:
        task.result()

    except asyncio.CancelledError:
        log(
            "Pass 1 cancelled."
        )

    except Exception:
        for line in (
            traceback.format_exc()
            .splitlines()
        ):
            log(line)


_task = asyncio.ensure_future(
    main()
)

_task.add_done_callback(
    done_callback
)

print(
    "Pass 1 camera-only recorder scheduled "
    "(ALL native Flow readback disabled)."
)
