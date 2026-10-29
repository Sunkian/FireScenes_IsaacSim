"""
test_render_differencing_masks_v1.py

Isaac Sim 5.1

PURPOSE
-------
Validate render-differencing masks on ONE FROZEN simulation state before
building the full 60-frame recorder.

This script:
1) pauses the timeline at the CURRENT fire/smoke state;
2) captures both cameras normally;
3) mutes only the Fire RENDER parameters, captures both cameras again;
4) restores Fire rendering;
5) mutes only the DenseSmoke RENDER parameters, captures both cameras again;
6) restores Smoke rendering;
7) computes:
       flame mask = |normal - no_fire|
       smoke mask = |normal - no_smoke|
8) merges those masks into the Replicator semantic maps.

It does NOT:
- touch Flow NanoVDB readback/export;
- change emitters;
- clear/reset the simulation;
- use rep.orchestrator.step_async();
- run a recording loop.

RUN THIS ONLY WHEN FIRE + SMOKE ARE VISIBLY DEVELOPED.
It is deliberately a one-frame validation step.

Output:
    /scratch/apagnoux/render_diff_test_v1/
"""

from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd


# =============================================================================
# SCENE PATHS
# =============================================================================

GEMINI_CAMERA = (
    "/Root/World/Meshes/SensorRig/Gemini335L/"
    "camera_rgb/Camera_rgb"
)

THERMAL_CAMERA = (
    "/Root/World/Meshes/SensorRig/"
    "ThermalCamera/FLIR_Lepton_3_1R"
)

FIRE_ROOT = "/Root/World/Meshes/hazards/Fire"
SMOKE_ROOT = "/Root/World/Meshes/hazards/DenseSmoke"


# =============================================================================
# OUTPUT
# =============================================================================

OUT = Path("/scratch/apagnoux/render_diff_test_v1")
OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "report.txt"


# =============================================================================
# CAMERA / MASK SETTINGS
# =============================================================================

GEMINI_RESOLUTION = (640, 360)
THERMAL_RESOLUTION = (160, 120)

# Start conservative. The script also saves the raw difference images, so
# these can be tuned from evidence rather than guessed repeatedly.
FLAME_DIFF_THRESHOLD = 10.0
SMOKE_DIFF_THRESHOLD = 4.0

# Remove isolated tiny regions in the binary masks.
MIN_COMPONENT_PIXELS_GEMINI = 15
MIN_COMPONENT_PIXELS_THERMAL = 3

FLAME_LABEL_ID = 250
SMOKE_LABEL_ID = 251

SETTLE_UPDATES = 4
CAMERA_RETRIES = 8


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
# BASIC HELPERS
# =============================================================================

async def update_once():
    await omni.kit.app.get_app().next_update_async()


async def updates(n):
    for _ in range(n):
        await update_once()


def require_prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {path}")
    return prim


def extract_semantic(output):
    if isinstance(output, dict):
        data = output.get("data", None)
        info = output.get("info", {})

        if data is not None:
            return np.asarray(data), info

        for _, value in output.items():
            if isinstance(value, np.ndarray):
                return np.asarray(value), info

        raise RuntimeError("Semantic annotator returned no data array")

    return np.asarray(output), {}


# =============================================================================
# FLOW RENDER MUTING
# =============================================================================

# These are render-only parameters documented by Flow's Render/RayMarch
# properties. We only touch values found below the Fire or DenseSmoke root.
#
# The matcher is deliberately broad because presets can author them either:
# - as attributes on a rayMarch child prim, e.g. "colorScale"
# - or as namespaced attrs, e.g. "rayMarch:colorScale"
#
# We never touch emitter/simulation attrs such as fuel, temperature, smoke,
# enabled, coupleRate*, etc.
RENDER_ZERO_NAMES = {
    "colorscale",
    "attenuation",
    "densitymultiplier",
    "volumecolormultiplier",
    "ambientmultiplier",
}

RENDER_PATH_HINTS = (
    "render",
    "raymarch",
    "offscreen",
)


def discover_render_controls(stage, root_path):
    root = require_prim(stage, root_path)

    found = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())

        if not (
            path == root_path
            or path.startswith(root_path + "/")
        ):
            continue

        path_lower = path.lower()

        # Render-related prims only.
        if not any(
            hint in path_lower
            for hint in RENDER_PATH_HINTS
        ):
            continue

        for a in prim.GetAttributes():
            name = a.GetName()
            norm = (
                name
                .replace(":", "")
                .replace("_", "")
                .lower()
            )

            if not any(
                target in norm
                for target in RENDER_ZERO_NAMES
            ):
                continue

            value = a.Get()

            # Only numerical scalar attrs are muted.
            if isinstance(value, (int, float)):
                found.append(
                    {
                        "prim_path": path,
                        "attr": a,
                        "attr_name": name,
                        "original": value,
                    }
                )

    return found


def mute_controls(controls):
    changed = []

    for item in controls:
        a = item["attr"]
        original = item["original"]

        try:
            # Use 0.0 even for float-like values.
            a.Set(0.0)
            changed.append(item)

        except Exception as exc:
            log(
                "Could not mute",
                item["prim_path"],
                item["attr_name"],
                ":",
                exc,
            )

    return changed


def restore_controls(controls):
    for item in controls:
        try:
            item["attr"].Set(
                item["original"]
            )
        except Exception as exc:
            log(
                "Could not restore",
                item["prim_path"],
                item["attr_name"],
                ":",
                exc,
            )


# =============================================================================
# CAMERA CAPTURE
# =============================================================================

async def read_rgb_valid(annotator, name):
    last_shape = None

    for attempt in range(CAMERA_RETRIES):
        await update_once()

        data = np.array(
            annotator.get_data(),
            copy=True,
        )

        last_shape = data.shape

        if (
            data.ndim == 3
            and data.shape[2] >= 3
            and data.size > 0
        ):
            return data[:, :, :3].astype(
                np.uint8
            )

        log(
            f"{name}: invalid RGB attempt "
            f"{attempt+1}/{CAMERA_RETRIES}, "
            f"shape={last_shape}"
        )

    raise RuntimeError(
        f"{name} RGB never became valid. "
        f"Last shape={last_shape}"
    )


async def capture_rgb_pair(
    gemini_rgb_ann,
    thermal_rgb_ann,
    label,
):
    await updates(SETTLE_UPDATES)

    gemini = await read_rgb_valid(
        gemini_rgb_ann,
        f"{label}/gemini",
    )

    thermal = await read_rgb_valid(
        thermal_rgb_ann,
        f"{label}/thermal",
    )

    log(
        label,
        "captured:",
        "Gemini",
        gemini.shape,
        "Thermal",
        thermal.shape,
    )

    return gemini, thermal


# =============================================================================
# IMAGE / MASK HELPERS
# =============================================================================

def save_rgb(path, arr):
    Image.fromarray(
        np.asarray(arr).astype(np.uint8)
    ).save(path)


def diff_strength(a, b):
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)

    # Maximum absolute RGB change at each pixel.
    return np.max(
        np.abs(a - b),
        axis=2,
    )


def save_diff(path, diff):
    d = np.asarray(diff).astype(np.float32)

    # Direct 0..255 visualization because RGB difference itself is in that range.
    vis = np.clip(
        d,
        0.0,
        255.0,
    ).astype(np.uint8)

    Image.fromarray(vis).save(path)


def connected_component_filter(mask, min_pixels):
    """
    Small dependency-free 8-connected component filter.
    Resolutions are small enough for this one-frame diagnostic.
    """
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape

    visited = np.zeros(
        (h, w),
        dtype=bool,
    )

    out = np.zeros(
        (h, w),
        dtype=bool,
    )

    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            component = []

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))

                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx

                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(component) >= min_pixels:
                for cy, cx in component:
                    out[cy, cx] = True

    return out


def make_mask(diff, threshold, min_component):
    raw = diff >= float(threshold)

    return connected_component_filter(
        raw,
        min_component,
    )


def save_mask(path, mask):
    Image.fromarray(
        np.asarray(mask).astype(
            np.uint8
        ) * 255
    ).save(path)


def semantic_preview(ids):
    ids = np.asarray(ids).astype(np.int32)

    if ids.ndim == 3 and ids.shape[2] == 1:
        ids = ids[:, :, 0]

    max_id = int(ids.max()) if ids.size else 0
    max_id = max(
        max_id,
        FLAME_LABEL_ID,
        SMOKE_LABEL_ID,
    )

    lut = np.zeros(
        (max_id + 1, 3),
        dtype=np.uint8,
    )

    for i in range(lut.shape[0]):
        lut[i] = [
            (37 * i) % 255,
            (67 * i) % 255,
            (97 * i) % 255,
        ]

    # Fixed visible colors for our two added classes.
    lut[FLAME_LABEL_ID] = [255, 80, 0]
    lut[SMOKE_LABEL_ID] = [180, 180, 180]

    return lut[
        np.clip(
            ids,
            0,
            lut.shape[0] - 1,
        )
    ]


def merge_semantics(base_ids, flame_mask, smoke_mask):
    ids = np.asarray(base_ids)

    if ids.ndim == 3 and ids.shape[2] == 1:
        ids = ids[:, :, 0]

    merged = ids.astype(np.int32).copy()

    if merged.shape != flame_mask.shape:
        raise RuntimeError(
            f"Flame mask shape mismatch: "
            f"{flame_mask.shape} vs {merged.shape}"
        )

    if merged.shape != smoke_mask.shape:
        raise RuntimeError(
            f"Smoke mask shape mismatch: "
            f"{smoke_mask.shape} vs {merged.shape}"
        )

    # Smoke first; flame wins where both overlap.
    merged[smoke_mask] = SMOKE_LABEL_ID
    merged[flame_mask] = FLAME_LABEL_ID

    return merged


def save_semantic_outputs(
    stem,
    base_ids,
    merged_ids,
    base_info,
):
    np.save(
        OUT / f"{stem}_base.npy",
        base_ids,
    )

    np.save(
        OUT / f"{stem}_merged.npy",
        merged_ids,
    )

    Image.fromarray(
        semantic_preview(base_ids)
    ).save(
        OUT / f"{stem}_base.png"
    )

    Image.fromarray(
        semantic_preview(merged_ids)
    ).save(
        OUT / f"{stem}_merged.png"
    )

    with open(
        OUT / f"{stem}_labels.json",
        "w",
    ) as f:
        json.dump(
            {
                "replicator_info": base_info,
                "added_labels": {
                    str(FLAME_LABEL_ID): "flame",
                    str(SMOKE_LABEL_ID): "smoke",
                },
            },
            f,
            indent=2,
            default=str,
        )


# =============================================================================
# MAIN
# =============================================================================

async def main():
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No USD stage open")

    timeline = omni.timeline.get_timeline_interface()

    # Freeze the current developed Flow state.
    timeline.pause()

    await updates(6)

    # Discover render-only controls.
    fire_controls = discover_render_controls(
        stage,
        FIRE_ROOT,
    )

    smoke_controls = discover_render_controls(
        stage,
        SMOKE_ROOT,
    )

    log(
        "Fire render controls found:",
        len(fire_controls),
    )

    for item in fire_controls:
        log(
            "  FIRE",
            item["prim_path"],
            item["attr_name"],
            "=",
            item["original"],
        )

    log(
        "Smoke render controls found:",
        len(smoke_controls),
    )

    for item in smoke_controls:
        log(
            "  SMOKE",
            item["prim_path"],
            item["attr_name"],
            "=",
            item["original"],
        )

    if not fire_controls:
        raise RuntimeError(
            "No Fire render controls found. "
            "See report.txt; do not continue to a full recorder yet."
        )

    if not smoke_controls:
        raise RuntimeError(
            "No Smoke render controls found. "
            "See report.txt; do not continue to a full recorder yet."
        )

    # Render products and annotators.
    gemini_rp = rep.create.render_product(
        GEMINI_CAMERA,
        GEMINI_RESOLUTION,
    )

    thermal_rp = rep.create.render_product(
        THERMAL_CAMERA,
        THERMAL_RESOLUTION,
    )

    gemini_rgb_ann = rep.AnnotatorRegistry.get_annotator(
        "rgb"
    )

    thermal_rgb_ann = rep.AnnotatorRegistry.get_annotator(
        "rgb"
    )

    gemini_sem_ann = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation"
    )

    thermal_sem_ann = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation"
    )

    gemini_rgb_ann.attach(gemini_rp)
    thermal_rgb_ann.attach(thermal_rp)
    gemini_sem_ann.attach(gemini_rp)
    thermal_sem_ann.attach(thermal_rp)

    await updates(10)

    # -----------------------------------------------------------------
    # 1. NORMAL
    # -----------------------------------------------------------------
    normal_gemini, normal_thermal = await capture_rgb_pair(
        gemini_rgb_ann,
        thermal_rgb_ann,
        "normal",
    )

    save_rgb(
        OUT / "gemini_normal.png",
        normal_gemini,
    )

    save_rgb(
        OUT / "thermal_normal.png",
        normal_thermal,
    )

    gemini_sem_raw = gemini_sem_ann.get_data()
    thermal_sem_raw = thermal_sem_ann.get_data()

    gemini_base, gemini_info = extract_semantic(
        gemini_sem_raw
    )

    thermal_base, thermal_info = extract_semantic(
        thermal_sem_raw
    )

    gemini_base = np.array(
        gemini_base,
        copy=True,
    )

    thermal_base = np.array(
        thermal_base,
        copy=True,
    )

    # -----------------------------------------------------------------
    # 2. FIRE MUTED -> flame difference
    # -----------------------------------------------------------------
    fire_changed = mute_controls(
        fire_controls
    )

    try:
        no_fire_gemini, no_fire_thermal = await capture_rgb_pair(
            gemini_rgb_ann,
            thermal_rgb_ann,
            "no_fire",
        )

    finally:
        restore_controls(
            fire_changed
        )

    await updates(SETTLE_UPDATES)

    save_rgb(
        OUT / "gemini_no_fire.png",
        no_fire_gemini,
    )

    save_rgb(
        OUT / "thermal_no_fire.png",
        no_fire_thermal,
    )

    # -----------------------------------------------------------------
    # 3. SMOKE MUTED -> smoke difference
    # -----------------------------------------------------------------
    smoke_changed = mute_controls(
        smoke_controls
    )

    try:
        no_smoke_gemini, no_smoke_thermal = await capture_rgb_pair(
            gemini_rgb_ann,
            thermal_rgb_ann,
            "no_smoke",
        )

    finally:
        restore_controls(
            smoke_changed
        )

    await updates(SETTLE_UPDATES)

    save_rgb(
        OUT / "gemini_no_smoke.png",
        no_smoke_gemini,
    )

    save_rgb(
        OUT / "thermal_no_smoke.png",
        no_smoke_thermal,
    )

    # -----------------------------------------------------------------
    # 4. DIFFERENCES + MASKS
    # -----------------------------------------------------------------
    gemini_flame_diff = diff_strength(
        normal_gemini,
        no_fire_gemini,
    )

    thermal_flame_diff = diff_strength(
        normal_thermal,
        no_fire_thermal,
    )

    gemini_smoke_diff = diff_strength(
        normal_gemini,
        no_smoke_gemini,
    )

    thermal_smoke_diff = diff_strength(
        normal_thermal,
        no_smoke_thermal,
    )

    save_diff(
        OUT / "gemini_flame_diff.png",
        gemini_flame_diff,
    )

    save_diff(
        OUT / "thermal_flame_diff.png",
        thermal_flame_diff,
    )

    save_diff(
        OUT / "gemini_smoke_diff.png",
        gemini_smoke_diff,
    )

    save_diff(
        OUT / "thermal_smoke_diff.png",
        thermal_smoke_diff,
    )

    gemini_flame = make_mask(
        gemini_flame_diff,
        FLAME_DIFF_THRESHOLD,
        MIN_COMPONENT_PIXELS_GEMINI,
    )

    thermal_flame = make_mask(
        thermal_flame_diff,
        FLAME_DIFF_THRESHOLD,
        MIN_COMPONENT_PIXELS_THERMAL,
    )

    gemini_smoke = make_mask(
        gemini_smoke_diff,
        SMOKE_DIFF_THRESHOLD,
        MIN_COMPONENT_PIXELS_GEMINI,
    )

    thermal_smoke = make_mask(
        thermal_smoke_diff,
        SMOKE_DIFF_THRESHOLD,
        MIN_COMPONENT_PIXELS_THERMAL,
    )

    save_mask(
        OUT / "gemini_flame_mask.png",
        gemini_flame,
    )

    save_mask(
        OUT / "thermal_flame_mask.png",
        thermal_flame,
    )

    save_mask(
        OUT / "gemini_smoke_mask.png",
        gemini_smoke,
    )

    save_mask(
        OUT / "thermal_smoke_mask.png",
        thermal_smoke,
    )

    # -----------------------------------------------------------------
    # 5. MERGE WITH REPLICATOR SEMANTICS
    # -----------------------------------------------------------------
    gemini_merged = merge_semantics(
        gemini_base,
        gemini_flame,
        gemini_smoke,
    )

    thermal_merged = merge_semantics(
        thermal_base,
        thermal_flame,
        thermal_smoke,
    )

    save_semantic_outputs(
        "gemini_semantic",
        gemini_base,
        gemini_merged,
        gemini_info,
    )

    save_semantic_outputs(
        "thermal_semantic",
        thermal_base,
        thermal_merged,
        thermal_info,
    )

    # -----------------------------------------------------------------
    # REPORT
    # -----------------------------------------------------------------
    log("")
    log("MASK RESULTS")
    log(
        "Gemini flame pixels:",
        int(gemini_flame.sum()),
    )
    log(
        "Gemini smoke pixels:",
        int(gemini_smoke.sum()),
    )
    log(
        "Thermal flame pixels:",
        int(thermal_flame.sum()),
    )
    log(
        "Thermal smoke pixels:",
        int(thermal_smoke.sum()),
    )

    log("")
    log(
        "Gemini flame diff max/mean:",
        float(gemini_flame_diff.max()),
        float(gemini_flame_diff.mean()),
    )
    log(
        "Gemini smoke diff max/mean:",
        float(gemini_smoke_diff.max()),
        float(gemini_smoke_diff.mean()),
    )
    log(
        "Thermal flame diff max/mean:",
        float(thermal_flame_diff.max()),
        float(thermal_flame_diff.mean()),
    )
    log(
        "Thermal smoke diff max/mean:",
        float(thermal_smoke_diff.max()),
        float(thermal_smoke_diff.mean()),
    )

    log("")
    log("DONE.")
    log("Output:", OUT)
    log(
        "If flame/smoke masks look correct, "
        "this exact mechanism can be folded into the sequence recorder."
    )


def done_callback(task):
    try:
        task.result()

    except asyncio.CancelledError:
        log("Render-difference test cancelled.")

    except Exception:
        for line in traceback.format_exc().splitlines():
            log(line)


_task = asyncio.ensure_future(
    main()
)

_task.add_done_callback(
    done_callback
)

print(
    "One-frame render-differencing test scheduled. "
    "Run it on a developed fire/smoke frame."
)
