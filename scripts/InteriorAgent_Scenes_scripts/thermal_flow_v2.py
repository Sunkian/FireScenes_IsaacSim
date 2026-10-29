"""
thermal_flow_v2_1_separated.py

Isaac Sim 5.1 — safer synthetic FLIR Lepton 3.1R thermal capture.

This version fixes the main weakness of V2:
- Fire and smoke are isolated with SEPARATE same-time render passes.
- Flame mask uses hysteresis thresholding + component filtering.
- Smoke mask uses its own smoke-only difference + morphology/component cleanup.
- The Flow simulation is paused before all comparison renders so every pass is
  taken at the same simulation state.
- No destructive Replicator teardown is performed at the end.

Run INSIDE Isaac Sim Script Editor.

Expected scene organization:
    .../hazards/Fire/...
    .../hazards/DenseSmoke/...

The script discovers Flow render/offscreen controls automatically from prim paths
containing "Fire" and "Smoke"/"DenseSmoke".

IMPORTANT:
This remains an empirical thermal model, not calibrated LWIR radiative transfer.
"""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd


# =============================================================================
# CONFIG
# =============================================================================

THERMAL_CAMERA = (
    "/Root/World/Meshes/SensorRig/"
    "ThermalCamera/FLIR_Lepton_3_1R"
)

OUTPUT_DIR = Path("/scratch/apagnoux/lepton_thermal_v2_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FLIR Lepton 3.1R output resolution
THERMAL_RESOLUTION = (160, 120)  # width, height

# Flow warm-up
FLOW_UPDATE_HZ = 60.0
FLOW_WARMUP_SECONDS = 3.0
FLOW_WARMUP_STEPS = int(round(FLOW_UPDATE_HZ * FLOW_WARMUP_SECONDS))

# RTX accumulation for each frozen-state pass
RT_SUBFRAMES = 4


# =============================================================================
# BASE TEMPERATURE PRIORS [degC]
# =============================================================================

AMBIENT_TEMP_C = 22.0

TEMPERATURE_C = {
    "wall": 23.0,
    "ceiling": 24.0,
    "floor": 24.0,
    "door": 23.0,
    "chair": 23.0,
    "sofa": 23.0,
    "table": 23.0,
    "cabinet": 23.0,
    "clutter": 23.0,
    "person": 33.0,
}


# =============================================================================
# FLAME MASK
# =============================================================================

# A pixel must first exceed CORE threshold; the mask then grows through
# neighboring pixels above GROW threshold. This keeps faint nearby flame while
# rejecting many isolated reflections.
FLAME_CORE_THRESHOLD = 90.0
FLAME_GROW_THRESHOLD = 45.0

# Remove tiny disconnected detections.
FLAME_MIN_COMPONENT_AREA = 8

FLOW_FLAME_MIN_TEMP_C = 450.0
FLOW_FLAME_MAX_TEMP_C = 850.0


# =============================================================================
# SMOKE MASK
# =============================================================================

# Smoke changes the image more subtly, so thresholds are lower.
SMOKE_CORE_THRESHOLD = 12.0
SMOKE_GROW_THRESHOLD = 5.0

# Remove isolated noise.
SMOKE_MIN_COMPONENT_AREA = 20

# One closing pass fills small gaps; one opening pass removes tiny speckles.
SMOKE_CLOSE_ITERATIONS = 1
SMOKE_OPEN_ITERATIONS = 1

SMOKE_SCORE_FOR_MAX = 80.0

SMOKE_MIN_TEMP_C = 35.0
SMOKE_MAX_TEMP_C = 110.0
SMOKE_MAX_OPACITY = 0.60


# =============================================================================
# LEPTON-LIKE SENSOR MODEL
# =============================================================================

LEPTON_MIN_TEMP_C = -10.0
LEPTON_MAX_TEMP_C = 400.0

TEMPORAL_NOISE_STD_C = 0.05
FIXED_PATTERN_STD_C = 0.08
RANDOM_SEED = 42


# =============================================================================
# PNG DISPLAY RANGE ONLY
# =============================================================================

DISPLAY_MIN_C = 20.0
DISPLAY_MAX_C = 60.0


# =============================================================================
# ARRAY / LABEL HELPERS
# =============================================================================

def extract_array(output: Any) -> np.ndarray:
    if isinstance(output, dict):
        output = output["data"]
    return np.asarray(output).copy()


def labels_to_text(labels: Any) -> str:
    if isinstance(labels, dict):
        return " ".join(str(v).lower() for v in labels.values())
    if isinstance(labels, (list, tuple)):
        return " ".join(str(v).lower() for v in labels)
    return str(labels).lower()


def build_surface_temperature(
    semantic_ids: np.ndarray,
    id_to_labels: dict,
) -> np.ndarray:
    temp = np.full(
        semantic_ids.shape,
        AMBIENT_TEMP_C,
        dtype=np.float32,
    )

    for raw_id, labels in id_to_labels.items():
        sid = int(raw_id)
        text = labels_to_text(labels)
        mask = semantic_ids == sid

        for class_name, temperature_c in TEMPERATURE_C.items():
            if class_name in text:
                temp[mask] = temperature_c
                break

    return temp


def luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb[:, :, :3].astype(np.float32)
    return (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)


# =============================================================================
# FLOW CONTROL DISCOVERY
# =============================================================================

def classify_flow_path(path: str) -> str | None:
    """
    Decide whether a Flow render/offscreen prim belongs to Fire or Smoke.
    """
    low = path.lower()
    tokens = [t for t in low.split("/") if t]

    # Test smoke first because names such as DenseSmoke are unambiguous.
    if any(("densesmoke" in t) or (t == "smoke") for t in tokens):
        return "smoke"

    if any(
        (t == "fire")
        or ("wispfire" in t)
        or ("whispyfire" in t)
        or ("whispy_fire" in t)
        for t in tokens
    ):
        return "fire"

    return None


def discover_flow_controls(stage):
    """
    Find the Flow rendering attributes that can suppress Fire and Smoke
    independently while leaving simulation state untouched.
    """
    groups = {
        "fire": [],
        "smoke": [],
    }

    # These are the same kinds of rendering controls used in the working
    # Flow subtraction pipeline.
    wanted = {
        ("flowrender/raymarch", "colorScale"),
        ("flowrender/raymarch", "attenuation"),
        ("flowoffscreen/colormap", "colorScale"),
        ("flowoffscreen/shadow", "attenuation"),
    }

    seen = set()

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        low = path.lower()

        group = classify_flow_path(path)
        if group is None:
            continue

        for suffix, attr_name in wanted:
            if suffix not in low:
                continue

            attr = prim.GetAttribute(attr_name)
            if not attr.IsValid():
                continue

            key = (group, path, attr_name)
            if key in seen:
                continue

            seen.add(key)

            groups[group].append(
                {
                    "path": path,
                    "name": attr_name,
                    "attr": attr,
                    "original": attr.Get(),
                }
            )

    return groups


def set_group_enabled(items, enabled: bool):
    """
    Suppress or restore one Flow visual group.

    The current presets have scalar colorScale/attenuation controls. If a
    property cannot be set to scalar zero, the exception is reported.
    """
    for item in items:
        attr = item["attr"]

        if enabled:
            attr.Set(item["original"])
        else:
            attr.Set(0.0)


async def settle_updates(count=2):
    for _ in range(count):
        await omni.kit.app.get_app().next_update_async()


# =============================================================================
# BINARY MASK OPERATIONS (NO SCIPY REQUIRED)
# =============================================================================

def binary_dilate(mask: np.ndarray, iterations=1) -> np.ndarray:
    out = mask.astype(bool).copy()

    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        nxt = np.zeros_like(out, dtype=bool)

        for dy in range(3):
            for dx in range(3):
                nxt |= p[
                    dy:dy + out.shape[0],
                    dx:dx + out.shape[1],
                ]

        out = nxt

    return out


def binary_erode(mask: np.ndarray, iterations=1) -> np.ndarray:
    out = mask.astype(bool).copy()

    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        nxt = np.ones_like(out, dtype=bool)

        for dy in range(3):
            for dx in range(3):
                nxt &= p[
                    dy:dy + out.shape[0],
                    dx:dx + out.shape[1],
                ]

        out = nxt

    return out


def binary_close(mask: np.ndarray, iterations=1) -> np.ndarray:
    return binary_erode(
        binary_dilate(mask, iterations),
        iterations,
    )


def binary_open(mask: np.ndarray, iterations=1) -> np.ndarray:
    return binary_dilate(
        binary_erode(mask, iterations),
        iterations,
    )


def hysteresis_mask(
    score: np.ndarray,
    grow_threshold: float,
    core_threshold: float,
) -> np.ndarray:
    """
    Keep only low-threshold regions that are connected to at least one
    high-threshold core pixel.
    """
    support = score >= grow_threshold
    seeds = score >= core_threshold

    h, w = score.shape
    result = np.zeros((h, w), dtype=bool)
    q = deque()

    ys, xs = np.nonzero(seeds & support)

    for y, x in zip(ys.tolist(), xs.tolist()):
        if not result[y, x]:
            result[y, x] = True
            q.append((y, x))

    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    while q:
        y, x = q.popleft()

        for dy, dx in neighbors:
            yy = y + dy
            xx = x + dx

            if yy < 0 or yy >= h or xx < 0 or xx >= w:
                continue

            if result[yy, xx] or not support[yy, xx]:
                continue

            result[yy, xx] = True
            q.append((yy, xx))

    return result


def filter_components(
    mask: np.ndarray,
    min_area: int,
):
    """
    Remove connected components smaller than min_area.
    Returns cleaned mask and a descending list of kept component areas.
    """
    mask = mask.astype(bool)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    cleaned = np.zeros((h, w), dtype=bool)

    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    kept_areas = []

    for y0 in range(h):
        for x0 in range(w):
            if visited[y0, x0] or not mask[y0, x0]:
                continue

            visited[y0, x0] = True
            q = deque([(y0, x0)])
            coords = []

            while q:
                y, x = q.popleft()
                coords.append((y, x))

                for dy, dx in neighbors:
                    yy = y + dy
                    xx = x + dx

                    if yy < 0 or yy >= h or xx < 0 or xx >= w:
                        continue

                    if visited[yy, xx] or not mask[yy, xx]:
                        continue

                    visited[yy, xx] = True
                    q.append((yy, xx))

            area = len(coords)

            if area >= min_area:
                kept_areas.append(area)

                for y, x in coords:
                    cleaned[y, x] = True

    kept_areas.sort(reverse=True)
    return cleaned, kept_areas


# =============================================================================
# DYNAMIC FLOW FIELD DERIVATION
# =============================================================================

def derive_flame(
    full_rgb: np.ndarray,
    fire_off_rgb: np.ndarray,
):
    """
    Fire-only difference.

    A high-threshold core is grown through a lower-threshold support region,
    then tiny disconnected components are removed.
    """
    lum_full = luminance(full_rgb)
    lum_fire_off = luminance(fire_off_rgb)

    score = np.maximum(
        lum_full - lum_fire_off,
        0.0,
    ).astype(np.float32)

    raw = hysteresis_mask(
        score,
        grow_threshold=FLAME_GROW_THRESHOLD,
        core_threshold=FLAME_CORE_THRESHOLD,
    )

    mask, component_areas = filter_components(
        raw,
        min_area=FLAME_MIN_COMPONENT_AREA,
    )

    strength = np.clip(
        (score - FLAME_GROW_THRESHOLD)
        / max(255.0 - FLAME_GROW_THRESHOLD, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)

    temp = (
        FLOW_FLAME_MIN_TEMP_C
        + strength
        * (FLOW_FLAME_MAX_TEMP_C - FLOW_FLAME_MIN_TEMP_C)
    ).astype(np.float32)

    temp[~mask] = 0.0

    return mask, score, temp, component_areas


def derive_smoke(
    full_rgb: np.ndarray,
    smoke_off_rgb: np.ndarray,
    flame_mask: np.ndarray,
):
    """
    Smoke-only difference.

    Because smoke may both attenuate and scatter visible light, use absolute
    luminance change after disabling ONLY the smoke Flow renderers.
    """
    lum_full = luminance(full_rgb)
    lum_smoke_off = luminance(smoke_off_rgb)

    score = np.abs(
        lum_full - lum_smoke_off
    ).astype(np.float32)

    raw = hysteresis_mask(
        score,
        grow_threshold=SMOKE_GROW_THRESHOLD,
        core_threshold=SMOKE_CORE_THRESHOLD,
    )

    if SMOKE_CLOSE_ITERATIONS > 0:
        raw = binary_close(
            raw,
            SMOKE_CLOSE_ITERATIONS,
        )

    if SMOKE_OPEN_ITERATIONS > 0:
        raw = binary_open(
            raw,
            SMOKE_OPEN_ITERATIONS,
        )

    # Flame pixels are not smoke pixels for the class mask.
    raw &= ~flame_mask

    mask, component_areas = filter_components(
        raw,
        min_area=SMOKE_MIN_COMPONENT_AREA,
    )

    strength = np.clip(
        score / max(SMOKE_SCORE_FOR_MAX, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)

    smoke_temp = (
        SMOKE_MIN_TEMP_C
        + strength
        * (SMOKE_MAX_TEMP_C - SMOKE_MIN_TEMP_C)
    ).astype(np.float32)

    opacity = (
        strength * SMOKE_MAX_OPACITY
    ).astype(np.float32)

    smoke_temp[~mask] = 0.0
    opacity[~mask] = 0.0

    return (
        mask,
        score,
        smoke_temp,
        opacity,
        component_areas,
    )


# =============================================================================
# THERMAL MODEL
# =============================================================================

def apply_dynamic_flow(
    surface_temp: np.ndarray,
    flame_mask: np.ndarray,
    flame_temp: np.ndarray,
    smoke_mask: np.ndarray,
    smoke_temp: np.ndarray,
    smoke_opacity: np.ndarray,
) -> np.ndarray:

    out = surface_temp.astype(np.float32).copy()

    # Smoke first: simplified emitting/attenuating layer.
    if np.any(smoke_mask):
        alpha = smoke_opacity[smoke_mask]

        out[smoke_mask] = (
            (1.0 - alpha) * out[smoke_mask]
            + alpha * smoke_temp[smoke_mask]
        )

    # Flame is hotter and takes precedence.
    if np.any(flame_mask):
        out[flame_mask] = np.maximum(
            out[flame_mask],
            flame_temp[flame_mask],
        )

    return out


def apply_lepton_sensor_model(
    thermal_gt_c: np.ndarray,
    fixed_pattern: np.ndarray,
):
    clipped = np.clip(
        thermal_gt_c,
        LEPTON_MIN_TEMP_C,
        LEPTON_MAX_TEMP_C,
    ).astype(np.float32)

    rng = np.random.default_rng(RANDOM_SEED + 1000)

    temporal = rng.normal(
        0.0,
        TEMPORAL_NOISE_STD_C,
        clipped.shape,
    ).astype(np.float32)

    return (
        clipped
        + fixed_pattern
        + temporal
    ).astype(np.float32)


def make_display(thermal_c: np.ndarray) -> np.ndarray:
    x = (
        thermal_c - DISPLAY_MIN_C
    ) / max(
        DISPLAY_MAX_C - DISPLAY_MIN_C,
        1e-6,
    )

    return np.round(
        np.clip(x, 0.0, 1.0) * 255.0
    ).astype(np.uint8)


# =============================================================================
# CAPTURE
# =============================================================================

async def capture_rgb(rgb_annot):
    await rep.orchestrator.step_async(
        delta_time=0.0,
        rt_subframes=RT_SUBFRAMES,
        pause_timeline=True,
    )
    return extract_array(rgb_annot.get_data())


async def run():

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No USD stage is open.")

    camera = stage.GetPrimAtPath(THERMAL_CAMERA)

    if not camera.IsValid():
        raise RuntimeError(
            f"Thermal camera not found:\n{THERMAL_CAMERA}"
        )

    # ---------------------------------------------------------------------
    # Discover independent Fire and Smoke controls.
    # ---------------------------------------------------------------------

    controls = discover_flow_controls(stage)

    print("\n========== FIRE CONTROLS ==========")
    for item in controls["fire"]:
        print(
            f'{item["path"]}.{item["name"]}'
            f' = {item["original"]}'
        )

    print("\n========== SMOKE CONTROLS ==========")
    for item in controls["smoke"]:
        print(
            f'{item["path"]}.{item["name"]}'
            f' = {item["original"]}'
        )

    if not controls["fire"]:
        raise RuntimeError(
            "No Fire Flow render controls were found.\n"
            "The Fire preset path must contain Fire/WispFire/WhispyFire."
        )

    if not controls["smoke"]:
        raise RuntimeError(
            "No Smoke Flow render controls were found.\n"
            "The smoke preset path must contain Smoke/DenseSmoke."
        )

    # ---------------------------------------------------------------------
    # Render product + annotators.
    # ---------------------------------------------------------------------

    render_product = rep.create.render_product(
        THERMAL_CAMERA,
        THERMAL_RESOLUTION,
    )

    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")

    semantic_annot = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation",
        init_params={"colorize": False},
    )

    depth_annot = rep.AnnotatorRegistry.get_annotator(
        "distance_to_camera"
    )

    rgb_annot.attach(render_product)
    semantic_annot.attach(render_product)
    depth_annot.attach(render_product)

    rep.orchestrator.set_capture_on_play(False)
    rep.orchestrator.preview()

    await settle_updates(3)

    timeline = omni.timeline.get_timeline_interface()

    # Fixed spatial pattern for this capture.
    fixed_rng = np.random.default_rng(RANDOM_SEED)

    fixed_pattern = fixed_rng.normal(
        0.0,
        FIXED_PATTERN_STD_C,
        (
            THERMAL_RESOLUTION[1],
            THERMAL_RESOLUTION[0],
        ),
    ).astype(np.float32)

    try:
        # -----------------------------------------------------------------
        # Warm up Flow.
        # -----------------------------------------------------------------

        timeline.play()
        await settle_updates(1)

        print(
            f"\nWarming Flow for {FLOW_WARMUP_SECONDS:.1f}s "
            f"({FLOW_WARMUP_STEPS} simulation steps)..."
        )

        for _ in range(FLOW_WARMUP_STEPS):
            await rep.orchestrator.step_async(
                delta_time=1.0 / FLOW_UPDATE_HZ,
                rt_subframes=1,
                pause_timeline=False,
            )

        # Freeze the exact simulation state for ALL comparison renders.
        timeline.pause()
        await settle_updates(3)

        print("Flow warm-up complete; simulation state frozen.")

        # -----------------------------------------------------------------
        # PASS A: FULL = Fire ON + Smoke ON
        # -----------------------------------------------------------------

        set_group_enabled(controls["fire"], True)
        set_group_enabled(controls["smoke"], True)
        await settle_updates(2)

        rgb_full = await capture_rgb(rgb_annot)

        # Semantics/depth are captured in the same frozen state.
        semantic_output = semantic_annot.get_data()

        semantic_ids = extract_array(
            semantic_output
        ).squeeze().astype(np.uint32)

        id_to_labels = (
            semantic_output
            .get("info", {})
            .get("idToLabels", {})
        )

        depth = extract_array(
            depth_annot.get_data()
        ).squeeze().astype(np.float32)

        # -----------------------------------------------------------------
        # PASS B: FIRE OFF, Smoke ON
        # -----------------------------------------------------------------

        set_group_enabled(controls["fire"], False)
        set_group_enabled(controls["smoke"], True)
        await settle_updates(3)

        rgb_fire_off = await capture_rgb(rgb_annot)

        # Restore fire before changing the next group.
        set_group_enabled(controls["fire"], True)
        await settle_updates(2)

        # -----------------------------------------------------------------
        # PASS C: Fire ON, SMOKE OFF
        # -----------------------------------------------------------------

        set_group_enabled(controls["fire"], True)
        set_group_enabled(controls["smoke"], False)
        await settle_updates(3)

        rgb_smoke_off = await capture_rgb(rgb_annot)

        # Restore full rendering.
        set_group_enabled(controls["smoke"], True)
        set_group_enabled(controls["fire"], True)
        await settle_updates(3)

        # -----------------------------------------------------------------
        # Base surfaces
        # -----------------------------------------------------------------

        surface_temp = build_surface_temperature(
            semantic_ids,
            id_to_labels,
        )

        # -----------------------------------------------------------------
        # Independent dynamic masks
        # -----------------------------------------------------------------

        (
            flame_mask,
            flame_score,
            flame_temp,
            flame_components,
        ) = derive_flame(
            rgb_full,
            rgb_fire_off,
        )

        (
            smoke_mask,
            smoke_score,
            smoke_temp,
            smoke_opacity,
            smoke_components,
        ) = derive_smoke(
            rgb_full,
            rgb_smoke_off,
            flame_mask,
        )

        thermal_gt = apply_dynamic_flow(
            surface_temp,
            flame_mask,
            flame_temp,
            smoke_mask,
            smoke_temp,
            smoke_opacity,
        )

        thermal_sensor = apply_lepton_sensor_model(
            thermal_gt,
            fixed_pattern,
        )

        thermal_preview = make_display(
            thermal_sensor
        )

        # -----------------------------------------------------------------
        # Save numeric outputs
        # -----------------------------------------------------------------

        np.save(
            OUTPUT_DIR / "semantic_ids.npy",
            semantic_ids,
        )

        np.save(
            OUTPUT_DIR / "depth_m.npy",
            depth,
        )

        np.save(
            OUTPUT_DIR / "surface_temperature_c.npy",
            surface_temp,
        )

        np.save(
            OUTPUT_DIR / "flow_flame_mask.npy",
            flame_mask.astype(np.uint8),
        )

        np.save(
            OUTPUT_DIR / "flow_flame_score.npy",
            flame_score,
        )

        np.save(
            OUTPUT_DIR / "flow_flame_temperature_c.npy",
            flame_temp,
        )

        np.save(
            OUTPUT_DIR / "flow_smoke_mask.npy",
            smoke_mask.astype(np.uint8),
        )

        np.save(
            OUTPUT_DIR / "flow_smoke_score.npy",
            smoke_score,
        )

        np.save(
            OUTPUT_DIR / "flow_smoke_temperature_c.npy",
            smoke_temp,
        )

        np.save(
            OUTPUT_DIR / "flow_smoke_opacity.npy",
            smoke_opacity,
        )

        np.save(
            OUTPUT_DIR / "thermal_gt_c.npy",
            thermal_gt,
        )

        np.save(
            OUTPUT_DIR / "thermal_lepton_c.npy",
            thermal_sensor,
        )

        # -----------------------------------------------------------------
        # Save diagnostic images
        # -----------------------------------------------------------------

        Image.fromarray(
            rgb_full[:, :, :3].astype(np.uint8),
            mode="RGB",
        ).save(
            OUTPUT_DIR / "pass_A_full.png"
        )

        Image.fromarray(
            rgb_fire_off[:, :, :3].astype(np.uint8),
            mode="RGB",
        ).save(
            OUTPUT_DIR / "pass_B_fire_off.png"
        )

        Image.fromarray(
            rgb_smoke_off[:, :, :3].astype(np.uint8),
            mode="RGB",
        ).save(
            OUTPUT_DIR / "pass_C_smoke_off.png"
        )

        Image.fromarray(
            flame_mask.astype(np.uint8) * 255,
            mode="L",
        ).save(
            OUTPUT_DIR / "flame_mask.png"
        )

        Image.fromarray(
            smoke_mask.astype(np.uint8) * 255,
            mode="L",
        ).save(
            OUTPUT_DIR / "smoke_mask.png"
        )

        # Score previews help threshold tuning.
        flame_score_preview = np.round(
            np.clip(flame_score / 150.0, 0.0, 1.0) * 255
        ).astype(np.uint8)

        smoke_score_preview = np.round(
            np.clip(smoke_score / 80.0, 0.0, 1.0) * 255
        ).astype(np.uint8)

        Image.fromarray(
            flame_score_preview,
            mode="L",
        ).save(
            OUTPUT_DIR / "flame_score_preview.png"
        )

        Image.fromarray(
            smoke_score_preview,
            mode="L",
        ).save(
            OUTPUT_DIR / "smoke_score_preview.png"
        )

        Image.fromarray(
            thermal_preview,
            mode="L",
        ).save(
            OUTPUT_DIR / "thermal_preview.png"
        )

        # -----------------------------------------------------------------
        # Report
        # -----------------------------------------------------------------

        print("\n========== SEMANTIC MAPPING ==========")
        for sid, labels in id_to_labels.items():
            print(sid, "->", labels)

        print("\n========== MASK COMPONENTS ==========")
        print(
            "Flame component areas:",
            flame_components[:10],
        )
        print(
            "Smoke component areas:",
            smoke_components[:10],
        )

        print("\n========== THERMAL RESULT ==========")
        print("Shape:", thermal_gt.shape)

        print(
            "Surface range:",
            float(surface_temp.min()),
            "to",
            float(surface_temp.max()),
            "C",
        )

        print(
            "GT range:",
            float(thermal_gt.min()),
            "to",
            float(thermal_gt.max()),
            "C",
        )

        print(
            "Lepton range:",
            float(thermal_sensor.min()),
            "to",
            float(thermal_sensor.max()),
            "C",
        )

        print(
            "Flame pixels:",
            int(flame_mask.sum()),
        )

        print(
            "Smoke pixels:",
            int(smoke_mask.sum()),
        )

        print("\nSaved to:", OUTPUT_DIR)

    finally:
        # -----------------------------------------------------------------
        # SAFE FINISH
        # -----------------------------------------------------------------

        print("\nFinishing safely...")

        # Restore BOTH visual groups no matter where an exception occurred.
        for group_name in ("fire", "smoke"):
            for item in controls[group_name]:
                try:
                    item["attr"].Set(item["original"])
                except Exception:
                    pass

        try:
            await rep.orchestrator.wait_until_complete_async()
        except Exception as exc:
            print(
                "wait_until_complete warning:",
                repr(exc),
            )

        # Keep simulation state paused; do NOT reset Flow with timeline.stop().
        try:
            timeline.pause()
        except Exception:
            pass

        await settle_updates(8)

        # Intentionally do NOT detach annotators or destroy the render product
        # immediately after RTX/Flow captures in Script Editor mode.
        print("Capture complete.")
        print("Timeline remains paused.")
        print("Flow rendering restored.")
        print("Isaac Sim should remain usable.")


def report_task(task):
    try:
        task.result()
    except asyncio.CancelledError:
        print("Thermal capture task cancelled.")
    except Exception:
        import traceback
        traceback.print_exc()


_task = asyncio.ensure_future(run())
_task.add_done_callback(report_task)

print("Thermal Flow V2.1 SEPARATED scheduled.")
