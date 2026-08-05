"""Record a synchronized Gemini + thermal-MVP sequence in Isaac Sim 5.1 UI.

Run this file from Isaac Sim's Script Editor, not with the system Python.
The script performs:
  1. A Flow warm-up period with no frames saved.
  2. A genuine-scene pass with the flame proxy hidden.
  3. A second, zero-time thermal-mask pass with the proxy visible.
  4. Synchronized saving of RGB, IR views, renderer depth, semantics,
     thermal ground truth, noisy thermal data, and metadata.

The thermal model is an MVP based on semantic classes. It is not a physical
LWIR/radiometric simulation. Flow smoke does not yet affect thermal values.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import UsdGeom


# =============================================================================
# USER CONFIGURATION
# =============================================================================

RGB_CAMERA = (
    "/World/SensorRig/Gemini335L/"
    "camera_rgb/Camera_rgb"
)

IR_LEFT_CAMERA = (
    "/World/SensorRig/Gemini335L/"
    "camera_ir_left/Camera_ir_left"
)

IR_RIGHT_CAMERA = (
    "/World/SensorRig/Gemini335L/"
    "camera_ir_right/Camera_ir_right"
)

DEPTH_CAMERA = (
    "/World/SensorRig/Gemini335L/"
    "camera_ir_left/Camera_Depth_Left"
)

THERMAL_CAMERA = "/World/SensorRig/ThermalCamera"
FLAME_PROXY = "/World/ThermalProxies/FlameProxy"

# Change the final folder name for each run.
OUTPUT_ROOT = Path(
    "/home/apagnoux/Foresight_Isaac/"
    "output/multimodal_mvp_warmup_001"
)

GEMINI_RESOLUTION = (640, 480)   # width, height
THERMAL_RESOLUTION = (160, 120)  # width, height

# A common logical output rate keeps the modalities synchronized in this MVP.
CAPTURE_FPS = 8.6
NUMBER_OF_FRAMES = 20

# Flow warm-up: no frames are saved during this period.
FLOW_UPDATE_HZ = 60.0
FLOW_WARMUP_SECONDS = 3.0
FLOW_WARMUP_STEPS = int(round(FLOW_UPDATE_HZ * FLOW_WARMUP_SECONDS))

# Rendering quality. Increase capture subframes if needed, but expect slower runs.
CAPTURE_RT_SUBFRAMES = 4
THERMAL_MASK_RT_SUBFRAMES = 1
WARMUP_RT_SUBFRAMES = 1

# Semantic-temperature MVP values. These are not calibrated radiometric values.
AMBIENT_TEMP_C = 22.0
PERSON_TEMP_C = 34.0
FLAME_TEMP_C = 850.0

# Simple placeholder sensor noise.
THERMAL_TEMPORAL_NOISE_STD_C = 0.05
THERMAL_FIXED_PATTERN_STD_C = 0.08
RANDOM_SEED = 42

# Display-only range. Fire/flame intentionally saturates white.
THERMAL_DISPLAY_MIN_C = 20.0
THERMAL_DISPLAY_MAX_C = 45.0

# Refuse to mix a new run into an existing non-empty output directory.
ALLOW_NONEMPTY_OUTPUT = False


# =============================================================================
# HELPERS
# =============================================================================


def require_camera(stage, path: str) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid camera prim path: {path}")
    if not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Prim is not a USD Camera: {path}")


def require_imageable(stage, path: str) -> UsdGeom.Imageable:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid imageable prim path: {path}")

    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        raise RuntimeError(f"Prim is not UsdGeom.Imageable: {path}")
    return imageable


def extract_array(output: Any) -> np.ndarray:
    """Copy annotator data before another render pass replaces the buffer."""
    if isinstance(output, dict):
        if "data" not in output:
            raise RuntimeError(
                f"Annotator dictionary has no 'data' key: {output.keys()}"
            )
        output = output["data"]
    return np.asarray(output).copy()


def save_rgb(path: Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"Unexpected camera image shape: {image.shape}")

    Image.fromarray(
        image[:, :, :3].astype(np.uint8),
        mode="RGB",
    ).save(path)


def labels_to_text(labels: Any) -> str:
    if isinstance(labels, dict):
        return " ".join(str(value).lower() for value in labels.values())
    if isinstance(labels, (list, tuple)):
        return " ".join(str(value).lower() for value in labels)
    return str(labels).lower()


def to_jsonable(value: Any) -> Any:
    """Convert numpy-heavy annotator output into JSON-compatible data."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)

    # Fall back to a readable representation for USD/carb/custom objects.
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def build_temperature_gt(
    semantic_ids: np.ndarray,
    id_to_labels: dict,
) -> tuple[np.ndarray, dict[str, int]]:
    """Create a noise-free semantic temperature image."""
    temperature_gt = np.full(
        semantic_ids.shape,
        AMBIENT_TEMP_C,
        dtype=np.float32,
    )

    counts = {
        "person": 0,
        "flame": 0,
    }

    for semantic_id_raw, labels in id_to_labels.items():
        semantic_id = int(semantic_id_raw)
        label_text = labels_to_text(labels)
        mask = semantic_ids == semantic_id

        if "person" in label_text:
            temperature_gt[mask] = PERSON_TEMP_C
            counts["person"] += int(mask.sum())
        elif "flame" in label_text:
            temperature_gt[mask] = FLAME_TEMP_C
            counts["flame"] += int(mask.sum())

    return temperature_gt, counts


def apply_thermal_sensor_mvp(
    temperature_gt_c: np.ndarray,
    fixed_pattern_c: np.ndarray,
    frame_id: int,
) -> np.ndarray:
    """Apply placeholder fixed-pattern and temporal noise."""
    rng = np.random.default_rng(RANDOM_SEED + 1000 + frame_id)
    temporal_noise = rng.normal(
        0.0,
        THERMAL_TEMPORAL_NOISE_STD_C,
        temperature_gt_c.shape,
    ).astype(np.float32)

    return (
        temperature_gt_c.astype(np.float32)
        + fixed_pattern_c
        + temporal_noise
    )


def make_thermal_display(thermal_c: np.ndarray) -> np.ndarray:
    normalized = (
        thermal_c - THERMAL_DISPLAY_MIN_C
    ) / (
        THERMAL_DISPLAY_MAX_C - THERMAL_DISPLAY_MIN_C
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2)


def make_annotator(name: str, **kwargs):
    return rep.AnnotatorRegistry.get_annotator(name, **kwargs)


# =============================================================================
# MAIN RECORDING COROUTINE
# =============================================================================


async def record_multimodal_sequence() -> None:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")

    for camera_path in (
        RGB_CAMERA,
        IR_LEFT_CAMERA,
        IR_RIGHT_CAMERA,
        DEPTH_CAMERA,
        THERMAL_CAMERA,
    ):
        require_camera(stage, camera_path)

    flame_imageable = require_imageable(stage, FLAME_PROXY)
    flame_imageable.MakeInvisible()

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        if not ALLOW_NONEMPTY_OUTPUT:
            raise RuntimeError(
                f"Output directory is not empty: {OUTPUT_ROOT}\n"
                "Change OUTPUT_ROOT or set ALLOW_NONEMPTY_OUTPUT=True."
            )

    frames_root = OUTPUT_ROOT / "frames"
    calibration_root = OUTPUT_ROOT / "calibration"
    frames_root.mkdir(parents=True, exist_ok=True)
    calibration_root.mkdir(parents=True, exist_ok=True)

    dataset_config = {
        "model_version": "thermal_mvp_v0_with_flow_warmup",
        "output_root": str(OUTPUT_ROOT),
        "number_of_frames": NUMBER_OF_FRAMES,
        "capture_fps": CAPTURE_FPS,
        "gemini_resolution": GEMINI_RESOLUTION,
        "thermal_resolution": THERMAL_RESOLUTION,
        "flow_warmup_seconds": FLOW_WARMUP_SECONDS,
        "flow_update_hz": FLOW_UPDATE_HZ,
        "flow_warmup_steps": FLOW_WARMUP_STEPS,
        "temperatures_c": {
            "ambient": AMBIENT_TEMP_C,
            "person": PERSON_TEMP_C,
            "flame": FLAME_TEMP_C,
        },
        "thermal_noise": {
            "temporal_std_c": THERMAL_TEMPORAL_NOISE_STD_C,
            "fixed_pattern_std_c": THERMAL_FIXED_PATTERN_STD_C,
            "random_seed": RANDOM_SEED,
        },
        "limitations": [
            "IR views are rendered camera outputs, not a calibrated physical IR model.",
            "Depth is renderer ground truth, not Gemini stereo reconstruction.",
            "Thermal values come from semantic class lookup.",
            "Flame uses a static geometry proxy.",
            "Flow smoke has no LWIR attenuation or emission effect.",
            "No emissivity, reflected radiance, or calibrated Lepton response.",
        ],
    }
    save_json(OUTPUT_ROOT / "dataset_config.json", dataset_config)

    # Stable fixed-pattern noise shared by every thermal frame.
    fixed_rng = np.random.default_rng(RANDOM_SEED)
    fixed_pattern_c = fixed_rng.normal(
        0.0,
        THERMAL_FIXED_PATTERN_STD_C,
        (THERMAL_RESOLUTION[1], THERMAL_RESOLUTION[0]),
    ).astype(np.float32)
    np.save(calibration_root / "thermal_fixed_pattern_c.npy", fixed_pattern_c)

    # -------------------------------------------------------------------------
    # Render products
    # -------------------------------------------------------------------------
    rp_rgb = rep.create.render_product(
        RGB_CAMERA,
        GEMINI_RESOLUTION,
        name="GeminiRgbRP",
    )
    rp_ir_left = rep.create.render_product(
        IR_LEFT_CAMERA,
        GEMINI_RESOLUTION,
        name="GeminiIrLeftRP",
    )
    rp_ir_right = rep.create.render_product(
        IR_RIGHT_CAMERA,
        GEMINI_RESOLUTION,
        name="GeminiIrRightRP",
    )
    rp_depth = rep.create.render_product(
        DEPTH_CAMERA,
        GEMINI_RESOLUTION,
        name="GeminiDepthRP",
    )
    rp_thermal = rep.create.render_product(
        THERMAL_CAMERA,
        THERMAL_RESOLUTION,
        name="ThermalRP",
    )

    render_products = [
        rp_rgb,
        rp_ir_left,
        rp_ir_right,
        rp_depth,
        rp_thermal,
    ]

    # -------------------------------------------------------------------------
    # Annotators: use a separate annotator instance for every render product.
    # -------------------------------------------------------------------------
    rgb_annotator = make_annotator("rgb")
    ir_left_annotator = make_annotator("rgb")
    ir_right_annotator = make_annotator("rgb")
    depth_annotator = make_annotator("distance_to_camera")

    rgb_semantic_annotator = make_annotator(
        "semantic_segmentation",
        init_params={"colorize": False},
    )
    thermal_semantic_annotator = make_annotator(
        "semantic_segmentation",
        init_params={"colorize": False},
    )

    rgb_camera_params = make_annotator("camera_params")
    ir_left_camera_params = make_annotator("camera_params")
    ir_right_camera_params = make_annotator("camera_params")
    depth_camera_params = make_annotator("camera_params")
    thermal_camera_params = make_annotator("camera_params")

    annotator_attachments = [
        (rgb_annotator, rp_rgb),
        (ir_left_annotator, rp_ir_left),
        (ir_right_annotator, rp_ir_right),
        (depth_annotator, rp_depth),
        (rgb_semantic_annotator, rp_rgb),
        (thermal_semantic_annotator, rp_thermal),
        (rgb_camera_params, rp_rgb),
        (ir_left_camera_params, rp_ir_left),
        (ir_right_camera_params, rp_ir_right),
        (depth_camera_params, rp_depth),
        (thermal_camera_params, rp_thermal),
    ]

    for annotator, render_product in annotator_attachments:
        annotator.attach(render_product)

    rep.orchestrator.set_capture_on_play(False)
    timeline = omni.timeline.get_timeline_interface()

    # Start from a stopped timeline. For a fully cold Flow state, reload the USD
    # scene before launching this script if the stage has already been played.
    timeline.stop()
    try:
        timeline.set_current_time(0.0)
    except Exception:
        # Some builds may not expose this method; stopping still leaves the
        # timeline ready for a new run.
        pass

    timeline.play()
    await omni.kit.app.get_app().next_update_async()

    try:
        # ---------------------------------------------------------------------
        # Flow warm-up: advance the simulation but do not save any frame.
        # ---------------------------------------------------------------------
        flame_imageable.MakeInvisible()

        print(
            f"Warming up Flow for {FLOW_WARMUP_SECONDS:.2f} s "
            f"({FLOW_WARMUP_STEPS} steps at {FLOW_UPDATE_HZ:.1f} Hz)..."
        )

        for warmup_step in range(FLOW_WARMUP_STEPS):
            await rep.orchestrator.step_async(
                delta_time=1.0 / FLOW_UPDATE_HZ,
                rt_subframes=WARMUP_RT_SUBFRAMES,
                pause_timeline=False,
            )

            completed = warmup_step + 1
            if completed == 1 or completed % 30 == 0 or completed == FLOW_WARMUP_STEPS:
                print(f"Flow warm-up: {completed}/{FLOW_WARMUP_STEPS}")

        warmup_end_time = float(timeline.get_current_time())
        print(
            "Flow warm-up complete at simulation time "
            f"{warmup_end_time:.4f} s. Starting saved capture."
        )

        # Capture camera calibration after the warm-up render products are live.
        save_json(
            calibration_root / "gemini_rgb_camera_params.json",
            rgb_camera_params.get_data(),
        )
        save_json(
            calibration_root / "gemini_ir_left_camera_params.json",
            ir_left_camera_params.get_data(),
        )
        save_json(
            calibration_root / "gemini_ir_right_camera_params.json",
            ir_right_camera_params.get_data(),
        )
        save_json(
            calibration_root / "gemini_depth_left_camera_params.json",
            depth_camera_params.get_data(),
        )
        save_json(
            calibration_root / "thermal_camera_params.json",
            thermal_camera_params.get_data(),
        )

        # ---------------------------------------------------------------------
        # Saved capture loop
        # ---------------------------------------------------------------------
        for frame_id in range(NUMBER_OF_FRAMES):
            # Frame 0 captures the exact developed state at the end of warm-up.
            # Subsequent frames advance at the requested capture rate.
            frame_delta_time = 0.0 if frame_id == 0 else 1.0 / CAPTURE_FPS

            # -------------------------------------------------------------
            # Pass A: genuine scene. Proxy hidden for Gemini modalities.
            # -------------------------------------------------------------
            flame_imageable.MakeInvisible()

            await rep.orchestrator.step_async(
                delta_time=frame_delta_time,
                rt_subframes=CAPTURE_RT_SUBFRAMES,
                pause_timeline=False,
            )

            simulation_time = float(timeline.get_current_time())
            sequence_time = frame_id / CAPTURE_FPS

            # Copy all buffers before running the second render pass.
            rgb = extract_array(rgb_annotator.get_data())
            ir_left = extract_array(ir_left_annotator.get_data())
            ir_right = extract_array(ir_right_annotator.get_data())
            depth_gt = extract_array(depth_annotator.get_data()).squeeze()

            rgb_semantic_output = rgb_semantic_annotator.get_data()
            rgb_semantic_ids = extract_array(rgb_semantic_output).squeeze()
            rgb_id_to_labels = (
                rgb_semantic_output.get("info", {}).get("idToLabels", {})
                if isinstance(rgb_semantic_output, dict)
                else {}
            )

            # -------------------------------------------------------------
            # Pass B: annotation-only thermal mask at the same time.
            # -------------------------------------------------------------
            flame_imageable.MakeVisible()

            await rep.orchestrator.step_async(
                delta_time=0.0,
                rt_subframes=THERMAL_MASK_RT_SUBFRAMES,
                pause_timeline=False,
            )

            thermal_semantic_output = thermal_semantic_annotator.get_data()
            thermal_semantic_ids = extract_array(
                thermal_semantic_output
            ).squeeze()
            thermal_id_to_labels = (
                thermal_semantic_output.get("info", {}).get("idToLabels", {})
                if isinstance(thermal_semantic_output, dict)
                else {}
            )

            # Restore the genuine scene before saving/advancing.
            flame_imageable.MakeInvisible()

            # -------------------------------------------------------------
            # Thermal MVP generation
            # -------------------------------------------------------------
            temperature_gt_c, thermal_counts = build_temperature_gt(
                thermal_semantic_ids,
                thermal_id_to_labels,
            )
            thermal_sensor_c = apply_thermal_sensor_mvp(
                temperature_gt_c,
                fixed_pattern_c,
                frame_id,
            )
            thermal_display = make_thermal_display(thermal_sensor_c)

            # -------------------------------------------------------------
            # Save one synchronized logical frame
            # -------------------------------------------------------------
            frame_dir = frames_root / f"{frame_id:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)

            save_rgb(frame_dir / "rgb.png", rgb)
            save_rgb(frame_dir / "ir_left.png", ir_left)
            save_rgb(frame_dir / "ir_right.png", ir_right)

            np.save(
                frame_dir / "depth_gt.npy",
                depth_gt.astype(np.float32),
            )
            np.save(
                frame_dir / "semantic_rgb.npy",
                rgb_semantic_ids.astype(np.uint32),
            )
            np.save(
                frame_dir / "thermal_semantic.npy",
                thermal_semantic_ids.astype(np.uint32),
            )
            np.save(
                frame_dir / "temperature_gt_celsius.npy",
                temperature_gt_c.astype(np.float32),
            )
            np.save(
                frame_dir / "thermal_sensor_celsius.npy",
                thermal_sensor_c.astype(np.float32),
            )
            Image.fromarray(thermal_display, mode="L").save(
                frame_dir / "thermal_display.png"
            )

            metadata = {
                "frame_id": frame_id,
                "simulation_time_seconds": simulation_time,
                "sequence_time_seconds": sequence_time,
                "warmup_end_time_seconds": warmup_end_time,
                "flow_warmup_seconds": FLOW_WARMUP_SECONDS,
                "capture_fps": CAPTURE_FPS,
                "thermal_model": "semantic_lookup_mvp",
                "thermal_counts": thermal_counts,
                "rgb_semantic_mapping": rgb_id_to_labels,
                "thermal_semantic_mapping": thermal_id_to_labels,
                "proxy_visible_in_saved_gemini_modalities": False,
                "proxy_used_for_thermal_mask": True,
                "smoke_thermal_effect": False,
                "depth_type": "renderer_ground_truth",
            }
            save_json(frame_dir / "metadata.json", metadata)

            print(
                f"Recorded {frame_id:06d}/{NUMBER_OF_FRAMES - 1:06d} | "
                f"sim={simulation_time:.4f}s | seq={sequence_time:.4f}s | "
                f"thermal={thermal_counts}"
            )

    finally:
        # Always leave the proxy hidden and the timeline stopped.
        flame_imageable.MakeInvisible()
        timeline.stop()

        for annotator, _ in annotator_attachments:
            try:
                annotator.detach()
            except Exception as exc:
                print(f"WARNING: could not detach annotator: {exc}")

        for render_product in render_products:
            try:
                render_product.destroy()
            except Exception:
                pass

        try:
            await rep.orchestrator.wait_until_complete_async()
        except Exception:
            pass

        # Refresh the UI after the timeline is stopped so the proxy-hidden state
        # is visible in the viewport without advancing the simulation.
        await omni.kit.app.get_app().next_update_async()

    print(f"Recording complete: {OUTPUT_ROOT}")


# Prevent accidental duplicate concurrent runs from the same execution.
asyncio.ensure_future(record_multimodal_sequence())


