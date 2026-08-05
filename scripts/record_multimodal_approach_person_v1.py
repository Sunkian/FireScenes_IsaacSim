"""Record a synchronized Gemini + Flow-based thermal sequence in Isaac Sim 5.1.

Run this file from Isaac Sim's Script Editor, not with the system Python.

This version does NOT use the FlameProxy cone. For each logical frame it:
  1. Captures Gemini RGB/IR/depth and the thermal-camera RGB view with Flow visible.
  2. Temporarily suppresses Flow rendering without advancing simulation time.
  3. Captures a second thermal-camera RGB view.
  4. Subtracts the two thermal-camera renders to estimate the dynamic Flow flame mask.
  5. Removes pixels labelled as floor from the flame mask.
  6. Moves the complete SensorRig closer to and above the person.
  7. Combines the remaining Flow flame mask with person semantics.

The Flow-render difference is an empirical MVP, not calibrated LWIR radiometry.
Flow smoke does not yet attenuate or emit in the generated thermal image.
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
from pxr import Gf, Usd, UsdGeom


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
SENSOR_RIG = "/World/SensorRig"
PERSON_PATH = "/World/Person/F_Medical_01"

# These are the exact Flow paths found in your scene.
FLOW_RENDER_ATTRIBUTES = (
    ("/World/Fire/Fire/flowRender/rayMarch", "colorScale"),
    ("/World/Fire/Fire/flowRender/rayMarch", "attenuation"),
    ("/World/Fire/Fire/flowOffscreen/colormap", "colorScale"),
    ("/World/Fire/Fire/flowOffscreen/shadow", "attenuation"),
)

# Change the final folder name for each run.
OUTPUT_ROOT = Path(
    "/home/apagnoux/Foresight_Isaac/"
    "output/multimodal_approach_person_001"
)

GEMINI_RESOLUTION = (640, 480)   # width, height
THERMAL_RESOLUTION = (160, 120)  # width, height

CAPTURE_FPS = 8.6
NUMBER_OF_FRAMES = 60

# Person-relative trajectory.
# The RGB optical centre finishes this far behind and above the person's
# world bounding-box centre. Orientation is unchanged in this version.
DISTANCE_BEHIND_PERSON_M = 0.70
HEIGHT_ABOVE_PERSON_M = 0.70

# Smooth motion stages: approach halfway, reach target, then pause.
PERSON_TRAJECTORY_SEGMENT_FRAMES = (25, 25, 9)

RESTORE_RIG_AFTER_RECORDING = True

# Flow warm-up: no frames are saved during this period.
FLOW_UPDATE_HZ = 60.0
FLOW_WARMUP_SECONDS = 3.0
FLOW_WARMUP_STEPS = int(round(FLOW_UPDATE_HZ * FLOW_WARMUP_SECONDS))

# Rendering quality.
CAPTURE_RT_SUBFRAMES = 4
FLOW_REFERENCE_RT_SUBFRAMES = 4
WARMUP_RT_SUBFRAMES = 1

# Surface thermal model.
AMBIENT_TEMP_C = 22.0
PERSON_TEMP_C = 34.0

# Dynamic Flow flame detection.
FLOW_FLAME_THRESHOLD = 60.0

FLOW_FLAME_MIN_TEMP_C = 450.0
FLOW_FLAME_MAX_TEMP_C = 1000.0

# Placeholder thermal sensor noise.
THERMAL_TEMPORAL_NOISE_STD_C = 0.05
THERMAL_FIXED_PATTERN_STD_C = 0.08
RANDOM_SEED = 42

# Display-only range. Flame intentionally saturates white.
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


def require_attribute(stage, prim_path: str, attribute_name: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid Flow prim path: {prim_path}")

    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        raise RuntimeError(
            f"Missing Flow attribute: {prim_path}.{attribute_name}"
        )
    return attribute


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


def semantic_class_mask(
    semantic_ids: np.ndarray,
    id_to_labels: dict,
    class_name: str,
) -> np.ndarray:
    """Return a boolean mask for one semantic class."""
    result = np.zeros(semantic_ids.shape, dtype=bool)
    expected = class_name.lower()

    for semantic_id_raw, labels in id_to_labels.items():
        semantic_id = int(semantic_id_raw)
        label_text = labels_to_text(labels)

        if expected in label_text:
            result |= semantic_ids == semantic_id

    return result


def smoothstep_01(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def get_translate_op(prim):
    """Return the existing translate op, or create one if needed."""
    xformable = UsdGeom.Xformable(prim)

    for operation in xformable.GetOrderedXformOps():
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return operation

    return xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)


def world_bbox_center(stage, prim_path: str) -> Gf.Vec3d:
    """Return the world-space centre of a prim hierarchy."""
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(f"Invalid person prim path: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )

    aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = aligned_box.GetMin()
    maximum = aligned_box.GetMax()

    return Gf.Vec3d(
        0.5 * (minimum[0] + maximum[0]),
        0.5 * (minimum[1] + maximum[1]),
        0.5 * (minimum[2] + maximum[2]),
    )


def compute_person_target_offset(
    stage,
    original_rig_position,
) -> tuple[Gf.Vec3d, dict]:
    """
    Compute the SensorRig translation offset needed to place the RGB camera
    behind and above the person while preserving the current camera rotation.
    """
    rgb_prim = stage.GetPrimAtPath(RGB_CAMERA)
    rig_prim = stage.GetPrimAtPath(SENSOR_RIG)

    if not rgb_prim.IsValid():
        raise RuntimeError(f"Invalid RGB camera path: {RGB_CAMERA}")

    rgb_world = UsdGeom.Xformable(
        rgb_prim
    ).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    current_rgb_position = rgb_world.ExtractTranslation()
    person_center = world_bbox_center(stage, PERSON_PATH)

    rgb_forward_world = rgb_world.TransformDir(
        Gf.Vec3d(0.0, 0.0, -1.0)
    ).GetNormalized()

    world_up = Gf.Vec3d(0.0, 0.0, 1.0)

    target_rgb_position = (
        person_center
        - DISTANCE_BEHIND_PERSON_M * rgb_forward_world
        + HEIGHT_ABOVE_PERSON_M * world_up
    )

    world_displacement = target_rgb_position - current_rgb_position

    rig_parent = rig_prim.GetParent()
    parent_world = UsdGeom.Xformable(
        rig_parent
    ).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    parent_world_inverse = parent_world.GetInverse()

    local_displacement = parent_world_inverse.TransformDir(
        world_displacement
    )

    target_rig_position = type(original_rig_position)(
        float(original_rig_position[0]) + float(local_displacement[0]),
        float(original_rig_position[1]) + float(local_displacement[1]),
        float(original_rig_position[2]) + float(local_displacement[2]),
    )

    details = {
        "person_center_world_m": [
            float(person_center[0]),
            float(person_center[1]),
            float(person_center[2]),
        ],
        "current_rgb_position_world_m": [
            float(current_rgb_position[0]),
            float(current_rgb_position[1]),
            float(current_rgb_position[2]),
        ],
        "target_rgb_position_world_m": [
            float(target_rgb_position[0]),
            float(target_rgb_position[1]),
            float(target_rgb_position[2]),
        ],
        "target_rig_local_translation_m": [
            float(target_rig_position[0]),
            float(target_rig_position[1]),
            float(target_rig_position[2]),
        ],
    }

    return target_rig_position, details


def validate_waypoint_configuration() -> None:
    if len(PERSON_TRAJECTORY_SEGMENT_FRAMES) != 3:
        raise RuntimeError(
            "PERSON_TRAJECTORY_SEGMENT_FRAMES must contain three stages."
        )

    if any(
        frame_count <= 0
        for frame_count in PERSON_TRAJECTORY_SEGMENT_FRAMES
    ):
        raise RuntimeError("Every trajectory stage must contain at least one frame.")

    expected_frames = 1 + sum(PERSON_TRAJECTORY_SEGMENT_FRAMES)

    if NUMBER_OF_FRAMES != expected_frames:
        raise RuntimeError(
            f"NUMBER_OF_FRAMES must be {expected_frames}, "
            f"but is {NUMBER_OF_FRAMES}."
        )


def set_rig_person_motion(
    translate_op,
    original_position,
    target_position,
    frame_id: int,
):
    """Move smoothly from the start to a point above and near the person."""
    stage_1, stage_2, stage_3 = PERSON_TRAJECTORY_SEGMENT_FRAMES

    if frame_id <= stage_1:
        segment_index = 0
        segment_progress = smoothstep_01(frame_id / stage_1)
        fraction = 0.5 * segment_progress

    elif frame_id <= stage_1 + stage_2:
        segment_index = 1
        local_frame = frame_id - stage_1
        segment_progress = smoothstep_01(local_frame / stage_2)
        fraction = 0.5 + 0.5 * segment_progress

    else:
        segment_index = 2
        segment_progress = 1.0
        fraction = 1.0

    vector_type = type(original_position)

    new_position = vector_type(
        float(original_position[0])
        + (
            float(target_position[0]) - float(original_position[0])
        )
        * fraction,
        float(original_position[1])
        + (
            float(target_position[1]) - float(original_position[1])
        )
        * fraction,
        float(original_position[2])
        + (
            float(target_position[2]) - float(original_position[2])
        )
        * fraction,
    )

    translate_op.Set(new_position)

    return {
        "segment_index": segment_index,
        "segment_progress": segment_progress,
        "global_progress": fraction,
        "local_position": new_position,
    }


def world_pose(stage, prim_path: str) -> dict:
    """Return a prim's world transform and position for frame metadata."""
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(f"Invalid pose prim path: {prim_path}")

    matrix = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    position = matrix.ExtractTranslation()

    return {
        "position_m": [
            float(position[0]),
            float(position[1]),
            float(position[2]),
        ],
        "matrix_row_major": [
            float(matrix[row][column])
            for row in range(4)
            for column in range(4)
        ],
    }


def to_jsonable(value: Any) -> Any:
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

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def build_surface_temperature_gt(
    semantic_ids: np.ndarray,
    id_to_labels: dict,
) -> tuple[np.ndarray, dict[str, int]]:
    """Create ambient/person temperatures; Flow flame is added separately."""
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

    return temperature_gt, counts


def extract_flow_flame(
    image_with_flow: np.ndarray,
    image_without_flow: np.ndarray,
    excluded_surface_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the visible Flow flame from positive rendered-light difference."""
    with_flow = image_with_flow[:, :, :3].astype(np.float32)
    without_flow = image_without_flow[:, :, :3].astype(np.float32)

    positive_rgb = np.maximum(with_flow - without_flow, 0.0)

    emission_score = (
        0.2126 * positive_rgb[:, :, 0]
        + 0.7152 * positive_rgb[:, :, 1]
        + 0.0722 * positive_rgb[:, :, 2]
    ).astype(np.float32)

    flame_mask = emission_score >= FLOW_FLAME_THRESHOLD

    # Reject Flow illumination/reflections on semantically labelled surfaces.
    # This remains valid when the rig moves because it follows the floor pixels.
    if excluded_surface_mask is not None:
        if excluded_surface_mask.shape != flame_mask.shape:
            raise RuntimeError(
                "Excluded-surface mask shape does not match thermal image: "
                f"{excluded_surface_mask.shape} vs {flame_mask.shape}"
            )
        flame_mask &= ~excluded_surface_mask.astype(bool)

    denominator = max(255.0 - FLOW_FLAME_THRESHOLD, 1.0)
    normalized_strength = np.clip(
        (emission_score - FLOW_FLAME_THRESHOLD) / denominator,
        0.0,
        1.0,
    )

    flame_temperature_c = (
        FLOW_FLAME_MIN_TEMP_C
        + normalized_strength
        * (FLOW_FLAME_MAX_TEMP_C - FLOW_FLAME_MIN_TEMP_C)
    ).astype(np.float32)

    # Zero outside the detected flame so the saved diagnostic array is clear.
    flame_temperature_c[~flame_mask] = 0.0

    return flame_mask, emission_score, flame_temperature_c


def make_difference_preview(
    image_with_flow: np.ndarray,
    image_without_flow: np.ndarray,
) -> np.ndarray:
    difference = np.maximum(
        image_with_flow[:, :, :3].astype(np.int16)
        - image_without_flow[:, :, :3].astype(np.int16),
        0,
    )
    return np.clip(difference * 4, 0, 255).astype(np.uint8)


def apply_thermal_sensor_mvp(
    temperature_gt_c: np.ndarray,
    fixed_pattern_c: np.ndarray,
    frame_id: int,
) -> np.ndarray:
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
    print("Recorder task started.")
    validate_waypoint_configuration()
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

    rig_prim = stage.GetPrimAtPath(SENSOR_RIG)
    if not rig_prim.IsValid():
        raise RuntimeError(f"Invalid SensorRig prim path: {SENSOR_RIG}")

    rig_translate_op = get_translate_op(rig_prim)
    original_rig_position = rig_translate_op.Get()

    if original_rig_position is None:
        original_rig_position = Gf.Vec3d(0.0, 0.0, 0.0)
        rig_translate_op.Set(original_rig_position)

    print("Original SensorRig translation:", original_rig_position)

    target_rig_position, person_target_details = (
        compute_person_target_offset(
            stage,
            original_rig_position,
        )
    )

    print("Target SensorRig translation:", target_rig_position)
    print(
        "Target RGB position:",
        person_target_details["target_rgb_position_world_m"],
    )

    flow_attributes = []
    for prim_path, attribute_name in FLOW_RENDER_ATTRIBUTES:
        attribute = require_attribute(stage, prim_path, attribute_name)
        flow_attributes.append((attribute, attribute.Get()))

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
        "model_version": "thermal_flow_render_difference_v1_5_approach_person",
        "output_root": str(OUTPUT_ROOT),
        "number_of_frames": NUMBER_OF_FRAMES,
        "capture_fps": CAPTURE_FPS,
        "gemini_resolution": GEMINI_RESOLUTION,
        "thermal_resolution": THERMAL_RESOLUTION,
        "flow_warmup_seconds": FLOW_WARMUP_SECONDS,
        "flow_update_hz": FLOW_UPDATE_HZ,
        "flow_warmup_steps": FLOW_WARMUP_STEPS,
        "rig_motion": {
            "enabled": True,
            "trajectory": "person-relative smoothstep translation",
            "person_path": PERSON_PATH,
            "distance_behind_person_m": DISTANCE_BEHIND_PERSON_M,
            "height_above_person_m": HEIGHT_ABOVE_PERSON_M,
            "segment_frames": PERSON_TRAJECTORY_SEGMENT_FRAMES,
            "moves_complete_sensor_rig": True,
            "rotation_enabled": False,
        },
        "flow_flame_detection": {
            "threshold": FLOW_FLAME_THRESHOLD,
            "excluded_semantic_class": "floor",
            "minimum_temperature_c": FLOW_FLAME_MIN_TEMP_C,
            "maximum_temperature_c": FLOW_FLAME_MAX_TEMP_C,
            "method": (
                "positive RGB luminance difference with "
                "thermal-camera floor semantic-mask rejection"
            ),
        },
        "temperatures_c": {
            "ambient": AMBIENT_TEMP_C,
            "person": PERSON_TEMP_C,
        },
        "thermal_noise": {
            "temporal_std_c": THERMAL_TEMPORAL_NOISE_STD_C,
            "fixed_pattern_std_c": THERMAL_FIXED_PATTERN_STD_C,
            "random_seed": RANDOM_SEED,
        },
        "limitations": [
            "IR views are rendered camera outputs, not a calibrated physical IR model.",
            "Depth is renderer ground truth, not Gemini stereo reconstruction.",
            "Person temperature comes from semantic class lookup.",
            "Flame temperature is estimated from rendered Flow brightness difference.",
            "The Flow scalar temperature field is not read directly.",
            (
                "Floor reflections are rejected using the thermal-camera "
                "semantic floor mask; unlabelled reflective surfaces may remain."
            ),
            "Flow smoke has no LWIR attenuation or emission effect.",
            "No emissivity, reflected radiance, or calibrated Lepton response.",
        ],
    }
    save_json(OUTPUT_ROOT / "dataset_config.json", dataset_config)

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
    # Annotators
    # -------------------------------------------------------------------------
    rgb_annotator = make_annotator("rgb")
    ir_left_annotator = make_annotator("rgb")
    ir_right_annotator = make_annotator("rgb")
    depth_annotator = make_annotator("distance_to_camera")
    thermal_rgb_annotator = make_annotator("rgb")

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
        (thermal_rgb_annotator, rp_thermal),
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

    timeline.stop()
    try:
        timeline.set_current_time(0.0)
    except Exception:
        pass

    timeline.play()
    await omni.kit.app.get_app().next_update_async()

    try:
        # ---------------------------------------------------------------------
        # Flow warm-up: advance but save nothing.
        # ---------------------------------------------------------------------
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
            frame_delta_time = 0.0 if frame_id == 0 else 1.0 / CAPTURE_FPS

            rig_motion_state = set_rig_person_motion(
                rig_translate_op,
                original_rig_position,
                target_rig_position,
                frame_id,
            )
            rig_local_position = rig_motion_state["local_position"]

            # Ensure Flow rendering is restored before the real capture.
            for attribute, original_value in flow_attributes:
                attribute.Set(original_value)

            # -------------------------------------------------------------
            # Pass A: real scene with Flow visible.
            # -------------------------------------------------------------
            await rep.orchestrator.step_async(
                delta_time=frame_delta_time,
                rt_subframes=CAPTURE_RT_SUBFRAMES,
                pause_timeline=False,
            )

            simulation_time = float(timeline.get_current_time())
            sequence_time = frame_id / CAPTURE_FPS

            rgb = extract_array(rgb_annotator.get_data())
            ir_left = extract_array(ir_left_annotator.get_data())
            ir_right = extract_array(ir_right_annotator.get_data())
            depth_gt = extract_array(depth_annotator.get_data()).squeeze()
            thermal_with_flow = extract_array(thermal_rgb_annotator.get_data())

            rgb_semantic_output = rgb_semantic_annotator.get_data()
            rgb_semantic_ids = extract_array(rgb_semantic_output).squeeze()
            rgb_id_to_labels = (
                rgb_semantic_output.get("info", {}).get("idToLabels", {})
                if isinstance(rgb_semantic_output, dict)
                else {}
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

            # -------------------------------------------------------------
            # Pass B: same simulation time, Flow rendering suppressed.
            # -------------------------------------------------------------
            try:
                for attribute, _ in flow_attributes:
                    attribute.Set(0.0)

                await rep.orchestrator.step_async(
                    delta_time=0.0,
                    rt_subframes=FLOW_REFERENCE_RT_SUBFRAMES,
                    pause_timeline=False,
                )

                thermal_without_flow = extract_array(
                    thermal_rgb_annotator.get_data()
                )

            finally:
                for attribute, original_value in flow_attributes:
                    attribute.Set(original_value)

            # -------------------------------------------------------------
            # Build thermal data from person semantics + Flow difference.
            # -------------------------------------------------------------
            temperature_gt_c, thermal_counts = build_surface_temperature_gt(
                thermal_semantic_ids,
                thermal_id_to_labels,
            )

            floor_mask = semantic_class_mask(
                thermal_semantic_ids,
                thermal_id_to_labels,
                "floor",
            )

            (
                flow_flame_mask,
                flow_emission_score,
                flow_flame_temperature_c,
            ) = extract_flow_flame(
                thermal_with_flow,
                thermal_without_flow,
                excluded_surface_mask=floor_mask,
            )

            temperature_gt_c[flow_flame_mask] = np.maximum(
                temperature_gt_c[flow_flame_mask],
                flow_flame_temperature_c[flow_flame_mask],
            )
            thermal_counts["flame"] = int(flow_flame_mask.sum())

            thermal_sensor_c = apply_thermal_sensor_mvp(
                temperature_gt_c,
                fixed_pattern_c,
                frame_id,
            )
            thermal_display = make_thermal_display(thermal_sensor_c)
            difference_preview = make_difference_preview(
                thermal_with_flow,
                thermal_without_flow,
            )

            # -------------------------------------------------------------
            # Save one synchronized logical frame.
            # -------------------------------------------------------------
            frame_dir = frames_root / f"{frame_id:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)

            save_rgb(frame_dir / "rgb.png", rgb)
            save_rgb(frame_dir / "ir_left.png", ir_left)
            save_rgb(frame_dir / "ir_right.png", ir_right)
            save_rgb(
                frame_dir / "thermal_flow_visible_rgb.png",
                thermal_with_flow,
            )
            save_rgb(
                frame_dir / "thermal_flow_suppressed_rgb.png",
                thermal_without_flow,
            )
            save_rgb(
                frame_dir / "flow_difference_preview.png",
                difference_preview,
            )

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
                frame_dir / "flow_flame_mask.npy",
                flow_flame_mask.astype(np.uint8),
            )
            np.save(
                frame_dir / "floor_semantic_mask.npy",
                floor_mask.astype(np.uint8),
            )
            np.save(
                frame_dir / "flow_emission_score.npy",
                flow_emission_score.astype(np.float32),
            )
            np.save(
                frame_dir / "flow_flame_temperature_celsius.npy",
                flow_flame_temperature_c.astype(np.float32),
            )
            np.save(
                frame_dir / "temperature_gt_celsius.npy",
                temperature_gt_c.astype(np.float32),
            )
            np.save(
                frame_dir / "thermal_sensor_celsius.npy",
                thermal_sensor_c.astype(np.float32),
            )

            Image.fromarray(
                flow_flame_mask.astype(np.uint8) * 255,
                mode="L",
            ).save(frame_dir / "flow_flame_mask_preview.png")

            Image.fromarray(
                floor_mask.astype(np.uint8) * 255,
                mode="L",
            ).save(frame_dir / "floor_semantic_mask_preview.png")

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
                "thermal_model": "flow_render_difference_v1_2_floor_semantic",
                "flow_flame_threshold": FLOW_FLAME_THRESHOLD,
                "excluded_semantic_class": "floor",
                "floor_pixels": int(floor_mask.sum()),
                "flow_flame_min_temp_c": FLOW_FLAME_MIN_TEMP_C,
                "flow_flame_max_temp_c": FLOW_FLAME_MAX_TEMP_C,
                "thermal_counts": thermal_counts,
                "rgb_semantic_mapping": rgb_id_to_labels,
                "thermal_semantic_mapping": thermal_id_to_labels,
                "proxy_used_for_thermal_mask": False,
                "flow_used_directly": True,
                "flow_source": "positive rendered RGB difference minus floor semantic mask",
                "flow_scalar_temperature_read_directly": False,
                "smoke_thermal_effect": False,
                "depth_type": "renderer_ground_truth",
                "rig_motion_progress": rig_motion_state["global_progress"],
                "rig_motion_segment_index": rig_motion_state["segment_index"],
                "rig_motion_segment_progress": (
                    rig_motion_state["segment_progress"]
                ),
                "person_target": person_target_details,
                "rig_local_translation_m": [
                    float(rig_local_position[0]),
                    float(rig_local_position[1]),
                    float(rig_local_position[2]),
                ],
                "sensor_rig_world_pose": world_pose(stage, SENSOR_RIG),
                "rgb_camera_world_pose": world_pose(stage, RGB_CAMERA),
                "thermal_camera_world_pose": world_pose(stage, THERMAL_CAMERA),
            }
            save_json(frame_dir / "metadata.json", metadata)

            print(
                f"Recorded {frame_id:06d}/{NUMBER_OF_FRAMES - 1:06d} | "
                f"sim={simulation_time:.4f}s | seq={sequence_time:.4f}s | "
                f"thermal={thermal_counts}"
            )

    finally:
        # Always restore the original Flow rendering values.
        for attribute, original_value in flow_attributes:
            try:
                attribute.Set(original_value)
            except Exception as exc:
                print(f"WARNING: could not restore Flow attribute: {exc}")

        if RESTORE_RIG_AFTER_RECORDING:
            try:
                rig_translate_op.Set(original_rig_position)
                await omni.kit.app.get_app().next_update_async()
                print("SensorRig restored to its original position.")
            except Exception as exc:
                print(f"WARNING: could not restore SensorRig: {exc}")

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

        await omni.kit.app.get_app().next_update_async()

    print(f"Recording complete: {OUTPUT_ROOT}")


# Launch asynchronously from Isaac Sim's Script Editor and report failures.
def _report_task_result(task):
    try:
        task.result()
    except asyncio.CancelledError:
        print("Recorder task was cancelled.")
    except Exception:
        import traceback
        print("Recorder failed with this error:")
        traceback.print_exc()


_recording_task = asyncio.ensure_future(record_multimodal_sequence())
_recording_task.add_done_callback(_report_task_result)
print("Recorder scheduled.")

