import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom


CAMERA_PATH = "/World/SensorRig/ThermalCamera"
FLAME_PROXY_PATH = "/World/ThermalProxies/FlameProxy"

OUTPUT_DIR = Path(
    "/home/apagnoux/Foresight_Isaac/output/thermal_mvp"
)

# Width, height
RESOLUTION = (160, 120)

# Initial approximations only.
TEMPERATURES_C = {
    "background": 22.0,
    "person": 34.0,
    "flame": 850.0,
}

NOISE_STD_C = 0.05
RANDOM_SEED = 42


def label_to_text(label: Any) -> str:
    """Normalize Replicator semantic-label structures."""
    if isinstance(label, dict):
        return " ".join(
            f"{key} {value}" for key, value in label.items()
        ).lower()

    if isinstance(label, (list, tuple)):
        return " ".join(str(value) for value in label).lower()

    return str(label).lower()


def fixed_display(
    temperature_c: np.ndarray,
    minimum_c: float = 20.0,
    maximum_c: float = 45.0,
) -> np.ndarray:
    """
    Produce an 8-bit display image.

    Values above maximum_c saturate. This intentionally makes fire and
    flames white while retaining contrast around human temperatures.
    """
    normalized = (
        (temperature_c - minimum_c)
        / (maximum_c - minimum_c)
    )
    normalized = np.clip(normalized, 0.0, 1.0)

    return np.round(normalized * 255.0).astype(np.uint8)


def percentile_agc(temperature_c: np.ndarray) -> np.ndarray:
    """
    Simple display-only automatic gain control.

    Never use this image as the only radiometric dataset output.
    """
    low = float(np.percentile(temperature_c, 1.0))
    high = float(np.percentile(temperature_c, 99.0))

    if high - low < 0.5:
        high = low + 0.5

    normalized = (temperature_c - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)

    return np.round(normalized * 255.0).astype(np.uint8)


async def capture_thermal_mvp() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage = omni.usd.get_context().get_stage()

    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not camera_prim.IsValid():
        raise RuntimeError(f"Invalid camera path: {CAMERA_PATH}")

    if not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(
            f"{CAMERA_PATH} is not a USD Camera prim"
        )

    # Temporarily expose the flame proxy for semantic capture.
    flame_proxy = stage.GetPrimAtPath(FLAME_PROXY_PATH)
    proxy_was_found = flame_proxy.IsValid()

    if proxy_was_found:
        UsdGeom.Imageable(flame_proxy).MakeVisible()
    else:
        print(
            "WARNING: Flame proxy not found. "
            "No flame pixels will be generated."
        )

    render_product = None
    semantic_annotator = None

    try:
        rep.orchestrator.set_capture_on_play(False)

        render_product = rep.create.render_product(
            CAMERA_PATH,
            RESOLUTION,
            name="ThermalMvpRenderProduct",
        )

        semantic_annotator = rep.annotators.get(
            "semantic_segmentation",
            init_params={"colorize": False},
        )
        semantic_annotator.attach(render_product)

        # Produce one synchronized rendered frame.
        await rep.orchestrator.step_async()

        output = semantic_annotator.get_data()

        if not isinstance(output, dict):
            raise RuntimeError(
                "Unexpected semantic annotator output"
            )

        semantic_ids = np.asarray(
            output["data"],
            dtype=np.uint32,
        ).squeeze()

        if semantic_ids.shape != (120, 160):
            raise RuntimeError(
                "Unexpected segmentation shape: "
                f"{semantic_ids.shape}"
            )

        id_to_labels = output.get(
            "info",
            {},
        ).get(
            "idToLabels",
            {},
        )

        print("Semantic IDs and labels:")
        for semantic_id, labels in id_to_labels.items():
            print(semantic_id, "->", labels)

        # All unlabelled pixels start at ambient temperature.
        temperature_c = np.full(
            semantic_ids.shape,
            TEMPERATURES_C["background"],
            dtype=np.float32,
        )

        class_pixel_counts = {
            "person": 0,
            "flame": 0,
        }

        for semantic_id_raw, labels in id_to_labels.items():
            semantic_id = int(semantic_id_raw)
            text = label_to_text(labels)
            mask = semantic_ids == semantic_id

            if "person" in text:
                temperature_c[mask] = TEMPERATURES_C["person"]
                class_pixel_counts["person"] += int(mask.sum())

            elif "flame" in text:
                temperature_c[mask] = TEMPERATURES_C["flame"]
                class_pixel_counts["flame"] += int(mask.sum())


        # Initial thermal-sensor noise approximation.
        rng = np.random.default_rng(RANDOM_SEED)
        noise = rng.normal(
            0.0,
            NOISE_STD_C,
            temperature_c.shape,
        ).astype(np.float32)

        thermal_raw_c = temperature_c + noise

        display_fixed = fixed_display(thermal_raw_c)
        display_agc = percentile_agc(thermal_raw_c)

        np.save(
            OUTPUT_DIR / "thermal_raw_celsius_0000.npy",
            thermal_raw_c,
        )

        np.save(
            OUTPUT_DIR / "semantic_ids_0000.npy",
            semantic_ids,
        )

        Image.fromarray(display_fixed).save(
            OUTPUT_DIR / "thermal_display_fixed_0000.png"
        )

        Image.fromarray(display_agc).save(
            OUTPUT_DIR / "thermal_display_agc_0000.png"
        )

        metadata = {
            "camera_path": CAMERA_PATH,
            "resolution": {
                "width": RESOLUTION[0],
                "height": RESOLUTION[1],
            },
            "temperature_units": "degrees Celsius",
            "temperature_model": TEMPERATURES_C,
            "noise_std_c": NOISE_STD_C,
            "semantic_mapping": {
                str(key): value
                for key, value in id_to_labels.items()
            },
            "class_pixel_counts": class_pixel_counts,
            "model_version": "thermal_mvp_v0",
            "limitations": [
                "Class-based temperatures",
                "No emissivity",
                "No reflected radiation",
                "No smoke attenuation",
                "No hot-smoke emission",
                "Approximate flame proxy",
                "No calibrated Lepton response",
            ],
        }

        with (
            OUTPUT_DIR / "thermal_metadata_0000.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                indent=2,
                default=str,
            )

        print("Thermal frame completed.")
        print("Output:", OUTPUT_DIR)
        print("Pixel counts:", class_pixel_counts)
        print(
            "Temperature range:",
            float(thermal_raw_c.min()),
            "to",
            float(thermal_raw_c.max()),
            "°C",
        )

    finally:
        if semantic_annotator is not None:
            semantic_annotator.detach()

        if render_product is not None:
            try:
                render_product.destroy()
            except AttributeError:
                pass

        # Hide the proxy again so it does not affect normal RGB captures.
        if proxy_was_found:
            UsdGeom.Imageable(flame_proxy).MakeInvisible()


asyncio.ensure_future(capture_thermal_mvp())
