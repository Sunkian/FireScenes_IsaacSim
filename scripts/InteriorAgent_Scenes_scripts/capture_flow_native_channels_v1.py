"""
capture_flow_native_channels_v1.py

Isaac Sim 5.1 / omni.flowusd 107.1.8

Capture the native Flow channels needed for the thermal pipeline:

FIRE layer:
    burn        -> flame mask
    temperature -> thermal contribution

DENSE SMOKE layer:
    smoke       -> smoke mask
    temperature -> thermal contribution

This script is designed to be safe in Script Editor:
- no timeline.stop()
- no render-product teardown
- native NanoVDB readback only
- one Flow exporter enabled at a time
- writes progress continuously to report.txt

Run AFTER:
1. configure_flow_sources_1m_channels_SAFE.py
2. save/reopen the scene
3. let Flow develop for ~3-5 s

Output:
    /scratch/apagnoux/flow_native_channels_v1/
        fire_burn.nvdb
        fire_temperature.nvdb
        smoke_density.nvdb
        smoke_temperature.nvdb
        report.txt
"""

from __future__ import annotations

import asyncio
import hashlib
import traceback
from pathlib import Path

import numpy as np

import omni.graph.core as og
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd


# =============================================================================
# CONFIG
# =============================================================================

FIRE_EXPORT = (
    "/Root/World/Meshes/hazards/Fire/"
    "flowSimulate/nanoVdbExport"
)

SMOKE_EXPORT = (
    "/Root/World/Meshes/hazards/DenseSmoke/"
    "flowSimulate/nanoVdbExport"
)

GRAPH_PATH = "/Root/World/FlowNativeChannelReadbackGraph"
NODE_NAME = "NanoVdbReadback"

OUT = Path("/scratch/apagnoux/flow_native_channels_v1")
OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "report.txt"

FLOW_HZ = 60.0
MAX_WAIT_SECONDS = 6.0
MAX_WAIT_STEPS = int(FLOW_HZ * MAX_WAIT_SECONDS)

RT_SUBFRAMES = 2
SETTLE_UPDATES = 12


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
# HELPERS
# =============================================================================

def require_prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {path}")
    return prim


def require_attr(prim, name):
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        raise RuntimeError(
            f"Missing attribute '{name}' on {prim.GetPath()}"
        )
    return attr


def configure_export(
    export_prim,
    *,
    readback_enabled,
    burn_enabled,
    smoke_enabled,
    temperature_enabled,
):
    require_attr(export_prim, "enabled").Set(True)
    require_attr(export_prim, "readbackOpenVDB").Set(False)
    require_attr(export_prim, "readbackEnabled").Set(readback_enabled)

    require_attr(export_prim, "burnEnabled").Set(burn_enabled)
    require_attr(export_prim, "smokeEnabled").Set(smoke_enabled)
    require_attr(export_prim, "temperatureEnabled").Set(temperature_enabled)

    for name in (
        "fuelEnabled",
        "velocityEnabled",
        "divergenceEnabled",
    ):
        attr = export_prim.GetAttribute(name)
        if attr.IsValid():
            attr.Set(False)

    for name in (
        "readbackClearOnStart",
        "readbackClearOnStop",
    ):
        attr = export_prim.GetAttribute(name)
        if attr.IsValid():
            attr.Set(False)


async def settle(n=SETTLE_UPDATES):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


def ensure_readback_node(stage):
    node_path = f"{GRAPH_PATH}/{NODE_NAME}"

    if not stage.GetPrimAtPath(node_path).IsValid():
        log("Creating readback node:", node_path)

        og.Controller.edit(
            {
                "graph_path": GRAPH_PATH,
                "evaluator_name": "push",
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    (
                        NODE_NAME,
                        "omni.flowusd.FlowNanoVdbReadback",
                    )
                ]
            },
        )

    node = og.Controller.node(node_path)

    if node is None or not node.is_valid():
        raise RuntimeError(
            f"Invalid OmniGraph node: {node_path}"
        )

    attrs = {}

    for channel in ("burn", "smoke", "temperature"):
        attr = node.get_attribute(f"outputs:{channel}")

        if attr is None or not attr.is_valid():
            raise RuntimeError(
                f"Missing outputs:{channel} on FlowNanoVdbReadback."
            )

        attrs[channel] = attr

    return attrs


def read_words(attr):
    value = og.Controller.get(attr)

    if value is None:
        return np.empty(0, dtype=np.uint32)

    return np.asarray(
        value,
        dtype=np.uint32,
    ).reshape(-1).copy()


def words_to_bytes(words):
    return words.astype(
        "<u4",
        copy=False,
    ).tobytes()


def sha1_words(words):
    if words.size == 0:
        return None

    return hashlib.sha1(
        words_to_bytes(words)
    ).hexdigest()


def save_buffer(filename, words):
    path = OUT / filename
    path.write_bytes(
        words_to_bytes(words)
    )

    log(
        "Saved",
        filename,
        "| words=",
        words.size,
        "| bytes=",
        words.size * 4,
        "| sha1=",
        sha1_words(words),
    )

    return path


async def wait_for_channels(
    *,
    label,
    exporter,
    other_exporter,
    readback_attrs,
    wanted_channels,
    export_config,
    timeline,
):
    log("")
    log("=" * 72)
    log("CAPTURE", label.upper())
    log("=" * 72)

    # Snapshot stale buffers before switching exporters.
    baseline = {
        ch: sha1_words(
            read_words(
                readback_attrs[ch]
            )
        )
        for ch in wanted_channels
    }

    log("Baseline signatures:", baseline)

    # Disable both first.
    configure_export(
        other_exporter,
        readback_enabled=False,
        burn_enabled=False,
        smoke_enabled=False,
        temperature_enabled=False,
    )

    configure_export(
        exporter,
        readback_enabled=False,
        **export_config,
    )

    await settle()

    # Enable only target exporter.
    configure_export(
        exporter,
        readback_enabled=True,
        **export_config,
    )

    await settle()

    timeline.play()

    fresh = {}

    for step in range(MAX_WAIT_STEPS):

        await rep.orchestrator.step_async(
            delta_time=1.0 / FLOW_HZ,
            rt_subframes=RT_SUBFRAMES,
            pause_timeline=False,
        )

        await omni.kit.app.get_app().next_update_async()

        for ch in wanted_channels:
            if ch in fresh:
                continue

            words = read_words(
                readback_attrs[ch]
            )

            if words.size == 0:
                continue

            sig = sha1_words(words)

            if sig != baseline[ch]:
                fresh[ch] = words

                log(
                    f"{label} {ch}: fresh after "
                    f"{step + 1} steps "
                    f"({(step + 1)/FLOW_HZ:.3f}s), "
                    f"{words.size * 4} bytes"
                )

        if len(fresh) == len(wanted_channels):
            timeline.pause()
            await settle(8)
            return fresh

        if (step + 1) % 60 == 0:
            log(
                "waiting...",
                f"{(step + 1)/FLOW_HZ:.1f}s",
                "| captured:",
                sorted(fresh.keys()),
            )

    timeline.pause()
    await settle(8)

    missing = [
        ch
        for ch in wanted_channels
        if ch not in fresh
    ]

    raise RuntimeError(
        f"{label}: no fresh native buffer for: {missing}"
    )


# =============================================================================
# MAIN
# =============================================================================

async def run():
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError(
            "No USD stage is open."
        )

    fire_export = require_prim(
        stage,
        FIRE_EXPORT,
    )

    smoke_export = require_prim(
        stage,
        SMOKE_EXPORT,
    )

    readback_attrs = ensure_readback_node(
        stage
    )

    timeline = omni.timeline.get_timeline_interface()

    try:
        log(
            "FLOW NATIVE CHANNEL CAPTURE V1"
        )

        try:
            timeline.pause()
        except Exception:
            pass

        await settle()

        # -----------------------------------------------------------------
        # FIRE: burn + temperature
        # -----------------------------------------------------------------

        fire = await wait_for_channels(
            label="fire",
            exporter=fire_export,
            other_exporter=smoke_export,
            readback_attrs=readback_attrs,
            wanted_channels=[
                "burn",
                "temperature",
            ],
            export_config={
                "burn_enabled": True,
                "smoke_enabled": False,
                "temperature_enabled": True,
            },
            timeline=timeline,
        )

        save_buffer(
            "fire_burn.nvdb",
            fire["burn"],
        )

        save_buffer(
            "fire_temperature.nvdb",
            fire["temperature"],
        )

        # -----------------------------------------------------------------
        # SMOKE: smoke + temperature
        # -----------------------------------------------------------------

        smoke = await wait_for_channels(
            label="smoke",
            exporter=smoke_export,
            other_exporter=fire_export,
            readback_attrs=readback_attrs,
            wanted_channels=[
                "smoke",
                "temperature",
            ],
            export_config={
                "burn_enabled": False,
                "smoke_enabled": True,
                "temperature_enabled": True,
            },
            timeline=timeline,
        )

        save_buffer(
            "smoke_density.nvdb",
            smoke["smoke"],
        )

        save_buffer(
            "smoke_temperature.nvdb",
            smoke["temperature"],
        )

        log("")
        log("=" * 72)
        log("SUCCESS")
        log("=" * 72)
        log("Saved to:", OUT)

    except Exception:
        log("")
        log("========== EXCEPTION ==========")

        for line in traceback.format_exc().splitlines():
            log(line)

        raise

    finally:
        # Leave native readback disabled and timeline paused.
        try:
            configure_export(
                fire_export,
                readback_enabled=False,
                burn_enabled=False,
                smoke_enabled=False,
                temperature_enabled=False,
            )
        except Exception:
            pass

        try:
            configure_export(
                smoke_export,
                readback_enabled=False,
                burn_enabled=False,
                smoke_enabled=False,
                temperature_enabled=False,
            )
        except Exception:
            pass

        try:
            timeline.pause()
        except Exception:
            pass

        await settle(8)

        log("")
        log("FINAL: timeline paused, native readback disabled.")


def report_task(task):
    try:
        task.result()

    except asyncio.CancelledError:
        log("Task cancelled.")

    except Exception:
        # Already written to report.txt.
        pass


_task = asyncio.ensure_future(run())
_task.add_done_callback(report_task)

log("Native channel capture scheduled.")
