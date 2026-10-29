"""
capture_fire_native_channels_warmup_v2.py

Isaac Sim 5.1 / omni.flowusd 107.1.8

Why this version exists
-----------------------
The previous Fire capture was taken almost immediately after simulation started.
Its temperature was uniformly ~0.319, which is consistent with the emitter
being sampled during the first coupling/update step, before combustion and
advection had time to develop. Burn was therefore still zero.

This script:
- isolates the FIRE Flow layer;
- exports fuel + burn + temperature;
- runs Fire for a full 3.0 s warm-up AFTER enabling native readback;
- captures the buffers only after warm-up;
- never calls timeline.stop();
- leaves the timeline paused at the end.

Outputs:
    /scratch/apagnoux/flow_native_fire_warm_v2/
        fire_fuel.nvdb
        fire_burn.nvdb
        fire_temperature.nvdb
        report.txt

Run INSIDE Isaac Sim Script Editor.
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


FIRE_EXPORT = (
    "/Root/World/Meshes/hazards/Fire/"
    "flowSimulate/nanoVdbExport"
)

SMOKE_EXPORT = (
    "/Root/World/Meshes/hazards/DenseSmoke/"
    "flowSimulate/nanoVdbExport"
)

GRAPH_PATH = "/Root/World/FlowFireWarmReadbackGraph"
NODE_NAME = "NanoVdbReadback"

OUT = Path("/scratch/apagnoux/flow_native_fire_warm_v2")
OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "report.txt"

FLOW_HZ = 60.0
WARMUP_SECONDS = 3.0
WARMUP_STEPS = int(FLOW_HZ * WARMUP_SECONDS)
RT_SUBFRAMES = 2


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


def require_attr(prim, name):
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        raise RuntimeError(f"Missing attribute '{name}' on {prim.GetPath()}")
    return attr


def set_export(
    prim,
    *,
    readback,
    fuel,
    burn,
    temperature,
):
    require_attr(prim, "enabled").Set(True)
    require_attr(prim, "readbackOpenVDB").Set(False)
    require_attr(prim, "readbackEnabled").Set(readback)
    require_attr(prim, "fuelEnabled").Set(fuel)
    require_attr(prim, "burnEnabled").Set(burn)
    require_attr(prim, "temperatureEnabled").Set(temperature)

    for name in ("smokeEnabled", "velocityEnabled", "divergenceEnabled"):
        attr = prim.GetAttribute(name)
        if attr.IsValid():
            attr.Set(False)


def ensure_node(stage):
    path = f"{GRAPH_PATH}/{NODE_NAME}"

    if not stage.GetPrimAtPath(path).IsValid():
        log("Creating readback node:", path)

        og.Controller.edit(
            {
                "graph_path": GRAPH_PATH,
                "evaluator_name": "push",
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    (NODE_NAME, "omni.flowusd.FlowNanoVdbReadback")
                ]
            },
        )

    node = og.Controller.node(path)

    if node is None or not node.is_valid():
        raise RuntimeError(f"Invalid readback node: {path}")

    result = {}

    for ch in ("fuel", "burn", "temperature"):
        attr = node.get_attribute(f"outputs:{ch}")

        if attr is None or not attr.is_valid():
            raise RuntimeError(f"Missing outputs:{ch}")

        result[ch] = attr

    return result


def read_words(attr):
    value = og.Controller.get(attr)

    if value is None:
        return np.empty(0, dtype=np.uint32)

    return np.asarray(
        value,
        dtype=np.uint32,
    ).reshape(-1).copy()


def sha1(words):
    if words.size == 0:
        return None

    return hashlib.sha1(
        words.astype("<u4", copy=False).tobytes()
    ).hexdigest()


def save_words(name, words):
    path = OUT / name

    path.write_bytes(
        words.astype("<u4", copy=False).tobytes()
    )

    log(
        "Saved",
        name,
        "| bytes=",
        words.size * 4,
        "| sha1=",
        sha1(words),
    )


async def settle(n=10):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


async def main():
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No USD stage is open.")

    fire_export = require_prim(stage, FIRE_EXPORT)
    smoke_export = require_prim(stage, SMOKE_EXPORT)
    attrs = ensure_node(stage)

    timeline = omni.timeline.get_timeline_interface()

    try:
        log("FIRE NATIVE CHANNEL CAPTURE — 3s WARMUP V2")

        try:
            timeline.pause()
        except Exception:
            pass

        # Disable Smoke readback completely so the global readback node
        # receives only Fire's native buffers.
        set_export(
            smoke_export,
            readback=False,
            fuel=False,
            burn=False,
            temperature=False,
        )

        # Prepare Fire exporter.
        set_export(
            fire_export,
            readback=False,
            fuel=True,
            burn=True,
            temperature=True,
        )

        await settle(12)

        # Enable Fire native readback.
        set_export(
            fire_export,
            readback=True,
            fuel=True,
            burn=True,
            temperature=True,
        )

        await settle(12)

        log(
            f"Running Fire warm-up for {WARMUP_SECONDS:.1f}s "
            f"({WARMUP_STEPS} simulation steps)..."
        )

        timeline.play()

        # IMPORTANT: ignore all early readbacks.
        # We only care about the developed field after the full warm-up.
        for step in range(WARMUP_STEPS):
            await rep.orchestrator.step_async(
                delta_time=1.0 / FLOW_HZ,
                rt_subframes=RT_SUBFRAMES,
                pause_timeline=False,
            )

            if (step + 1) % 60 == 0:
                log(
                    "warm-up:",
                    f"{(step + 1)/FLOW_HZ:.1f}s / {WARMUP_SECONDS:.1f}s"
                )

        # Give asynchronous readback a little time to catch the latest state.
        for _ in range(20):
            await omni.kit.app.get_app().next_update_async()

        timeline.pause()

        await settle(12)

        log("")
        log("CAPTURE AFTER WARM-UP")

        captured = {}

        for ch in ("fuel", "burn", "temperature"):
            words = read_words(attrs[ch])

            captured[ch] = words

            log(
                ch,
                "| words=",
                words.size,
                "| bytes=",
                words.size * 4,
                "| sha1=",
                sha1(words),
            )

        if captured["fuel"].size == 0:
            raise RuntimeError("Fuel readback is empty.")

        if captured["temperature"].size == 0:
            raise RuntimeError("Temperature readback is empty.")

        # Burn is allowed to remain empty/zero at this stage; the next
        # inspector will tell us whether the actual grid values are positive.
        if captured["burn"].size == 0:
            log("WARNING: burn readback byte buffer is empty.")

        save_words(
            "fire_fuel.nvdb",
            captured["fuel"],
        )

        if captured["burn"].size:
            save_words(
                "fire_burn.nvdb",
                captured["burn"],
            )

        save_words(
            "fire_temperature.nvdb",
            captured["temperature"],
        )

        log("")
        log("SUCCESS")
        log("Saved to:", OUT)

    except Exception:
        log("")
        log("========== EXCEPTION ==========")

        for line in traceback.format_exc().splitlines():
            log(line)

        raise

    finally:
        try:
            set_export(
                fire_export,
                readback=False,
                fuel=False,
                burn=False,
                temperature=False,
            )
        except Exception:
            pass

        try:
            timeline.pause()
        except Exception:
            pass

        await settle(8)

        log("")
        log("FINAL: timeline paused, Fire readback disabled.")


def callback(task):
    try:
        task.result()
    except asyncio.CancelledError:
        log("Task cancelled.")
    except Exception:
        pass


_task = asyncio.ensure_future(main())
_task.add_done_callback(callback)

log("Fire 3-second warm-up capture scheduled.")
