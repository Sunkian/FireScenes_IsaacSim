"""
capture_smoke_native_channels_only_v2.py

Isaac Sim 5.1 / omni.flowusd 107.1.8

Purpose
-------
Recover the Smoke part only, after Fire has already been captured successfully.

This version is intentionally more tolerant than v1:
- it captures SMOKE DENSITY as the primary target;
- it does not fail just because the temperature channel stays unchanged/stale;
- it uses a fresh OmniGraph readback node path;
- it leaves the timeline paused at the end.

Run AFTER:
1. Fire/Smoke scene is open
2. DenseSmoke is visible in the scene
3. Let the simulation run ~3-5 s before launching, or launch and let it run

Outputs
-------
/scratch/apagnoux/flow_native_channels_v1/
    smoke_density.nvdb
    smoke_temperature.nvdb   (if available)
    smoke_only_report.txt
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


SMOKE_EXPORT = "/Root/World/Meshes/hazards/DenseSmoke/flowSimulate/nanoVdbExport"
FIRE_EXPORT = "/Root/World/Meshes/hazards/Fire/flowSimulate/nanoVdbExport"

GRAPH_PATH = "/Root/World/FlowSmokeOnlyReadbackGraph"
NODE_NAME = "NanoVdbReadback"

OUT = Path("/scratch/apagnoux/flow_native_channels_v1")
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "smoke_only_report.txt"

FLOW_HZ = 60.0
MAX_WAIT_SECONDS = 8.0
MAX_WAIT_STEPS = int(FLOW_HZ * MAX_WAIT_SECONDS)
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


def configure_export(export_prim, *, readback_enabled, burn_enabled, smoke_enabled, temperature_enabled):
    require_attr(export_prim, "enabled").Set(True)
    require_attr(export_prim, "readbackOpenVDB").Set(False)
    require_attr(export_prim, "readbackEnabled").Set(readback_enabled)
    require_attr(export_prim, "burnEnabled").Set(burn_enabled)
    require_attr(export_prim, "smokeEnabled").Set(smoke_enabled)
    require_attr(export_prim, "temperatureEnabled").Set(temperature_enabled)

    for name in ("fuelEnabled", "velocityEnabled", "divergenceEnabled"):
        attr = export_prim.GetAttribute(name)
        if attr.IsValid():
            attr.Set(False)


def words_to_bytes(words):
    return words.astype("<u4", copy=False).tobytes()


def sha1_words(words):
    if words.size == 0:
        return None
    return hashlib.sha1(words_to_bytes(words)).hexdigest()


def save_buffer(filename, words):
    path = OUT / filename
    path.write_bytes(words_to_bytes(words))
    log("Saved", filename, "| words=", words.size, "| bytes=", words.size * 4, "| sha1=", sha1_words(words))


def ensure_readback_node(stage):
    node_path = f"{GRAPH_PATH}/{NODE_NAME}"

    if not stage.GetPrimAtPath(node_path).IsValid():
        log("Creating smoke-only readback node:", node_path)
        og.Controller.edit(
            {"graph_path": GRAPH_PATH, "evaluator_name": "push"},
            {og.Controller.Keys.CREATE_NODES: [(NODE_NAME, "omni.flowusd.FlowNanoVdbReadback")]}
        )

    node = og.Controller.node(node_path)
    if node is None or not node.is_valid():
        raise RuntimeError(f"Invalid OmniGraph node: {node_path}")

    attrs = {}
    for ch in ("smoke", "temperature"):
        attr = node.get_attribute(f"outputs:{ch}")
        if attr is None or not attr.is_valid():
            raise RuntimeError(f"Missing outputs:{ch} on readback node.")
        attrs[ch] = attr

    return attrs


def read_words(attr):
    value = og.Controller.get(attr)
    if value is None:
        return np.empty(0, dtype=np.uint32)
    return np.asarray(value, dtype=np.uint32).reshape(-1).copy()


async def settle(n=10):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


async def run():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")

    smoke_export = require_prim(stage, SMOKE_EXPORT)
    fire_export = require_prim(stage, FIRE_EXPORT)
    readback_attrs = ensure_readback_node(stage)
    timeline = omni.timeline.get_timeline_interface()

    try:
        log("SMOKE-ONLY NATIVE CHANNEL CAPTURE V2")

        try:
            timeline.pause()
        except Exception:
            pass

        # Disable fire readback completely, enable smoke readback only.
        configure_export(
            fire_export,
            readback_enabled=False,
            burn_enabled=False,
            smoke_enabled=False,
            temperature_enabled=False,
        )

        configure_export(
            smoke_export,
            readback_enabled=False,
            burn_enabled=False,
            smoke_enabled=True,
            temperature_enabled=True,
        )

        await settle()

        baseline_smoke = read_words(readback_attrs["smoke"])
        baseline_temp = read_words(readback_attrs["temperature"])

        baseline_smoke_sha1 = sha1_words(baseline_smoke)
        baseline_temp_sha1 = sha1_words(baseline_temp)

        log("Baseline smoke SHA1:", baseline_smoke_sha1)
        log("Baseline temp SHA1 :", baseline_temp_sha1)
        log("Baseline smoke bytes:", baseline_smoke.size * 4)
        log("Baseline temp bytes :", baseline_temp.size * 4)

        configure_export(
            smoke_export,
            readback_enabled=True,
            burn_enabled=False,
            smoke_enabled=True,
            temperature_enabled=True,
        )

        await settle()

        timeline.play()

        captured_smoke = None
        captured_temp = None

        for step in range(MAX_WAIT_STEPS):
            await rep.orchestrator.step_async(
                delta_time=1.0 / FLOW_HZ,
                rt_subframes=RT_SUBFRAMES,
                pause_timeline=False,
            )
            await omni.kit.app.get_app().next_update_async()

            smoke_words = read_words(readback_attrs["smoke"])
            temp_words = read_words(readback_attrs["temperature"])

            if captured_smoke is None and smoke_words.size > 0:
                sig = sha1_words(smoke_words)
                if sig != baseline_smoke_sha1:
                    captured_smoke = smoke_words
                    log(
                        f"Fresh smoke density after {step+1} steps "
                        f"({(step+1)/FLOW_HZ:.3f}s), bytes={smoke_words.size*4}, sha1={sig}"
                    )

            if captured_temp is None and temp_words.size > 0:
                sig = sha1_words(temp_words)
                # More tolerant: accept either a changed buffer OR any non-empty
                # temp after smoke has appeared.
                if sig != baseline_temp_sha1 or captured_smoke is not None:
                    captured_temp = temp_words
                    log(
                        f"Accepted smoke temperature after {step+1} steps "
                        f"({(step+1)/FLOW_HZ:.3f}s), bytes={temp_words.size*4}, sha1={sig}"
                    )

            if captured_smoke is not None and captured_temp is not None:
                break

            if (step + 1) % 60 == 0:
                log(
                    f"waiting... {(step+1)/FLOW_HZ:.1f}s | "
                    f"smoke={'yes' if captured_smoke is not None else 'no'} "
                    f"temp={'yes' if captured_temp is not None else 'no'}"
                )

        timeline.pause()
        await settle(8)

        if captured_smoke is None:
            raise RuntimeError("No fresh native smoke-density buffer captured.")

        save_buffer("smoke_density.nvdb", captured_smoke)

        if captured_temp is not None:
            save_buffer("smoke_temperature.nvdb", captured_temp)
        else:
            log("No smoke_temperature buffer captured; smoke_density was saved successfully.")

        log("")
        log("SUCCESS")
        log("Saved outputs to:", OUT)

    except Exception:
        log("")
        log("========== EXCEPTION ==========")
        for line in traceback.format_exc().splitlines():
            log(line)
        raise

    finally:
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
        log("FINAL: timeline paused, smoke native readback disabled.")


def done_callback(task):
    try:
        task.result()
    except asyncio.CancelledError:
        log("Task cancelled.")
    except Exception:
        pass


_task = asyncio.ensure_future(run())
_task.add_done_callback(done_callback)

log("Smoke-only capture scheduled.")
