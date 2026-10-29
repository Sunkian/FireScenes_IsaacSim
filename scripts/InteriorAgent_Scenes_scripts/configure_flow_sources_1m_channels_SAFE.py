"""
configure_flow_sources_1m_channels_SAFE.py

Isaac Sim 5.1 — SAFE configuration only.

This version avoids the operation that was most likely causing the native crash:
    timeline.stop()

It also avoids changing NanoVDB export settings in the same script.

What it changes
---------------
FIRE:
    radius = 1.0 m
    radiusIsWorldSpace = True
    fuel = 0.8
    temperature = 2.0
    burn = 0.0

DENSE SMOKE:
    radius = 1.0 m
    radiusIsWorldSpace = True
    smoke = 1.0
    coupleRateSmoke = 50.0
    fuel = 0.0
    coupleRateFuel = 0.0
    temperature = 0.5
    velocity = (0, 0, 0.8)
    velocityIsWorldSpace = True

IMPORTANT
---------
1. Run this with the timeline PAUSED.
2. This script pauses the timeline itself, but does NOT stop/reset it.
3. It does NOT touch nanoVdbExport.
4. After it completes, SAVE the USD manually.
5. Then reopen the scene / restart Isaac before generating fresh Flow state.
"""

from pathlib import Path
import asyncio

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf


FIRE_EMITTER = "/Root/World/Meshes/hazards/Fire/flowEmitterSphere"
SMOKE_EMITTER = "/Root/World/Meshes/hazards/DenseSmoke/flowEmitterSphere"

REPORT = Path("/scratch/apagnoux/flow_sources_1m_safe_report.txt")


def require_prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {path}")
    return prim


def set_attr(prim, name, value, lines):
    attr = prim.GetAttribute(name)

    if not attr.IsValid():
        lines.append(f"[MISSING] {prim.GetPath()} :: {name}")
        return

    before = attr.Get()
    attr.Set(value)
    after = attr.Get()

    lines.append(
        f"{prim.GetPath()} :: {name}: {before} -> {after}"
    )


async def main():
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No USD stage is open.")

    timeline = omni.timeline.get_timeline_interface()

    lines = [
        "========== FLOW 1M SAFE CONFIG ==========",
        "",
        "Pausing timeline safely...",
    ]

    try:
        timeline.pause()
    except Exception as exc:
        lines.append(f"pause warning: {exc}")

    for _ in range(8):
        await omni.kit.app.get_app().next_update_async()

    fire = require_prim(stage, FIRE_EMITTER)
    smoke = require_prim(stage, SMOKE_EMITTER)

    lines += ["", "FIRE", "----"]

    set_attr(fire, "radius", 1.0, lines)
    set_attr(fire, "radiusIsWorldSpace", True, lines)
    set_attr(fire, "fuel", 0.8, lines)
    set_attr(fire, "coupleRateFuel", 2.0, lines)
    set_attr(fire, "temperature", 2.0, lines)
    set_attr(fire, "coupleRateTemperature", 10.0, lines)
    set_attr(fire, "burn", 0.0, lines)
    set_attr(fire, "coupleRateBurn", 0.0, lines)

    lines += ["", "DENSE SMOKE", "-----------"]

    set_attr(smoke, "radius", 1.0, lines)
    set_attr(smoke, "radiusIsWorldSpace", True, lines)
    set_attr(smoke, "smoke", 1.0, lines)
    set_attr(smoke, "coupleRateSmoke", 50.0, lines)
    set_attr(smoke, "temperature", 0.5, lines)
    set_attr(smoke, "coupleRateTemperature", 50.0, lines)
    set_attr(smoke, "fuel", 0.0, lines)
    set_attr(smoke, "coupleRateFuel", 0.0, lines)
    set_attr(smoke, "velocity", Gf.Vec3f(0.0, 0.0, 0.8), lines)
    set_attr(smoke, "velocityIsWorldSpace", True, lines)

    for _ in range(8):
        await omni.kit.app.get_app().next_update_async()

    lines += [
        "",
        "DONE.",
        "Timeline remains paused.",
        "nanoVdbExport was NOT modified.",
        "",
        "NEXT:",
        "1. Save the USD manually.",
        "2. Reopen the scene (or restart Isaac).",
        "3. Press Play and let Flow develop for 3-5 seconds.",
        "4. Then capture fresh native channels.",
    ]

    REPORT.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nReport:", REPORT)


def report_task(task):
    try:
        task.result()
    except asyncio.CancelledError:
        print("Safe Flow config task cancelled.")
    except Exception:
        import traceback
        traceback.print_exc()


_task = asyncio.ensure_future(main())
_task.add_done_callback(report_task)

print("SAFE 1m Flow configuration scheduled.")
