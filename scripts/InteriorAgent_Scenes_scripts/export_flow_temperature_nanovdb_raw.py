"""
export_flow_temperature_nanovdb_raw.py

Isaac Sim 5.1 — diagnostic export of Flow temperature as NATIVE NanoVDB.

Purpose
-------
The previous exports produced 65-byte OpenVDB headers only. This script tests
whether the failure is specifically in the NanoVDB -> OpenVDB conversion path.

Key difference:
    readbackOpenVDB = False

Everything else stays close to the known-working Flow readback setup.

Output:
    /scratch/apagnoux/flow_temperature_nanovdb_raw/
        fire/
        smoke/

Run INSIDE Isaac Sim Script Editor.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd


# =============================================================================
# CONFIG
# =============================================================================

FIRE_SIM = "/Root/World/Meshes/hazards/Fire/flowSimulate"
SMOKE_SIM = "/Root/World/Meshes/hazards/DenseSmoke/flowSimulate"

OUTPUT_ROOT = Path("/scratch/apagnoux/flow_temperature_nanovdb_raw")
FIRE_OUT = OUTPUT_ROOT / "fire"
SMOKE_OUT = OUTPUT_ROOT / "smoke"

SIM_HZ = 60.0
SIM_SECONDS = 4.0
SIM_STEPS = int(round(SIM_HZ * SIM_SECONDS))

# Keep several renderer subframes while Flow/readback is active.
RT_SUBFRAMES = 5

# Extra Kit updates after pausing so async readback can flush.
FINAL_SETTLE_UPDATES = 120


# =============================================================================
# HELPERS
# =============================================================================

def require_attr(prim, name):
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        raise RuntimeError(f"Missing '{name}' on {prim.GetPath()}")
    return attr


def configure_temperature_export(sim_prim, out_dir: Path):
    export_prim = sim_prim.GetChild("nanoVdbExport")

    if not export_prim.IsValid():
        raise RuntimeError(
            f"Missing nanoVdbExport under {sim_prim.GetPath()}"
        )

    require_attr(export_prim, "enabled").Set(True)
    require_attr(export_prim, "readbackEnabled").Set(True)

    # IMPORTANT TEST:
    # Keep the GPU-produced NanoVDB format. Do NOT convert to OpenVDB.
    require_attr(export_prim, "readbackOpenVDB").Set(False)

    require_attr(export_prim, "readbackDirectory").Set(str(out_dir))
    require_attr(export_prim, "readbackClearOnStart").Set(False)
    require_attr(export_prim, "readbackClearOnStop").Set(False)
    require_attr(export_prim, "statisticsEnabled").Set(True)

    require_attr(export_prim, "temperatureEnabled").Set(True)

    # Disable unrelated channels for this diagnostic.
    for name in (
        "burnEnabled",
        "smokeEnabled",
        "fuelEnabled",
        "velocityEnabled",
        "divergenceEnabled",
    ):
        attr = export_prim.GetAttribute(name)
        if attr.IsValid():
            attr.Set(False)

    return export_prim


async def settle(n):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


def inspect_files(label, directory):
    files = sorted(directory.glob("*.vdb"))

    print(f"\n========== {label.upper()} ==========")
    print("Directory:", directory)
    print("Count:", len(files))

    large = []

    for p in files[-30:]:
        size = p.stat().st_size
        if size > 128:
            large.append(p)

        print(
            f"{p.name:45s} "
            f"{size:10d} bytes "
            f"{'DATA' if size > 128 else 'HEADER/EMPTY'}"
        )

    return large


# =============================================================================
# MAIN
# =============================================================================

async def run():
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No USD stage is open.")

    fire_sim = stage.GetPrimAtPath(FIRE_SIM)
    smoke_sim = stage.GetPrimAtPath(SMOKE_SIM)

    if not fire_sim.IsValid():
        raise RuntimeError(f"Missing: {FIRE_SIM}")

    if not smoke_sim.IsValid():
        raise RuntimeError(f"Missing: {SMOKE_SIM}")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    FIRE_OUT.mkdir(parents=True, exist_ok=True)
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)

    configure_temperature_export(fire_sim, FIRE_OUT)
    configure_temperature_export(smoke_sim, SMOKE_OUT)

    print("========== RAW NANOVDB EXPORT TEST ==========")
    print("Fire layer:", fire_sim.GetAttribute("layer").Get())
    print("Smoke layer:", smoke_sim.GetAttribute("layer").Get())
    print("readbackOpenVDB = False")
    print("Output:", OUTPUT_ROOT)

    timeline = omni.timeline.get_timeline_interface()

    try:
        # Let USD/Flow notice the export settings.
        await settle(10)

        timeline.play()
        await settle(3)

        print(
            f"\nRunning {SIM_SECONDS:.1f}s "
            f"({SIM_STEPS} Flow steps)..."
        )

        for i in range(SIM_STEPS):
            await rep.orchestrator.step_async(
                delta_time=1.0 / SIM_HZ,
                rt_subframes=RT_SUBFRAMES,
                pause_timeline=False,
            )

            if (i + 1) % 60 == 0:
                print(f"  simulated {(i + 1) / SIM_HZ:.1f}s")

        timeline.pause()

        print("\nPaused. Waiting for async readback to flush...")
        await settle(FINAL_SETTLE_UPDATES)

        fire_data = inspect_files("fire", FIRE_OUT)
        smoke_data = inspect_files("smoke", SMOKE_OUT)

        print("\n========== RESULT ==========")

        if fire_data:
            print(
                "FIRE: native NanoVDB contains data.",
                "Largest:",
                max(p.stat().st_size for p in fire_data),
                "bytes",
            )
        else:
            print("FIRE: still no non-empty NanoVDB found.")

        if smoke_data:
            print(
                "SMOKE: native NanoVDB contains data.",
                "Largest:",
                max(p.stat().st_size for p in smoke_data),
                "bytes",
            )
        else:
            print("SMOKE: still no non-empty NanoVDB found.")

        print("\nOutput root:", OUTPUT_ROOT)

    finally:
        try:
            timeline.pause()
        except Exception:
            pass

        await settle(8)
        print("\nTimeline remains paused.")


def report_task(task):
    try:
        task.result()
    except asyncio.CancelledError:
        print("Raw NanoVDB export task cancelled.")
    except Exception:
        import traceback
        traceback.print_exc()


_task = asyncio.ensure_future(run())
_task.add_done_callback(report_task)

print("Raw Flow NanoVDB temperature export scheduled.")
