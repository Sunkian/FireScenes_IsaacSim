"""
flow_native_nanovdb_readback.py

Isaac Sim 5.1 / omni.flowusd 107.1.8
Native Flow NanoVDB readback diagnostic.

This uses the exact behavior confirmed from NVIDIA's local C++ implementation:

    iFace->mapLatestNanoVdbReadback(&readback)

The FlowNanoVdbReadback node has no inputs and exposes the latest global Flow
readback. Because this scene has TWO Flow simulations, this script isolates them
by enabling native readback on only ONE nanoVdbExport at a time.

It saves the raw temperature NanoVDB buffer for:
    Fire      -> layer 1
    DenseSmoke -> layer 2

Outputs:
    /scratch/apagnoux/flow_native_readback/
        fire_temperature.nvdb
        fire_temperature_uint32.npy
        smoke_temperature.nvdb
        smoke_temperature_uint32.npy
        report.txt

Run INSIDE Isaac Sim Script Editor.
"""

from __future__ import annotations

import asyncio
import hashlib
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

GRAPH_PATH = "/Root/World/FlowNativeReadbackGraph"
NODE_NAME = "NanoVdbReadback"

OUT = Path("/scratch/apagnoux/flow_native_readback")
OUT.mkdir(parents=True, exist_ok=True)

FLOW_HZ = 60.0

# Maximum simulated time allowed while waiting for each fresh readback.
MAX_WAIT_SECONDS = 4.0
MAX_WAIT_STEPS = int(round(FLOW_HZ * MAX_WAIT_SECONDS))

# Give asynchronous Flow readback a little breathing room.
RT_SUBFRAMES = 2
UPDATES_AFTER_SWITCH = 8


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


def configure_native_temperature_readback(export_prim, enabled: bool):
    """
    Native readback only:
      readbackEnabled = target state
      readbackOpenVDB = False
      temperature only
    """
    require_attr(export_prim, "enabled").Set(True)
    require_attr(export_prim, "readbackOpenVDB").Set(False)
    require_attr(export_prim, "readbackEnabled").Set(enabled)

    # We are reading through OmniGraph, not writing OpenVDB files.
    require_attr(export_prim, "temperatureEnabled").Set(True)

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


async def settle(n):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


def ensure_readback_node(stage):
    """
    Create the push OmniGraph used by NVIDIA's FlowNanoVdbReadback node.
    Reuse it if this script was already run.
    """
    node_path = f"{GRAPH_PATH}/{NODE_NAME}"

    existing = stage.GetPrimAtPath(node_path)

    if not existing.IsValid():
        print("Creating FlowNanoVdbReadback OmniGraph node...")

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
            f"Could not create/find OmniGraph node: {node_path}"
        )

    temperature_attr = node.get_attribute("outputs:temperature")

    if temperature_attr is None or not temperature_attr.is_valid():
        raise RuntimeError(
            "FlowNanoVdbReadback has no valid outputs:temperature attribute."
        )

    return node, temperature_attr


def read_temperature_words(attr):
    """
    OmniGraph uint[] -> independent NumPy uint32 array.
    """
    value = og.Controller.get(attr)

    if value is None:
        return np.empty(0, dtype=np.uint32)

    arr = np.asarray(value, dtype=np.uint32).reshape(-1).copy()
    return arr


def signature(words: np.ndarray):
    if words.size == 0:
        return None

    return hashlib.sha1(
        words.astype("<u4", copy=False).tobytes()
    ).hexdigest()


def raw_bytes(words: np.ndarray) -> bytes:
    return words.astype("<u4", copy=False).tobytes()


def printable_prefix(data: bytes, n=16):
    x = data[:n]
    return "".join(
        chr(b) if 32 <= b <= 126 else "."
        for b in x
    )


def save_buffer(label: str, words: np.ndarray):
    data = raw_bytes(words)

    nvdb_path = OUT / f"{label}_temperature.nvdb"
    npy_path = OUT / f"{label}_temperature_uint32.npy"

    nvdb_path.write_bytes(data)
    np.save(npy_path, words)

    digest = hashlib.sha1(data).hexdigest()

    print(f"\nSaved {label}:")
    print("  uint32 words :", words.size)
    print("  bytes        :", len(data))
    print("  SHA1         :", digest)
    print("  first bytes  :", data[:16].hex(" "))
    print("  ASCII prefix :", printable_prefix(data))
    print("  raw file     :", nvdb_path)
    print("  npy file     :", npy_path)

    return {
        "label": label,
        "words": int(words.size),
        "bytes": len(data),
        "sha1": digest,
        "prefix_hex": data[:16].hex(" "),
        "prefix_ascii": printable_prefix(data),
        "nvdb": str(nvdb_path),
        "npy": str(npy_path),
    }


async def wait_for_fresh_buffer(
    label,
    target_export,
    other_export,
    temperature_attr,
    timeline,
):
    """
    Since FlowNanoVdbReadback maps the global *latest* readback, only the target
    exporter is allowed to read back while we wait.

    A SHA1 change ensures we do not accidentally save the stale buffer that was
    already present before switching exporters.
    """

    # Snapshot whatever the node currently holds before switching.
    baseline = read_temperature_words(temperature_attr)
    baseline_sig = signature(baseline)

    print(f"\n========== CAPTURING {label.upper()} ==========")
    print("Baseline words:", baseline.size)
    print("Baseline SHA1 :", baseline_sig)

    # Disable both first, then enable only the selected Flow exporter.
    configure_native_temperature_readback(other_export, False)
    configure_native_temperature_readback(target_export, False)

    await settle(UPDATES_AFTER_SWITCH)

    configure_native_temperature_readback(target_export, True)

    await settle(UPDATES_AFTER_SWITCH)

    timeline.play()

    latest = np.empty(0, dtype=np.uint32)

    for step in range(MAX_WAIT_STEPS):

        await rep.orchestrator.step_async(
            delta_time=1.0 / FLOW_HZ,
            rt_subframes=RT_SUBFRAMES,
            pause_timeline=False,
        )

        # Let the push graph receive the async readback callback.
        await omni.kit.app.get_app().next_update_async()

        current = read_temperature_words(temperature_attr)

        if current.size == 0:
            continue

        current_sig = signature(current)
        latest = current

        # Fresh means the node changed after this target exporter was enabled.
        if current_sig != baseline_sig:
            print(
                f"Fresh {label} readback after "
                f"{step + 1} Flow steps "
                f"({(step + 1) / FLOW_HZ:.3f}s simulated)."
            )

            timeline.pause()
            await settle(8)
            return current

        if (step + 1) % 60 == 0:
            print(
                f"  waiting... {(step + 1) / FLOW_HZ:.1f}s, "
                f"current words={current.size}"
            )

    timeline.pause()
    await settle(8)

    if latest.size:
        raise RuntimeError(
            f"{label}: a non-empty buffer exists, but it never changed after "
            "switching to this exporter. Refusing to save a possibly stale "
            "buffer."
        )

    raise RuntimeError(
        f"{label}: FlowNanoVdbReadback remained empty for "
        f"{MAX_WAIT_SECONDS:.1f}s."
    )


# =============================================================================
# MAIN
# =============================================================================

async def run():
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No USD stage is open.")

    fire_export = require_prim(stage, FIRE_EXPORT)
    smoke_export = require_prim(stage, SMOKE_EXPORT)

    _, temperature_attr = ensure_readback_node(stage)

    timeline = omni.timeline.get_timeline_interface()

    results = []

    try:
        # Start with both disk/native readbacks disabled, then isolate each one.
        configure_native_temperature_readback(fire_export, False)
        configure_native_temperature_readback(smoke_export, False)

        await settle(10)

        # -----------------------------------------------------------------
        # FIRE LAYER 1
        # -----------------------------------------------------------------

        fire_words = await wait_for_fresh_buffer(
            label="fire",
            target_export=fire_export,
            other_export=smoke_export,
            temperature_attr=temperature_attr,
            timeline=timeline,
        )

        results.append(
            save_buffer("fire", fire_words)
        )

        # -----------------------------------------------------------------
        # DENSE SMOKE LAYER 2
        # -----------------------------------------------------------------

        smoke_words = await wait_for_fresh_buffer(
            label="smoke",
            target_export=smoke_export,
            other_export=fire_export,
            temperature_attr=temperature_attr,
            timeline=timeline,
        )

        results.append(
            save_buffer("smoke", smoke_words)
        )

        # -----------------------------------------------------------------
        # REPORT
        # -----------------------------------------------------------------

        lines = [
            "FLOW NATIVE NANOVDB READBACK",
            "",
        ]

        for r in results:
            lines += [
                f"{r['label']}:",
                f"  words: {r['words']}",
                f"  bytes: {r['bytes']}",
                f"  sha1: {r['sha1']}",
                f"  prefix_hex: {r['prefix_hex']}",
                f"  prefix_ascii: {r['prefix_ascii']}",
                f"  nvdb: {r['nvdb']}",
                f"  npy: {r['npy']}",
                "",
            ]

        report_path = OUT / "report.txt"
        report_path.write_text("\n".join(lines))

        print("\n========== SUCCESS ==========")
        print(report_path.read_text())

    finally:
        # Stop requesting new native readbacks, but preserve Flow itself.
        try:
            configure_native_temperature_readback(fire_export, False)
            configure_native_temperature_readback(smoke_export, False)
        except Exception:
            pass

        try:
            timeline.pause()
        except Exception:
            pass

        await settle(8)

        print("Timeline paused.")
        print("Native readback disabled on both Flow simulations.")
        print("The OmniGraph readback node is intentionally kept alive.")


def report_task(task):
    try:
        task.result()

    except asyncio.CancelledError:
        print("Native NanoVDB readback task cancelled.")

    except Exception:
        import traceback
        traceback.print_exc()


_task = asyncio.ensure_future(run())
_task.add_done_callback(report_task)

print("Flow native NanoVDB readback scheduled.")
