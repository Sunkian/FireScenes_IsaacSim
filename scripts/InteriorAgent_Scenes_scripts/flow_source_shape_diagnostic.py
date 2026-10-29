"""
flow_source_shape_diagnostic.py

Run INSIDE Isaac Sim Script Editor.

Purpose:
The native NanoVDB projection is now correct, but the projected Fire/Smoke
fields are almost spherical. This script inspects the source emitters and
transport settings that determine the actual Flow field shape.

Output:
    /scratch/apagnoux/flow_source_shape_diagnostic.txt
"""

from pathlib import Path

import omni.usd
from pxr import Usd, UsdGeom


OUT = Path("/scratch/apagnoux/flow_source_shape_diagnostic.txt")

ROOTS = [
    "/Root/World/Meshes/hazards/Fire",
    "/Root/World/Meshes/hazards/DenseSmoke",
]

KEYWORDS = [
    "radius",
    "enabled",
    "layer",
    "temperature",
    "smoke",
    "burn",
    "fuel",
    "velocity",
    "couplerate",
    "alloc",
    "damping",
    "buoy",
    "vortic",
    "cool",
    "fade",
    "advection",
    "diverg",
    "pressure",
    "densitycellsize",
    "substep",
]


stage = omni.usd.get_context().get_stage()

if stage is None:
    raise RuntimeError("No USD stage is open.")

lines = []
lines.append("========== FLOW SOURCE / SHAPE DIAGNOSTIC ==========")

for root_path in ROOTS:
    root = stage.GetPrimAtPath(root_path)

    lines.append("")
    lines.append("=" * 80)
    lines.append(root_path)
    lines.append("=" * 80)

    if not root.IsValid():
        lines.append("MISSING ROOT")
        continue

    for prim in stage.Traverse():
        path = prim.GetPath().pathString

        if not (
            path == root_path
            or path.startswith(root_path + "/")
        ):
            continue

        low_path = path.lower()

        interesting_prim = any(
            token in low_path
            for token in [
                "flowemittersphere",
                "flowsimulate",
                "advection",
                "vorticity",
                "pressure",
                "summaryallocate",
                "flowrender",
                "flowoffscreen",
            ]
        )

        if not interesting_prim:
            continue

        lines.append("")
        lines.append(path)
        lines.append(f"type: {prim.GetTypeName()}")

        try:
            xf = UsdGeom.Xformable(prim)
            m = xf.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            t = m.ExtractTranslation()

            lines.append(
                "world translation: "
                f"[{float(t[0]):.6f}, {float(t[1]):.6f}, {float(t[2]):.6f}]"
            )
            lines.append(f"local-to-world: {m}")
        except Exception:
            pass

        for attr in prim.GetAttributes():
            name = attr.GetName()
            low = name.lower()

            if not any(k in low for k in KEYWORDS):
                continue

            try:
                value = attr.Get()
            except Exception as exc:
                value = f"<ERROR: {exc}>"

            lines.append(f"  {name} = {value}")

OUT.write_text("\n".join(lines) + "\n")

print("Saved diagnostic to:", OUT)
