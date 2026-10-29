"""
inspect_fire_combustion_SAFE.py

Isaac Sim 5.1 — read-only diagnostic.
No timeline changes. No NanoVDB loading. No Warp.

Checks the Fire combustion settings that determine whether the native
burn channel can become non-zero.

Output:
    /scratch/apagnoux/fire_combustion_safe_report.txt
"""

from pathlib import Path
import omni.usd

FIRE_EMITTER = "/Root/World/Meshes/hazards/Fire/flowEmitterSphere"
FIRE_ADV = "/Root/World/Meshes/hazards/Fire/flowSimulate/advection"
FIRE_BURN_ADV = "/Root/World/Meshes/hazards/Fire/flowSimulate/advection/burn"
FIRE_TEMP_ADV = "/Root/World/Meshes/hazards/Fire/flowSimulate/advection/temperature"
FIRE_FUEL_ADV = "/Root/World/Meshes/hazards/Fire/flowSimulate/advection/fuel"

OUT = Path("/scratch/apagnoux/fire_combustion_safe_report.txt")

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is open.")

lines = ["========== FIRE COMBUSTION SAFE DIAGNOSTIC =========="]

checks = [
    (FIRE_EMITTER, [
        "enabled",
        "radius",
        "radiusIsWorldSpace",
        "fuel",
        "coupleRateFuel",
        "temperature",
        "coupleRateTemperature",
        "burn",
        "coupleRateBurn",
        "velocity",
        "coupleRateVelocity",
    ]),
    (FIRE_ADV, [
        "enabled",
        "combustionEnabled",
        "ignitionTemp",
        "burnPerTemp",
        "fuelPerBurn",
        "tempPerBurn",
        "smokePerBurn",
        "coolingRate",
        "buoyancyPerTemp",
        "gravity",
    ]),
    (FIRE_BURN_ADV, [
        "damping",
        "fade",
        "secondOrderBlendFactor",
        "secondOrderBlendThreshold",
    ]),
    (FIRE_TEMP_ADV, [
        "damping",
        "fade",
    ]),
    (FIRE_FUEL_ADV, [
        "damping",
        "fade",
    ]),
]

for path, names in checks:
    prim = stage.GetPrimAtPath(path)
    lines += ["", path]

    if not prim.IsValid():
        lines.append("  MISSING PRIM")
        continue

    for name in names:
        attr = prim.GetAttribute(name)

        if not attr.IsValid():
            lines.append(f"  {name} = <MISSING>")
            continue

        try:
            value = attr.Get()
        except Exception as exc:
            value = f"<ERROR: {exc}>"

        lines.append(f"  {name} = {value}")

OUT.write_text("\n".join(lines) + "\n")

print("\n".join(lines))
print("\nSaved:", OUT)
