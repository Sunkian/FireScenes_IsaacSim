"""
grow_fire_smoke_progressively_v1.py

Isaac Sim 5.1

Goal
----
Start from a small ignition and progressively grow BOTH fire and smoke,
while keeping 1.0 m as the FINAL emitter radius.

This is a scene controller, not a diagnostic.

Behavior (default 10 s)
-----------------------
0-2 s   : ignition, small source
2-5 s   : developing flame/plume
5-8 s   : strong vertical growth toward ceiling
8-10 s  : fully developed source, smoke persists and spreads laterally

Important
---------
- Does NOT call timeline.stop().
- Leaves timeline PAUSED at the end.
- Assumes the ceiling/walls have working Physics colliders.
  Without a ceiling collider, smoke/fire will pass through instead of
  spreading under the ceiling.
- Does NOT save the USD automatically.
- Final source radius is 1.0 m for both Fire and DenseSmoke.

Run INSIDE Isaac Sim Script Editor.
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf


# =============================================================================
# PATHS
# =============================================================================

FIRE_EMITTER = "/Root/World/Meshes/hazards/Fire/flowEmitterSphere"
SMOKE_EMITTER = "/Root/World/Meshes/hazards/DenseSmoke/flowEmitterSphere"

FIRE_VEL_ADV = "/Root/World/Meshes/hazards/Fire/flowSimulate/advection/velocity"

SMOKE_VEL_ADV = "/Root/World/Meshes/hazards/DenseSmoke/flowSimulate/advection/velocity"
SMOKE_DENSITY_ADV = "/Root/World/Meshes/hazards/DenseSmoke/flowSimulate/advection/smoke"

REPORT = Path("/scratch/apagnoux/fire_smoke_growth_v1_report.txt")


# =============================================================================
# TIMING / GROWTH
# =============================================================================

DURATION_S = 10.0

# Keep 1 m as the final physical source radius.
FIRE_RADIUS_START_M = 0.10
FIRE_RADIUS_END_M = 1.00

SMOKE_RADIUS_START_M = 0.15
SMOKE_RADIUS_END_M = 1.00

# Final source strengths are close to your existing setup.
FIRE_FUEL_START = 0.15
FIRE_FUEL_END = 0.80

FIRE_TEMP_START = 0.35
FIRE_TEMP_END = 2.00

SMOKE_START = 0.15
SMOKE_END = 1.00

SMOKE_TEMP_START = 0.10
SMOKE_TEMP_END = 0.50

# World-up source velocities.
FIRE_UP_START = 0.15
FIRE_UP_END = 0.85

SMOKE_UP_START = 0.25
SMOKE_UP_END = 1.00

# Gentle horizontal motion only after the plume is developed.
# The ceiling collider should do the main sideways spreading.
LATERAL_VELOCITY_MAX = 0.18


# =============================================================================
# HELPERS
# =============================================================================

def require_prim(stage, path):
    prim = stage.GetPrimAtPath(path)

    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {path}")

    return prim


def attr(prim, name):
    a = prim.GetAttribute(name)

    if not a.IsValid():
        raise RuntimeError(
            f"Missing attribute '{name}' on {prim.GetPath()}"
        )

    return a


def optional_attr(prim, name):
    a = prim.GetAttribute(name)
    return a if a.IsValid() else None


def lerp(a, b, x):
    return a + (b - a) * x


def smoothstep01(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def growth_curve(t):
    """
    Piecewise growth:
      0-2s  : 0 -> 0.18
      2-5s  : 0.18 -> 0.50
      5-8s  : 0.50 -> 0.82
      8-10s : 0.82 -> 1.00
    """
    if t <= 2.0:
        q = smoothstep01(t / 2.0)
        return lerp(0.0, 0.18, q)

    if t <= 5.0:
        q = smoothstep01((t - 2.0) / 3.0)
        return lerp(0.18, 0.50, q)

    if t <= 8.0:
        q = smoothstep01((t - 5.0) / 3.0)
        return lerp(0.50, 0.82, q)

    q = smoothstep01((t - 8.0) / 2.0)
    return lerp(0.82, 1.00, q)


async def settle(n=8):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


# =============================================================================
# MAIN
# =============================================================================

async def main():
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No USD stage is open.")

    fire = require_prim(stage, FIRE_EMITTER)
    smoke = require_prim(stage, SMOKE_EMITTER)

    fire_vel_adv = require_prim(stage, FIRE_VEL_ADV)
    smoke_vel_adv = require_prim(stage, SMOKE_VEL_ADV)
    smoke_density_adv = require_prim(stage, SMOKE_DENSITY_ADV)

    timeline = omni.timeline.get_timeline_interface()

    lines = [
        "========== PROGRESSIVE FIRE + SMOKE GROWTH V1 ==========",
        "",
        f"Duration: {DURATION_S:.1f} s",
        "Final Fire radius: 1.0 m",
        "Final Smoke radius: 1.0 m",
        "",
    ]

    # ---------------------------------------------------------------------
    # SAFE INITIAL STATE
    # ---------------------------------------------------------------------

    try:
        timeline.pause()
    except Exception:
        pass

    await settle(8)

    # Start small.
    attr(fire, "radius").Set(FIRE_RADIUS_START_M)
    attr(fire, "radiusIsWorldSpace").Set(True)

    attr(fire, "fuel").Set(FIRE_FUEL_START)
    attr(fire, "temperature").Set(FIRE_TEMP_START)

    attr(fire, "velocity").Set(
        Gf.Vec3f(0.0, 0.0, FIRE_UP_START)
    )
    attr(fire, "velocityIsWorldSpace").Set(True)

    attr(smoke, "radius").Set(SMOKE_RADIUS_START_M)
    attr(smoke, "radiusIsWorldSpace").Set(True)

    attr(smoke, "smoke").Set(SMOKE_START)
    attr(smoke, "temperature").Set(SMOKE_TEMP_START)

    # DenseSmoke should not behave as a second fuel source.
    fuel_attr = optional_attr(smoke, "fuel")
    if fuel_attr:
        fuel_attr.Set(0.0)

    couple_fuel = optional_attr(smoke, "coupleRateFuel")
    if couple_fuel:
        couple_fuel.Set(0.0)

    attr(smoke, "velocity").Set(
        Gf.Vec3f(0.0, 0.0, SMOKE_UP_START)
    )
    attr(smoke, "velocityIsWorldSpace").Set(True)

    # ---------------------------------------------------------------------
    # PERSISTENCE / TRANSPORT
    # ---------------------------------------------------------------------
    # Your previous DenseSmoke settings were quite dissipative:
    # damping ~0.30, fade ~0.65.
    # Reduce them so smoke can accumulate, rise, hit the ceiling and spread.

    damp = optional_attr(smoke_density_adv, "damping")
    if damp:
        damp.Set(0.05)

    fade = optional_attr(smoke_density_adv, "fade")
    if fade:
        fade.Set(0.08)

    # Keep velocity from disappearing too quickly.
    for vel_adv in (fire_vel_adv, smoke_vel_adv):
        damp = optional_attr(vel_adv, "damping")
        if damp:
            damp.Set(0.01)

        fade = optional_attr(vel_adv, "fade")
        if fade:
            fade.Set(0.15)

    # Make sure both emitters are enabled only when growth starts.
    attr(fire, "enabled").Set(False)
    attr(smoke, "enabled").Set(False)

    await settle(8)

    # ---------------------------------------------------------------------
    # IGNITION
    # ---------------------------------------------------------------------

    attr(fire, "enabled").Set(True)
    attr(smoke, "enabled").Set(True)

    timeline.play()

    lines.append("t=0.0 s: Fire + DenseSmoke enabled")

    start = time.monotonic()
    last_logged_second = -1

    while True:
        now = time.monotonic()
        t = now - start

        if t >= DURATION_S:
            break

        g = growth_curve(t)

        # -------------------------------------------------------------
        # FIRE
        # -------------------------------------------------------------

        fire_radius = lerp(
            FIRE_RADIUS_START_M,
            FIRE_RADIUS_END_M,
            g,
        )

        fire_fuel = lerp(
            FIRE_FUEL_START,
            FIRE_FUEL_END,
            g,
        )

        fire_temp = lerp(
            FIRE_TEMP_START,
            FIRE_TEMP_END,
            g,
        )

        fire_up = lerp(
            FIRE_UP_START,
            FIRE_UP_END,
            g,
        )

        attr(fire, "radius").Set(float(fire_radius))
        attr(fire, "fuel").Set(float(fire_fuel))
        attr(fire, "temperature").Set(float(fire_temp))

        # Small asymmetric movement after ~5s adds natural-looking motion.
        lateral_phase = smoothstep01((t - 5.0) / 3.0)

        fire_lat = (
            LATERAL_VELOCITY_MAX
            * 0.45
            * lateral_phase
            * math.sin(1.7 * t)
        )

        attr(fire, "velocity").Set(
            Gf.Vec3f(
                float(fire_lat),
                0.0,
                float(fire_up),
            )
        )

        # -------------------------------------------------------------
        # SMOKE
        # -------------------------------------------------------------

        smoke_radius = lerp(
            SMOKE_RADIUS_START_M,
            SMOKE_RADIUS_END_M,
            g,
        )

        smoke_value = lerp(
            SMOKE_START,
            SMOKE_END,
            g,
        )

        smoke_temp = lerp(
            SMOKE_TEMP_START,
            SMOKE_TEMP_END,
            g,
        )

        smoke_up = lerp(
            SMOKE_UP_START,
            SMOKE_UP_END,
            g,
        )

        # Once developed, inject a mild changing horizontal component.
        # Collision with the ceiling should then make the plume spread further.
        smoke_lat_x = (
            LATERAL_VELOCITY_MAX
            * lateral_phase
            * math.sin(0.9 * t)
        )

        smoke_lat_y = (
            LATERAL_VELOCITY_MAX
            * lateral_phase
            * math.cos(0.7 * t)
        )

        attr(smoke, "radius").Set(float(smoke_radius))
        attr(smoke, "smoke").Set(float(smoke_value))
        attr(smoke, "temperature").Set(float(smoke_temp))

        attr(smoke, "velocity").Set(
            Gf.Vec3f(
                float(smoke_lat_x),
                float(smoke_lat_y),
                float(smoke_up),
            )
        )

        current_second = int(t)

        if current_second != last_logged_second:
            last_logged_second = current_second

            text = (
                f"t={t:4.1f}s | "
                f"growth={g:0.2f} | "
                f"fire r={fire_radius:0.2f}m T={fire_temp:0.2f} "
                f"| smoke r={smoke_radius:0.2f}m density={smoke_value:0.2f}"
            )

            print(text)
            lines.append(text)
            REPORT.write_text("\n".join(lines) + "\n")

        await omni.kit.app.get_app().next_update_async()

    # ---------------------------------------------------------------------
    # FINAL DEVELOPED STATE
    # ---------------------------------------------------------------------

    attr(fire, "radius").Set(FIRE_RADIUS_END_M)
    attr(fire, "fuel").Set(FIRE_FUEL_END)
    attr(fire, "temperature").Set(FIRE_TEMP_END)
    attr(fire, "velocity").Set(
        Gf.Vec3f(0.0, 0.0, FIRE_UP_END)
    )

    attr(smoke, "radius").Set(SMOKE_RADIUS_END_M)
    attr(smoke, "smoke").Set(SMOKE_END)
    attr(smoke, "temperature").Set(SMOKE_TEMP_END)
    attr(smoke, "velocity").Set(
        Gf.Vec3f(0.0, 0.0, SMOKE_UP_END)
    )

    # Safe end: PAUSE, never STOP.
    timeline.pause()

    await settle(8)

    lines += [
        "",
        "DONE.",
        "Timeline paused.",
        "Fire radius = 1.0 m",
        "Smoke radius = 1.0 m",
        "",
        "If the plume passes through the ceiling instead of spreading sideways,",
        "the next thing to fix is the CEILING COLLIDER, not the emitter growth.",
    ]

    REPORT.write_text("\n".join(lines) + "\n")

    print("\n".join(lines[-8:]))
    print("\nReport:", REPORT)


def done_callback(task):
    try:
        task.result()

    except asyncio.CancelledError:
        print("Growth controller cancelled.")

    except Exception:
        import traceback
        traceback.print_exc()


_task = asyncio.ensure_future(main())
_task.add_done_callback(done_callback)

print("Progressive Fire + Smoke growth scheduled.")
