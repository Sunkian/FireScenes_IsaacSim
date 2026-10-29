#!/usr/bin/env python3
"""
make_lepton_from_v3_1.py

Offline post-processing for the working v3.1 capture.

It DOES NOT run Isaac Sim and DOES NOT modify the recorder.

Input sequence:
  /scratch/apagnoux/record_dual_pov_semantics_v3_1/sequence_001/

Required per frame:
  thermal_semantic_usd/<frame>.npy
  thermal_semantic_usd/<frame>_labels.json
  thermal_flow_burn/<frame>.npy
  thermal_flow_smoke_tau/<frame>.npy

Outputs:
  thermal_gt_c/<frame>.npy
  thermal_lepton_c/<frame>.npy
  thermal_lepton_preview/<frame>.png

The product is a physics-inspired Lepton-like LWIR simulation, not a
full spectral FLIR radiative-transfer model.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image

SEQ = Path("/scratch/apagnoux/record_dual_pov_semantics_v3_1/sequence_001")

SEM_DIR = SEQ / "thermal_semantic_usd"
BURN_DIR = SEQ / "thermal_flow_burn"
SMOKE_DIR = SEQ / "thermal_flow_smoke_tau"

GT_DIR = SEQ / "thermal_gt_c"
LEPTON_DIR = SEQ / "thermal_lepton_c"
PREVIEW_DIR = SEQ / "thermal_lepton_preview"

for d in (GT_DIR, LEPTON_DIR, PREVIEW_DIR):
    d.mkdir(parents=True, exist_ok=True)

AMBIENT_C = 22.0

SURFACE_TEMP_C = {
    "background": 22.0,
    "unlabelled": 22.0,
    "unlabeled": 22.0,
    "person": 33.0,
    "floor": 25.0,
    "wall": 23.5,
    "ceiling": 24.0,
    "chair": 24.5,
    "sofa": 25.0,
    "door": 23.5,
}

# From the earlier warmed Fire burn diagnostics.
BURN_MIN = 0.005
BURN_FULL = 0.063
FIRE_MIN_C = 120.0
FIRE_MAX_C = 850.0

# Dense smoke: partial LWIR attenuation/emission, not opaque RGB smoke.
SMOKE_IR_COEFF = 0.70
SMOKE_MIN_C = 35.0
SMOKE_MAX_C = 180.0

# Lepton-like low-gain sensor approximation.
LEPTON_MIN_C = -10.0
LEPTON_MAX_C = 400.0
NOISE_SIGMA_C = 0.05
FPN_SIGMA_C = 0.03
QUANT_C = 0.01
RNG_SEED = 3101


def parse_id_to_label(info):
    if not isinstance(info, dict):
        return {}

    mapping = info.get("idToLabels", {})

    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except Exception:
            mapping = {}

    result = {}

    if isinstance(mapping, dict):
        for k, v in mapping.items():
            try:
                sid = int(k)
            except Exception:
                continue

            if isinstance(v, str):
                label = v
            elif isinstance(v, dict):
                label = (
                    v.get("class")
                    or v.get("semanticLabel")
                    or v.get("label")
                    or v.get("name")
                )
                if label is None and v:
                    label = next(iter(v.values()))
            else:
                label = None

            if label is not None:
                result[sid] = str(label).lower()

    return result


def surface_temp_map(ids, labels):
    out = np.full(ids.shape, AMBIENT_C, dtype=np.float32)

    for sid in np.unique(ids):
        sid = int(sid)
        name = labels.get(sid, "")
        temp = AMBIENT_C

        for key, value in SURFACE_TEMP_C.items():
            if key in name:
                temp = float(value)
                break

        out[ids == sid] = temp

    return out


def c_to_k(x):
    return np.asarray(x, dtype=np.float32) + 273.15


def make_gt(surface_c, burn, smoke_tau):
    burn = np.asarray(burn, dtype=np.float32)
    smoke_tau = np.asarray(smoke_tau, dtype=np.float32)

    flame_alpha = np.clip(
        (burn - BURN_MIN) / max(BURN_FULL - BURN_MIN, 1e-6),
        0.0,
        1.0,
    )
    flame_temp = FIRE_MIN_C + flame_alpha * (FIRE_MAX_C - FIRE_MIN_C)

    smoke_alpha = 1.0 - np.exp(
        -SMOKE_IR_COEFF * np.clip(smoke_tau, 0.0, None)
    )
    smoke_alpha = np.clip(smoke_alpha, 0.0, 0.90)

    smoke_strength = np.clip(smoke_tau / 1.5, 0.0, 1.0)
    smoke_temp = (
        SMOKE_MIN_C
        + smoke_strength * (SMOKE_MAX_C - SMOKE_MIN_C)
    )

    # Approximate radiance mixing in Kelvin^4 space.
    L = c_to_k(surface_c) ** 4

    fire_L = c_to_k(flame_temp) ** 4
    L = (1.0 - flame_alpha) * L + flame_alpha * fire_L

    smoke_L = c_to_k(smoke_temp) ** 4
    L = (1.0 - smoke_alpha) * L + smoke_alpha * smoke_L

    return (np.maximum(L, 1.0) ** 0.25 - 273.15).astype(np.float32)


def blur3x3(x):
    x = np.asarray(x, dtype=np.float32)
    p = np.pad(x, ((1, 1), (1, 1)), mode="edge")

    return (
        p[:-2, :-2] + 2*p[:-2, 1:-1] + p[:-2, 2:]
        + 2*p[1:-1, :-2] + 4*p[1:-1, 1:-1] + 2*p[1:-1, 2:]
        + p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:]
    ) / 16.0


def make_lepton(gt, fixed_pattern, rng):
    x = blur3x3(gt)
    x += fixed_pattern
    x += rng.normal(0.0, NOISE_SIGMA_C, x.shape).astype(np.float32)
    x = np.clip(x, LEPTON_MIN_C, LEPTON_MAX_C)
    x = np.round(x / QUANT_C) * QUANT_C
    return x.astype(np.float32)


def save_preview(path, x):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)

    vis = np.zeros(x.shape, dtype=np.uint8)

    if np.any(finite):
        vals = x[finite]
        lo = min(float(np.percentile(vals, 2)), 20.0)
        hi = max(float(np.percentile(vals, 98)), 40.0)

        if hi <= lo:
            hi = lo + 1.0

        norm = np.zeros_like(x)
        norm[finite] = (x[finite] - lo) / (hi - lo)
        norm = np.clip(norm, 0.0, 1.0)
        vis[finite] = np.round(norm[finite] * 255).astype(np.uint8)

    Image.fromarray(vis, mode="L").save(path)


frames = sorted(p.stem for p in BURN_DIR.glob("*.npy"))

if not frames:
    raise RuntimeError(
        f"No burn maps found in {BURN_DIR}. "
        "Run the v3.1 recorder first."
    )

rng = np.random.default_rng(RNG_SEED)

# Fixed spatial pattern, constant across the sequence.
first_burn = np.load(BURN_DIR / f"{frames[0]}.npy")
fixed_pattern = rng.normal(
    0.0,
    FPN_SIGMA_C,
    size=first_burn.shape,
).astype(np.float32)

np.save(SEQ / "thermal_fixed_pattern_c.npy", fixed_pattern)

for i, stem in enumerate(frames):
    sem_path = SEM_DIR / f"{stem}.npy"
    label_path = SEM_DIR / f"{stem}_labels.json"
    burn_path = BURN_DIR / f"{stem}.npy"
    smoke_path = SMOKE_DIR / f"{stem}.npy"

    if not (sem_path.exists() and burn_path.exists() and smoke_path.exists()):
        print("SKIP incomplete frame:", stem)
        continue

    ids = np.load(sem_path).astype(np.int32)

    if label_path.exists():
        with open(label_path) as f:
            info = json.load(f)
    else:
        info = {}

    labels = parse_id_to_label(info)
    surface = surface_temp_map(ids, labels)

    burn = np.load(burn_path).astype(np.float32)
    smoke_tau = np.load(smoke_path).astype(np.float32)

    gt = make_gt(surface, burn, smoke_tau)
    lepton = make_lepton(gt, fixed_pattern, rng)

    np.save(GT_DIR / f"{stem}.npy", gt)
    np.save(LEPTON_DIR / f"{stem}.npy", lepton)
    save_preview(PREVIEW_DIR / f"{stem}.png", lepton)

    if i % 10 == 0:
        print(
            stem,
            "GT[min/max]=",
            float(np.min(gt)),
            float(np.max(gt)),
            "Lepton[min/max]=",
            float(np.min(lepton)),
            float(np.max(lepton)),
        )

print("DONE")
print("GT:", GT_DIR)
print("Lepton:", LEPTON_DIR)
print("Preview:", PREVIEW_DIR)
