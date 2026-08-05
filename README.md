# FireScenes Isaac Sim

Isaac Sim Version: 5.1.0-rc.19 Kit SDK Version: 107.3.3 Operating System: GPU: NVIDIA GeForce RTX 4070


Synthetic multimodal indoor-fire scenes and synchronized dataset recording for **Isaac Sim 5.1**.

The project currently combines:

- an Orbbec Gemini 335L virtual sensor rig;
- a provisional FLIR Lepton 3.1R-like thermal camera;
- NVIDIA Flow fire and smoke;
- a prone human character;
- RGB, IR-view, renderer-depth, semantic, and synthetic thermal outputs;
- scripted sensor-rig motion.

> **Project status:** research prototype / MVP. The generated thermal images are not yet calibrated LWIR radiometry.

---

## Environment used

| Component | Version / configuration |
|---|---|
| Isaac Sim | `5.1.0-rc.19` |
| Kit SDK | `107.3.3` |
| GPU used during development | NVIDIA GeForce RTX 4070-class GPU |
| Stage units | metres |
| Stage up axis | Z |
| Gemini render resolution | `640 × 480` |
| Thermal render resolution | `160 × 120` |
| Capture rate | `8.6 FPS` |
| Flow warm-up | `3.0 s`, normally 180 unsaved updates at 60 Hz |

The `.usda` scenes reference NVIDIA-hosted Isaac Sim and Flow assets, so those assets must be reachable when opening a scene.

---

## Current sensor rig

The main hierarchy is:

```text
/World
└── SensorRig
    ├── Gemini335L
    │   ├── camera_rgb/Camera_rgb
    │   ├── camera_ir_left/Camera_ir_left
    │   ├── camera_ir_right/Camera_ir_right
    │   └── camera_ir_left/Camera_Depth_Left
    └── ThermalCamera
```

The thermal camera is currently configured as a provisional Lepton 3.1R-like pinhole camera:

- resolution: `160 × 120`;
- horizontal FOV approximation: `95°`;
- focal length in the current USD camera: approximately `9.60086`;
- optical axis parallel to the Gemini RGB camera;
- optical centre placed provisionally **20 mm above** the Gemini RGB optical centre;
- zero provisional sideways and forward/back offset.

This 20 mm transform is useful for virtual development, but it is **not the final physical calibration**. Once the real bracket is assembled, the physical RGB-to-thermal extrinsics and thermal lens distortion must replace it.

Move only:

```text
/World/SensorRig
```

Do not independently move the RGB, IR, depth, or thermal child cameras after the relative transform has been fixed.

---

## What has been completed

1. Added the Gemini 335L sensor asset and a custom thermal camera.
2. Added a prone human character.
3. Added NVIDIA Flow fire and smoke.
4. Added semantic classes for the person, fire/smoke test objects, and floor surfaces.
5. Built an initial cone-based thermal MVP.
6. Added a 3-second Flow warm-up before recording.
7. Replaced the cone-based flame mask with a mask derived from the changing Flow render.
8. Tuned the Flow difference threshold to `60`.
9. Replaced the fixed image-row floor-reflection filter with a semantic floor mask.
10. Aligned the virtual thermal camera 20 mm above the Gemini RGB camera with parallel optical axes.
11. Recorded synchronized RGB, IR views, renderer depth, semantics, thermal arrays, camera parameters, and metadata.
12. Tested rigid motion of the complete sensor rig.
13. Added multi-waypoint motion.
14. Added a trajectory that approaches and moves above the person.
15. Prepared a further script that also rotates the full rig to look at the person.

---

# Scene files

All current scenes are under:

```text
scenes/base/
```

## `simple_room_base.usda`

Minimal starting scene.

Contains:

- the NVIDIA Isaac Sim Simple Room asset;
- default lighting and render configuration;
- no sensor rig;
- no person;
- no fire or smoke.

Use this only as the clean room baseline.

---

## `simple_room_sensor_rig.usda`

Adds the sensor rig to the Simple Room.

Contains:

- `/World/SensorRig`;
- Gemini 335L;
- custom `ThermalCamera`;
- Simple Room and lighting;
- no person;
- no fire or smoke.

Use it for camera-path, render-product, and calibration tests without incident effects.

---

## `simple_room_labeled.usda`

Early sensor-rig and Replicator setup scene.

Contains:

- Gemini 335L and thermal camera;
- Simple Room;
- Replicator stage objects/render products created during early testing;
- no person;
- no fire or smoke.

Despite the filename, this file is mainly an early rig/recording checkpoint rather than the current semantic-floor scene.

---

## `simple_room_person_fire.usda`

Adds a prone person and a Flow fire.

Contains:

- complete sensor rig;
- Simple Room;
- `F_Medical_01` lying on the floor;
- NVIDIA Flow fire;
- no separate DenseSmoke preset;
- no flame proxy.

Use it to test person/fire visibility without the additional smoke volume.

---

## `simple_room_person_fire_smoke.usda`

Adds the main multimodal incident elements.

Contains:

- complete sensor rig;
- prone person;
- Flow fire;
- DenseSmoke Flow preset;
- semantic labels for `person`, `fire`, and `smoke`;
- no flame-proxy mesh.

Use it as a clean no-proxy incident baseline.

---

## `simple_room_person_fire_smoke_proxy.usda`

Historical proxy-based thermal scene.

Contains:

- complete sensor rig;
- prone person;
- Flow fire;
- DenseSmoke;
- visible `/World/ThermalProxies/FlameProxy`;
- `class=flame` on the proxy.

This scene was used to validate the first semantic temperature lookup:

```text
background → 22 °C
person     → 34 °C
proxy      → 850 °C
```

It should not be used for the current no-proxy Flow-difference pipeline.

---

## `simple_room_person_fir_smoke_noproxy.usda`

No-proxy development checkpoint. Note the existing filename typo: `fir`.

Contains:

- sensor rig;
- person;
- Flow fire and smoke;
- semantic labels;
- a legacy `ThermalProxies/FlameProxy` prim that is stored invisible.

The current no-proxy recorder scripts do not use the legacy proxy. It remains only as a backward-compatible scene artifact.

---

## `fire_scene_rig_aligned_20mm_v1.usda`

Current aligned fire-scene checkpoint.

Contains:

- Gemini 335L and thermal camera;
- thermal camera aligned parallel to RGB and 20 mm above it;
- `160 × 120`, approximately `95°` thermal camera configuration;
- prone person with `class=person`;
- Flow fire;
- smoke scene objects;
- floor semantics used to reject reflected fire illumination;
- hidden legacy thermal proxy retained in the stage but not required by current scripts.

This is the recommended starting scene for the current recording pipeline.

---

# Script files

All scripts are under:

```text
scripts/
```

Run the recording scripts from **Isaac Sim's Script Editor**, not from the system Python interpreter.

## Empty placeholders

The following files are currently empty placeholders:

- `control_fire.py`
- `generate_dataset.py`
- `launch_scene.py`
- `record_baseline.py`
- `thermal_model.py`

They are reserved for later refactoring into smaller reusable modules.

---

## `thermal_mvp_ui.py`

First one-frame thermal proof of concept.

What it does:

- renders the `ThermalCamera` at `160 × 120`;
- temporarily shows the `FlameProxy`;
- reads semantic segmentation;
- maps background, person, and proxy classes to fixed temperatures;
- adds simple Gaussian noise;
- saves raw Celsius values, semantic IDs, a fixed-range image, an AGC image, and metadata;
- hides the proxy again.

Status: **historical MVP**.

Limitations:

- proxy-based flame;
- no Flow-derived dynamic flame;
- no synchronized Gemini outputs;
- no emissivity, reflected radiation, smoke attenuation, or calibrated Lepton response.

---

## `record_multimodal_mvp.py`

First full synchronized proxy-based sequence recorder.

What it adds:

- 3-second Flow warm-up;
- synchronized Gemini RGB, left/right IR views, renderer depth, and thermal outputs;
- cone hidden during Gemini capture;
- cone temporarily visible during a zero-time thermal semantic pass;
- separate ideal temperature and noisy thermal-sensor arrays;
- calibration files and frame metadata.

Status: **historical but stable proxy baseline**.

---

## `record_multimodal_proxy_backup.py`

Backup copy of the proxy-based multimodal recorder.

It is intentionally retained so the known-working cone workflow remains available while the no-proxy approach evolves.

---

## `record_multimodal_flow_no_proxy.py`

First recorder that removes the proxy from thermal generation.

For each logical frame it:

1. captures Gemini outputs and a thermal-camera RGB render with Flow visible;
2. suppresses Flow rendering without advancing simulation time;
3. captures a second thermal-camera render;
4. subtracts the second image from the first;
5. thresholds the positive difference at `FLOW_FLAME_THRESHOLD = 60`;
6. maps detected flame brightness to an empirical `450–1000 °C` range;
7. combines it with person semantics.

Status: **first working Flow-based no-proxy recorder**.

Known issue: bright fire illumination and reflections can also appear in the difference mask.

---

## `record_multimodal_flow_no_proxy_floor_filtered_v2.py`

Intermediate fixed-camera reflection filter.

It extends the first no-proxy recorder by removing detected flame pixels below a fixed image row.

Status: **debug/history only**.

Why it is obsolete:

- the image-row cutoff works only for one fixed thermal-camera viewpoint;
- it breaks when the sensor rig moves.

Use the semantic-floor version instead.

---

## `record_multimodal_flow_no_proxy_floor_semantic_v1.py`

Current stationary no-proxy thermal baseline.

It replaces the fixed row cutoff with:

```text
Flow difference mask
− thermal-camera pixels labelled as floor
= final flame mask
```

This makes floor-reflection rejection follow the floor when the viewpoint changes.

It also saves:

- `floor_semantic_mask.npy`;
- `floor_semantic_mask_preview.png`;
- Flow flame masks and scores;
- synchronized multimodal frames;
- calibration and metadata.

Status: **recommended fixed-rig baseline**.

---

## `record_multimodal_moving_rig_test_v1.py`

First synchronized moving-rig test.

It extends the semantic-floor recorder by moving the complete `/World/SensorRig` smoothly by `0.20 m` while recording 20 frames.

It verifies that:

- all cameras move rigidly together;
- RGB-to-thermal alignment is preserved;
- floor semantics continue to follow the viewpoint;
- per-frame world poses are stored;
- the rig is restored after recording.

Status: **validated motion test**.

---

## `record_multimodal_waypoint_translation_v1.py`

Multi-waypoint translation test.

It records 60 frames and moves the complete rig through a sequence of smooth translation waypoints:

```text
start
→ forward
→ farther forward
→ slightly sideways and forward
→ pause
```

It stores:

- segment index;
- segment progress;
- global motion progress;
- rig and camera poses.

Rotation is not included.

Status: **validated waypoint-translation prototype**.

---

## `record_multimodal_approach_person_v1.py`

Person-relative approach trajectory.

It computes the world-space bounding-box centre of:

```text
/World/Person/F_Medical_01
```

The RGB camera then moves smoothly toward a target approximately:

- `0.70 m` behind the person along the initial camera-forward direction;
- `0.70 m` above the person's centre.

The complete rig moves together and pauses at the target. Its orientation remains unchanged.

Status: **latest person-approach script currently present in the repository**.

---

## `record_multimodal_approach_and_look_at_person_v1.py`

Latest local development script; add it to `scripts/` if it has not yet been pushed.

It extends `record_multimodal_approach_person_v1.py` by:

- moving the complete rig closer to and above the person;
- smoothly rotating the complete rig;
- keeping the RGB/IR/depth/thermal calibration rigid;
- making the RGB camera progressively look toward the person.

Status: **next runtime validation target**.

---

# Current thermal generation method

The current no-proxy thermal pipeline is:

```text
Flow warm-up
    ↓
Capture thermal-camera RGB with Flow visible
    ↓
Set Flow render color/attenuation values to zero
    ↓
Capture thermal-camera RGB again at delta_time = 0
    ↓
Positive luminance difference
    ↓
Threshold at 60
    ↓
Remove floor-semantic pixels
    ↓
Map flame strength to empirical 450–1000 °C
    ↓
Combine with ambient and person temperature classes
    ↓
Add fixed-pattern and temporal noise
    ↓
Save Celsius arrays and 8-bit display
```

The Flow attributes temporarily suppressed are currently taken from:

```text
/World/Fire/Fire/flowRender/rayMarch.colorScale
/World/Fire/Fire/flowRender/rayMarch.attenuation
/World/Fire/Fire/flowOffscreen/colormap.colorScale
/World/Fire/Fire/flowOffscreen/shadow.attenuation
```

All original Flow values are restored after the reference render.

---

# Recorded dataset structure

A typical output is:

```text
<output_root>/
├── dataset_config.json
├── calibration/
│   ├── gemini_rgb_camera_params.json
│   ├── gemini_ir_left_camera_params.json
│   ├── gemini_ir_right_camera_params.json
│   ├── gemini_depth_left_camera_params.json
│   ├── thermal_camera_params.json
│   └── thermal_fixed_pattern_c.npy
└── frames/
    └── 000000/
        ├── rgb.png
        ├── ir_left.png
        ├── ir_right.png
        ├── depth_gt.npy
        ├── semantic_rgb.npy
        ├── thermal_semantic.npy
        ├── thermal_flow_visible_rgb.png
        ├── thermal_flow_suppressed_rgb.png
        ├── flow_difference_preview.png
        ├── flow_flame_mask.npy
        ├── flow_flame_mask_preview.png
        ├── flow_emission_score.npy
        ├── flow_flame_temperature_celsius.npy
        ├── floor_semantic_mask.npy
        ├── floor_semantic_mask_preview.png
        ├── temperature_gt_celsius.npy
        ├── thermal_sensor_celsius.npy
        ├── thermal_display.png
        └── metadata.json
```

Exact files vary slightly between historical recorder versions.

---

# How to run a recorder

1. Open the desired `.usda` scene in Isaac Sim.
2. Keep the timeline stopped.
3. Open **Window → Script Editor**.
4. Use a fresh output directory in the script.
5. Run the script with `runpy`.

Example:

```python
from pathlib import Path
import runpy

SCRIPT_PATH = Path(
    "/home/apagnoux/Foresight_Isaac/scripts/"
    "record_multimodal_approach_person_v1.py"
)

runpy.run_path(
    str(SCRIPT_PATH),
    run_name="__main__",
)
```

The recorder starts the timeline, performs Flow warm-up, captures the sequence, restores Flow render settings, restores the rig when configured, and stops the timeline.

Do not run these scripts with:

```bash
python3 script.py
```

They depend on Isaac Sim's embedded Python environment and Omniverse modules.

---

# Semantic requirements

Current no-proxy scripts expect:

```text
/World/Person/F_Medical_01 → class = person
```

Visible walking-floor surfaces should use:

```text
class = floor
```

Do not label wall trims, skirting, or vertical floor-rim pieces as floor unless they are genuinely part of the walking surface.

The no-proxy pipeline does not require:

```text
class = flame
```

because flame pixels are derived from the Flow render difference.

---

# Calibration notes

The repository includes diagnostic text files produced during virtual rig alignment:

- `camera_separation.txt`
- `camera_rig_alignment.txt`
- `camera_alignment_after_20mm.txt`
- `camera_offset_components.txt`

They document the progression from the original large virtual separation to the provisional aligned setup.

The current virtual result is approximately:

```text
right/left offset:     0 mm
up/down offset:       20 mm
forward/back offset:   0 mm
optical-axis angle:    0°
```

Physical calibration is still required after assembling the real Gemini + Lepton bracket:

1. thermal intrinsics and distortion;
2. Gemini RGB intrinsics verification;
3. RGB-to-thermal rotation and translation;
4. insertion of the measured transform into the virtual rig;
5. regeneration of calibration files.

---

# Important limitations

The present dataset is useful for pipeline development, synchronization, annotation testing, and synthetic perception experiments. It should not yet be described as physically accurate thermal sensing.

Current limitations:

- Flow render brightness is used as an empirical flame proxy;
- Flow's internal temperature/burn/smoke NanoVDB fields are not read directly;
- flame values mapped to `450–1000 °C` are assumptions, not calibrated measurements;
- person temperature is class-based and spatially uniform;
- background is initially uniform at `22 °C`;
- no material emissivity;
- no reflected LWIR radiation;
- no hot-surface heating model;
- no smoke attenuation in LWIR;
- no hot-smoke thermal emission;
- no Lepton lens distortion;
- only basic fixed-pattern and temporal noise;
- IR images are rendered camera outputs, not a calibrated Gemini NIR model;
- `depth_gt.npy` is perfect renderer distance, not Gemini stereo reconstruction;
- semantic floor rejection can erase true flame pixels where flame projects directly over a labelled floor;
- building degradation, secondary ignition, and structural failure are not implemented yet.

---

# Recommended current workflow

For a static rig:

```text
fire_scene_rig_aligned_20mm_v1.usda
+
record_multimodal_flow_no_proxy_floor_semantic_v1.py
```

For a simple motion test:

```text
fire_scene_rig_aligned_20mm_v1.usda
+
record_multimodal_moving_rig_test_v1.py
```

For a scripted path:

```text
fire_scene_rig_aligned_20mm_v1.usda
+
record_multimodal_waypoint_translation_v1.py
```

For approaching the victim:

```text
fire_scene_rig_aligned_20mm_v1.usda
+
record_multimodal_approach_person_v1.py
```

For approaching and turning toward the victim:

```text
fire_scene_rig_aligned_20mm_v1.usda
+
record_multimodal_approach_and_look_at_person_v1.py
```

---

# Next development steps

1. Validate the approach-and-look-at-person recorder at runtime.
2. Refactor repeated recorder code into reusable modules.
3. Add a configurable trajectory/scenario file instead of hard-coded constants.
4. Add animated people, beginning with crawling or small body motion.
5. Add one burnable object with states:
   ```text
   intact → heating → burning → charred
   ```
6. Spawn or move Flow emitters onto burning objects.
7. Add one pre-fractured falling object such as a ceiling tile.
8. Add PhysX rigid bodies, colliders, and a release event for debris.
9. Save object fire state, damage state, and actor state in each frame's metadata.
10. Replace Flow render differencing with NanoVDB temperature/burn/smoke sampling.
11. Add a physically motivated LWIR radiance and Lepton sensor model.
12. Validate synthetic outputs against the physical Gemini + Lepton rig.

---

# Suggested repository cleanup

The project currently preserves intermediate scripts because each records a milestone. As the pipeline stabilizes, consider:

```text
scripts/
├── archive/
│   ├── proxy/
│   ├── fixed_row_filter/
│   └── motion_tests/
├── record_dataset.py
├── scenario_controller.py
├── rig_motion.py
├── flow_thermal.py
├── thermal_sensor.py
└── config/
```

Also consider renaming:

```text
simple_room_person_fir_smoke_noproxy.usda
```

to:

```text
simple_room_person_fire_smoke_noproxy.usda
```

after updating any references.

---

## License

No license file is currently documented in this repository. Add one before redistributing project code or derived assets.
