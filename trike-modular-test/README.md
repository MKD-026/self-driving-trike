# Modular route pipeline

The live modular launcher uses three independent TensorRT contexts on the same
1120x416 stitched panorama:

```text
detection     models/yolo26s_q32.engine
segmentation  models/yolo26s-sem_q32.engine
depth         models/yolo26s-depth_q32.engine
```

They are submitted concurrently. Detection supplies BEV objects and each box samples the fused metric depth map. YOLO depth supplies dense relative/metric depth.
Valid OAK stereo pixels are projected into the stitched panorama, used to fit a
robust scale and offset for YOLO depth, and then retained as the preferred metric
values in the fused map.

The semantic segmentation engine supplies dense road and sidewalk classes. The road mask is projected into BEV with fused depth and constrains the avoidance target before pure pursuit. Fixed road-width limits remain as a safety fallback.

Route loading, VectorNav parsing, stale-fix rejection, monotonic route progress,
and interpolated lookahead selection use the same implementation as the VLA
pipeline. The default lookahead is 5 m. A stale/missing GPS fix or a completed
route disarms steering.

Start with the ESP disabled by omitting the ESP launcher and using dry-run:

```bash
./run_autonomous.sh --dry-run --no-tts
```

After verifying the BEV and route direction, use the ESP launcher:

For a read-only live browser view using the same camera/model pass, add
`--web-dashboard --no-imshow` and open `http://<jetson-ip>:8766` on the same network.

```bash
./run_autonomous_esp.sh --enable-esp --no-tts
```

Add `--auto-arm` to make that specific launch arm itself whenever route,
stitched depth, and road-corridor safety checks are valid. It centers
automatically if any check becomes invalid; no `A` key is required.

Useful overrides:

```text
--route-file PATH
--route-name NAME
--route-lookahead-m 5
--vn-port PATH
--vn-baud 115200
--gps-max-age 1.0
--no-route
```

The process starts disarmed. Press `A` only after cameras, fused depth, and a
fresh route target are ready.


## Active source map

- `autonomous_driving.py`: live orchestration, safety state, preview, and pure-pursuit command loop.
- `autonomous_driving_esp.py`: explicit ESP hardware adapter and calibration preflight.
- `parallel_perception.py`: parallel detection, semantic segmentation, depth, and OAK/YOLO fusion.
- `bev.py`: metric obstacle footprints and width-aware pure-pursuit avoidance.
- `road_bev.py`: valid-panorama masking and semantic road boundaries.
- `route_following.py`: monotonic `route1` progress and metric lookahead target.
- `gps.py`: VectorNav/NMEA serial reader and stale-fix rejection.
- `oak_live_inputs.py`: synchronized three-OAK RGB/depth acquisition.
- `image_processing.py`: validated Mac-equivalent cylindrical SIFT renderer for RGB and OAK depth.
- `sift_stitch_config.json`: fixed frame-300 rig calibration, cam0=left/cam1=center/cam2=right.
- `inputs.py`: camera packet types and nonblocking terminal controls.
- `esp_controller.py`: fail-centered serial command stream.
- `replay_dataset.py`: offline dataset validation with video, JSONL, and waypoint CSV outputs.
- `models/`: the generic detection type and three active TensorRT engines.

`run_autonomous.sh` is dry-run-only. `run_autonomous_esp.sh --enable-esp` is the only hardware-actuating path.
