# Third-party components and their licences
This project is licensed AGPL-3.0 (see LICENSE) so it can combine components under GPL-3.0 / AGPL-3.0 as well as permissive licences.
Third-party code and models keep their own licences. Large model weights are NOT stored in this repository; download them from their sources.

| Component | Use | Licence |
|---|---|---|
| Depth Anything V2 Small (metric) | monocular depth (fine-tuned locally; fine-tuned weights are private) | Apache-2.0 |
| Depth Anything V2 Base/Large, other CC BY-NC models | offline comparison / label teachers only, never redistributed | CC BY-NC 4.0 |
| Ultralytics YOLOv8n | person detection | AGPL-3.0 |
| OSNet (torchreid) | person re-identification | MIT |
| OpenCV | feature tracking, geometry | Apache-2.0 |
| ONNX Runtime (+ QNN execution provider) | on-phone inference | MIT |
| Rerun | 3-D viewer | MIT / Apache-2.0 |
| COLMAP | 3-D reconstruction for depth labels | BSD-3-Clause |
| Project AirSim / AirSim | simulator | MIT |
| Stable-Baselines3 | RL training (earlier phases) | MIT |
| DJI Mobile SDK v4 | aircraft link (app) | DJI SDK terms - DJI files and the app key are never committed |
Ideas (not code) borrowed and cited in the docs: FASTER, SUPER, Primitive-Planner, YOPO, Elastic-Tracker, BoT-SORT, OctoMap, Bonxai.
