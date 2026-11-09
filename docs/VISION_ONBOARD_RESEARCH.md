# Onboard vision + flying on the phone: research and plan (Claude Code, 2026-11-09)

the pilot's goal: the trained AI flies the real Mavic Mini through the controller, sees (cars, people, bicycles), follows a target, and avoids obstacles. He is fine with the OFFICIAL DJI Mobile SDK (v4) and will use his Galaxy Z Fold 7 as the phone. This note decides WHERE each piece runs and what to build and test, and says what is proven vs not.

## 1. Where does each piece run? (the core design question)
Chain: aircraft camera -> remote -> phone (SDK v4 decodes video) -> [perception] -> [pilot policy] -> [safety layer] -> virtual-stick command -> phone SDK -> remote -> aircraft.
- The VIDEO already lands on the phone. Running perception THERE avoids sending 720p video over Wi-Fi to the laptop (saves tens of ms and a failure point). The laptop is optional in the field.
- Two stages:
  A. "Laptop brain" (first): phone app = SDK relay + perception. Phone sends small results (boxes, tracks, coarse depth grid, telemetry) to the laptop at 10-15 Hz; laptop runs the pilot + safety and sends velocity setpoints back. Easy to debug, log and replay.
  B. "Phone brain" (goal): the pilot policy (tiny MLP) also runs on the phone, so the Fold 7 + remote is the whole flying system. Laptop only for logging/training.
- Safety layer ALWAYS runs closest to the aircraft (on the phone) and can veto any command: geofence/hover box, distance-to-obstacle limit from depth, link-loss -> hover/RTH, battery, and the pilot's physical sticks override (to be tested props-off). The AI proposes; the safety layer disposes.

## 2. What the Fold 7 can run (evidence, not promises)
- SoC is a Snapdragon 8 Elite class chip. Qualcomm AI Hub lists Depth Anything V2 Small (518x518, ~24.7M params) at ~46-48 ms on a Galaxy S25 NPU (TFLite or ONNX), ~41 ms on a newer chip, ~60 ms on S24 (https://huggingface.co/qualcomm/Depth-Anything-V2). So ~20 Hz depth is plausible; numbers vary between model-card revisions and the Fold 7 itself is UNMEASURED: bench it (card S4).
- LiteRT (new TFLite) with the Qualcomm AI Engine Direct (QNN) accelerator supports 8 Elite; Google reports many models under 5 ms on NPU (vendor numbers) (https://developers.google.com/edge/litert/next/qualcomm). Small detectors should be easily real time; the policy MLP (15->64->64->3) is negligible on CPU.
- Runtime options to compare in S4: LiteRT+QNN NPU, ONNX Runtime Mobile (QNN EP), NCNN (Vulkan GPU), plain CPU. Thermal throttling during 10+ minute runs is a real Android issue: log temperature and fps over time.
- Samsung/Android 16 compatibility of DJI SDK v4 is separately unproven (card S1).

## 3. Models (license-checked where I could)
- Detector: YOLOX-Nano/Tiny/S (Apache-2.0, Megvii; mobile-friendly, ONNX/NCNN/TFLite exports) or RF-DETR (Apache-2.0). Ultralytics YOLOv8/11/26 are AGPL-3.0: avoid for anything shipped/public unless the pilot accepts it (https://playground.roboflow.com/models/compare/yolo26-vs-yolox). Classes: person, bicycle, car (+truck/bus, motorcycle). Train/fine-tune: COCO-pretrained + VisDrone (aerial) + our AirSim synthetic frames (card Q4/Q5 stay the dataset/trainer cards).
- Tracker: ByteTrack (MIT, runs on boxes only, cheap on CPU) to keep "that bicycle" identity across frames; handle loss with re-acquire by appearance (Technion work uses template matching when the tracker restarts) (https://crml.eelabs.technion.ac.il/?p=5533).
- Depth: Depth Anything V2 Small (Apache-2.0; Base/Large are non-commercial) -> relative depth map -> reduce to a coarse grid (e.g. 3x5 of nearest-obstacle scores) as the obstacle observation. Monocular depth is RELATIVE (scale unknown); calibrate scale against telemetry/known distances, and never treat it as a laser. Reference results: a student policy on monocular Depth Anything V2 depth runs at 10 Hz with sim-to-real via domain randomisation (arXiv 2603.01999, ground robot); monocular-depth RL flight transferred sim-to-real (Expert Syst. Appl. 2022); depth-transfer with a sim-trained encoder (arXiv 2505.12428). Our plan follows the "depth as the middle layer" decision already in VISION_AVOIDANCE_RESEARCH.md.
- Pilot policy: SAC actor (MLP) exports to ONNX/LiteRT trivially; verify bit-level parity against the Python policy on recorded observations (card S6).

## 4. Follow-a-target and avoid-obstacles behaviour
- Baseline controller (no learning, write first): image-based visual servoing. Target box centre error -> yaw rate / lateral velocity; box height (or width) vs desired -> forward/back speed; hover when the target is lost, re-acquire; open-source reference: Texas A&M ardrone2_ibvs (BSD) commands at 15-25 Hz from the tracker box (https://unmanned.tamu.edu/projects/ardrone2_ibvs/); Stanley (OAK-D + MAVLink) for subsumption-style "follow nearest person" (https://github.com/Matchstic/stanley).
- Learned version: the RL pilot gets the target box (cx, cy, h, confidence, age) and the depth grid as extra observations next to the 15 state numbers; reward = keep target centred at set distance, never closer than d_min to any obstacle, smooth commands. Trained in AirSim with scripted moving targets (people/cars/bicycle models) and obstacles, with the REAL measured latency/gain/noise (S2a/S2b) and detector noise (dropouts, jitter) injected. Compare against the IBVS baseline on the same seeds.
- Layering (subsumption): safety/avoidance > follow > hover. Avoidance always wins.
- Limits to say out loud: Mavic Mini 1 has a single forward 720p camera, no obstacle sensors, ~13 min battery, weak wind tolerance, Wi-Fi-class video with ~200-300 ms glass-to-glass latency (to be measured); following a fast bicycle may exceed its speed/latency envelope. Airspace/LAANC (Class D nearby) and the pilot present for every test.

## 5. Getting frames out of the Mini via the official SDK (what is known)
- SDK v4 exposes decoded frames through DJICodecManager.YuvDataCallback.onYuvDataReceived(MediaFormat, ByteBuffer yuvFrame, int size, int w, int h) (https://developer.dji.com/api-reference/android-api/Components/CodecManager/DJICodecManager_YuvDataCallbackInterface.html); DJI's Android-VideoStreamDecodingSample uses FFmpeg parsing + MediaCodec and outputs YUV; DJI documents a hardware-decode problem on Mavic Mini / Mini SE and recommends software decoding (https://sdk-forum.dji.net/hc/en-us/articles/4674075760153). Convert YUV(NV12/I420)->RGB in native code, resize, feed the model. Unknown for the Mini 1: exact resolution/fps (expect 720p), and decoder behaviour on a Fold 7. YUV callbacks reportedly stop in some setups: log frame rate and add a watchdog.
- Alternative path: Pi bridge + DJI Link claims H.265 video decode on the PC (unproven here).

## 6. Evidence ladder (do not skip rungs; each rung is a card)
1. S4 bench (no aircraft): run detector + depth + policy on the Fold 7 against recorded/sim video; table of latency, fps, power, temperature over 10 min.
2. S5 video: SDK relay app decodes live Mini video, runs perception, logs detections (shadow mode, the pilot hand-flying or aircraft on the ground, props off); measure glass-to-glass latency.
3. S6 policy export: ONNX/LiteRT policy on the phone matches Python on 10k recorded observations; the safety layer unit-tested.
4. S7 follow/avoid in sim: IBVS baseline then RL with detections; report vs baseline, same seeds.
5. S2a/S2b real latency/gain into the sim; S3 staged real flight only with the pilot present, props off first, per-stage OK.
## 7. Following at a chosen distance (added 2026-11-09, the owner: "stay 5 or 10 feet away")
The Mini has ONE forward camera and no range sensor, so distance has to be inferred. Method on the phone (app 0.3.0, object `Range`):
- range = real_height / (box_height_fraction * C), C = 2*tan(vertical FOV / 2). Typical real heights: person 1.7 m, bicycle 1.0 m, car 1.5 m (editable). The default C (0.815) assumes full-width 16:9 video; the app has a "Calibrate range" button: select the target at a measured distance, enter feet, C is rescaled and saved.
- Smoothing: EMA per track; boxes cut by the frame edge are ignored; no forward motion without a usable range; never advance inside 1.5 m of the target.
- Honest accuracy: roughly +-20-30% (people vary, bicycle boxes may or may not include the rider, pitch of the aircraft/gimbal changes apparent height). Good enough for "about 15 ft", not for "exactly 15 ft". A second estimate can later come from the depth model, anchored to the box-height range (relative depth gets a scale from a known anchor).
- Controller (baseline, in the app): yaw from the horizontal image error, forward speed = 0.35 m/s per metre of range error, deadband 0.4 m, +-1 m/s caps via the safety layer.

### What this means for training (cards S7, S5, Q3)
1. The learned follower must be GOAL-CONDITIONED on distance: observation = [target bearing (box centre x), target range ESTIMATE, desired range, target visible flag, time since seen, plus the usual 15 state numbers], action = forward/right/down velocity + yaw rate. Desired range is sampled per episode (1.5 - 15 m) so one policy serves any distance the owner dials in.
2. The sim must give the policy the SAME measurement the phone gives, NOT ground truth: render the target at real size (person 1.7 m, bicycle, car), project it through the camera model to a box, compute range with the same formula, add noise (+-20%), box-edge dropouts and the real detector's latency/misses. A policy trained on perfect distance will fail on the real estimate.
3. Reward: -|true_range - desired| (shaped), keep the target centred, penalise jerk, big penalty inside 1.5 m, no reward for fast motion. Evaluate against the app's baseline controller on the same scripted target paths (walking, stop/start, turning).
4. Calibrate C on the real camera (S5) BEFORE trusting any real range number; log estimate vs tape-measured distance at 3, 5, 10, 15 m to put the real error bars into the sim noise.
