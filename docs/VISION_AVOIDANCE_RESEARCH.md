# Camera-based obstacle avoidance for the Mavic Mini — research note (Claude Code, 2026-11-08)

the pilot's goal: train a policy that flies the real Mavic Mini (Mini 1) and, looking through the drone's own camera, has a computer process the video in real time to see obstacles and avoid them. Free / open-source only.

## 1. The hardware chain (what is actually possible with a Mini 1)
- The Mini 1 has ONE forward camera on a gimbal and a downward vision sensor (used by DJI for position hold only). No forward obstacle sensors, no SDK access to raw sensors. So all forward obstacle sensing has to come from the camera video. That is exactly the gap this project fills.
- Video path: drone --(DJI Wi-Fi link)--> Mini 1 RC --(USB cable)--> Fold 7 running our MSDK v4 app (card S1). The SDK hands the app the live H.264 video (VideoFeeder; DJI recommends the transcoded feed for decoding, bitrate settable 0.3-20 Mbps).
- The PC cannot talk to the RC directly (MSDK is Android/iOS only). So the laptop gets the video FROM THE PHONE over the network and sends velocity commands back the same way (the S1 WebSocket contract, extended with a video stream).
- The Fold 7's only USB-C port is taken by the RC cable, so phone <-> laptop must be Wi-Fi: laptop joined to the phone's hotspot (lowest latency, works in the field), not Tailscale over the internet.
- Latency budget (estimates, MUST be measured with a stopwatch-on-screen test): DJI video link ~150-250 ms + phone re-encode/send ~30-60 ms + laptop decode + depth model ~30-50 ms (RTX 4050) + command back to drone ~50-100 ms => roughly 0.3-0.45 s from "obstacle appears" to "drone reacts". At 1 m/s that is ~0.4 m travelled blind: fly slowly (<= 1 m/s), stop distance >= 2 m. This is the hard limit of this hardware.
- Later option: run the depth model ON the Fold 7 (Snapdragon NPU; Depth Anything V2 Small is small enough) and only send a tiny depth image to the laptop, removing a video hop.

## 2. Recommended architecture: "depth as the middle layer" (modular, matches the Q10 decision)
camera video -> monocular depth model -> small depth image (e.g. 64x48) + drone state + goal direction -> RL policy -> velocity command -> DJI virtual stick (velocity mode) -> DJI's own flight controller.
Why: the policy never sees raw pixels, so the sim-to-real gap is in depth (much smaller and easier to fake in sim) instead of in colours/textures/lighting. This is the standard approach in the open literature (UZH "Learning High-Speed Flight in the Wild", NTNU Aerial Gym depth navigation, "Depth Transfer" 2025).
Plus a NON-learned safety layer that always runs: if the centre of the depth image says anything is closer than X m -> brake + hover; geofence; low battery; link loss -> hover/land (S1 watchdog). The learned policy steers; the rule-based layer has the final say.

## 3. Open-source building blocks (license checked where noted)
Perception (monocular depth):
- Depth Anything V2 — Small (25M) is Apache-2.0 [checked]; Base/Large/Giant are CC-BY-NC-4.0 (personal/non-commercial only) [checked]. Metric-depth fine-tunes exist for Small/Base (check license in the repo). Small at ~256-384 px runs in real time on an RTX 4050. FIRST CHOICE.
- Video Depth Anything (temporally stable depth for video) — worth testing for flicker; check license/speed.
- Metric scale: relative depth + DJI telemetry (height above takeoff, velocity) to fix scale, or a metric variant. Absolute accuracy is not needed for "something is close in front".
Odometry / mapping (for Q10 discovery flights, not needed for basic avoidance):
- DPVO / DPV-SLAM (Princeton, deep visual odometry; MIT as far as I know — verify), MASt3R-SLAM (CVPR 2025, monocular dense SLAM ~15 FPS; NAVER checkpoints are non-commercial — fine for personal), ORB-SLAM3 (GPLv3, classic).
- Map fusion: nvblox (NVIDIA, Apache-2.0, GPU TSDF/ESDF) or OctoMap/voxblox for the occupancy map the planner and the reward use.
- Mini 1 caveat: no hardware-synced IMU stream, so VIO-style methods (VINS) are out; vision-only odometry + DJI telemetry for scale.
Policy training (the camera policy):
- Aerial Gym Simulator (NTNU, open source, Isaac Gym, GPU-parallel depth rendering, ships depth-based RL navigation examples, sim2real shown) — thousands of parallel drones, trains depth-avoidance policies in hours. Needs an NVIDIA GPU: the Yoga's RTX 4050 (6 GB; keep env counts modest). Our Project AirSim setup runs ~0.7 s/step real time — far too slow for a depth-image policy that needs millions of steps; keep AirSim for VALIDATION (realistic visuals, the pilot's yard).
- Differentiable-physics depth->control policies (2025-26: e.g. "Simple but Stable, Fast and Safe", github Fanxing-LI/avoidance; DiffRacing) — very fast to train; check code/licenses; output thrust/body rates, which DJI does not accept, so we would keep a velocity-command action instead.
- Sim-to-real for depth: train a teacher on perfect sim depth, then a student on depth that looks like Depth Anything's errors: either add realistic noise/blur/holes/scale error, or (better) run Depth Anything on the sim's RGB renders during student training ("Depth Transfer", arXiv 2505.12428). Domain-randomize lighting, textures, obstacle shapes.
Detection (Q5, separate): RF-DETR / YOLOX for "what is it", not needed for avoidance itself.

## 4. How it fits the plan we already have
1. Q3 (now): hover with SAC on state only. Teaches the command interface; no camera.
2. NEW stage "A1 depth-avoidance policy": Aerial Gym, depth image + state + goal -> velocity command (same ±1 m/s velocity interface as Q3 and S1). Teacher on GT depth, student on Depth-Anything-style depth.
3. NEW "A2 validate in Project AirSim + a real-world test area": AirSim depth camera config already exists (robot_quadrotor_fastphysics_depth_lidar.jsonc); run the student with Depth Anything on AirSim's RGB camera, in Blocks then in the pilotYard (Q11). Measure collisions = 0 over N flights at <= 1 m/s.
4. S1 extension: the phone app also streams video (H.264 over the hotspot) to the laptop; laptop runs Depth Anything + policy + safety layer; measure end-to-end latency with a stopwatch filmed through the drone camera.
5. S3 extension: real flights props-off -> tethered -> outdoor toward a soft obstacle (cardboard box / pool noodle on a stand), slow, the pilot on the sticks.
6. CCW-Net (friend's paper) on the avoidance policy: concepts like obstacle_close_ahead, gap_left/right, closing_speed, goal_blocked — "why did it turn left?" is exactly the paper's question, and the clearest demo of it in this project.
7. Q10 discovery flights add the SLAM + map layer on top.

## 5. Honest limits
- 0.3-0.45 s latency + one camera + no forward sensors => slow, cautious flight only; thin branches and wires are near-invisible to monocular depth (and to a real-world test area scan): never rely on it near power lines.
- DJI's flight controller sits under our commands (velocity mode) — good for safety, but the policy can't do aggressive manoeuvres.
- Everything above is free/open; the only registration is the DJI developer app key MSDK needs.

Sources: DJI MSDK VideoFeeder docs (developer.dji.com android-api DJIVideoFeeder); Depth Anything V2 repo (github.com/DepthAnything/Depth-Anything-V2); Aerial Gym (arXiv 2503.01471, github.com/ntnu-arl/aerial_gym_simulator); Agilicious (arXiv 2307.06100); "Simple but Stable, Fast and Safe" (arXiv 2604.10548); DiffRacing (arXiv 2603.08019); Depth Transfer (arXiv 2505.12428); Flying in Clutter on Monocular RGB (arXiv 2512.17349); MASt3R-SLAM (arXiv 2412.12392, CVPR 2025).
