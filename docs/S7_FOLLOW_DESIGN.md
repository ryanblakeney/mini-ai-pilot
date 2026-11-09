# S7 design: follow-a-target + obstacle avoidance in the sim (2026-11-09)

Grounds: VISION_ONBOARD_RESEARCH.md §4, VISION_AVOIDANCE_RESEARCH.md, Q3 env family
(gym_env.py / waypoint_env.py / obstacle_env.py). Card rules honored: survival-shaped
reward + terminal-penalty dominance proof BEFORE launch; compare vs scripted baseline
on identical seeds; layering avoidance > follow > hover; ONE training task at a time;
sim work only after the Q3 waypoint_v2 relaunch releases the sims.

## D1 Detector = synthetic GT projection (phase 1, no rendering, no real detector)
The RL obs is a target box + depth grid, never pixels (research decision "depth as
the middle layer"). Phase 1 computes the box analytically: target 3D pose (scripted)
+ camera pose (rigid to drone body) -> pinhole projection -> box (cx, cy, h_px).
Rationale: the rig runs ~0.7 s/step; a real detector in the training loop does not
fit and is not needed to train a box-conditioned policy. Detector error model
(D6) stands in for YOLO errors. Real-detector validation (YOLOX/RF-DETR on rendered
frames) is a LATER validation rung (cards S5/Q5), and camera capture through this
projectairsim rig is itself unproven — do not conflate the two.

## D2 Moving targets are virtual (kinematic script), not spawned meshes
Phase 1 target = a scripted 3D position + physical height (no scene object).
The drone never collides with the target, so physics interaction is irrelevant for
training, and the projectairsim move/spawn API for scene objects is unproven on this
rig (risk item). Scripts (per-episode, from env rng):
- person : speed 0.8 m/s, straight, heading U[0,2pi), height 1.7 m
- bicycle: speed 2.5 m/s, heading swings +-45 deg, period 8 s, height 1.1 m
- car    : speed 6.0 m/s, straight road, height 1.5 m (often exits FOV -> loss/re-acquire)
Spawn start: ground level, horizontal range U[6,12] m, inside the camera FOV.
FPV_TARGET_SPEED_SCALE (default 1.0) multiplies all script speeds: at 1.0 the
bicycle and car structurally outrun the +-1 m/s drone (short chase windows by
design); training smoke should also try ~0.3 to put all kinds inside the
velocity envelope. Recorded as script['speed_scale'].
Visual CC0 meshes (kids-open-assets rules, e.g. Quaternius person/bike/car, or Blocks'
built-in Skele if it loads) are DEFERRED to a video/validation run once the move-object
API is smoke-tested. Honest limitation, stated in the report.

## D3 Action space 4-D: [v_n, v_e, v_d, yaw_rate]
v* m/s +-1 exactly as Q3 (inner velocity controller reused verbatim). New dim
yaw_rate = action[3]*0.5 rad/s held for the step (Mini reality: fixed forward cam,
pointing is the only way to keep cx centered). Implementation: try the client's
rotate-by-velocity / rotate-to-yaw async API — UNTESTED on this rig; first smoke
must verify it (props-off equivalent: sim only). Fallback FPV_YAW=0 -> 3-D
translation-only follow (strafe), action space reverts to 3 dims; IBVS baseline
adapts automatically (it just commands no yaw).

## D4 Camera model (synthetic)
Rigid to body, optical axis forward, pitch -20 deg below horizon (Mini gimbal-neutral
approx). 720x480, fx=fy=640, cx0=360, cy0=240. Box: project target base P and top
P+[0,0,-H]; h_px = f*H/dist (clip); cy = midpixel of the two projections (clip to
frame); in_fov = both points in front of camera plane and cy/cx inside frame.
All parameters env-overridable.

## D5 Observation (Box 35 when yaw on)
[state 15] as Q3 but pos_err rebased is unused for follow (keep raw hover err),
then [5]: cx_err = (cx-cx0)/cx0 clipped [-1,1]; cy_err same; log(h_px/h_star)
(h_star = box height at desired range, per target kind); conf [0..1]; age = steps
since last hit /10 clipped [0..1].
then [15] depth grid 3 rows x 5 cols (row-major): for each pixel cell, min distance
1..20 m to any scene-object AABB intersecting that pixel cone; normalized
(d-1)/19; no obstacle in cone = 1.0 (free). Grid from the same scene object list the
stage-3 geometry collision route uses (list_objects + poses + AABB).

## D6 Detector noise (phase 1 model, replaceable by measured S5 stats)
Per step, on the GT box: hit = bernoulli(1 - p_miss), p_miss=0.15 default;
jitter cx,cy ~ N(0, 0.5% of frame dim); h *= (1+N(0,0.05)); conf = clip(N(0.85,0.08)).
On miss: box frozen at last value, age++. Latency: the box the agent sees at step t
is the GT box of step t-L, L ~ U{0,1} steps (0.5 s dt; matches the research
0.3-0.45 s estimate; when S2a/S2b measure real end-to-end latency set FPV_LAT_MS and
it dominates: L = round(lat/step_dt) sampled in [0, lat_hi]). Domain randomization
switch FPV_DR=1 randomizes p_miss U[0.05,0.25], jitter 0.25-1%, latency per episode.

## D7 Reward v5 (follow) — survival-shaped, dominance-checked (check_follow_reward.py)
in_view = box detected (not aged out > K=20 steps)
center  = max(0, 1 - hypot(cx_err, cy_err))          # 1 at dead center
range_t = max(0, 1 - |log(dist/dist_star)| )        # dist_star: 6 m default
r_step  = 0.5*in_view + 0.4*center + 0.2*range_t
          - 0.05*||a|| - 0.03*||a - a_prev|| - 0.02*||v||
terminals (bad-only => early term never helps): landed -10; drone >12 m from origin
-10; collision -20. trunc at max_steps=100. No positive terminal anywhere.
Per-step ceiling = 1.1 - penalties < r_step > -0.0266-ish floor... the checker
asserts: (a) camping/hover-at-start sum < any decent-tracking full episode;
(b) every terminal penalty magnitude > max remaining harvest (bound: 100*1.1);
(c) ordering under gamma 0.98 preserved. Pure function follow_reward(), numpy-only,
runs on bare sable python3 like the Q3 checkers.

## D8 IBVS baseline (ibvs_follow.py, pure numpy, unit-tested offline)
Inputs: box (cx, cy, h_px, conf, age) + state. Outputs action (same 4-D box):
- yaw_rate = clip(kp_yaw * cx_err, +-1) * 0.5 rad/s  (kp_yaw=0.6, only when |cx_err|>deadband 0.05)
- forward (body-forward axis): v_f = clip(kp_f * (h_px/h_star - 1), +-1); too big -> back off
- vertical: keep h-consistent range error small: v_d = clip(kp_z*(h_px/h_star -1)*0.5, +-1)*0.5 (hold alt mostly; z trim small)
- lateral: v_lat = clip(kp_lat * cy_err?, +-1) — with yaw servo + forward servo, cy error drives lateral (kp_lat=0.8)
- hover-on-loss: age > K_loss(=4): v=0, yaw holds last; re-acquire implicit (scene moves/target keeps moving).
Anti-double-move discipline: one integrated action per step (Q3 gimbal lesson).
All gains FPV_IBVS_* env-overridable for tuning. Unit tests: synthetic box sequences
(plateau test: static target -> converges; moving target -> keeps error bounded).

## D9 Files (local mirror sim/, later deployed to D:\fpv\rl after Q3 relaunch)
- sim/follow_common.py  — pure: pinhole projection, target scripts, detector noise,
  depth grid, follow_reward (numpy only)
- sim/check_follow_reward.py — dominance proof + projection/noise property tests (assert-based, exits 0)
- sim/ibvs_follow.py    — pure IBVS controller (numpy only)
- sim/test_ibvs_follow.py — offline unit tests on synthetic box streams
- sim/follow_env.py     — FollowDroneEnv(BlocksDroneEnv), sim-bound part only
- sim/rl_eval_follow.py — eval rig: zero + IBVS baselines + policy, identical seeds;
  verdict FOLLOW_OK iff mean|center err| < baseline's AND collisions 0 AND loss time <
  baseline's; emits EP/CSV lines in the rl_eval_waypoint.py house style
- sim/rl_train_follow_multi.py — copy of rl_train_waypoint_multi.py (already carries
  the CkptEvery fix), env swapped, FPV_RUN default runs/follow_v1
No file touched that the live Q3 relaunch imports. Nothing runs on the Yoga until the
Q3 relaunch's TRAIN_DONE.

## D10 Launch/eval protocol (later, when sims free)
Pre-flight (unchanged Q3 gates): FPVTrainV4 Ready + FPVEvalV4 Ready + no Blocks.exe +
Q3 relaunch TRAIN_DONE. Smoke: 1 sim, 3 eps, FPV_YAW API check, collision route log.
Train: SAC v7 recipe on N=6 sims (Q3p), 48k global steps, ckpt/2500.
Eval: >=4 eps x 120 steps per condition, seeds shared with baselines, over
person/bike/car scripts; report center-err, range-err, loss-frag-time, collisions,
smoothness; learned vs IBVS vs zero, identical seeds; failures reported honestly;
page_journal entry per milestone.

## Implementation notes (2026-11-09 code pass, offline-verified)
- IBVS forward-servo sign: spec D8's printed literal contradicted its own prose;
  prose (too-big -> back off) implemented, documented in ibvs_follow.py.
- Dominance formalization: with v5 every terminal is bad-only and a zero-action
  idle step always yields r >= 0, so any terminal is a strict |P| loss from every
  reachable state — that, plus per-kind EXPECTED CAMP < TRACK (60 natural scripts
  through the real D6 detector: person 30.8<48.3, bike 23.7<53.3, car
  18.9<46.8), is what check_follow_reward.py proves. STRICT per-script dominance
  is impossible for the rare receding-along-LOS scripts (hover farms in_view+center
  for free) — that is exactly open-risk 3's loss/re-acquire family; eval scores
  per-visible-step, loss time is a first-class column.
- Checker imports follow_common directly (pure by design); no regex extraction.

## Open risks (tracked)
1. projectairsim rotate API untested (fallback D3).
2. Scene-object list may include the drone itself / odd AABBs — filter by name, verify in smoke.
3. Target scripts can outrun the drone (car) — by design for loss/re-acquire, but the
   eval metric must separate "left frame" from "lost forever": score per-visible-step.
4. Depth grid from AABB cones may be near-useless in the sparse Blocks scene — if so,
   avoidance degrades to collision-terminal pressure only; document, don't fake it.
