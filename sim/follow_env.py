"""FollowDroneEnv: S7 follow-a-target task (ProjectAirSim Blocks rig).

Subclasses BlocksDroneEnv (stage-1) and reuses ALL battle-tested plumbing
VERBATIM: persistent asyncio loop thread, sim-clock(ts)-gated steps, inverted
LandedState decode LANDED=0, takeoff+settle reset, disconnect-before-exit.
Subclass only -- the parent plumbing is load-bearing and must never be
rewritten (waypoint_env.py / obstacle_env.py precedent).

Task (docs/S7_FOLLOW_DESIGN.md, decisions D1-D14 LOCKED -- no redesign):
a virtual scripted target (person/bicycle/car, D2) moves on the ground;
the agent sees a synthetic detector box (D4 pinhole projection + D6 noise),
the phone-app apparent-size RANGE ESTIMATE of that box (D12, same formula +
EMA + +-noise + edge dropouts -- never ground truth), a per-episode DESIRED
follow range (D11, goal-conditioned), plus a 3x5 depth grid (D5), and must
keep the target centered at the desired range.

  action  = [v_n, v_e, v_d, yaw_rate] Box(4) +-1 (D3; yaw dim = +-0.5 rad/s
            via the client rotate API -- UNTESTED on this rig per D3, smoke
            verifies; FPV_YAW=0 falls back to Box(3) translation-only and
            LOGS once). step[3] applies for the step after issuing (rate
            held across the next step window).
  obs     = Box(37) per D13: [15] raw hover state (err3 vel3 rpy3 ang3
            prev-action-trans3) + [7] box feats: cx_err, cy_err (clipped
            [-1,1]), clip(log(range_est/desired), -2, 2) (0.0 with no valid
            est yet), desired/15.0, range_valid 0/1 (usable THIS step),
            conf, age/10 (clipped) + [15] depth grid row-major
            ((d-1)/19, free=1.0).
  reward  = reward v7, PURE follow_reward() in follow_common.py (D7 + D11 +
            D17: goal-conditioned range term DOMINANT (0.2/0.3/0.6 in_view/
            center/range, no |a|/|v| taxes), close-guard -2.0/step inside
            1.5 m, NO positive terminal); dominance PROVEN offline by
            check_follow_reward.py before any launch (card hard rule -- the
            Q3 hover killers).
  term    = collision -20 | landed -10 | > max(12, desired+8) m from START
            -10 (bad-only, radius scales with the D11 goal); trunc =
            max_steps (100).

Collision route: IDENTICAL pattern to obstacle_env.py -- PRIMARY the real
collision_info pub/sub topic (client.subscribe on its own recv thread);
FPV_COLL_ROUTE=geometry forces the fallback (world.list_objects +
get_object_pose AABBs, trailing '_<cm>' name suffix = full edge). The SAME
enumerated AABBs feed the D5 depth grid, with the drone itself excluded by
name (cube-name regex already excludes "Drone1"; floor/ground excluded too).

Env knobs (D2/D3/D6/D11/D12): FPV_TARGET_KIND person|bike|bicycle|car|rand,
FPV_PMISS, FPV_JIT (frame fraction), FPV_LAT_MS ("400" or "300-500"),
FPV_YAW (0 -> 3-D action, logged fallback), FPV_DR (1 = per-episode
domain randomization), FPV_GROUND_Z (default -2.7, Blocks scene ground),
FPV_COLL_ROUTE (auto|geometry), FPV_RANGE_MIN / FPV_RANGE_MAX (D11 desired
sweep bounds, default 1.5 / 15.0 m), FPV_DIST_STAR set => FIXED desired
every episode (back-compat), FPV_RANGE_NOISE / FPV_RANGE_EMA (D12),
FPV_DETOUR (D15: 1 = evaluation rung -- start the scripted target behind a
cube dead ahead; off by default, reset() stream bit-identical when off),
plus the parent's FPV_DISTURB_* / FPV_PORT_*.

Coordinates: AirSim NED. Run ONLY where Blocks.exe is listening
(scheduled-task console-session pattern). py_compile-only on non-Windows.
"""
import asyncio
import os
import re
import threading

import numpy as np
from gymnasium import spaces

import follow_common as fc
from gym_env import BlocksDroneEnv, quat_to_rpy

SETTLED_GUARD = 10          # mirror parent landed guard: takeoff/settle bump
LEAVE_RADIUS = 12.0         # D7 leave radius FLOOR; D11: per-episode it is
#                             max(LEAVE_RADIUS, desired + 8.0)
CUBE_NAME_RE = re.compile(r"block|cube", re.IGNORECASE)
EXCLUDE_NAME_RE = re.compile(r"floor|ground|plane|pad|light|sky|fog|volume",
                             re.IGNORECASE)
GEOM_DEFAULT_HALF = 1.0     # same fallback half-extent as obstacle_env.py


class FollowDroneEnv(BlocksDroneEnv):
    """BlocksDroneEnv + scripted moving target + synthetic detector + depth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   # connects on the loop thread
        # ---- knobs (D2/D3/D6) ----
        self._yaw_on = os.environ.get("FPV_YAW", "1").strip() != "0"
        self._yaw_logged = False
        kind = os.environ.get("FPV_TARGET_KIND", "rand").strip().lower()
        self._kind = fc.KIND_ALIAS.get(kind, "rand")
        self._dr = os.environ.get("FPV_DR", "0").strip() == "1"
        # D15: detour-required eval rung (default OFF => reset() bit-identical)
        self._detour = os.environ.get("FPV_DETOUR", "0").strip() == "1"
        self._p_miss = float(os.environ.get("FPV_PMISS", "0.15"))
        self._jit = float(os.environ.get("FPV_JIT", "0.005"))
        lat = os.environ.get("FPV_LAT_MS", "").strip()
        self._lat_lo = self._lat_hi = None
        if lat:
            parts = [p for p in re.split(r"[-,]", lat) if p]
            self._lat_lo = float(parts[0])
            self._lat_hi = float(parts[-1])
        self._ground_z = float(os.environ.get("FPV_GROUND_Z", "-2.7"))
        # D11: FPV_DIST_STAR set => FIXED desired every episode (back-compat
        # eval); otherwise desired ~ U[FPV_RANGE_MIN, FPV_RANGE_MAX] sampled
        # per-episode in reset() from the SEEDED rng (same seed => same
        # stream for policy AND baselines, rl_eval_follow contract).
        ds = os.environ.get("FPV_DIST_STAR", "").strip()
        self._fixed_desired = float(ds) if ds else None
        self._desired = fc.DIST_STAR          # overwritten every reset()
        self._leave_radius = LEAVE_RADIUS     # recomputed per episode (D11)
        self._cam = fc.default_cam()
        # ---- action/obs spaces (D3/D13) ----
        n_a = 4 if self._yaw_on else 3
        self.action_space = spaces.Box(-1.0, 1.0, (n_a,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (37,),
                                           dtype=np.float32)
        # ---- episode state ----
        self._script = None
        self._noise = None
        self._range = None                    # fc.RangeEstimator (D12)
        self._h_star = fc.h_star_for(1.7, self._desired, self._cam)
        self._start = None
        self._prev_a4 = np.zeros(4, dtype=np.float32)
        self._loss_steps = 0
        self._box_feats = np.zeros(7, dtype=np.float32)
        self._grid_objs = []
        # ---- collision route (IDENTICAL to obstacle_env.py) ----
        self._collide_evt = threading.Event()
        self._last_collision = None
        self._boxes = []           # geometry route: list of (name, c, half)
        self._boxes_logged = False
        route = os.environ.get("FPV_COLL_ROUTE", "auto").strip().lower()
        if route == "geometry":
            self.collision_route = "geometry"
        else:
            try:
                self._lp.sync(self._subscribe_collision(), timeout=30)
                self.collision_route = "pubsub"
            except Exception:
                self.collision_route = "geometry"
        if not self._yaw_on and not self._yaw_logged:
            self._yaw_logged = True
            print("FPV_YAW=0: translation-only follow, action space Box(3)",
                  flush=True)

    # ---- client calls: collision pub/sub + object enumeration ----
    # (verbatim pattern from obstacle_env.py: subscribe topic from
    # robot_info["collision_info"], callback only sets an event flag; the
    # fallback enumerates world.list_objects/get_object_pose AABBs -- the
    # exact same calls now also feed the D5 depth grid.)
    async def _subscribe_collision(self):
        topic = self.drone.robot_info["collision_info"]

        def _on_collision(_topic, payload):
            self._last_collision = payload
            self._collide_evt.set()

        self.client.subscribe(topic, _on_collision)

    async def _enumerate_boxes(self):
        boxes = []
        try:
            names = self.world.list_objects(".*") or []
        except Exception:
            return []
        for name in list(names):
            try:
                if not CUBE_NAME_RE.search(str(name)):
                    continue          # excludes the drone itself by name
                if EXCLUDE_NAME_RE.search(str(name)):
                    continue
                pose = self.world.get_object_pose(str(name))
                tr = getattr(pose, "translation", None)
                if tr is None and isinstance(pose, dict):
                    tr = pose.get("translation")
                if tr is None:
                    continue
                if isinstance(tr, dict):
                    x, y, z = float(tr["x"]), float(tr["y"]), float(tr["z"])
                else:
                    x, y, z = float(tr.x), float(tr.y), float(tr.z)
                half = GEOM_DEFAULT_HALF
                m = re.search(r"(\d+)\s*$", str(name))
                if m:                       # e.g. TemplateCube_Rounded_150
                    edge_m = int(m.group(1)) / 100.0
                    if 0.1 <= edge_m <= 20.0:
                        half = edge_m / 2.0
                boxes.append((str(name), np.array([x, y, z], dtype=np.float32),
                              half))
            except Exception:
                continue
        return boxes

    async def _apply_yaw_async(self, yaw_rate):
        # D3: rotate_by_yaw_rate_async(yaw_rate, duration) -- verified present
        # on the rig 2026-11-10 (RotateByYawRate RPC).
        await asyncio.wait_for(
            self.drone.rotate_by_yaw_rate_async(float(yaw_rate),
                                                self.step_dt),
            timeout=15)

    def _geom_hit(self, pos):
        for _name, center, half in self._boxes:
            d = np.maximum(np.abs(np.asarray(pos) - center) - half, 0.0)
            if float(np.linalg.norm(d)) < 0.25:
                return True
        return False

    # ---- episode ----
    def reset(self, *, seed=None, options=None):
        # Parent plumbing verbatim: world reload, arm, takeoff, settle >= 2 m
        # AGL, self._target = settled pos, self._n = 0, _rng reseeded on seed.
        obs_raw, _ = super().reset(seed=seed, options=options)
        k0 = self._kin()
        p0, o0 = k0["pose"]["position"], k0["pose"]["orientation"]
        pos0 = np.array([p0["x"], p0["y"], p0["z"]], dtype=np.float32)
        _, _, yaw0 = quat_to_rpy(o0["w"], o0["x"], o0["y"], o0["z"])
        self._start = pos0.copy()
        self._prev_a4 = np.zeros(4, dtype=np.float32)
        self._loss_steps = 0
        self._box_feats = np.zeros(7, dtype=np.float32)
        self._collide_evt.clear()
        # D11: draw the episode desired AFTER seeding (fixed draw order:
        # desired FIRST, then detector knobs, then the script) so same-seed
        # streams stay identical across conditions; FPV_DIST_STAR override
        # pins it (back-compat eval).
        if self._fixed_desired is not None:
            self._desired = float(self._fixed_desired)
        else:
            lo = fc._env_f("FPV_RANGE_MIN", 1.5)
            hi = fc._env_f("FPV_RANGE_MAX", 15.0)
            self._desired = float(self._rng.uniform(lo, hi))
        self._leave_radius = max(LEAVE_RADIUS, self._desired + 8.0)
        # detector knobs: FIXED draw order after seed => identical box
        # streams for policy/baselines on the same seed (rl_eval_follow).
        lat = fc.sample_latency_steps(self._rng, self.step_dt,
                                      self._lat_lo, self._lat_hi)
        jit, pmiss = self._jit, self._p_miss
        if self._dr:
            drp = fc.dr_detector_params(self._rng)
            pmiss, jit = drp["p_miss"], drp["jitter"]
        self._noise = fc.DetectorNoise(self._rng, p_miss=pmiss, jitter=jit,
                                       lat_steps=lat, params=self._cam)
        self._script = fc.sample_target(self._rng, self._kind,
                                        drone_start=pos0, drone_yaw=yaw0,
                                        ground_z=self._ground_z,
                                        params=self._cam,
                                        desired=self._desired)
        # D12: the phone-identical range estimator (draws from the SAME rng
        # after seeding -- same-seed streams stay identical).
        self._range = fc.RangeEstimator(self._rng, self._script["height"],
                                        params=self._cam)
        self._h_star = fc.h_star_for(self._script["height"],
                                     self._desired, self._cam)
        if self.collision_route == "geometry":
            self._boxes = self._lp.sync(self._enumerate_boxes(), timeout=120)
            if not self._boxes_logged:
                self._boxes_logged = True
                print(f"FOLLOW_COLL_ROUTE=geometry cubes={len(self._boxes)}",
                      flush=True)
        # scene static within an episode: enumerate the depth-grid AABBs
        # ONCE per episode from the same client calls (D5).
        if self.collision_route == "geometry":
            src = self._boxes
        else:
            src = self._lp.sync(self._enumerate_boxes(), timeout=120)
        self._grid_objs = [(n, c, np.array([h, h, h], dtype=np.float32))
                           for (n, c, h) in src]
        # D15 (FPV_DETOUR=1, default off): detour-required eval rung --
        # AFTER the script draw (fixed order desired -> detector knobs ->
        # script -> detour) so same-seed streams are untouched when off.
        # Pure geometry over the enumerated AABBs (NO world mutation): pick
        # one cube 3-10 m dead ahead (+-60 deg of start yaw) and shift the
        # target start behind it; honest no_cube fallback runs unchanged.
        if self._detour:
            blk = fc.detour_blocker(self._grid_objs, pos0, yaw0)
            if blk is None:
                print("FOLLOW_DETOUR=no_cube", flush=True)
            else:
                d_ray = fc.detour_target(self._script, blk[1], pos0,
                                         self._rng)
                print(f"FOLLOW_DETOUR=applied cube={blk[0]} d={d_ray:.1f}",
                      flush=True)
        obs = self._obs37(self._obs_from_kin(k0)[0], k0)
        return obs, {"dist": 0.0, "in_view": False, "center_err": 0.0,
                     "loss_steps": 0, "target_kind": self._script["kind"],
                     "desired": self._desired,
                     "range_est": float("nan"), "range_valid": False}

    def _obs37(self, state15, k):
        feats = self._box_feats
        grid = fc.depth_grid(np.asarray(state15[:3], dtype=np.float64)
                             + np.asarray(self._target, dtype=np.float64),
                             float(state15[8]), self._grid_objs,
                             params=self._cam)
        return np.concatenate([state15, feats, grid]).astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape[0] == 3:
            action = np.concatenate([action, [0.0]]).astype(np.float32)
        a3 = action[:3]
        self._n += 1
        self._last_action = a3          # parent obs block 12:15 stays 3-D
        k, landed = self._lp.sync(self._step_async(a3))
        if self._yaw_on:
            try:
                self._lp.sync(self._apply_yaw_async(float(action[3]) * 0.5),
                              timeout=20)
            except Exception as exc:    # D3 risk item: untested rotate API
                print(f"FPV_YAW_FALLBACK rotate_by_yaw_rate_async failed "
                      f"({exc!r}); yaw dim ignored this episode", flush=True)
                self._yaw_on = False
        # parent raw hover-error obs (err=pos-target, vel, rpy, ang, prev a3)
        state15, _ = self._obs_from_kin(k)
        pos = np.asarray(self._target, dtype=np.float64) \
            + np.asarray(state15[:3], dtype=np.float64)
        yaw = float(state15[8])
        # advance the scripted target by the WALL-FREE sim step counter
        t_now = self._n * self.step_dt
        tpos = fc.target_at(self._script, t_now)
        gt = fc.project_target(pos, yaw, tpos, self._script["height"],
                               params=self._cam)
        box = self._noise.update(gt)
        in_view = box is not None and self._noise.age <= fc.K_VIEW
        if not in_view:
            self._loss_steps += 1
        # D12/D13: feed the phone-identical range estimator the FRESH noisy
        # box (frozen/stale boxes are unusable, like the app) with the GT
        # edge_cut flag; range_valid = usable measurement THIS step.
        fresh = box is not None and box["age"] == 0
        est, est_valid = self._range.update(box if fresh else None,
                                            bool(gt.get("edge_cut", False)))
        range_valid = bool(est_valid) and fresh
        if box is not None:
            cx_err = float(np.clip((box["cx"] - self._cam["cx0"])
                                   / self._cam["cx0"], -1.0, 1.0))
            cy_err = float(np.clip((box["cy"] - self._cam["cy0"])
                                   / self._cam["cy0"], -1.0, 1.0))
            conf = float(box["conf"])
            age10 = min(float(box["age"]) / 10.0, 1.0)
        else:
            cx_err = cy_err = 0.0
            conf = 0.0
            age10 = min(float(self._noise.age) / 10.0, 1.0)
        # obs v2 (D13) box-feats block, 7-D
        log_r = (float(np.clip(np.log(est / self._desired), -2.0, 2.0))
                 if (range_valid and np.isfinite(est)) else 0.0)
        self._box_feats = np.array(
            [cx_err, cy_err, log_r, self._desired / 15.0,
             1.0 if range_valid else 0.0, conf, age10], dtype=np.float32)
        obs = self._obs37(state15, k)
        # terminals (D7, bad-only)
        dist_start = float(np.linalg.norm(pos - self._start))
        if self._n <= SETTLED_GUARD:
            self._collide_evt.clear()
            collided = False
        elif self.collision_route == "pubsub":
            collided = bool(self._collide_evt.is_set())
        else:
            collided = self._geom_hit(pos)
        flew_out = bool(dist_start > self._leave_radius)   # D11 scaled radius
        reward = fc.follow_reward(
            float(self._box_feats[0]), float(self._box_feats[1]),
            gt["dist"], in_view,
            float(np.linalg.norm(action)),
            float(np.linalg.norm(action - self._prev_a4)),
            float(np.linalg.norm(state15[3:6])),
            landed=landed, flew_out=flew_out, collided=collided,
            desired=self._desired)
        self._prev_a4 = action
        terminated = bool(collided or landed or flew_out)
        truncated = bool(self._n >= self.max_steps)
        center_err = float(np.hypot(self._box_feats[0], self._box_feats[1]))
        return obs, reward, terminated, truncated, {
            "center_err": center_err, "dist": gt["dist"], "in_view": in_view,
            "loss_steps": self._loss_steps, "collided": collided,
            "landed": landed, "flew_out": flew_out,
            "dist_start": dist_start,
            "target_kind": self._script["kind"],
            "age": self._noise.age,
            "h_star": self._h_star,
            "desired": self._desired,
            "range_est": float(est) if np.isfinite(est) else float("nan"),
            "range_valid": range_valid}
