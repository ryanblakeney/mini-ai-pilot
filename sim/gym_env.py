"""BlocksDroneEnv: gymnasium wrapper around Project AirSim Blocks (Q3 stage 1).

Stage 1 task: hover stabilization under random external-force disturbances.
  action  = [v_north, v_east, v_down] m/s velocity setpoint held for step_dt
  obs     = [pos_err(3), lin_vel(3), rpy(3), ang_vel(3), prev_action(3)]
  reward  = reward v2 (2026-11-08 review: survival reward + terminal penalty
            kills the termination exploit; action box ±1 m/s):
              r = 1 - min(err,4)/4 - 0.05*||action|| - 0.02*||lin_vel||
              if terminal (err > 8 m or landed): r = -10
  term    = ||pos_err|| > 8 m or landed mid-episode; trunc = max_steps.

Coordinates: AirSim NED (x north, y east, z DOWN; more negative z = higher).

All projectairsim calls are funneled through ONE persistent asyncio loop thread
(pynng sockets are loop-affine; per-step asyncio.run breaks them).

Run ONLY where Blocks.exe is listening (scheduled-task console-session pattern).
"""
import asyncio
import math
import os
import random
import threading
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from projectairsim import ProjectAirSimClient, Drone, World

SCENE = "scene_basic_drone.jsonc"


def quat_to_rpy(w, x, y, z):
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(s)
    sinp = 2.0 * (w * z + x * y)
    cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sinp, cosp)
    return roll, pitch, yaw


async def _chain(coro):
    """Await move_*_async's inner RPC, then the returned command task."""
    return await (await coro)


class _LoopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def sync(self, coro, timeout=90):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)


class BlocksDroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="127.0.0.1", step_dt=0.5, max_steps=100,
                 disturb_every=None, disturb_scale=None):
        super().__init__()
        # v4: disturb defaults overridable via env (probe_b evidence 2026-11-08:
        # clean-sim hover holds 3 cm; +/-1 N every 4 steps was the real wall).
        if disturb_every is None:
            disturb_every = int(os.environ.get("FPV_DISTURB_EVERY", "4"))
        if disturb_scale is None:
            disturb_scale = float(os.environ.get("FPV_DISTURB_SCALE", "1.0"))
        self.step_dt = step_dt
        self.max_steps = max_steps
        self.disturb_every = disturb_every
        self.disturb_scale = disturb_scale
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (15,), dtype=np.float32)
        self._lp = _LoopThread()
        self._lp.start()
        self._lp.sync(self._connect())
        self._last_action = np.zeros(3, dtype=np.float32)
        self._n = 0
        self._target = None
        self._last_ts = None
        self._rng = random.Random()

    # ---- client plumbing (all on loop thread) ----
    async def _connect(self):
        # Q3p (prepared): per-instance RPC ports via env vars.
        # FPV_PORT_TOPICS / FPV_PORT_SERVICES default to 8989/8990, so with no
        # env set the client behaves exactly as before (ProjectAirSimClient()
        # defaults are port_topics=8989, port_services=8990 in client.py line 32).
        _tp = int(os.environ.get("FPV_PORT_TOPICS", "8989"))
        _sp = int(os.environ.get("FPV_PORT_SERVICES", str(_tp + 1)))
        self.client = ProjectAirSimClient(port_topics=_tp, port_services=_sp)
        self.client.connect()
        self.world = World(self.client, SCENE, delay_after_load_sec=3)
        self.drone = Drone(self.client, self.world, "Drone1")
        self.drone.enable_api_control()

    def _kin(self):
        try:
            k = self.drone.get_ground_truth_kinematics()
        except AttributeError:
            k = None
        if not k:
            k = self.drone.get_kinematics()
        return k

    async def _reset_async(self):
        # Fresh World reload = true episode restart (proven respawn path from smoke).
        self.world = World(self.client, SCENE, delay_after_load_sec=3)
        self.drone = Drone(self.client, self.world, "Drone1")
        self.drone.enable_api_control()
        self.drone.arm()
        await self.drone.takeoff_async(timeout_sec=40)
        # Settle: kinematics read right after takeoff_async can be stale (hover_v1
        # video caught target z=-2.69 ~= ground level -> "hover" at pad height).
        # Enforce >= 2 m AGL (scene ground z ~ -2.7).
        await asyncio.sleep(2)
        k = self._kin()
        p = k["pose"]["position"]
        if p["z"] > -4.7:
            await asyncio.wait_for(_chain(
                self.drone.move_to_position_async(p["x"], p["y"], -4.7, velocity=1.5)),
                timeout=60)
            await asyncio.sleep(1)
            k = self._kin()
            p = k["pose"]["position"]
        # hover target = settled, properly elevated takeoff position
        self._target = np.array([p["x"], p["y"], p["z"]], dtype=np.float32)

    async def _step_async(self, action):
        if self.disturb_every and self._n % self.disturb_every == 0:
            f = [self._rng.uniform(-1, 1) * self.disturb_scale for _ in range(3)]
            self.drone.set_external_force(f)
        ts0 = self._last_ts or 0
        # minimum real-time advance so the sim physics runs during the step
        await asyncio.sleep(self.step_dt * 0.7)
        await asyncio.wait_for(
            self.drone.move_by_velocity_async(
                float(action[0]), float(action[1]), float(action[2]),
                duration=self.step_dt),
            timeout=15)
        # move_by_velocity_async returns as soon as the RPC is sent; gate on the
        # sim's nanosecond time_stamp so each step really advances ~step_dt sim-time.
        target_ts = ts0 + int(self.step_dt * 1e9)
        deadline = time.monotonic() + self.step_dt * 4 + 2.0
        k = self._kin()
        while time.monotonic() < deadline:
            k = self._kin()
            ts = k.get("time_stamp") or 0
            if ts >= target_ts:
                break
            await asyncio.sleep(0.05)
        self._last_ts = k.get("time_stamp") or ts0
        landed = False
        # LandedState enum is INVERTED vs naive bool: LANDED=0, FLYING=1, and
        # the raw payload may be a dict (truthy either way — caused the bogus
        # step-0/step-10 terminations, 00:52 + 01:42 runs). Decode properly;
        # also keep the settled-steps guard for the takeoff transition.
        if self._n > 10:
            try:
                st = self.drone.get_landed_state()
                if isinstance(st, dict):
                    st = st.get("landed_state", st.get("state", 1))
                landed = int(st) == 0  # LandedState.LANDED == 0
            except Exception:
                landed = False
        return k, landed

    async def _close_async(self):
        # No land_async here: the sim process is killed right after close and a
        # pending Land task segfaults python teardown (seen 2026-11-08 00:45).
        try:
            self.drone.set_external_force([0.0, 0.0, 0.0])
            self.drone.cancel_last_task()
        except Exception:
            pass
        try:
            self.client.disconnect()  # mandatory or python hangs
        except Exception:
            pass

    # ---- obs/reward ----
    def _obs_from_kin(self, k):
        p = k["pose"]["position"]
        o = k["pose"]["orientation"]
        v = k["twist"]["linear"]
        w = k["twist"]["angular"]
        pos = np.array([p["x"], p["y"], p["z"]], dtype=np.float32)
        vel = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        ang = np.array([w["x"], w["y"], w["z"]], dtype=np.float32)
        rpy = np.array(quat_to_rpy(o["w"], o["x"], o["y"], o["z"]), dtype=np.float32)
        if self._target is None:
            self._target = pos
        err = pos - self._target
        obs = np.concatenate([err, vel, rpy, ang, self._last_action]).astype(np.float32)
        return obs, float(np.linalg.norm(err))

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self._lp.sync(self._reset_async())
        self._n = 0
        self._last_action = np.zeros(3, dtype=np.float32)
        k0 = self._kin()
        self._last_ts = k0.get("time_stamp")
        obs, err = self._obs_from_kin(k0)
        return obs, {"pos_err": err}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._last_action = action
        self._n += 1
        k, landed = self._lp.sync(self._step_async(action))
        obs, err = self._obs_from_kin(k)
        # reward v2 (2026-11-08 review): all-negative per-step reward made early
        # termination optimal; survival-shaped reward + terminal penalty fixes it.
        vel_norm = float(np.linalg.norm(obs[3:6]))
        reward = float(1.0 - min(err, 4.0) / 4.0 - 0.05 * np.linalg.norm(action)
                       - 0.02 * vel_norm)
        terminated = bool(err > 8.0 or landed)
        if terminated:
            reward = -10.0
        truncated = bool(self._n >= self.max_steps)
        return obs, reward, terminated, truncated, {"pos_err": err}

    def close(self):
        try:
            self._lp.sync(self._close_async(), timeout=120)
        finally:
            self._lp.stop()
