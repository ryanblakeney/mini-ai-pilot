"""WaypointDroneEnv: Q3 stage-2 waypoint-reach task (ProjectAirSim Blocks rig).

Subclasses BlocksDroneEnv (stage-1 hover) and reuses ALL of its projectairsim
plumbing verbatim (persistent asyncio loop thread, ts-gated steps, inverted
LandedState decode LANDED=0, disconnect-before-exit). Subclass only -- the
parent plumbing is load-bearing and must never be rewritten.

Task: after the parent takeoff+settle (self._target = settled position), a
waypoint is sampled per episode: bearing uniform [0, 2pi), horizontal distance
d uniform [3.0, 10.0] m at the settled altitude (NED):
    wp = target + [d*cos(b), d*sin(b), 0]
  obs     = [wp_err(3), lin_vel(3), rpy(3), ang_vel(3), prev_action(3)]
            with wp_err = pos - wp   (Box(15), unchanged shape)
  dist    = ||pos - wp|| (3D norm)
  arrived = dist <= 0.75 m
  failed  = dist > 12.0 m or landed mid-episode
  term    = arrived or failed;  truncated = step count >= max_steps (100)

Reward v3 (waypoint, 2026-11-08 design -- LOCKED arithmetic; verify with
check_waypoint_reward.py BEFORE any launch):

    def wp_reward(dist, a_norm, v_norm, arrived, failed):
        if arrived: return 60.0
        if failed: return -10.0
        return 0.1 + 0.4*(1 - min(dist, 10.0)/10.0) - 0.05*a_norm - 0.02*v_norm

Card's hard rule: survival-shaped positive per-step reward + terminal penalty
larger than any possible gain from ending early. Dominance arithmetic
(offline-verified, see check_waypoint_reward.py):
  * CAMP bound: per-step max while not arriving (dist > 0.75; penalties only
    subtract) is 0.1 + 0.4*(1 - 0.76/10) = 0.4696, so any 100-step camping
    episode earns <= 46.96.
  * ARRIVAL return: covering the max d0 = 10 m needs >= 20 steps even at a
    conservative 1 m/s average; worst per-step floor is -0.0266
    (||a|| <= sqrt(3), ||v|| <= 2 m/s), so arrival return >=
    20*(-0.0266) + 60 = 59.47 > 46.96. Arriving always beats camping, and
    also beats hovering in place at the start (<= 100 x 0.38 = 38, since
    d0 >= 3 makes the per-step ceiling 0.1 + 0.4*0.7 = 0.38).
  * FLY-AWAY: from d0 = 3 m, flying away crosses dist > 12 only after >= 9 m
    of progress (>= 9 steps at 1 m/s); the ceiling sum of the shrinking
    per-step reward plus the -10 terminal lands at <= -6 < every baseline.
Disturbance mechanism inherited exactly as-is (FPV_DISTURB_EVERY /
FPV_DISTURB_SCALE env vars, see gym_env.py).

Coordinates: AirSim NED. Run ONLY where Blocks.exe is listening
(scheduled-task console-session pattern).
"""
import math

import numpy as np

from gym_env import BlocksDroneEnv


def wp_reward(dist, a_norm, v_norm, arrived, failed):
    """Stage-2 waypoint reward v3 -- PURE function, no sim imports needed.

    Dominance arithmetic for these constants is documented in the module
    header and asserted offline by check_waypoint_reward.py.
    """
    if arrived:
        return 60.0
    if failed:
        return -10.0
    return 0.1 + 0.4 * (1.0 - min(dist, 10.0) / 10.0) - 0.05 * a_norm - 0.02 * v_norm


class WaypointDroneEnv(BlocksDroneEnv):
    """BlocksDroneEnv + randomized horizontal waypoint goal (stage 2)."""

    ARRIVE_RADIUS = 0.75
    FAIL_RADIUS = 12.0
    D_MIN, D_MAX = 3.0, 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wp = None   # waypoint (NED); None until reset() samples one
        self._d0 = 0.0    # start distance for this episode

    def reset(self, *, seed=None, options=None):
        # Parent plumbing verbatim: fresh world reload, arm, takeoff, settle;
        # sets self._target to the settled takeoff position.
        obs, info = super().reset(seed=seed, options=options)
        b = self._rng.uniform(0.0, 2.0 * math.pi)
        d = self._rng.uniform(self.D_MIN, self.D_MAX)
        self._wp = (self._target + np.array(
            [d * math.cos(b), d * math.sin(b), 0.0], dtype=np.float32
        )).astype(np.float32)
        self._d0 = d
        k0 = self._kin()
        obs, dist = self._obs_from_kin(k0)
        return obs, {"dist": dist, "arrived": False, "failed": False,
                     "start_dist": d}

    def _obs_from_kin(self, k):
        # Parent builds obs[:3] = pos - target; rebase that block on the
        # waypoint so the whole obs layout stays Box(15).
        obs, _ = super()._obs_from_kin(k)
        if self._wp is None:
            return obs, float(np.linalg.norm(obs[:3]))
        pos = obs[:3] + self._target
        wp_err = (pos - self._wp).astype(np.float32)
        obs = obs.copy()
        obs[:3] = wp_err
        return obs, float(np.linalg.norm(wp_err))

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._last_action = action
        self._n += 1
        k, landed = self._lp.sync(self._step_async(action))
        obs, dist = self._obs_from_kin(k)
        arrived = bool(dist <= self.ARRIVE_RADIUS)
        failed = bool(dist > self.FAIL_RADIUS or landed)
        reward = wp_reward(dist, float(np.linalg.norm(action)),
                           float(np.linalg.norm(obs[3:6])), arrived, failed)
        terminated = arrived or failed
        truncated = bool(self._n >= self.max_steps)
        return obs, reward, terminated, truncated, {
            "dist": dist, "arrived": arrived, "failed": failed,
            "start_dist": self._d0}
