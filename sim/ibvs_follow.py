"""ibvs_follow.py -- S7 IBVS follow baseline controller (spec D8 + D14).

PURE module: numpy + stdlib only (runs on bare sable python3). The scripted
baseline rl_eval_follow.py compares the learned policy against, on identical
obs streams. Inputs are what the AGENT gets (D13 obs v2): the noisy box, the
apparent-size range estimate, its validity flag and the per-episode DESIRED
range; output is the same 4-D action the env consumes:

    a = [v_north, v_east, v_down, yaw_rate_action]     (all in [-1, 1])

D14: the baseline is aligned to the PHONE APP 0.3.0 FollowBehavior
(VISION_ONBOARD_RESEARCH.md section 7) so eval measures the learned policy
against the controller the owner actually ships:
  yaw       = clip(kp_yaw*cx_err, -1, 1) with deadband 0.05 (unchanged D8):
            yaw comes from the HORIZONTAL image error; action dim +-1 ==
            +-0.5 rad/s NED, positive = turn toward a right-of-center target;
            on loss the last yaw command is HELD.
  forward   v_f (body frame): NO forward motion without a usable range
            (range_valid false => v_f = 0). Else err = est - desired;
            |err| <= deadband_m (FPV_IBVS_DEADBAND_M, 0.4 m) => v_f = 0;
            else v_f = clip(0.35 * err, -1, 1) m/s -- the app's 0.35 m/s per
            metre of range error (gain FPV_IBVS_KP_F). Close guard: never
            advance inside 1.5 m of the target (v_f = min(v_f, 0) when
            est < 1.5 m) -- the app's hard floor.
  vertical  v_d = clip(-kp_z*(est/desired - 1)*0.5, -1, 1)*0.5 when the range
            is usable, else 0: small z trim to help the range servo.
  lateral   v_lat = clip(-kp_lat*cy_err, -1, 1) (kp_lat=0.8, unchanged): cy
            error drives the strafe on the -20 deg pitched cam; body
            (v_f, v_lat) are rotated to world by the current yaw.
  loss:     box is None or age > K_loss(4): v = 0, yaw HOLDS its last
            command (hover-on-loss, D8). NEW D14 give-up: after >= giveup
            CONSECUTIVE loss steps (FPV_IBVS_GIVEUP_S, 8.0 s / step_dt -> 16
            steps at 0.5 s) the controller has GIVEN UP: all-zero action
            permanently until reset() -- the app stops a hopeless follow.
Anti-double-move discipline (Q3 gimbal lesson): exactly ONE integrated
action per step -- the controller is stateless except hover-hold / give-up
state.
"""
import os

import numpy as np


def _gain(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


NEVER_ADVANCE_M = 1.5   # app hard floor: never advance inside 1.5 m (D14)


class IBVSFollow:
    """Pure IBVS follow controller (D8 + D14). reset() between episodes."""

    def __init__(self, gains=None, cam=None, cx0=360.0, cy0=240.0,
                 step_dt=0.5):
        g = {
            "kp_yaw": _gain("FPV_IBVS_KP_YAW", 0.6),
            "kp_f":   _gain("FPV_IBVS_KP_F", 0.35),      # m/s per m range err
            "kp_lat": _gain("FPV_IBVS_KP_LAT", 0.8),
            "kp_z":   _gain("FPV_IBVS_KP_Z", 1.0),
            "deadband": _gain("FPV_IBVS_DEADBAND", 0.05),      # yaw (frame)
            "deadband_m": _gain("FPV_IBVS_DEADBAND_M", 0.4),   # range (m)
            "k_loss": int(_gain("FPV_IBVS_K_LOSS", 4)),
        }
        if gains:
            g.update(gains)
        self.g = g
        self.cx0, self.cy0 = cx0, cy0
        # D14 give-up: continuous loss >= FPV_IBVS_GIVEUP_S / step_dt steps
        self.giveup_steps = max(
            1, int(round(_gain("FPV_IBVS_GIVEUP_S", 8.0) / float(step_dt))))
        self.reset()

    def reset(self):
        self._hold_yaw = 0.0        # yaw action held on loss (D8)
        self._loss_run = 0          # consecutive loss steps (D14)
        self._given_up = False      # permanent all-zero after give-up (D14)

    def action(self, box, est, range_valid, desired, drone_yaw=0.0):
        """One 4-D action. box: current noisy box dict (cx, cy, age) or None.
        est: apparent-size range estimate (D12); range_valid: it is usable
        THIS step; desired: per-episode follow range (D11)."""
        if self._given_up:                      # D14: given up = zeros forever
            return np.zeros(4, dtype=np.float32)
        lost = box is None or float(box.get("age", 0)) > self.g["k_loss"]
        if lost:
            self._loss_run += 1
            if self._loss_run >= self.giveup_steps:
                self._given_up = True           # app gives up after 8 s loss
                return np.zeros(4, dtype=np.float32)
            # hover-on-loss (D8): v = 0, yaw holds its last command
            return np.array([0.0, 0.0, 0.0, self._hold_yaw], dtype=np.float32)
        self._loss_run = 0
        cx_err = (float(box["cx"]) - self.cx0) / self.cx0
        cy_err = (float(box["cy"]) - self.cy0) / self.cy0
        # yaw servo (deadband, D8)
        if abs(cx_err) > self.g["deadband"]:
            yaw_a = float(np.clip(self.g["kp_yaw"] * cx_err, -1.0, 1.0))
        else:
            yaw_a = 0.0
        self._hold_yaw = yaw_a
        est_f = None                          # usable numeric estimate?
        if bool(range_valid) and est is not None:
            e = float(est)
            if np.isfinite(e):
                est_f = e
        if est_f is not None:
            # forward servo = app 0.3.0: 0.35 m/s per m range error, deadband
            err = est_f - float(desired)
            if abs(err) <= self.g["deadband_m"]:
                v_f = 0.0
            else:
                v_f = float(np.clip(self.g["kp_f"] * err, -1.0, 1.0))
            if est_f < NEVER_ADVANCE_M:       # never advance inside 1.5 m
                v_f = min(v_f, 0.0)
            v_d = float(np.clip(-self.g["kp_z"]
                                * (est_f / float(desired) - 1.0) * 0.5,
                                -1.0, 1.0)) * 0.5
        else:
            v_f = 0.0                           # no forward w/o usable range
            v_d = 0.0
        v_lat = float(np.clip(-self.g["kp_lat"] * cy_err, -1.0, 1.0))
        # body -> world (NED): forward=(cos,sin), right=(-sin,cos)
        c, s = np.cos(drone_yaw), np.sin(drone_yaw)
        vn = float(np.clip(v_f * c - v_lat * s, -1.0, 1.0))
        ve = float(np.clip(v_f * s + v_lat * c, -1.0, 1.0))
        return np.array([vn, ve, v_d, yaw_a], dtype=np.float32)
