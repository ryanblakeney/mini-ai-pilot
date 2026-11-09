"""ibvs_follow.py -- S7 IBVS follow baseline controller (spec D8).

PURE module: numpy + stdlib only (runs on bare sable python3). The scripted
baseline rl_eval_follow.py compares the learned policy against, on identical
box streams. Inputs are the NOISY box the agent sees (D6 stream) plus current
drone yaw; output is the same 4-D action the env consumes:

    a = [v_north, v_east, v_down, yaw_rate_action]     (all in [-1, 1])

Servo structure (D8, gains FPV_IBVS_* overridable):
  yaw_rate  = clip(kp_yaw*cx_err, -1, 1)   (kp_yaw=0.6; zero while
            |cx_err| <= deadband 0.05; action dim +-1 == +-0.5 rad/s NED,
            positive = yaw toward east/right, i.e. turn toward a target that
            sits right-of-center in the image)
  forward   v_f   = clip(-kp_f*(h_px/h_star - 1), -1, 1)
            too big (close) -> back off, too small -> close in. NOTE: D8's
            printed formula literal (kp_f*(h/h*-1)) has the opposite sign of
            D8's own prose "too big -> back off" and of the mandated test
            (too-big box => forward <= 0); prose+test implemented, deviation
            documented in the deliverables report.
  lateral   v_lat = clip(-kp_lat*cy_err, -1, 1)        (kp_lat=0.8)
            on the -20 deg pitched cam cy_err>0 (target low in frame) means
            too close/low => strafe "back" against the forward axis; body
            (v_f, v_lat) are rotated to world by the current yaw.
  vertical  v_d   = clip(-kp_z*(h_px/h_star - 1)*0.5, -1, 1)*0.5
            small z trim to help the h range servo (hold altitude mostly).
  loss:     box is None or age > K_loss(4): v = 0, yaw HOLDS its last
            command (hover-on-loss, D8; re-acquire is implicit).
  Anti-double-move discipline (Q3 gimbal lesson): exactly ONE integrated
  action per step -- the controller is stateless except hover-hold state.
"""
import os

import numpy as np


def _gain(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


class IBVSFollow:
    """Pure IBVS follow controller (D8). reset() between episodes."""

    def __init__(self, gains=None, cam=None, cx0=360.0, cy0=240.0):
        g = {
            "kp_yaw": _gain("FPV_IBVS_KP_YAW", 0.6),
            "kp_f":   _gain("FPV_IBVS_KP_F", 1.0),
            "kp_lat": _gain("FPV_IBVS_KP_LAT", 0.8),
            "kp_z":   _gain("FPV_IBVS_KP_Z", 1.0),
            "deadband": _gain("FPV_IBVS_DEADBAND", 0.05),
            "k_loss": int(_gain("FPV_IBVS_K_LOSS", 4)),
        }
        if gains:
            g.update(gains)
        self.g = g
        self.cx0, self.cy0 = cx0, cy0
        self.reset()

    def reset(self):
        self._hold_yaw = 0.0        # yaw action held on loss (D8)

    def action(self, box, h_star, drone_yaw=0.0):
        """One 4-D action from the current noisy box (dict cx,cy,h_px[,age])
        or None. h_star = expected h_px at the desired range (D5)."""
        lost = box is None or float(box.get("age", 0)) > self.g["k_loss"]
        if lost:
            # hover-on-loss (D8): v = 0, yaw holds its last command
            return np.array([0.0, 0.0, 0.0, self._hold_yaw], dtype=np.float32)
        cx_err = (float(box["cx"]) - self.cx0) / self.cx0
        cy_err = (float(box["cy"]) - self.cy0) / self.cy0
        ratio = float(box["h_px"]) / max(float(h_star), 1e-6) - 1.0
        # yaw servo (deadband)
        if abs(cx_err) > self.g["deadband"]:
            yaw_a = float(np.clip(self.g["kp_yaw"] * cx_err, -1.0, 1.0))
        else:
            yaw_a = 0.0
        self._hold_yaw = yaw_a
        # forward / lateral / vertical servos (body frame)
        v_f = float(np.clip(-self.g["kp_f"] * ratio, -1.0, 1.0))
        v_lat = float(np.clip(-self.g["kp_lat"] * cy_err, -1.0, 1.0))
        v_d = float(np.clip(-self.g["kp_z"] * ratio * 0.5, -1.0, 1.0)) * 0.5
        # body -> world (NED): forward=(cos,sin), right=(-sin,cos)
        c, s = np.cos(drone_yaw), np.sin(drone_yaw)
        vn = float(np.clip(v_f * c - v_lat * s, -1.0, 1.0))
        ve = float(np.clip(v_f * s + v_lat * c, -1.0, 1.0))
        return np.array([vn, ve, v_d, yaw_a], dtype=np.float32)
