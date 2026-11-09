#!/usr/bin/env python3
"""Offline unit tests for the S7 IBVS baseline controller (ibvs_follow.py, D8).

Pure numpy, NO sim. Synthetic box streams, assert-based:
  1. static centered target            -> actions ~ 0
  2. target drifting right             -> yaw action consistently POSITIVE
     (convention: positive yaw action = positive NED yaw rate = turning
     right/east, i.e. toward a target right-of-center in the image)
  3. box too big (too close)           -> body-forward component <= 0
  4. loss (no box / age > K_loss)      -> v = 0 and yaw holds the last cmd
  5. lateral: target below center      -> body-forward-lateral consistent
     (strafe "back" against forward axis, per documented sign convention)

Run: /usr/bin/python3 test_ibvs_follow.py   -> prints PASS, exit 0.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ibvs_follow import IBVSFollow

H_STAR = 181.333   # person at 6 m: 640*1.7/6


def main():
    ctl = IBVSFollow()

    # 1. static, centered, at range -> everything ~0
    ctl.reset()
    for _ in range(5):
        a = ctl.action({"cx": 360.0, "cy": 240.0, "h_px": H_STAR, "age": 0},
                       H_STAR, drone_yaw=0.0)
        assert a.shape == (4,)
        assert np.allclose(a, 0.0, atol=1e-6), a
    assert np.isfinite(a).all()

    # 2. target drifting right: cx_err > 0 => yaw action > 0 EVERY step
    #    (positive yaw action = turn right/east toward the target)
    ctl.reset()
    ys = []
    for i in range(8):
        cx = 360.0 + 24.0 * (i + 1)          # first step already past deadband
        a = ctl.action({"cx": cx, "cy": 240.0, "h_px": H_STAR, "age": 0},
                       H_STAR, drone_yaw=0.0)
        ys.append(a[3])
    assert all(y > 0.0 for y in ys), ys
    assert ys[-1] >= ys[0] - 1e-9            # grows with the error (saturates)
    assert ys[-1] <= 1.0 + 1e-9              # clipped to +-1 action units
    # mirrored: target drifting LEFT => yaw action consistently negative
    a = ctl.action({"cx": 200.0, "cy": 240.0, "h_px": H_STAR, "age": 0},
                   H_STAR, drone_yaw=0.0)
    assert a[3] < 0.0
    # deadband: tiny error => no yaw action
    a = ctl.action({"cx": 361.0, "cy": 240.0, "h_px": H_STAR, "age": 0},
                   H_STAR, drone_yaw=0.0)
    assert a[3] == 0.0                       # |cx_err|=0.0028 < 0.05

    # 3. too big (too close): forward component <= 0 (at yaw=0 vn IS forward)
    ctl.reset()
    for mult in (1.2, 1.5, 3.0):
        a = ctl.action({"cx": 360.0, "cy": 240.0, "h_px": H_STAR * mult,
                        "age": 0}, H_STAR, drone_yaw=0.0)
        assert a[0] <= 0.0, (mult, a)
        assert a[2] <= 0.0, (mult, a)        # z trim goes UP (NED down +)
    # too small (too far): forward > 0
    a = ctl.action({"cx": 360.0, "cy": 240.0, "h_px": H_STAR * 0.5, "age": 0},
                   H_STAR, drone_yaw=0.0)
    assert a[0] > 0.0 and a[2] > 0.0

    # 4. loss handling: None box or age > K_loss=4 => hover + yaw hold
    ctl.reset()
    a_seen = ctl.action({"cx": 500.0, "cy": 240.0, "h_px": H_STAR, "age": 0},
                        H_STAR, drone_yaw=0.0)
    hold = a_seen[3]
    assert hold > 0.0
    for box in (None, {"cx": 500.0, "cy": 240.0, "h_px": H_STAR, "age": 5}):
        a = ctl.action(box, H_STAR, drone_yaw=0.3)
        assert a[0] == 0.0 and a[1] == 0.0 and a[2] == 0.0, (box, a)
        assert a[3] == hold, (box, a)        # yaw HOLDS last command (D8)
    a = ctl.action({"cx": 500.0, "cy": 240.0, "h_px": H_STAR, "age": 4},
                   H_STAR, drone_yaw=0.0)    # age == K_loss is NOT loss yet
    assert a[3] > 0.0

    # 5. lateral convention: cy below center (too low/close on pitched cam)
    #    => negative lateral => at yaw=0 lateral is the EAST component
    ctl.reset()
    a = ctl.action({"cx": 360.0, "cy": 400.0, "h_px": H_STAR, "age": 0},
                   H_STAR, drone_yaw=0.0)
    assert a[1] < 0.0, a                     # v_lat = -kp_lat*cy_err
    # and with body yaw=90deg (nose east), a body-left strafe maps to NORTH:
    # vn = -v_lat*sin(yaw) = -a[1]
    a90 = ctl.action({"cx": 360.0, "cy": 400.0, "h_px": H_STAR, "age": 0},
                     H_STAR, drone_yaw=np.pi / 2)
    assert abs(a90[0] + a[1]) < 1e-5 and abs(a90[1]) < 1e-5

    # saturation: everything stays inside the action box
    ctl.reset()
    for cx in (0.0, 719.0):
        a = ctl.action({"cx": cx, "cy": 0.0, "h_px": 10000.0, "age": 0},
                       H_STAR, drone_yaw=1.0)
        assert np.all(np.abs(a) <= 1.0 + 1e-6), a

    print("PASS")


if __name__ == "__main__":
    main()
    sys.exit(0)
