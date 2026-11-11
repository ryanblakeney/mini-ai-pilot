#!/usr/bin/env python3
"""Offline unit tests for the S7 IBVS baseline (ibvs_follow.py, D8 + D14)
and the D16 AvoidIBVSFollow depth-grid veto.

Pure numpy, NO sim. Synthetic (box, est, range_valid, desired) streams,
assert-based -- the D14 app-0.3.0 alignment: yaw from horizontal image error,
forward 0.35 m/s per metre of range error, deadband 0.4 m, never advance
inside 1.5 m, no forward without a usable range, hover-on-loss with yaw hold,
and permanent all-zero give-up after 8 s (16 steps @ 0.5 s) of continuous
loss.

Run: /usr/bin/python3 test_ibvs_follow.py   -> prints PASS, exit 0.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ibvs_follow import AvoidIBVSFollow, IBVSFollow

DES = 6.0        # per-episode desired range (m)


def box(cx=360.0, cy=240.0, age=0):
    return {"cx": cx, "cy": cy, "age": age}


def main():
    ctl = IBVSFollow()

    # 1. centered, est == desired -> everything 0
    ctl.reset()
    for _ in range(5):
        a = ctl.action(box(), DES, True, DES, drone_yaw=0.0)
        assert a.shape == (4,)
        assert np.allclose(a, 0.0, atol=1e-6), a
    assert np.isfinite(a).all()

    # 2. yaw: target right of centre => yaw action POSITIVE every step
    #    (toward the target; + = NED yaw rate toward east/right)
    ctl.reset()
    ys = []
    for i in range(8):
        a = ctl.action(box(cx=360.0 + 24.0 * (i + 1)), DES, True, DES,
                       drone_yaw=0.0)
        ys.append(a[3])
    assert all(y > 0.0 for y in ys), ys
    assert ys[-1] >= ys[0] - 1e-9 and ys[-1] <= 1.0 + 1e-9
    a = ctl.action(box(cx=200.0), DES, True, DES, drone_yaw=0.0)
    assert a[3] < 0.0                                # mirrored: left => neg
    a = ctl.action(box(cx=361.0), DES, True, DES, drone_yaw=0.0)
    assert a[3] == 0.0                               # |cx_err| < yaw deadband

    # 3. range servo (app 0.3.0): 0.35 m/s per m error, at yaw=0 vn IS forward
    ctl.reset()
    # est > desired (too far) => chase: forward POSITIVE
    for est in (DES + 1.0, DES + 5.0, DES + 50.0):
        a = ctl.action(box(), est, True, DES, drone_yaw=0.0)
        assert a[0] > 0.0, (est, a)
        assert a[0] <= 1.0                           # +-1 m/s cap
    # est < desired (too close) => back off: forward NEGATIVE
    a = ctl.action(box(), DES - 2.0, True, DES, drone_yaw=0.0)
    assert a[0] < 0.0, a
    # vertical trim sign: est > desired (too far) => v_d < 0? no: est>desired
    # means too far -> (est/desired-1)>0 -> v_d = -k*.. <0 -> NED down + means
    # SINK; the trim descends toward a ground target that is far. OK by design.
    # deadband |err| <= 0.4 m => forward exactly 0 (kept off the exact 0.4
    # boundary: 6.4-6.0 rounds to 0.4000000000000004 > 0.4 in binary floats)
    for est in (DES + 0.35, DES - 0.3):
        a = ctl.action(box(), est, True, DES, drone_yaw=0.0)
        assert a[0] == 0.0 and a[1] == 0.0, (est, a)
    # gain check: err=2 m => v_f = clip(0.35*2) = 0.7 m/s
    a = ctl.action(box(), DES + 2.0, True, DES, drone_yaw=0.0)
    assert abs(a[0] - 0.7) < 1e-6, a

    # 4. close guard: est < 1.5 m with err > 0 => forward clamped <= 0
    #    (never advance inside 1.5 m)
    for est, des in ((1.4, 0.8), (1.45, 1.0), (1.0, 0.5)):
        a = ctl.action(box(), est, True, des, drone_yaw=0.0)
        assert a[0] <= 0.0, (est, des, a)
    # but backing OUT is still allowed when inside 1.5 and err < 0
    a = ctl.action(box(), 1.4, True, 3.0, drone_yaw=0.0)
    assert a[0] < 0.0, a

    # 5. no usable range => NO forward motion (v = 0, z trim 0)
    for est, valid in ((float("inf"), True), (99.0, False),
                       (float("nan"), False)):
        a = ctl.action(box(), est, valid, DES, drone_yaw=0.0)
        assert a[0] == 0.0 and a[1] == 0.0 and a[2] == 0.0, (est, valid, a)
    # yaw still works without a range (app: yaw comes from image, not range)
    a = ctl.action(box(cx=500.0), float("inf"), False, DES, drone_yaw=0.0)
    assert a[3] > 0.0

    # 6. loss handling: no box / age > K_loss=4 => hover + yaw HOLD
    ctl.reset()
    a_seen = ctl.action(box(cx=500.0), DES, True, DES, drone_yaw=0.0)
    hold = a_seen[3]
    assert hold > 0.0
    for b in (None, box(age=5)):
        a = ctl.action(b, DES, True, DES, drone_yaw=0.3)
        assert a[0] == 0.0 and a[1] == 0.0 and a[2] == 0.0, (b, a)
        assert a[3] == hold, (b, a)                  # yaw HOLDS (D8)
    a = ctl.action(box(cx=500.0, age=4), DES, True, DES, drone_yaw=0.0)
    assert a[3] > 0.0                                # age == K_loss: not loss

    # 7. give-up: 16 CONSECUTIVE loss steps (8 s / 0.5 s) => all zeros
    #    permanently, yaw 0 (not even held), until reset()
    ctl.reset()
    ctl.action(box(cx=500.0), DES, True, DES)       # establish a held yaw
    for i in range(15):                              # 15 losses: still hover
        a = ctl.action(None, DES, True, DES, drone_yaw=0.0)
        assert a[3] > 0.0, (i, a)                    # yaw HELD, not given up
        assert a[0] == 0.0 and a[1] == 0.0
    a = ctl.action(None, DES, True, DES, drone_yaw=0.0)   # 16th: GIVEN UP
    assert np.allclose(a, 0.0), a
    a = ctl.action(box(cx=500.0), 12.0, True, DES)   # a fresh box changes
    assert np.allclose(a, 0.0), a                    # NOTHING -- zeros forever
    ctl.reset()                                      # reset() revives it
    a = ctl.action(box(cx=500.0), 12.0, True, DES)
    assert a[0] > 0.0 and a[3] > 0.0                 # chase + yaw restored

    # 8. loss SHORTER than 8 s then re-acquire => works normally again
    ctl.reset()
    ctl.action(box(cx=500.0), DES, True, DES)
    for _ in range(10):                              # 5 s of loss (< 8 s)
        a = ctl.action(None, DES, False, DES)
        assert a[0] == 0.0
    a = ctl.action(box(), 12.0, True, DES, drone_yaw=0.0)   # re-acquired,
    assert a[0] > 0.0, a                             # too far => chase again

    # 9. lateral convention (unchanged D8): cy below center (target low on the
    #    -20 deg pitched cam) => strafe "back" against the forward axis; body
    #    (v_f, v_lat) rotate to world by yaw. est at deadband => forward 0.
    ctl.reset()
    a = ctl.action(box(cy=400.0), DES, True, DES, drone_yaw=0.0)
    assert a[1] < 0.0, a                             # v_lat = -kp_lat*cy_err
    a90 = ctl.action(box(cy=400.0), DES, True, DES, drone_yaw=np.pi / 2)
    assert abs(a90[0] + a[1]) < 1e-5 and abs(a90[1]) < 1e-5

    # saturation: everything stays inside the action box
    ctl.reset()
    for cx in (0.0, 719.0):
        a = ctl.action(box(cx=cx, cy=0.0), 1e6, True, 1.0, drone_yaw=1.0)
        assert np.all(np.abs(a) <= 1.0 + 1e-6), a

    # --- D16 AvoidIBVSFollow: depth-grid forward veto (velocity only) ----
    # grid = the agent's obs[22:37]: 3x5 row-major, val=(d-1)/19, free=1.0.
    # float64 grids here so the threshold math in test (d) is EXACT.
    VETO = 4.0                       # default veto_m
    THR = (VETO - 1.0) / 19.0        # col_min < THR  <=>  obstacle < veto_m

    def gblock(cols, val):
        g = np.ones(15, dtype=np.float64)
        for r in range(3):
            for c in cols:
                g[r * 5 + c] = val
        return g

    av = AvoidIBVSFollow()

    # (a) free grid => EXACT passthrough of the inner IBVS action
    ref = IBVSFollow()
    for est in (DES + 2.0, DES - 2.0, DES + 0.35):
        a_av = av.action(box(), est, True, DES, drone_yaw=0.4,
                         grid=np.ones(15, dtype=np.float64))
        a_ref = ref.action(box(), est, True, DES, drone_yaw=0.4)
        assert np.allclose(a_av, a_ref), (est, a_av, a_ref)
    a_av = av.action(box(cx=500.0), DES, True, DES, 0.0,
                     np.ones(15, dtype=np.float64))
    a_ref = ref.action(box(cx=500.0), DES, True, DES, 0.0)
    assert np.allclose(a_av, a_ref)
    # side column only (c=0/c=4) never vetoes the centre corridor
    a_av = av.action(box(), DES + 2.0, True, DES, drone_yaw=0.0,
                     grid=gblock([0, 4], 0.02))
    assert a_av[0] > 0.5, a_av

    # (b) near-centre obstacle (all rows, cols 1-3 at 0.05 << THR) vetoes
    #     forward: inner would charge 0.7 m/s north at yaw=0 (est = DES+2).
    inner_ref = IBVSFollow()
    a_in = inner_ref.action(box(), DES + 2.0, True, DES, drone_yaw=0.0)
    assert a_in[0] > 0.5                              # inner DOES charge
    a = av.action(box(), DES + 2.0, True, DES, drone_yaw=0.0,
                  grid=gblock([1, 2, 3], 0.05))
    fwd = np.array([np.cos(0.0), np.sin(0.0)])        # blocker direction
    assert float(a[:2] @ fwd) <= 1e-9, a              # no motion toward it
    assert abs(a[0]) <= 1e-9, a                       # forward zeroed
    assert 0.0 < abs(a[1]) <= 0.6 + 1e-6, a           # lateral strafe <= 0.6
    assert a[2] == a_in[2] and a[3] == a_in[3], (a, a_in)  # v_d/yaw untouched

    # (c) strafe side = the MORE OPEN outer column (col_min[0] vs col_min[4])
    a_left = av.action(box(), DES + 2.0, True, DES, drone_yaw=0.0,
                       grid=gblock([1, 2, 3, 4], 0.05))   # right wall -> go left
    a_right = av.action(box(), DES + 2.0, True, DES, drone_yaw=0.0,
                        grid=gblock([0, 1, 2, 3], 0.05))   # left wall -> go right
    assert a_left[1] < 0.0 and abs(a_left[0]) <= 1e-9, a_left
    assert a_right[1] > 0.0 and abs(a_right[0]) <= 1e-9, a_right
    assert abs(a_left[1]) == min(0.6, a_in[0]) or abs(abs(a_left[1]) - 0.6) < 1e-6
    a_90 = av.action(box(), DES + 2.0, True, DES, drone_yaw=np.pi / 2,
                     grid=gblock([0, 1, 2, 3], 0.05))  # right of east = -x
    assert a_90[0] < 0.0 and abs(a_90[1]) < 1e-6, a_90

    # (d) threshold math: obstacle AT exactly veto_m NOT vetoed (val < THR
    #     is strict); at veto_m - 0.1 IS vetoed.
    a = av.action(box(), DES + 2.0, True, DES, drone_yaw=0.0,
                  grid=gblock([1, 2, 3], THR))
    assert a[0] > 0.5, a                              # at veto_m: still charges
    a = av.action(box(), DES + 2.0, True, DES, drone_yaw=0.0,
                  grid=gblock([1, 2, 3], (VETO - 0.1 - 1.0) / 19.0))
    assert abs(a[0]) <= 1e-9, a                       # inside veto_m: vetoed

    # hover-on-loss under a vetoed grid: nothing to veto => inner passthrough
    a = av.action(None, DES, True, DES, drone_yaw=0.0, grid=gblock([1, 2, 3], 0.0))
    assert np.allclose(a[:3], 0.0), a

    # reset() propagates to the inner give-up state
    for _ in range(16):
        av.action(None, DES, True, DES, 0.0, np.ones(15, dtype=np.float64))
    assert np.allclose(av.action(box(cx=500.0), DES, True, DES, 0.0,
                                 np.ones(15, dtype=np.float64)), 0.0)  # given up
    av.reset()
    a = av.action(box(), DES + 2.0, True, DES, 0.0,
                  np.ones(15, dtype=np.float64))
    assert a[0] > 0.0, a                              # revived

    print("PASS")


if __name__ == "__main__":
    main()
    sys.exit(0)
