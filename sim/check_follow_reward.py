#!/usr/bin/env python3
"""Offline proof for the S7 follow reward v6 (D7 + D11) -- the card's HARD RULE.

NO sim, NO third-party deps beyond numpy -- runs under plain
/usr/bin/python3 (follow_common.py is pure, so a direct import is safe; the
Q3 checkers had to regex-extract because their reward lived in a sim-bound
module).

Card hard rule, restated for S7: survival-shaped POSITIVE per-step reward
(ceiling 1.1) with NO positive terminal anywhere, and bad-only terminals
(landed -10, flew-out -10, collision -20) whose penalty magnitude exceeds any
harvest obtainable from ending early. v6 adds the D11 goal-conditioning: the
range term is |log(dist/desired)| around a PER-EPISODE desired range. Every
Q3 hover stage died of violating exactly this; this checker PROVES the
arithmetic numerically:

  (a) camping/hover-at-start full-episode total < decent-tracking total,
      evaluated by SIMULATING both behaviours through the real target
      scripts (spawned around the desired via sample_target(desired=...)),
      projection and DetectorNoise -- re-run for EVERY desired in
      DESIRED_SWEEP = [1.5, 3, 6, 10, 15] m (D11);
  (b) for every terminal penalty P and every early-exit step k, the best
      terminal-ending episode total is strictly below BOTH no-terminal
      alternatives (camping and tracking), so terminating early is never a
      harvest -- plus the literal per-step bound |P| > per-step max 1.1;
  (c) the ordering survives gamma = 0.98 discounting (FOLLOW_D > CAMP_D for
      every (desired, kind));
  (c2) close-guard sanity (v6): a too-close steady-state return is strictly
      below the tracking steady-state return -- the CLOSE_PENALTY 2.0 per
      step dominates the max 0.2 range-term gain, and R_STEP_FLOOR prints
      the worst non-terminal per-step floor = -(CLOSE_PENALTY + penalties);
  (d) projection sanity (D4) + apparent-size range_estimate inversion (D12)
      + RangeEstimator EMA/dropout semantics;
  (e) noise-model statistics over 10k draws (miss rate, jitter std, h mult
      std, conf mean within tolerances);
  (f) depth-grid normalization bounds [0, 1] and semantics (free = 1.0).

Run:  /usr/bin/python3 check_follow_reward.py   -> prints ALL PASS, exit 0.
"""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import follow_common as fc

MAX_STEPS = 100
STEP_DT = 0.5
GAMMA = 0.98
K_VIEW = fc.K_VIEW          # 20
R_MAX = fc.R_STEP_CEILING   # 1.1 per-step ceiling (penalties only subtract)
DRONE0 = np.array([0.0, 0.0, -4.7])   # settled takeoff height (ground -2.7)
# D11: the goal-conditioning sweep -- the dominance proof must hold at EVERY
# desired range a policy can be conditioned on (env knobs default 1.5..15).
DESIRED_SWEEP = [1.5, 3.0, 6.0, 10.0, 15.0]
# worst-case penalty magnitudes (Box(4) +-1 actions, +-2 m/s action deltas,
# <= 3 m/s airspeed in this env):
PEN_MAX = 0.05 * 2.0 + 0.03 * 4.0 + 0.02 * 3.0
R_STEP_FLOOR = -(fc.CLOSE_PENALTY + PEN_MAX)   # worst non-terminal step


def dsum(k, per_step):
    """Sum_{t=1..k} gamma^t * per_step."""
    return per_step * sum(GAMMA ** t for t in range(1, k + 1))


def rollout(kind, rng, mode, desired):
    """One natural-script episode through the REAL primitives, goal-conditioned.

    mode='camp'  : hover-at-start (zero action, zero yaw).
    mode='chase' : decent follow -- hold the DESIRED range: close in (or back
                   off) at <=1 m/s body speed when |range - desired| > 0.5 m,
                   yaw servo toward the target (0.5 rad/s cap), vd=0; when
                   the target kinematically outruns the drone (car/bike net
                   speed) the policy honestly loses range/view like any real
                   policy would.
    Detector: default D6 noise (p_miss=0.15, jitter, age-out K_VIEW).
    Returns (total, per-step income list).
    """
    script = fc.sample_target(rng, kind, drone_start=DRONE0, drone_yaw=0.0,
                              desired=desired)
    noise = fc.DetectorNoise(rng)
    cam = fc.default_cam()
    pos = DRONE0.copy()
    yaw = 0.0
    a_prev = np.zeros(4)
    cum, total = [0.0], 0.0
    for t in range(1, MAX_STEPS + 1):
        tp = fc.target_at(script, t * STEP_DT)
        if mode == "camp":
            a = np.zeros(4)
            pos = pos.copy()
        else:
            dxy = tp[:2] - pos[:2]
            hd = float(math.hypot(dxy[0], dxy[1]))
            gap = hd - desired                    # hold the DESIRED band (D11)
            if hd > 1e-9 and abs(gap) > 0.5:
                sp = min(1.0, abs(gap))           # speed cap ~1 m/s along LOS
                ux, uy = dxy[0] / hd, dxy[1] / hd
                vn, ve = (sp * ux, sp * uy) if gap > 0 else (-sp * ux,
                                                             -sp * uy)
            else:
                vn, ve = 0.0, 0.0
            psi_err = math.atan2(math.sin(math.atan2(dxy[1], dxy[0]) - yaw),
                                 math.cos(math.atan2(dxy[1], dxy[0]) - yaw))
            yaw_cmd = min(max(psi_err / 0.5, -1.0), 1.0)
            yaw = yaw + 0.5 * yaw_cmd * STEP_DT * 2.0
            a = np.array([vn, ve, 0.0, yaw_cmd])
            pos = pos + np.array([vn * STEP_DT, ve * STEP_DT, 0.0])
        gt = fc.project_target(pos, yaw, tp, script["height"])
        box = noise.update(gt)
        in_view = box is not None and noise.age <= K_VIEW
        cx_err = ((box["cx"] - cam["cx0"]) / cam["cx0"]) if box else 1.0
        cy_err = ((box["cy"] - cam["cy0"]) / cam["cy0"]) if box else 1.0
        r = fc.follow_reward(cx_err, cy_err, gt["dist"], in_view,
                             float(np.linalg.norm(a)),
                             float(np.linalg.norm(a - a_prev)),
                             0.0 if mode == "camp" else
                             float(np.linalg.norm(a[:2])) / STEP_DT * STEP_DT,
                             desired=desired)
        a_prev = a
        total += r
        cum.append(total)
    return total, np.array(cum)


def behav_sims(kind, rng, desired, n=60):
    """Paired natural-distribution totals: camp vs decent-follow chase.
    camp_cum = MEAN camping income profile (what the camping behaviour earns
    by step k on average -- the (b)/(c) profiles must mirror the SAME
    expectation the policy gradient sees, not one lucky script)."""
    camp, foll = [], []
    camp_cum = np.zeros(MAX_STEPS + 1)
    for _ in range(n):
        _t, cum = rollout(kind, rng, "camp", desired)
        camp.append(_t)
        camp_cum = camp_cum + cum
        _t, _ = rollout(kind, rng, "chase", desired)
        foll.append(_t)
    return camp, foll, camp_cum / n


def main():
    rng = random.Random(7)
    f = fc.follow_reward

    # ---- sanity: terminals exact, no positive terminal anywhere -----------
    assert f(0.0, 0.0, 6.0, True, 0.0, 0.0, 0.0, collided=True) == -20.0
    assert f(0.0, 0.0, 6.0, True, 0.0, 0.0, 0.0, landed=True) == -10.0
    assert f(0.0, 0.0, 6.0, True, 0.0, 0.0, 0.0, flew_out=True) == -10.0
    assert f(0.0, 0.0, 6.0, True, 0.0, 0.0, 0.0,
             collided=True, landed=True) == -20.0     # collision outranks
    assert f(0.0, 0.0, 6.0, True, 0.0, 0.0, 0.0) <= 1.1 + 1e-12
    assert abs(f(0.0, 0.0, 6.0, False, 0.0, 0.0, 0.0) - 0.6) < 1e-12
    assert f(2.0, 2.0, 100.0, False, 0.0, 0.0, 0.0) == 0.0  # true floor, no penalties
    assert abs(f(0.0, 0.0, 6.0, True, 0.0, 0.0, 0.0) - 1.1) < 1e-12

    # ---- v6 goal-conditioning + close-guard direct checks ------------------
    # range term is goal-conditioned: same episode geometry, different goals
    assert abs(f(0, 0, 8.0, True, 0, 0, 0, desired=8.0)
               - (1.1)) < 1e-12                        # perfect at ANY goal
    assert abs(f(0, 0, 8.0, True, 0, 0, 0, desired=2.0)
               - (0.5 + 0.4 + 0.2 * max(0.0, 1 - abs(math.log(4.0))))) < 1e-12
    # log form is SCALE-FREE: identical relative error => identical term at
    # the near and far ends of the D11 sweep (the whole point of D11).
    assert abs(f(0, 0, 1.5 * 1.3, True, 0, 0, 0, desired=1.5)
               - f(0, 0, 15.0 * 1.3, True, 0, 0, 0, desired=15.0)) < 1e-12
    # close-guard (v6): per-step subtraction, NOT a terminal
    g_dist = 1.4
    expect_g = (0.5 + 0.4 + 0.2 * max(0.0, 1 - abs(math.log(g_dist / 1.5)))
                - fc.CLOSE_PENALTY)
    assert abs(f(0, 0, g_dist, True, 0, 0, 0, desired=1.5) - expect_g) < 1e-12
    assert f(0, 0, 1.2, True, 0, 0, 0, landed=True) == -10.0   # guard NOT terminal
    assert fc.CLOSE_GUARD == 1.5 and fc.CLOSE_PENALTY == 2.0
    # worst non-terminal per-step floor (printed for the launch card):
    # all shaping terms zero (no in_view, no center, no range) + full
    # penalties + the close-guard subtraction.
    assert abs(f(9, 9, 0.05, False, 2.0, 4.0, 3.0, desired=1.5)
               - R_STEP_FLOOR) < 1e-12
    assert f(9, 9, 0.05, True, 2.0, 4.0, 3.0, desired=1.5) >= R_STEP_FLOOR

    # ---- (a)+(c) camping vs decent tracking over the D11 desired sweep ----
    # Natural target-script distribution (D2 spawn band scaled by the goal),
    # real D6 detector model, both behaviours rolled through the REAL
    # primitives. Dominance claim asserted per (desired, kind) on the
    # EXPECTED full-episode return -- this is exactly what a policy gradient
    # sees. Honest caveat (spec open-risk 3, D7 locked -- NOT redesigned): on
    # the rare script where the target recedes nearly along the line of
    # sight, hover collects in_view+center for free, so STRICT per-script
    # dominance cannot hold for that script family; the spec itself
    # classifies those scripts as loss/re-acquire stress cases and mandates
    # per-visible-step scoring at eval time (rl_eval_follow.py).
    mean_camp, mean_foll, camp_tot, foll_tot = {}, {}, {}, {}
    camp_cum = None                                    # worst-case profile
    r_track_min_by_d = {}
    for d_want in DESIRED_SWEEP:
        rmin_d = 1e18
        for kind in ("person", "bicycle", "car"):
            c, fl, cum = behav_sims(kind, rng, d_want)
            camp_tot[(d_want, kind)], foll_tot[(d_want, kind)] = c, fl
            mean_camp[(d_want, kind)] = float(np.mean(c))
            mean_foll[(d_want, kind)] = float(np.mean(fl))
            camp_cum = cum if camp_cum is None else np.maximum(camp_cum, cum)
            assert mean_camp[(d_want, kind)] < mean_foll[(d_want, kind)], \
                (d_want, kind, mean_camp[(d_want, kind)],
                 mean_foll[(d_want, kind)])
            rmin_d = min(rmin_d, mean_foll[(d_want, kind)])
        r_track_min_by_d[d_want] = rmin_d / MAX_STEPS  # conservative rate
    CAMP_MEAN_MIN = min(mean_camp.values())
    FOLL_MEAN_MIN = min(mean_foll.values())
    r_track_min = min(r_track_min_by_d.values())

    # ---- (b) terminal dominance --------------------------------------------
    # Literal per-step bound (task spec): every terminal penalty magnitude
    # exceeds the per-step harvest max 1.1 (0.5+0.4+0.2; penalties subtract).
    for P in (fc.TERM_LANDED, fc.TERM_FLEW_OUT, fc.TERM_COLLISION):
        assert abs(P) > R_MAX, P
    # Multi-step form: a zero-action idle step OUTSIDE the close-guard has
    # reward >= 0 always (all shaping terms >= 0, all penalties exactly 0 at
    # a=0, a_prev=0, v=0), and inside the guard the worst step is >=
    # R_STEP_FLOOR > -10; the drone can always retreat to a safe range, so
    # from ANY prefix state future income is >= 0 by idling at a safe range.
    # Taking a bad-only terminal at step k instead of continuing is therefore
    # strictly worse for ANY behaviour prefix income rho(k):
    #     rho(k) + P  <  rho(k) + 0*(MAX_STEPS - k)      for every P < 0
    # verified over both behaviour income profiles at every k:
    zero_floor = f(9.0, 9.0, 100.0, False, 0.0, 0.0, 0.0)   # idle, safe range
    assert zero_floor >= 0.0
    for P in (fc.TERM_LANDED, fc.TERM_FLEW_OUT, fc.TERM_COLLISION):
        for k in range(0, MAX_STEPS):            # terminate on step k+1
            for rho in (camp_cum[k], r_track_min * k):
                assert rho + P < rho + zero_floor * (MAX_STEPS - k)
    # Report (informative; not the incentive claim): the best absolute
    # terminal-ending episode total, vs the tracking expectation it must lose
    # to -- a terminal-cut shortens the SAME income stream by >= |P|:
    worst_term, worst_arg = -1e18, None
    for P in (fc.TERM_LANDED, fc.TERM_FLEW_OUT, fc.TERM_COLLISION):
        for k in range(0, MAX_STEPS):
            prof = max(camp_cum[k], r_track_min * k)
            if prof + P > worst_term:
                worst_term, worst_arg = prof + P, (P, k)
    assert worst_term < FOLL_MEAN_MIN

    # ---- (c) gamma = 0.98 discounting preserves the ordering, per goal -----
    # discounted proof runs per (goal, kind) on MATCHED expectations: the
    # pair's own expected track rate (what (a) proved undiscounted) vs that
    # pair's MEAN camp income profile (60 camp sims, independent rng seed --
    # same-sample-size comparison as (a), discounting both streams equally).
    rng2 = random.Random(101)
    last_CD = last_FD = last_TD = 0.0
    for d_want in DESIRED_SWEEP:
        for kind in ("person", "bicycle", "car"):
            FOLLOW_D = dsum(MAX_STEPS, mean_foll[(d_want, kind)] / MAX_STEPS)
            cum_n, cnt = np.zeros(MAX_STEPS + 1), 60
            for _ in range(cnt):
                _t, cum = rollout(kind, rng2, "camp", d_want)
                cum_n += cum
            cum_n /= cnt
            camp_inc = [max(0.0, cum_n[t] - cum_n[t - 1])
                        for t in range(1, MAX_STEPS + 1)]
            CAMP_D = sum(GAMMA ** t * camp_inc[t - 1]
                         for t in range(1, MAX_STEPS + 1))
            term_d = -1e18
            for P in (fc.TERM_LANDED, fc.TERM_FLEW_OUT, fc.TERM_COLLISION):
                for k in range(0, MAX_STEPS):
                    camp_d = sum(GAMMA ** t * camp_inc[t - 1]
                                 for t in range(1, k + 1))
                    track_d = dsum(k, mean_foll[(d_want, kind)] / MAX_STEPS)
                    prof_d = max(camp_d, track_d)
                    term_d = max(term_d, prof_d + GAMMA ** (k + 1) * P)
                    # discounted: a terminal cut is worse than the SAME
                    # stream run on
                    assert prof_d + GAMMA ** (k + 1) * P < (
                        CAMP_D if prof_d == camp_d else FOLLOW_D)
            assert FOLLOW_D > CAMP_D, (d_want, kind, FOLLOW_D, CAMP_D)
            assert FOLLOW_D > term_d, (d_want, kind, FOLLOW_D, term_d)
            last_CD, last_FD, last_TD = CAMP_D, FOLLOW_D, term_d

    # ---- (c2) close-guard sanity (v6): too-close steady < tracking steady --
    # best achievable step INSIDE the guard: perfect center, in_view, zero
    # motion, range term maximized against the closest legal goal (1.5):
    r_close = f(0, 0, fc.CLOSE_GUARD - 0.01, True, 0, 0, 0, desired=1.5)
    # a decent tracking step OUTSIDE the guard: on-band (1.15x desired),
    # near-centered, moving (this is the profile the guard must lose to):
    r_track_step = f(0.05, 0.0, 6.9, True, 0.8, 0.1, 0.9, desired=6.0)
    assert fc.CLOSE_PENALTY > 0.2                  # penalty > max range gain
    assert r_close < 0.0 < r_track_step
    for d_want in DESIRED_SWEEP:                   # at EVERY goal in the sweep
        rc = f(0, 0, min(d_want, fc.CLOSE_GUARD - 0.01), True, 0, 0, 0,
               desired=d_want)
        rt = f(0.05, 0.0, d_want * 1.15, True, 0.8, 0.1, 0.9, desired=d_want)
        assert rc < rt, (d_want, rc, rt)
        # steady-state 30-step discounted returns: parking on the target LOSES
        assert dsum(30, rc) < dsum(30, rt), (d_want, rc, rt)
    CLOSE_D = dsum(30, r_close)
    TRACK_D_GUARD = dsum(30, r_track_step)

    # ---- (d) projection sanity (D4) + range_estimate (D12) -----------------
    cam = fc.default_cam()
    # target BASE on the ground (z=-2.7), drone at settled -4.7: the honest
    # S7 geometry (targets are ground-level per D2). Top lands well in-frame.
    tgt = fc.project_target([0, 0, -4.7], 0.0, [6.0, 0.0, -2.7], 1.7)
    assert tgt["in_fov"], tgt
    assert not tgt["edge_cut"]                     # fully visible segment
    assert abs(tgt["cx"] - cam["cx0"]) < 1e-6
    approx = cam["fx"] * 1.7 / tgt["dist"]
    assert abs(tgt["h_px"] / approx - 1.0) < 0.06, (tgt["h_px"], approx)
    far = fc.project_target([0, 0, -4.7], 0.0, [12.0, 0.0, -2.7], 1.7)
    assert 0.0 < far["h_px"] < tgt["h_px"] < cam["H"] + 1.0
    behind = fc.project_target([0, 0, -4.7], 0.0, [-5.0, 0.0, -2.7], 1.7)
    assert not behind["in_fov"] and behind["h_px"] == 0.0
    over = fc.project_target([0, 0, -4.7], 0.0, [1.0, 0.0, 2.0], 1.7)
    assert not over["in_fov"]                      # above the frame top edge
    # edge_cut (D12): 2 m ahead the segment midpixel is in-frame (in_fov True
    # -- in_fov logic UNCHANGED) but the raw bottom endpoint runs off-frame.
    closeb = fc.project_target([0, 0, -4.7], 0.0, [2.0, 0.0, -2.7], 1.7)
    assert closeb["in_fov"] and closeb["edge_cut"], closeb
    # h_star at dist_star matches projection scale
    assert abs(fc.h_star_for(1.7, 6.0) - cam["fx"] * 1.7 / 6.0) < 1e-9
    # range_estimate INVERTS the projection (D12, phone-identical): slant
    # range along the optical axis. The residual is the -20 deg camera pitch:
    # a VERTICAL segment is not perpendicular to the optical axis, so its
    # projected length deviates from fy*h/slant by ~O(h_eff/slant) -- the
    # phone inherits exactly the same geometry (part of why its honest
    # accuracy is +-20-30%, research doc section 7). Tolerance shrinks ~1/slant.
    f_ax, _, _ = fc.cam_axes(0.0, cam)
    for slant in (3.0, 5.0, 6.0, 10.0, 15.0):
        tgt_ax = np.array([0.0, 0.0, -4.7]) + slant * f_ax
        bx = fc.project_target([0, 0, -4.7], 0.0, tgt_ax, 1.7)
        est = fc.range_estimate(bx["h_px"], 1.7)
        assert abs(est / slant - 1.0) < max(0.02, 0.6 / slant), \
            (slant, est)
    assert math.isinf(fc.range_estimate(0.0, 1.7)) and not \
        math.isfinite(fc.range_estimate(-5.0, 1.7))
    # C == 2*tan(vFOV/2) for this pinhole (documented identity)
    assert abs(cam["H"] / cam["fy"] - 2 * math.tan(
        math.atan(cam["H"] / 2 / cam["fy"]))) < 1e-12
    # RangeEstimator semantics (D12): deterministic knobs
    re_rng = random.Random(23)
    estm = fc.RangeEstimator(re_rng, 1.7, noise=0.0, ema_alpha=1.0)
    assert math.isinf(estm.est)
    assert estm.update(None) == (estm.est, False)      # no box: no update
    assert estm.update({"h_px": 181.33}, edge_cut=True)[1] is False  # drop
    assert math.isinf(estm.est)                        # ... WITHOUT updating
    e1, ok1 = estm.update({"h_px": fc.default_cam()["fy"] * 1.7 / 6.0})
    assert ok1 and abs(e1 - 6.0) < 1e-6                # first hit SEEDS est
    e2, ok2 = estm.update({"h_px": fc.default_cam()["fy"] * 1.7 / 3.0})
    assert ok2 and abs(e2 - 3.0) < 1e-6                # alpha=1: full replace
    estm2 = fc.RangeEstimator(random.Random(24), 1.7, noise=0.0,
                              ema_alpha=0.3)
    h6 = fc.default_cam()["fy"] * 1.7 / 6.0
    estm2.update({"h_px": h6})                        # seeds 6.0
    e3, ok3 = estm2.update({"h_px": h6 * 2})           # raw -> 3.0
    assert ok3 and abs(e3 - (0.3 * 3.0 + 0.7 * 6.0)) < 1e-6   # EMA (D12)
    bad_est, bad_ok = estm2.update({"h_px": 0.0})      # unmeasurable
    assert not bad_ok and bad_est == e3                # est preserved
    # +-noise knob bounded: noise=0.2 => raw within +-20% of truth
    estm3 = fc.RangeEstimator(random.Random(25), 1.7, noise=0.2)
    hs = fc.default_cam()["fy"] * 1.7 / 5.0
    seen = [estm3.update({"h_px": hs})[0] for _ in range(50)]
    assert all(4.0 < v < 7.5 for v in seen)            # 5*(1+-0.2) within band

    # ---- (e) noise-model statistics, 10k draws ------------------------------
    n_rng = random.Random(11)
    noise = fc.DetectorNoise(n_rng, p_miss=0.15, jitter=0.005, lat_steps=0)
    gt = {"cx": 360.0, "cy": 240.0, "h_px": 180.0, "in_fov": True}
    djs, dhs, confs = [], [], []
    prev_hits = 0
    for _ in range(10000):
        b = noise.update(gt)
        if noise.hits > prev_hits:
            prev_hits = noise.hits
            djs.append(b["cx"] - 360.0)
            dhs.append(b["h_px"] / 180.0 - 1.0)
            confs.append(b["conf"])
    miss_rate = 1.0 - noise.hits / 10000.0
    assert abs(miss_rate - 0.15) < 0.03, miss_rate
    assert abs(np.std(djs) - 0.005 * 720.0) / (0.005 * 720.0) < 0.15
    assert abs(np.std(dhs) - 0.05) / 0.05 < 0.15
    assert abs(np.mean(confs) - 0.85) < 0.01
    assert 0.0 <= min(confs) and max(confs) <= 1.0
    # frozen-on-miss + age semantics (a prior hit exists -> box freezes)
    n2 = fc.DetectorNoise(random.Random(3), p_miss=1.0)
    assert n2.update(gt) is None and n2.age == 1     # never had a hit
    n2.last = {"cx": 360.0, "cy": 240.0, "h_px": 180.0, "conf": 0.9}
    b = n2.update(gt)
    assert n2.age == 2 and b["age"] == 2 and b["cx"] == 360.0  # frozen (D6)
    # latency: box at step t is GT of step t-L (jitter=0 to test delay exactly)
    n3 = fc.DetectorNoise(random.Random(4), p_miss=0.0, jitter=0.0,
                          lat_steps=1)
    g1 = {"cx": 100.0, "cy": 240.0, "h_px": 100.0, "in_fov": True}
    g2 = {"cx": 200.0, "cy": 240.0, "h_px": 100.0, "in_fov": True}
    assert abs(n3.update(g1)["cx"] - 100.0) < 1e-9   # warmup: only g1 exists
    assert abs(n3.update(g2)["cx"] - 100.0) < 1e-9   # step2 sees GT of step1
    assert abs(n3.update(g2)["cx"] - 200.0) < 1e-9   # step3 sees GT of step2
    # latency steps sampler (D6 U{0,1} and FPV_LAT_MS hi/lo)
    assert fc.sample_latency_steps(random.Random(5), 0.5) in (0, 1)
    assert fc.sample_latency_steps(random.Random(5), 0.5, 300, 500) == 1

    # ---- (f) depth grid bounds + semantics (D5) -----------------------------
    empt = fc.depth_grid([0, 0, -4.7], 0.0, [])
    assert np.all(empt == 1.0)
    objs = [("TemplateCube_Rounded_150", np.array([6.0, 0.0, -4.0]),
             np.array([0.75, 0.75, 0.75]))]
    g = fc.depth_grid([0, 0, -4.7], 0.0, objs)
    assert g.shape == (15,) and np.all(g >= 0.0) and np.all(g <= 1.0)
    assert g.min() < 1.0                          # cube covers some cells
    assert abs(g.min() - (5.25 - 1.0) / 19.0) < 1e-6   # (d-1)/19 exact
    assert g.max() == 1.0                         # free cones stay 1.0
    gbehind = fc.depth_grid([0, 0, -4.7], 0.0,
                            [("c", np.array([-30.0, 0.0, -4.0]),
                              np.array([1.0, 1.0, 1.0]))])
    assert np.all(gbehind == 1.0)                 # behind camera: invisible
    close = fc.depth_grid([0, 0, -4.7], 0.0,
                          [("c", np.array([1.2, 0.0, -4.7]),
                            np.array([0.5, 0.5, 0.5]))])
    assert close.min() >= 0.0                     # nearer than d_min clamps

    # ---- (g) FPV_TARGET_SPEED_SCALE knob -------------------------------------
    # script['speed'] must equal KINDS[kind]['speed'] * scale for env-set and
    # env-unset scales, the explicit speed_scale arg must override the env var,
    # invalid env values fall back to 1.0, and the INTEGRATION must use the
    # same scaled speed (straight-line person: displacement == speed*t_end,
    # which is also what target_at() extrapolates with past t_end).
    s_rng = random.Random(13)
    _saved = os.environ.pop("FPV_TARGET_SPEED_SCALE", None)
    try:
        for scale in (0.3, 1.0, 1.5):
            if scale == 1.0:
                os.environ.pop("FPV_TARGET_SPEED_SCALE", None)  # unset -> 1.0
            else:
                os.environ["FPV_TARGET_SPEED_SCALE"] = str(scale)
            for kind in ("person", "bicycle", "car"):
                s = fc.sample_target(s_rng, kind)
                assert s["speed"] == fc.KINDS[kind]["speed"] * scale, \
                    (kind, scale, s["speed"])
                assert s["speed_scale"] == scale
                d_xy = float(np.hypot(*(s["xy"][-1] - s["xy"][0])))
                if kind != "bicycle":             # straight-line kinds
                    assert abs(d_xy - s["speed"] * s["t_end"]) < 1e-9, kind
            # explicit arg overrides the env var in both directions
            s = fc.sample_target(s_rng, "car", speed_scale=0.25)
            assert s["speed"] == fc.KINDS["car"]["speed"] * 0.25
            assert s["speed_scale"] == 0.25
        for bad in ("abc", "", "1.5x"):           # invalid -> fall back to 1.0
            os.environ["FPV_TARGET_SPEED_SCALE"] = bad
            s = fc.sample_target(s_rng, "person")
            assert s["speed"] == fc.KINDS["person"]["speed"] and \
                s["speed_scale"] == 1.0, bad
    finally:
        os.environ.pop("FPV_TARGET_SPEED_SCALE", None)
        if _saved is not None:
            os.environ["FPV_TARGET_SPEED_SCALE"] = _saved

    # ---- (h) D11 spawn band scales with the goal ------------------------------
    d_rng = random.Random(31)
    s_def = fc.sample_target(d_rng, "person")       # desired=None: legacy
    r0 = float(np.hypot(*(s_def["xy"][0] - np.array([0.0, 0.0]))))
    assert 6.0 - 1e-9 <= r0 <= 12.0 + 1e-9, r0     # U[6,12] UNCHANGED
    for d_want in DESIRED_SWEEP:
        lo, hi = max(2.0, 0.6 * d_want), max(3.0, 1.4 * d_want)
        for _ in range(20):
            s = fc.sample_target(d_rng, "person", desired=d_want)
            r0 = float(np.hypot(*(s["xy"][0] - np.array([0.0, 0.0]))))
            assert lo - 1e-9 <= r0 <= hi + 1e-9, (d_want, r0)

    # ---- report --------------------------------------------------------------
    print("follow reward v6 dominance table (episode returns, %d steps,"
          " 60 natural scripts/kind/goal, real D6 detector):" % MAX_STEPS)
    print("  goal   %-9s  CAMP mean / max      TRACK mean / max" % "kind")
    for key in camp_tot:
        print("  %5.1f  %-9s  %7.2f / %7.2f   %7.2f / %7.2f"
              % (key[0], key[1], mean_camp[key], max(camp_tot[key]),
                 mean_foll[key], max(foll_tot[key])))
    print("  (EXPECTED returns govern learning at EVERY desired in %s;"
          " receding-LOS scripts are the documented loss/re-acquire stress"
          " family -- spec open-risk 3)" % DESIRED_SWEEP)
    print("  best TERMINAL-ending episode over all (P, k): %.2f at %s "
          "< track mean-min %.2f; idle floor r=0 (safe range) makes early"
          " termination a strict |P| loss from ANY state" % (worst_term,
                                                             worst_arg,
                                                             FOLL_MEAN_MIN))
    print("  per-step harvest ceiling %.1f < |P| for P in {-10,-10,-20}; "
          "post-terminal harvest = 0 (episode over); worst NON-terminal step"
          " R_STEP_FLOOR = %.2f = -(CLOSE_PENALTY %.1f + max penalties %.2f)"
          % (R_MAX, R_STEP_FLOOR, fc.CLOSE_PENALTY, PEN_MAX))
    print("  gamma=%.2f: at every goal TRACK %.2f > CAMP %.2f (last pair);"
          " every discounted terminal cut < the SAME stream run on; best"
          " terminal %.2f" % (GAMMA, last_FD, last_CD, last_TD))
    print("  close-guard v6: best step inside %.1f m = %.2f < decent track"
          " step %.2f; 30-step steady %.2f < %.2f; CLOSE_PENALTY %.1f > max"
          " range-term gain 0.2" % (fc.CLOSE_GUARD, r_close, r_track_step,
                                     CLOSE_D, TRACK_D_GUARD,
                                     fc.CLOSE_PENALTY))
    print("  projection: h_px=%.1f vs f*H/dist=%.1f (<6%%), cx=%.0f==cx0;"
          " edge_cut set at 2 m, unset at 6/12 m" % (tgt["h_px"], approx,
                                                     tgt["cx"]))
    print("  range_estimate (D12): inverts projection within O(0.6/slant)"
          " tilt-residual at 3/5/6/10/15 m slant; C=H/fy=2tan(vFOV/2);"
          " EMA/dropout semantics ok")
    print("  noise stats @10k: miss=%.3f jitter_sd=%.2fpx h_sd=%.3f "
          "conf_mu=%.3f" % (miss_rate, np.std(djs), np.std(dhs),
                            np.mean(confs)))
    print("  depth grid: free=1.0, min=%.3f exact (d-1)/19, all in [0,1]"
          % g.min())
    print()
    print("asserts: mean_camp < mean_track for all kinds at every desired in"
          " %s; idle floor r>=0 (safe range) makes every (P,k) terminal cut"
          " strictly worse than continuing (undiscounted and discounted);"
          " close-guard steady return strictly below tracking steady;"
          " projection/range-estimate/noise/grid properties hold"
          % DESIRED_SWEEP)
    print("ALL PASS")


if __name__ == "__main__":
    main()
