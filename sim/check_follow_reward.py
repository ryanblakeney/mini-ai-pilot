#!/usr/bin/env python3
"""Offline proof for the S7 follow reward v5 (D7) -- the card's HARD RULE.

NO sim, NO third-party deps beyond numpy -- runs under plain
/usr/bin/python3 (follow_common.py is pure, so a direct import is safe; the
Q3 checkers had to regex-extract because their reward lived in a sim-bound
module).

Card hard rule, restated for S7: survival-shaped POSITIVE per-step reward
(ceiling 1.1) with NO positive terminal anywhere, and bad-only terminals
(landed -10, >12 m from start -10, collision -20) whose penalty magnitude
exceeds any harvest obtainable from ending early. Every Q3 hover stage died
of violating exactly this; this checker PROVES the arithmetic numerically:

  (a) camping/hover-at-start full-episode total < decent-tracking total,
      evaluated by SIMULATING both behaviours through the real target
      scripts, projection and DetectorNoise (not hand-waved constants);
  (b) for every terminal penalty P and every early-exit step k, the best
      terminal-ending episode total is strictly below BOTH no-terminal
      alternatives (camping and tracking), so terminating early is never a
      harvest -- plus the literal per-step bound |P| > per-step max 1.1;
  (c) the ordering survives gamma = 0.98 discounting;
  (d) projection sanity (D4): target straight ahead at 6 m, 1.7 m tall ->
      h_px within 6% of f*H/dist and cx == cx0;
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


def dsum(k, per_step):
    """Sum_{t=1..k} gamma^t * per_step."""
    return per_step * sum(GAMMA ** t for t in range(1, k + 1))


def rollout(kind, rng, mode):
    """One natural-script episode through the REAL primitives.

    mode='camp'  : hover-at-start (zero action, zero yaw).
    mode='chase' : decent follow -- fly toward the target at <=1 m/s body
                   speed, yaw servo toward it (0.5 rad/s cap), vd=0; when the
                   target kinematically outruns the drone (car/bike net
                   speed) the policy honestly loses range/view like any
                   real policy would.
    Detector: default D6 noise (p_miss=0.15, jitter, age-out K_VIEW).
    Returns (total, per-step income list).
    """
    script = fc.sample_target(rng, kind, drone_start=DRONE0, drone_yaw=0.0)
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
            vn, ve = np.clip(dxy, -1.0, 1.0)
            # speed cap ~1 m/s along track (diagonal strafe would exceed it)
            s = math.hypot(vn, ve)
            if s > 1.0:
                vn, ve = vn / s, ve / s
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
                             float(np.linalg.norm(a[:2])) / STEP_DT * STEP_DT)
        a_prev = a
        total += r
        cum.append(total)
    return total, np.array(cum)


def behav_sims(kind, rng, n=60):
    """Paired natural-distribution totals: camp vs decent-follow chase.
    camp_cum = MEAN camping income profile (what the camping behaviour earns
    by step k on average -- the (b)/(c) profiles must mirror the SAME
    expectation the policy gradient sees, not one lucky script)."""
    camp, foll = [], []
    camp_cum = np.zeros(MAX_STEPS + 1)
    for _ in range(n):
        _t, cum = rollout(kind, rng, "camp")
        camp.append(_t)
        camp_cum = camp_cum + cum
        _t, _ = rollout(kind, rng, "chase")
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
             collided=True, landed=True) == -20.0      # collision outranks
    assert f(0.0, 0.0, 6.0, True, 0.0, 0.0, 0.0) <= 1.1 + 1e-12
    assert abs(f(0.0, 0.0, 6.0, False, 0.0, 0.0, 0.0) - 0.6) < 1e-12  # no in_view
    assert f(2.0, 2.0, 100.0, False, 0.0, 0.0, 0.0) == 0.0  # true floor, no penalties
    # perfect-tracking step is the ceiling:
    assert abs(f(0.0, 0.0, 6.0, True, 0.0, 0.0, 0.0) - 1.1) < 1e-12

    # ---- (a) camping vs decent tracking, SIMULATED -------------------------
    # Natural target-script distribution (D2), real D6 detector model, both
    # behaviours rolled through the REAL primitives. Dominance claim asserted
    # per kind on the EXPECTED full-episode return -- this is exactly what a
    # policy gradient sees. Honest caveat (spec open-risk 3, D7 locked -- NOT
    # redesigned): on the rare script where the target recedes nearly along
    # the line of sight, hover collects in_view+center (~0.9/step) for free,
    # so STRICT per-script dominance cannot hold for that script family; the
    # spec itself classifies those scripts as loss/re-acquire stress cases
    # and mandates per-visible-step scoring at eval time (rl_eval_follow.py).
    camp_tot, foll_tot, camp_cum = {}, {}, None
    for kind in ("person", "bicycle", "car"):
        c, fl, cum = behav_sims(kind, rng)
        camp_tot[kind], foll_tot[kind] = c, fl
        camp_cum = cum if camp_cum is None else np.maximum(camp_cum, cum)
    mean_camp = {k: float(np.mean(v)) for k, v in camp_tot.items()}
    mean_foll = {k: float(np.mean(v)) for k, v in foll_tot.items()}
    CAMP_MAX = max(max(v) for v in camp_tot.values())
    CAMP_MEAN_MIN = min(mean_camp.values())
    FOLL_MAX = max(max(v) for v in foll_tot.values())
    FOLL_MEAN_MIN = min(mean_foll.values())
    r_track_min = FOLL_MEAN_MIN / MAX_STEPS   # conservative track step rate
    for kind in camp_tot:
        assert mean_camp[kind] < mean_foll[kind], (kind, mean_camp[kind],
                                                   mean_foll[kind])

    # ---- (b) terminal dominance --------------------------------------------
    # Literal per-step bound (task spec): every terminal penalty magnitude
    # exceeds the per-step harvest max 1.1 (0.5+0.4+0.2; penalties subtract).
    for P in (fc.TERM_LANDED, fc.TERM_FLEW_OUT, fc.TERM_COLLISION):
        assert abs(P) > R_MAX, P
    # Multi-step form: a zero-action idle step has reward >= 0 ALWAYS (all
    # shaping terms >= 0, all penalties exactly 0 at a=0,a_prev=0,v=0) -- so
    # from ANY prefix state the agent can guarantee future income >= 0 by
    # idling. Taking a bad-only terminal at step k instead of continuing is
    # therefore strictly worse for ANY behaviour prefix income rho(k):
    #     rho(k) + P  <  rho(k) + 0*(MAX_STEPS - k)      for every P < 0
    # verified over both behaviour income profiles at every k:
    zero_floor = f(9.0, 9.0, 0.5, False, 0.0, 0.0, 0.0)   # idle worst case
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

    # ---- (c) gamma = 0.98 discounting preserves the ordering ---------------
    FOLLOW_D = dsum(MAX_STEPS, r_track_min)
    camp_inc = [max(0.0, camp_cum[t] - camp_cum[t - 1])
                for t in range(1, MAX_STEPS + 1)]
    CAMP_D = sum(GAMMA ** t * camp_inc[t - 1] for t in range(1, MAX_STEPS + 1))
    term_d = -1e18
    for P in (fc.TERM_LANDED, fc.TERM_FLEW_OUT, fc.TERM_COLLISION):
        for k in range(0, MAX_STEPS):
            camp_d = sum(GAMMA ** t * camp_inc[t - 1]
                         for t in range(1, k + 1))
            track_d = dsum(k, r_track_min)
            prof_d = max(camp_d, track_d)
            term_d = max(term_d, prof_d + GAMMA ** (k + 1) * P)
            # discounted: a terminal cut is worse than the SAME stream run on
            assert prof_d + GAMMA ** (k + 1) * P < (
                CAMP_D if prof_d == camp_d else FOLLOW_D)
    assert FOLLOW_D > CAMP_D, (FOLLOW_D, CAMP_D)
    assert FOLLOW_D > term_d, (FOLLOW_D, term_d)

    # ---- (d) projection sanity (D4) ----------------------------------------
    cam = fc.default_cam()
    # target BASE on the ground (z=-2.7), drone at settled -4.7: the honest
    # S7 geometry (targets are ground-level per D2). Top lands well in-frame.
    tgt = fc.project_target([0, 0, -4.7], 0.0, [6.0, 0.0, -2.7], 1.7)
    assert tgt["in_fov"], tgt
    assert abs(tgt["cx"] - cam["cx0"]) < 1e-6
    approx = cam["fx"] * 1.7 / tgt["dist"]
    assert abs(tgt["h_px"] / approx - 1.0) < 0.06, (tgt["h_px"], approx)
    far = fc.project_target([0, 0, -4.7], 0.0, [12.0, 0.0, -2.7], 1.7)
    assert 0.0 < far["h_px"] < tgt["h_px"] < cam["H"] + 1.0
    behind = fc.project_target([0, 0, -4.7], 0.0, [-5.0, 0.0, -2.7], 1.7)
    assert not behind["in_fov"] and behind["h_px"] == 0.0
    over = fc.project_target([0, 0, -4.7], 0.0, [1.0, 0.0, 2.0], 1.7)
    assert not over["in_fov"]                    # above the frame top edge
    # h_star at dist_star matches projection scale
    assert abs(fc.h_star_for(1.7, 6.0) - cam["fx"] * 1.7 / 6.0) < 1e-9

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

    # ---- report --------------------------------------------------------------
    print("follow reward v5 dominance table (episode returns, %d steps,"
          " 60 natural scripts/kind, real D6 detector):" % MAX_STEPS)
    print("  %-9s  CAMP mean / max      TRACK mean / max" % "kind")
    for k in camp_tot:
        print("  %-9s  %7.2f / %7.2f   %7.2f / %7.2f"
              % (k, mean_camp[k], max(camp_tot[k]),
                 mean_foll[k], max(foll_tot[k])))
    print("  (EXPECTED returns govern learning; receding-LOS scripts are the"
          " documented loss/re-acquire stress family -- spec open-risk 3)")
    print("  best TERMINAL-ending episode over all (P, k): %.2f at %s "
          "< track mean-min %.2f; idle floor r=0 makes early termination a"
          " strict |P| loss from ANY state" % (worst_term, worst_arg,
                                               FOLL_MEAN_MIN))
    print("  per-step harvest ceiling %.1f < |P| for P in {-10,-10,-20}; "
          "post-terminal harvest = 0 (episode over)" % R_MAX)
    print("  gamma=%.2f: TRACK %.2f > CAMP %.2f; every discounted terminal"
          " cut < the SAME stream run on; best terminal %.2f"
          % (GAMMA, FOLLOW_D, CAMP_D, term_d))
    print("  projection: h_px=%.1f vs f*H/dist=%.1f (<6%%), cx=%.0f==cx0"
          % (tgt["h_px"], approx, tgt["cx"]))
    print("  noise stats @10k: miss=%.3f jitter_sd=%.2fpx h_sd=%.3f "
          "conf_mu=%.3f" % (miss_rate, np.std(djs), np.std(dhs),
                            np.mean(confs)))
    print("  depth grid: free=1.0, min=%.3f exact (d-1)/19, all in [0,1]"
          % g.min())
    print()
    print("asserts: mean_camp[k] < mean_track[k] for all kinds; idle floor"
          " r>=0 makes every (P,k) terminal cut strictly worse than"
          " continuing (undiscounted and discounted); projection/noise/grid"
          " properties hold")
    print("ALL PASS")


if __name__ == "__main__":
    main()
