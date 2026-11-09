#!/usr/bin/env python3
"""Task D: offline arithmetic verification for the stage-2 waypoint reward v3.

NO sim, NO third-party deps -- runs under plain /usr/bin/python3. It does NOT
import waypoint_env (that would pull projectairsim); instead it regex-extracts
the pure wp_reward() source from waypoint_env.py and exec's just that.

Asserts the card's hard rule (survival reward + terminal penalty bigger than
any early-exit gain) as episode-return dominance:
  ARRIVAL return > CAMP return, > HOVER-IN-PLACE return, and FLY-AWAY return
  is dominated by everything.

Run:  /usr/bin/python3 check_waypoint_reward.py
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "waypoint_env.py")

# constants mirrored from the LOCKED design (waypoint_env.py header)
ARRIVE_R = 0.75
FAIL_R = 12.0
D_MIN, D_MAX = 3.0, 10.0
MAX_STEPS = 100
STEP_DT = 0.5


def load_wp_reward():
    src = open(SRC, encoding="utf-8").read()
    m = re.search(r"(?ms)^def wp_reward\(.*?(?=^def |^class |\Z)", src)
    if not m:
        sys.exit("FAIL: could not regex-extract wp_reward from " + SRC)
    ns = {}
    exec(m.group(0), ns)
    return ns["wp_reward"]


def main():
    wp_reward = load_wp_reward()

    # sanity: terminal values exactly as designed
    assert wp_reward(0.5, 0.0, 0.0, True, False) == 60.0
    assert wp_reward(13.0, 0.0, 0.0, False, True) == -10.0
    # sanity: per-step reward monotone decreasing in dist, positive near goal
    assert wp_reward(1.0, 0.0, 0.0, False, False) > wp_reward(5.0, 0.0, 0.0, False, False) \
        > wp_reward(12.0, 0.0, 0.0, False, False) > 0.0

    # ---- (1) CAMP bound -----------------------------------------------------
    # Best per-step while NOT arriving: dist -> just above ARRIVE_R, zero
    # penalties (penalties only subtract). dist=0.76 is the task-quoted bound;
    # dist -> 0.75+ is the true supremum. Use the sup for rigor.
    camp_step = wp_reward(ARRIVE_R + 1e-9, 0.0, 0.0, False, False)
    camp_step_task = wp_reward(0.76, 0.0, 0.0, False, False)
    CAMP_TOTAL = MAX_STEPS * camp_step
    assert abs(camp_step_task - 0.4696) < 1e-9

    # ---- minimum ARRIVAL return ---------------------------------------------
    # Worst case: d0 = 10 m at a conservative 1 m/s average speed
    # (max cmd is 1 m/s/axis => real speeds are higher) => >= 20 steps of
    # approach at the worst possible per-step floor: dist capped at 10 in the
    # shaping, ||a|| <= sqrt(3), ||v|| <= 2 m/s, then +60 for the arrival.
    min_steps = math.ceil(D_MAX / (1.0 * STEP_DT))          # >= 20
    r_min_step = wp_reward(10.0, math.sqrt(3.0), 2.0, False, False)
    ARRIVAL_MIN = min_steps * r_min_step + 60.0

    # ---- max zero-action-at-start return -------------------------------------
    # Zero velocity holds the drone at the start point; dist stays d0 >= 3,
    # so per-step ceiling = 0.1 + 0.4*(1 - 3/10) = 0.38 (no penalties at 0 cmd,
    # and 0.02*||v|| only subtract under gusts).
    ZERO_TOTAL = MAX_STEPS * wp_reward(D_MIN, 0.0, 0.0, False, False)

    assert ARRIVAL_MIN > CAMP_TOTAL, (ARRIVAL_MIN, CAMP_TOTAL)
    assert ARRIVAL_MIN > ZERO_TOTAL, (ARRIVAL_MIN, ZERO_TOTAL)

    # ---- (2) FLY-AWAY dominated ----------------------------------------------
    # From d0 = 3 m fly straight away: separation grows >= STEP_DT * 1 m/s
    # = 0.5 m per step (conservative), so crossing dist > 12 takes >= 18 steps
    # and fails on step 19. Ceiling sum of the shrinking per-step reward on
    # the way out (penalties omitted) plus the -10 terminal:
    worst = 0.0
    d, t = D_MIN, 0
    while True:
        t += 1
        d += STEP_DT * 1.0
        if d > FAIL_R:
            worst += wp_reward(d, 0.0, 0.0, False, True)
            break
        worst += 0.1 + 0.4 * max(0.0, 1.0 - d / 10.0)
    assert worst < 0.0 and worst < ZERO_TOTAL and worst < CAMP_TOTAL
    # task-quoted rough: 9 steps x ~0.08 - 10 = -9.28 (same conclusion)

    # ---- (3) dominance table ---------------------------------------------------
    rows = [
        ("ARRIVE (worst case)", ">= %.2f" % ARRIVAL_MIN,
         "%d steps @ %.4f + 60" % (min_steps, r_min_step)),
        ("CAMP just outside r=%.2f m" % ARRIVE_R, "<= %.2f" % CAMP_TOTAL,
         "%d x %.4f (task-quoted 0.4696 @ dist=0.76)" % (MAX_STEPS, camp_step)),
        ("HOVER-IN-PLACE at start", "<= %.2f" % ZERO_TOTAL,
         "%d x %.4f (d0>=%.1f ceiling)" % (MAX_STEPS, wp_reward(D_MIN, 0, 0, False, False), D_MIN)),
        ("FLY-AWAY (best case)", "<= %.2f" % worst,
         "ceil-sum %d steps then -10" % t),
    ]
    print("waypoint reward v3 dominance table (episode returns):")
    print("%-28s %-12s  %s" % ("behavior", "bound", "derivation"))
    for r in rows:
        print("%-28s %-12s  %s" % r)
    print()
    print("asserts: ARRIVAL_MIN=%.4f > CAMP_TOTAL=%.4f > 0, ZERO_TOTAL=%.2f,"
          " FLYAWAY_MAX=%.4f < 0  => all dominance checks hold"
          % (ARRIVAL_MIN, CAMP_TOTAL, ZERO_TOTAL, worst))
    print("ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
