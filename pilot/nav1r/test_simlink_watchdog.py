#!/usr/bin/env python3
"""test_simlink_watchdog.py - pure-python unit tests for the NAV1-STALL watchdog.

No projectairsim, no sim, no sockets: drives StallWatchdog (the decision core
of nav1_simlink.py's velocity_loop watchdog) with a scripted clock and a
scripted motion/stall feed. Exit 0 = all checks PASS.
Run: python3 test_simlink_watchdog.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from nav1_simlink import StallWatchdog
except Exception as ex:  # import-skip guard (stdlib-only import expected)
    print("SKIP: cannot import nav1_simlink: %r" % ex)
    sys.exit(0)

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def drive(wd, t, dur, commanding=True, cmd=0.5, obs=0.0, step=0.2):
    """Run wd.update over [t, t+dur); returns (events, end_time)."""
    evs = []
    end = t + dur
    now = t
    while now < end:
        evs.extend(wd.update(now, commanding, cmd, obs))
        now += step
    return evs, now


CMD, OBS_OK, OBS_STALL = 0.5, 0.5, 0.0

# ---- (iii) no false stall during normal flight ----
wd = StallWatchdog()
evs, _ = drive(wd, 0.0, 10.0, cmd=CMD, obs=OBS_OK)
check(evs == [], "normal flight (cmd 0.5, obs 0.5) over 10 s -> no events")

# ---- (iii) no false stall during commanded-zero dwell (the 12 s rehearsal dwell) ----
wd = StallWatchdog()
evs, _ = drive(wd, 0.0, 30.0, cmd=0.0, obs=0.0)
check(evs == [], "commanded-zero dwell 30 s (cmd 0, obs 0) -> no events")
check(wd.stall_sticky is False, "commanded-zero dwell leaves no sticky flag")

# ---- no false stall when control not commanding (stale command / disabled) ----
wd = StallWatchdog()
evs, _ = drive(wd, 0.0, 10.0, commanding=False, cmd=CMD, obs=OBS_STALL)
check(evs == [], "not commanding (cmd stale / control off) -> no events")

# ---- stall right at the threshold boundary (cmd exactly 0.35 counts) ----
wd = StallWatchdog()
evs, _ = drive(wd, 0.0, 3.0, cmd=StallWatchdog.CMD_MIN, obs=0.11)
check("detected" in evs, "cmd == CMD_MIN, obs <= OBS_STALL -> still detected")

# ---- (i) stall flags after exactly the prescribed window ----
wd = StallWatchdog()
evs, _ = drive(wd, 0.0, 2.4, cmd=CMD, obs=OBS_STALL)
check(evs == [], "no stall before the 2.5 s window elapses (2.4 s)")
evs, _ = drive(wd, 2.4, 0.4, cmd=CMD, obs=OBS_STALL)
check("detected" in evs, "stall detected within the window (t=2.4..2.8 s)")
check(evs[0] == "detected", "'detected' emitted before any ladder step")
check(evs[1] == "step_hover", "ladder starts with hover")
check(wd.last_detect is not None and abs(wd.last_detect["cmd"] - CMD) < 1e-9,
      "last_detect carries the cmd norm for STALL_DETECTED logging")

# ---- (ii) recovery ladder order hover -> enable_api, each once (no thrash) ----
# NOTE detection fires on the first tick with (now - stall_since) >= 2.5, i.e.
# the tick at t=2.6; verify steps fire on ticks 2.0 s after each ladder step.
wd = StallWatchdog()
evs, t = drive(wd, 0.0, 2.8, cmd=CMD, obs=OBS_STALL)   # ticks through 2.6
check("step_hover" in evs, "step_hover issued at detection")
evs, t = drive(wd, t, 1.8, cmd=CMD, obs=OBS_STALL)     # ticks through 4.4 (< 2 s verify)
check(evs == [], "hover step gets its full 2 s re-issue window (no second step early)")
evs, t = drive(wd, t, 0.4, cmd=CMD, obs=OBS_STALL)     # tick at 4.6 = hover verify + 2.0
check(evs == ["step_enable"], "step_enable issued right after the hover verify window")
evs, t = drive(wd, t, 1.6, cmd=CMD, obs=OBS_STALL)     # ticks through 6.4 (< 2 s verify)
check(evs == [], "enable step gets its full 2 s verify window")
evs, t = drive(wd, t, 0.4, cmd=CMD, obs=OBS_STALL)     # tick at 6.6/6.8 = enable verify + 2.0
check(evs == ["unrecovered"], "unrecovered after both ladder steps failed")
check(wd.stall_sticky is True, "sticky stall flag set after unrecovered")
evs, t = drive(wd, t, 20.0, cmd=CMD, obs=OBS_STALL)
check(evs == [], "no ladder thrash while sticky (20 s more of stall, < 30 s re-arm)")

# ---- re-arm after 30 s, then re-detect + re-run the ladder ----
evs, t = drive(wd, t, 10.2, cmd=CMD, obs=OBS_STALL)    # crosses failed_at+30
check("rearmed" in evs, "watchdog re-arms 30 s after unrecovered")
evs, t = drive(wd, t, 3.0, cmd=CMD, obs=OBS_STALL)
check("detected" in evs and "step_hover" in evs, "re-detection after re-arm runs the ladder again")

# ---- sticky flag clears when motion returns on its own ----
wd2 = StallWatchdog()
drive(wd2, 0.0, 2.8, cmd=CMD, obs=OBS_STALL)
drive(wd2, 2.8, 4.2, cmd=CMD, obs=OBS_STALL)           # run through the full ladder
evs_f, t = drive(wd2, 8.0, 5.0, cmd=CMD, obs=OBS_STALL)
check("unrecovered" in evs_f or wd2.stall_sticky, "fixture: wd2 stuck (failed/sticky)")
evs = wd2.update(t, True, CMD, 0.6)
check(evs == ["recovered:auto"], "motion returning while stuck -> recovered:auto + reset")
check(wd2.stall_sticky is False, "sticky flag cleared when motion observed")

# ---- recovery via hover (motion returns during the hover verify window) ----
wd = StallWatchdog()
evs, t = drive(wd, 0.0, 2.8, cmd=CMD, obs=OBS_STALL)
check(evs == ["detected", "step_hover"], "hover fixture armed")
evs, t = drive(wd, t, 1.0, cmd=CMD, obs=OBS_STALL)     # still stalled 1 s
evs, _ = drive(wd, t, 0.6, cmd=CMD, obs=OBS_OK)        # motion returns
check("recovered:hover" in evs, "motion during hover verify -> STALL_RECOVERED via=hover")
check(wd.phase == "idle" and wd.stall_sticky is False, "watchdog idle after recovery")

# ---- recovery via enable_api (hover failed, motion after enable step) ----
wd = StallWatchdog()
evs, t = drive(wd, 0.0, 2.8, cmd=CMD, obs=OBS_STALL)
evs, t = drive(wd, t, 2.0, cmd=CMD, obs=OBS_STALL)     # hover verify expires -> step_enable
check("step_enable" in evs, "enable fixture: step_enable issued")
evs, _ = drive(wd, t, 0.6, cmd=CMD, obs=OBS_OK)
check("recovered:enable_api" in evs, "motion after enable step -> via=enable_api")

# ---- brief stall blips below the window never flag ----
wd = StallWatchdog()
t = 0.0
for _ in range(10):  # ten 2.2 s stall blips separated by 1 s of motion
    evs, t = drive(wd, t, 2.2, cmd=CMD, obs=OBS_STALL)
    if evs:
        break
    evs, t = drive(wd, t, 1.0, cmd=CMD, obs=OBS_OK)
    if evs:
        break
else:
    evs = []
check(evs == [], "2.2 s stall blips (with motion between) never flag")

print("%d checks failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
