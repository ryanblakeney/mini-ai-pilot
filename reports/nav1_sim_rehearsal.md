# NAV1 sim rehearsal report — 2026-11-15 (attempt 9)

**Card:** t_70db6699 (NAV1-R simulator rehearsal of fly-the-plan before any real flight)
**Verdict: the rehearsal DID NOT FLY. Zero waypoint runs, so there are NO per-waypoint
arrival distances to report against REACH_M = 2.5 m. Activation of the plan is blocked
by an app-side SimLink defect found by this rehearsal — that is the honest result, in
the same style as the S6 report.**

Everything upstream of the defect was proven live against the real Unreal sim today;
the defect is the first thing after the operator taps "Yes" on take-control.

## What was actually run (all real, all today, US Central)

Rig: FPVS6Loop-style Blocks sim on the Yoga (06:00 free per patrol), SimLink bridge
`pilot/nav1r/nav1_simlink.py` under task FPVNAV1R, Drone Relay **0.8.0** rehearsal
build on the strix emulator (AVD small_phone), plan `nav1r-2wp` (2 HOVER wps, 20 m N
hover 12 s, +20 m E hover 15 s, 5 m alt, 1.0 m/s, no LANDING).

1. Bridge connected to the LIVE sim (not --no-sim): `BRIDGE_LISTENING mode=sim`,
   kinematics read from Blocks (drone at n=-1.00 e=8.00 alt 7.63 yaw 315), telemetry
   mapped through the Geo.kt-compatible flat-earth transform, `homeSet=true` proven
   in the wire log. Wire log: `nav1_wire_20261115T134118.jsonl` (2.2 MB, 9024 seqs,
   5 Hz tx-T cadence verified).
2. End-to-end wire path proven: Yoga bridge -> ssh tunnel -> emulator app telemetry.
   App HUD went live; the app's own preflight confirmed it independently.
3. Preflight ran 7x: **all PASS every time** — plan has wp, Telemetry live (last 2 s),
   GPS 12 sats (sim constant), home set, battery 100% (sim constant), wp <= 30 m,
   wp <= 40 m of home, wp count OK. Airspace checkbox ticked, hold-to-activate armed
   (`HOLD 1.2 s`) and completed (`map: hold complete - activating plan`).
4. **The defect:** after the `Take control` confirmation, every attempt (n=7, including
   one only 14 s after a fresh `sim connected`) logs
   `link: sim enable: not connected` and refuses activation. No `wp N reached`,
   no `route finished (hover)`, ever emitted. Pause/skip/STOP could not be exercised
   because no route ever started.

## Defect analysis (evidence, not guesswork)

- App side: `SimLink.control("enable")` reaches the send-failure branch, i.e.
  `send('C', ...)` returned false (`out == null` or the socket write threw) — while the
  telemetry half of the very same link was demonstrably delivering fresh frames
  (preflight Telemetry-live PASS, HUD alt readout).
- Bridge side: the wire log contains **zero inbound `C` and zero `V` frames** — the app
  never wrote a single byte up the link that the bridge saw. The bridge was healthy the
  whole time (client=1, T frames streaming, `homeSet` present).
- Consistent with (not yet proven): the Pilot activating the route holds a *stale*
  SimLink object from an earlier `Simulator` tap while the live socket belongs to a
  newer instance — telemetry reaches the UI through the live instance while
  `enableControl()` writes to the dead one. The session pattern supports this: every
  refusal followed a link re-dial (`pc stopAll: switching link`) earlier in the same
  app session. Fixing/demonstrating this belongs to the follow-up card.
- Not a wire-format issue: framing was already proven byte-level
  (type byte + 4-byte BE length) against the app's own reader, and inbound frames of
  that framing are consumed correctly by the app.

## Fixes shipped to the rig today (all deployed, hash-verified)

1. `nav1_simlink.py` NameError scoping bug: `World`/`Drone` were imported inside
   `SimDrone.connect()` but referenced in `_init_world()` -> first real-sim run died
   with `name 'World' is not defined`. Moved the lazy import into `_init_world()`.
   Deployed sha256 `768fa5cf...` (replaced by 2. below).
2. `nav1_simlink.py` telemetry throttle: the 5 Hz app-facing T stream kept the app's
   window permanently non-idle for uiautomator. App-facing sends now throttled to
   `TELEM_SEND_INTERVAL_S = 0.70` (chosen inside the app's 1000 ms telemTimeoutMs);
   internal sampling and the wire log stay 5 Hz (verified: socket gap 0.8 s, wire rows
   0.20 s). Deployed sha256 `40b66c04...`. Smoke-tested live via `--no-sim` on port 19931.
3. Operability notes proven the hard way (saved for the next driver):
   - Once the sim link is connected the emulator window **never** reaches uiautomator
     idle — even at 0.8 s telemetry cadence and with the IME properly dismissed. UI
     automation of the connected app must be screenshot-driven (screenshots + OCR/vision
     worked; that is how the 7 activation attempts were driven).
   - The soft IME window swallows the Simulator button tap: dismiss the IME
     (`input keyevent 111`) BEFORE tapping Simulator, and type the host each session —
     the host field is NOT persisted to prefs.
   - BACK key from the Map page exits the app and drops the link (`pc stopAll: app
     left the screen`); there is no visible Map -> Fly affordance in 0.8.0.
   - Stale 9930 ssh tunnels survive worker restarts: check `ss -ltnp` before assuming
     a fresh tunnel is needed; the app auto-reconnects within seconds of the bridge
     coming back, but NOT after link-switch events.

## Rig state after this run (clean)

- Yoga: FPVNAV1R task ended + deleted, end_fpv_jobs run, Blocks=0, sim released at
  ~08:33 CT. Wire log + status copied off the Yoga.
- strix: emulator + app + plan + driver intact under ~/nav1r; evidence under
  ~/nav1r/v_run*.
- Rehearsal code (`pilot/nav1r/`, this report) committed in mini-ai-pilot; nothing
  pushed to the public repo from this run without a secret scan.

## What must happen before ANY arrival-error measurement exists

Follow-up card (created): make the app's control half of SimLink usable —
`sim enable: sent` must appear, then `control_on=1` on the bridge LIVE line —
with a regression test that catches the stale-link case. Only then can this card's
measurement plan be executed: clean run, pause/resume + skip run, STOP run;
per-waypoint mean arrival distance vs REACH_M = 2.5 m from the wire log; dwell timing.
Props-off ground test and any real flight remain separate stages needing the owner's
explicit OK in chat — nothing in this rehearsal touched a real aircraft.

---

# Attempt 10 — 2026-11-15 (later the same day): THE ROUTE FLEW. Real numbers replace the honest-failure above.

**Verdict: PASS for the clean-run measurements.** After two app fixes (below), run 1 flew the
nav1r-2wp plan end-to-end on the live Blocks sim and the wire log yields real per-waypoint
arrival distances against REACH_M = 2.5 m. Run 2 exposed a sim-side command-application stall
(after run 1 the bridge kept accepting V frames but the Blocks drone stopped moving) — the
pause/resume/skip loop is therefore NOT yet verified in-loop, honestly flagged below.

## The two real defects (both found on this card, both fixed + tested + committed)

1. **The 'sim enable: not connected' blocker (badd9b5 on wt/nav1-r-rehearsal).**
   Reproduced deterministically on a local rig (bridge --no-sim on sable + strix emulator):
   a fresh session with ONE Simulator tap reproduced the refusal; `sim connected` logged on the
   link thread, refusal logged on the MAIN thread (tid==pid), bridge socket alive with zero rx.
   Root cause: `Pilot.engage` runs the confirm-dialog callback on the Android main thread;
   `SimLink.send()`'s raw socket write threw `NetworkOnMainThreadException`, swallowed by
   `catch (Exception)` -> `false` -> "not connected". The stale-instance hypothesis from
   attempt 9 is DISPROVEN (single-instance sessions reproduce it 1/1).
   Fix: `control()` (and disableControl's zero frame) now run on a per-instance single daemon
   executor; send() logs the real write-failure reason; link death logs via onLog. In-loop proof
   the new diagnostic fires: `sim write failed: NetworkOnMainThreadException: null` from the
   remaining main-thread path in Pilot.stop's sendVel(ZERO) (harmless, still caught by design).
2. **Cross-machine clock skew tripped the 1 s safety (65aa052).** First real-sim engage logged
   `sim enable: sent` then `SAFETY: telemetry lost` 1 ms later: bridge CMD_ENABLE at
   bridge-time 15:20:03Z while the app's clock read 15:20:04.746Z — Yoga runs ~1.7 s behind.
   `SimLink.parseTelemetry` used the bridge-declared `t` for freshness; any skew > telemTimeoutMs
   (1000 ms) makes every sample look stale (preflight's 2 s window masked it). Fix: telemetry is
   now stamped at RECEIVE time (local); the bridge's declared clock is never trusted for age.
   Regression test sends a T frame declaring `t = now-5000`; delivered age must be < 1 s.
   Tests: 123 unit tests green (`:app:testDebugUnitTest`), SimLinkTest 4/4 (enable reaches the
   fake bridge as a real C frame; redial writes only to the new socket; foreign-thread calls
   deliver; receive-stamping).

In-loop acceptance from the task: **`link: sim enable: sent` in logcat + `control_on=1` on the
bridge LIVE line — both observed** (local rig 09:01 CT, real sim 09:24:59 CT: `CMD_ENABLE
control_on=1`, `cmd_age_s=0.0` while V frames streamed at ~20 Hz).

## Run 1 (clean) — REAL measurements from the wire log (5 Hz tx-T, 203 ms median gap)

Plan nav1r-2wp as actually seeded: wp0 = home+(19.96 N, 0 E) hover 12 s, wp1 = home+(19.96 N,
19.96 E) hover 15 s, alt 5 m, 1.0 m/s (the seed json put wp0 north of the bridge HOME point
(LAT0/LON0), not north of the spawn point — spawn sits at n=-1.00 e=8.00 so leg 1 flew ~9 m
across plus the settle). App markers: `wp 0 reached` 09:25:21.44, `wp 1 reached` 09:26:01.68,
`route finished (hover)` 09:26:16.72 (CT).

| metric | value | bar |
|---|---|---|
| wp0 arrival distance at reached (wire-nearest sample, 13 ms off) | **1.66 m** | <= 2.5 m PASS |
| wp0 min distance in +-1 s window | 1.11 m | |
| wp1 arrival distance at reached (17 ms off) | **2.46 m** | <= 2.5 m PASS |
| wp1 min distance in +-1 s window | 2.44 m | |
| mean per-waypoint arrival | **2.06 m** | < 2.5 m PASS |
| wp0 dwell + leg-2 timing | reached->reached 40.2 s = 12 s hover + ~20 m leg at <=1 m/s | matches plan |
| wp1 dwell until `route finished (hover)` | **15.0 s** exactly | matches plan hoverSec 15 |
| V commands leg-1 | 419 frames, all nonzero, cadence ~20 Hz (pilot loop) | as designed |
| telemetry cadence at bridge | median 203 ms (5 Hz), app-facing throttled 0.70 s | unchanged |

wp1's 2.46 m sits just inside REACH_M — the last leg ended with ~2 m residual east error at the
reach check; honest to note the arrival tolerance was NOT comfortable on this run.

## Run 2 (pause/resume/skip) — app side exercised, sim side stalled: NOT VERIFIED end-to-end

Pause + Resume taps at 09:27:01/09:27:05 landed on the (pixel-verified) Pause button. The wire
log shows the app streamed 718 V frames in the route-2 window, **656 of them non-zero toward
the wp0 target** — so the pause->resume command path on the APP side worked. But the Blocks
drone did not move at all for 36 s (constant n=19.43 e=17.59): command application in the
bridge->sim stalled after run 1 (V frames were consumed and echoed with converted vn/ve but
never reached the sim's velocity control). Skip-at-dwell therefore never got a live dwell to
skip. Candidate next step for the follow-up: bridge-side check of cmd application after a
completed route (deadman/`cmd_ne` latching in nav1_simlink.py after `disable`-less route end).
`pause`/`resume`/`skip` themselves are code-reviewed sound (RouteBehavior volatile flags; skip
logs `wp N skipped` and advances).

## Run 3 (STOP) — exercised on the live route at 09:30:39.98

While mode=ROUTE, STOP tap -> `pilot: STOP: STOP pressed -> remote sticks have the aircraft`,
`sim disable: sent` (bridge CMD_DISABLE control_on=0 hover(0,0,0) at bridge time 15:30:39Z),
V frames stopped within 4 frames (~200 ms) after disable. Note: this STOP fired during the
stalled run-2 route rather than a fresh run-3 activation (the re-activation path returns
'route already running' while mode=ROUTE) — the app->bridge STOP effect is nonetheless proven
live (control released, commands stop).

## Rig + evidence trail

FPVNAV1R re-registered + fired 09:02 CT (blocks up 8989/8990 in <34 s, BRIDGE_LISTENING mode=sim
15:02:39Z), sable->Yoga 9930 tunnel recreated, app = 0.8.2-nav1rfix2 build (wt/nav1-r-rehearsal
@ 65aa052). Sim released + task deleted + end_fpv_jobs run after the runs (Blocks=0).
Wire log: nav1_wire_20261115T150235.jsonl (14425 rows) and full activation logcat are attached
to card t_a0ef7a74; measurement script pilot/nav1r/measure_arrivals.py in this repo.
UI reality unchanged from attempt 9: uiautomator can never idle while telemetry streams, but
`uiautomator dump` DOES succeed while an AlertDialog is up — dialogs were anchored that way;
custom-drawn map buttons (Pause 2245,645 / Skip 2247,481 / STOP 1263,974 / Check-Hold
2247,806 @ 2400x1080) were located by fill-color segmentation and view-bounds dumps.

## What this unblocks

The fly-the-plan activation + clean 2-waypoint arrival vs REACH_M=2.5 m: PASS with real numbers.
Still open before the props-off stage: bridge V-application stall after a completed route
(run 2), skip-at-dwell live check, and the remaining main-thread sendVel in Pilot.stop (cosmetic
log noise, behaviorally harmless). Props-off ground test and any real flight remain separate
stages needing the owner's explicit OK in chat. Nothing in this rehearsal touched a real aircraft.
