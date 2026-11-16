# NAV1-R sim rehearsal rig — locked design (orchestrator decisions)

Gate for NAV1-R card t_70db6699: Drone Relay app (Simulator link) flies a 2-waypoint plan
against the Unreal Blocks sim over the SimLink wire protocol. The S8 bridge was never built
(S8 card t_16357295 still blocked), so this rig builds it.

## Topology
```
[small_phone emulator on strix <emulator-host>]   (adb via ~/android-sdk on strix)
   Drone Relay 0.8.0 + home-field SimLink patch  → TCP to <lab-server>:9930
[sable-lxc <lab-server>]
   ssh -f -N -g -L 9930:localhost:9930 <laptop>   (public listener → tunnel)
[Yoga <laptop> (tailscale), LAN-Wi-Fi, console unlocked to 2026-11-12 07:30 CT]
   interactive one-shot task FPVNAV1R → Blocks.exe (ports 8989/8990) + nav1_simlink.py
   bridge binds 127.0.0.1:9930 (access only via the ssh tunnel)
```

## Wire protocol (must match SimLink.kt exactly)
Framing both directions (per SimLink.kt send()/read loop): [type:1 byte ASCII]['T'|'F'|'V'|'C'][length:4 bytes BIG-endian][payload bytes]
- bridge→phone `T`: telemetry JSON:
  {"t":epoch_ms,"pitch":deg,"roll":deg,"yaw":deg,"lat":deg,"lon":deg,"alt":mAGL,
   "vn":m/s,"ve":m/s,"vd":m/s,"batt":100,"gimbal":0.0,"flying":bool,"sats":12,
   "homeSet":true,"homeLat":30.395,"homeLon":-85.670}
  (homeSet/homeLat/homeLon are NEW optional fields the patched app reads; optDouble/optBoolean defaults keep old builds safe.)
- bridge→phone `F`: JPEG frame bytes — NOT sent in this rehearsal (documented limitation).
- phone→bridge `V`: {"fwd":m/s,"right":m/s,"down":+down m/s,"yaw":deg/s} BODY frame,
  sent at 20 Hz while the route runs (Pilot loop 50 ms). SafetyLayer already clipped ±1/±0.5.
- phone→bridge `C`: {"cmd":"enable|disable|takeoff|land|home|cancel_home"}

## Bridge `nav1_simlink.py` behaviour (on Yoga)
- Python: use D:\fpv\venv312\Scripts\python.exe; import projectairsim lazily (only when --no-sim not set).
- Reuse the PROVEN client pattern from D:\fpv\rl\gym_env.py (also mirrored at
  ~/projects/mini-ai-pilot/sim/gym_env.py on sable):
  `from projectairsim import ProjectAirSimClient, Drone, World`, ONE persistent asyncio loop
  thread, all client calls via asyncio.run_coroutine_threadsafe(...).result(timeout),
  `_chain(coro) = await (await coro)`, World() needs CWD/WorkingDirectory = ...example_user_scripts,
  kinematics via drone.get_ground_truth_kinematics() (try/except fallback get_kinematics),
  client.disconnect() before exit + os._exit(0) after flush. Kill sim via PID, never land-on-teardown.
- TCP server on 127.0.0.1:9930, accepts ONE app client at a time (extra conns refused/closed after log).
  A second connection replaces the first ONLY after the rehearsal restart; simplest: refuse with log.
- Telemetry push at 5 Hz to the connected app (freshness gate in the app preflight is 2 s;
  SafetyLayer telem timeout is 1 s).
- Coordinate mapping (flat earth — MUST match Geo.kt exactly):
  lat0=30.395, lon0=-85.670 (fictional Blocks spot).
  app math:   n=(lat-lat0)*111320 ; e=(lon-lon0)*111320*cos(radians(lat0))
  bridge inv: lat=lat0+n/111320 ; lon=lon0+e/(111320*cos(radians(lat0)))
  sim world is NED with spawn at (0,0,~0): n=x, e=y, alt=-z. home=(0,0) ↔ lat0,lon0.
- Command apply semantics: hold last 'V' for the sim: on every 'V' set (vn,ve,vd)=bodyToNe(fwd,right,yaw)
  and re-issue the velocity hold every 0.2 s while command age < 1.5 s (deadman → hover via zero velocity).
  bodyToNe: vn=fwd*cos(yaw) - right*sin(yaw); ve=fwd*sin(yaw) + right*cos(yaw); vd=down.
- 'C' semantics: enable → controlOn=true, then if not flying: takeoffAsync → hover 5 m, log AUTO_TOOK_OFF.
  disable → controlOn=false + hover(0,0,0). takeoff/land → takeoffAsync/landAsync with log.
  home → move to (0,0) at current alt (RTL sim) + log. cancel_home → stop RTL.
  All 'C' and 'V' messages logged to the wire log regardless of controlOn.
- Wire log JSONL D:\fpv\ops\nav1_wire_<UTCYYYYMMDDTHHMMSS>.jsonl, one line per message:
  {"seq":n,"t":epoch_ms,"dir":"rx|tx","k":"V|C|T",...payload fields}. Flush every line.
- Status file D:\fpv\ops\nav1_r_status.txt, last line sentinel: BRIDGE_LISTENING →
  APP_CONNECTED → AUTO_TOOK_OFF → (live lines: last-cmd age etc.) → NAV1R_DONE / NAV1R_SIM_LOST / NAV1R_ABORT.
- Attitude: quaternion → euler (projectairsim kinematics pose.orientation). Match the euler
  convention gym_env.py uses for its obs (yaw deg, 0=north, CW+) so telemetry yaw is degrees clockwise from north.
  pitch/roll just for display — be honest about sign convention in a comment.
- velocity vn/ve/vd: prefer drone.get_ground_truth_velocity() if the installed client has it
  (CHECK the venv's projectairsim API); else finite-difference kinematics at the telemetry rate and note it.
- Battery: constant 100 (sim has none) — note in log line at connect.
- --no-sim dry mode (no projectairsim needed): synthetic drone orbiting the origin at 5 m,
  applies 'V' by integrating velocity at 10 Hz, still writes wire log + status sentinels. Used to validate wiring from sable.

## Wrapper + task (mirror the proven S6 pair; keep CRLF line endings!)
Local mirror to base on: ~/projects/mini-ai-pilot/pilot/loopback/s6_loop.ps1 + register_s6.ps1
(already on D:\fpv\ops\ too). Same shape, renamed for NAV1R:
- nav1_r_loop.ps1: FIRST line after param block:
  `if (& 'D:\fpv\ops\fpv_busy_guard.ps1 -Self $PSCommandPath') { exit 0 }` (exact `if (& ...)` form;
  the guard script already exists at D:\fpv\ops\fpv_busy_guard.ps1). Then: start Blocks
  (same exe/flags/WorkingDirectory as s6_loop.ps1), poll ports 8989/8990 up to 300 s
  (sentinel PORT_TIMEOUT_8989_300S on timeout), run nav1_simlink.py with venv312 python
  (Start-Process -PassThru -NoNewWindow -RedirectStandardOutput/-RedirectStandardError),
  wait until bridge exits (it runs the whole rehearsal; the ops side ends it with end_fpv_jobs),
  finally: kill ONLY own Blocks PID via `taskkill /PID <id> /T /F`, write NAV1R_DONE.
- nav1_r_register.ps1: registers interactive one-shot FPVNAV1R (InteractiveToken, same as
  register_s6.ps1 incl. -WorkingDirectory example_user_scripts pitfall), StartAt default ~1 min after register.
- After the rehearsal end with: schtasks /end FPVNAV1R + D:\fpv\ops\end_fpv_jobs.ps1 -Task FPVNAV1R.

## Safety rules for this rig (hard)
- NOTHING here touches the real aircraft. Blocks-only. Props-off rule is untouched.
- Do NOT fire the task unless `Get-Process Blocks*` is empty AND no FPV* task Running (RIG-H gate).
- Console override window ends 2026-11-12 07:30 CT; do not start sim jobs after 06:30 CT that day.

## App-side change (separate task, minimal + disclosed)
Branch wt/nav1-r-rehearsal from wt/nav1-map-page @ 8c57939 in ~/projects/dji-relay (worktree dir
~/projects/dji-relay/.worktrees/t_70db6699-nav1r). One commit: SimLink.parseTelemetry also reads
j.optBoolean("homeSet") / optDouble("homeLat") / optDouble("homeLon") into Telemetry (fields exist
already in FlightLink.kt:15). This unblocks the preflight home-set item in sim (preflight uses
t.homeSet at MainActivity.kt:186); DjiLink path untouched. Build with ./build.sh :app:assembleDebug.
Deliver APK to strix: scp to <emulator-host>:~/nav1r/app-debug.apk (install done by orchestrator).

## Rehearsal script (orchestrator drives via adb on strix)
1. Fire FPVNAV1R on Yoga; confirm NAV1R_READY-ish sentinels + `ssh -g -L` tunnel up on sable.
2. Boot strix AVD small_phone (-no-window -gpu swiftshader_indirect), install APK, launch <app-package>.
3. UI via `adb shell input` + uiautomator dumps: link = Simulator, host = <lab-server> → Connect;
   open Map page; load pushed plan file; Check → tick airspace → Fly plan (hold) ≥1.2 s.
   Capture `adb logcat -s DroneRelay:D` (all Diag lines incl. "wp N reached"/"route finished") to a file.
4. Plan file (pushed to filesDir/plans/ via adb, PlanStore Open reads it):
   {"name":"nav1r-2wp","waypoints":[
     {"lat":30.3951797,"lon":-85.67, "heightM":5.0,"speedMps":1.0,"hoverSec":12,"type":"HOVER"},   # 20 m north
     {"lat":30.3951797,"lon":-85.6697919,"heightM":5.0,"speedMps":1.0,"hoverSec":15,"type":"HOVER"} # +20 m east
   ]}
   (e=20m → lon delta = 20/(111320*cos(radians(30.395))) ; n=20m → lat delta 20/111320. No LANDING type.)
5. Runs: (1) clean full route; (2) PAUSE mid leg1→Resume + SKIP at leg2; (3) STOP mid-route.
   Between runs: STOP gives control back; drone hovers; re-engage re-homes at current pos →
   use 'C' home (map page RTH button or rehome via bridge) or re-check plan distances stay inside fence 40 m.
6. Measure from bridge wire log + logcat (analysis script, delegated): per-WP closest-approach distance,
   first entry into 2.5 m + ±1 m alt band, time-in-radius, dwell displacement (should hover <~1 m),
   command rate, deadman behaviour, STOP→last-command latency. Compare app 'wp reached' vs bridge truth.

## Honesty requirements (S6-report style)
Report must state: synthetic lat/lon (map is fictional spot, no OSM truth), no camera frames sent,
battery/sats are constants, velocity source (velocity API vs differencing), sim step realism limits,
the app is 8c57939 + exactly one disclosed home-field commit, dwell timing measured via bridge clock,
and what the rehearsal does NOT prove (real GPS error, wind, latency, geofence trip, RTH on battery).
