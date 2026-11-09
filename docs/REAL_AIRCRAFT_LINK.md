## How it works
- The remote is a USB host only toward a phone. A PC plugged in gets a limited serial port that never reaches the aircraft. So a Raspberry Pi Zero 2 W (<pi-address>, key auth) pretends to be the phone using DJI Link's bridge (~/djilink/bridge.py on the Pi, localhost-bound, UDC 3f980000.usb, HIGH speed; full speed fails).
- Cabling that worked: remote -> the adapter chain -> cable to the Pi MIDDLE "USB" port; Pi powered from PWR. A USB hub in the middle can never work. After any bridge restart the pilot must unplug and replug the cable at the Pi.
- Start/restart: `ssh pi@<pi-address> 'sudo ~/rs.sh'` (never pkill -f over ssh: it kills your own shell). Status: `cat /sys/class/udc/*/state` should say configured; ~/djilink/bridge.out shows AOA strings DJI/WM160.
- Tunnel from the lab server: `ssh -f -N -L 9910:127.0.0.1:9910 pi@<pi-address>`.
- Wire format: send `55 CC 49 57 <len u32 LE> <DUML frame>`; the same container comes back. DUML crc8 init 0x77, crc16 init 0x3692 (proven against known frames).

## Use the helper
`python3 tools/mavic_link.py read` (pitch) and `... gimbal -90` / `gimbal 90` (one relative move, clipped to -90..+20, reads the pushed angle to verify). Needs the drone on and linked.

## Facts you must remember
- Aircraft telemetry arrives on its own (flight controller sender 0x03, gimbal 0x04, camera 0x01). Gimbal push is set 04 id 05; int16 at payload offset 0 is pitch in 0.1 degree.
- Gimbal command: sender 0x02 (APP; the PC sender 0x0a is silently ignored), receiver 0x04, type 0x40, set 0x04 id 0x14, payload `<hhh yaw roll pitch> + 00 01`, units 0.1 degree, RELATIVE (delta). ONE command per step: sending it twice doubled the move and overshot to +20 degrees. Value 0 does nothing. Range -90..+20. ACK: type 0x80 from 0x04.
- Read-only get-version (set 0 id 1) got no reply from the aircraft; not needed.
- Always close the loop: read the pushed angle/position before the next command, one change at a time.

## Rules
- Only the gimbal has ever been commanded. Flight commands (virtual stick, takeoff, motors) need the pilot's explicit per-stage OK in chat, props OFF, the pilot at the remote. Not implemented in the helper on purpose.
- Next steps are on cards S1b (decision table), S2 (adapter: +-1 m/s setpoints to stick commands with telemetry feedback), S3 (staged real flight). Find the stick/virtual-stick command in DUML_COMMANDS_FULL.md (DataFlycJoystick) and test it with props off, reading telemetry back.

## How the AI will fly the real aircraft (plan; NOTHING below the line is tested on the aircraft yet)
Pilot (hover_v7 SAC, later waypoint/vision) -> action = velocity setpoint in m/s (NED, +-1) at dt 0.5 s, 15-number observation (position error, velocity, tilt, rate, last action) -> adapter -> DUML virtual-stick command (DataFlycJoystick, set 0x03 id 0x8E: flag byte + float32 roll, pitch, yaw, throttle; UNVERIFIED on the Mini, check DUML_COMMANDS_FULL.md and FLIGHT_GATING.md) -> aircraft -> telemetry back -> next observation.
1. Measure first, props OFF: log telemetry (flight controller pushes, set 0x03: ids 0x43/0x09/0x53/0x67 seen) while the pilot moves the remote sticks, to learn which fields give position/velocity/attitude/battery and their units. Write the decoder in mavic_link.py (read-only) and verify against known motion (the pilot tilts the aircraft by hand).
2. Identify the plant: stick-level -> resulting velocity/attitude response per flight mode (props off can only show attitude/command echo; real velocity needs the pilot's staged flight). Fit a simple model: velocity = gain x stick with first-order lag, plus measured link latency (timestamp command vs telemetry change).
3. Put the REAL numbers into the simulator before any more training: action scaling, command delay, observation noise/latency, the sim's dt 0.5. Retrain with domain randomisation over those ranges (gain +-20%, delay 0-300 ms, wind). A policy trained with perfect instant velocity control will NOT transfer.
4. The pilot never touches the link directly: adapter owns safety. Clip commands, rate-limit, geofence (hover box), dead-man (stop sending = hover/brake; the aircraft's own RTH/failsafe stays armed), kill on telemetry loss or battery low, the pilot's remote sticks override (test that handover with props off FIRST).
5. Order: read-only telemetry -> gimbal (done) -> virtual-stick accepted with props off (read the echo) -> tethered/netted hover indoors with the pilot holding the sticks -> outdoor hover (airspace/LAANC check) -> waypoints. Each stage needs the pilot's explicit OK in chat.
6. Verification: same metric as the sim (mean distance from hover spot, 0.335 m pass mark in sim; real GPS/vision error bars wider), compared with the pilot hand-flying and with zero-action drift. Report failures honestly in the page journal.
What training must learn from the real device: control rate 2 Hz is slow for the real aircraft; if the link/latency demands it, train at the aircraft's native rate or add action smoothing/repeat.

## Official DJI SDK path (the pilot 2026-11-09: fully OK to use it)
Mobile SDK v4 supports the Mavic Mini 1 (since 4.13) incl. virtual stick; known issue: use SOFTWARE video decoding on Mini/Mini SE; no Mini waypoint missions. Needs a free DJI developer account/App Key. Use it as the second link (card S1) next to the Pi bridge: a phone relay app streams video+telemetry to the laptop and applies virtual-stick setpoints; the AI program stays on the laptop. The Pi route is proven for telemetry+gimbal; video and flight commands are unproven on both routes.

S1 status 2026-11-09: DJI app created (package com.example.dronerelay), key kept out of the repo, phone = the pilot's Fold 7.
