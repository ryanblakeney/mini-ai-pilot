# pilot-phone — SAC hover policy loopback runner + SafetyGate (pure Kotlin JVM)

Self-contained Gradle project (no Android SDK). Runs the exported SAC actor
(`pilot/export/sac_actor_hover_v7.onnx`, fp32, input `obs` `[1,15]` float32,
output `action` `[1,3]` float32 in [-1,1], deterministic `tanh(mu)` — never
sample) behind a pure-Kotlin safety gate, and serves it to the sim bridge over
a TCP JSON-lines protocol.

Obs layout (float32, NED, z DOWN), length 15:
`[0..2]` pos error vs hover target (m) · `[3..5]` linear vel (m/s) ·
`[6..8]` roll,pitch,yaw (rad) · `[9..11]` angular vel (rad/s) ·
`[12..14]` last action (velocity setpoints m/s).
Action = `[v_north, v_east, v_down]` m/s, clipped to [-1,1], held 0.5 s/step.
The environment feeds back the **safety-filtered** action as the next
`prev_action`.

## Build & test

All gradle commands must stay memory-capped on sable:

```bash
cd ~/projects/mini-ai-pilot/pilot/phone
systemd-run --user --scope --quiet --collect -p MemoryHigh=4G -p MemoryMax=5G \
  -p CPUWeight=50 nice -n 10 gradle --no-daemon test
```

Verified: **37 tests, 0 failures** (`SafetyGateTest`). Toolchain: Gradle 8.5,
Kotlin JVM plugin 1.9.22, JVM 17, JUnit 5 (`useJUnitPlatform()`).
Repos: mavenCentral + google.

## Run the loopback runner

The runner is a **TCP server**; the sim bridge on the Windows laptop dials IN
across tailscale. Single connected client; after it disconnects the runner
waits for the next one.

```bash
# from pilot/phone (so the default --model ../export/... path resolves):
systemd-run --user --scope --quiet --collect -p MemoryHigh=4G -p MemoryMax=5G \
  -p CPUWeight=50 nice -n 10 gradle --no-daemon run --args="--listen 0.0.0.0:9931 --log loop.jsonl"

# or the installed distribution:
gradle --no-daemon installDist
./build/install/pilot-phone/bin/pilot-phone --listen 0.0.0.0:9931 --log loop.jsonl
```

Args (manual parsing, no library):

| Flag | Default | Meaning |
|---|---|---|
| `--model <path>` | `../export/sac_actor_hover_v7.onnx` (rel. to pilot/phone) | model file |
| `--backend onnx\|tflite\|auto` | `auto` (prefers onnx) | inference backend |
| `--listen <host:port>` | `0.0.0.0:9931` | TCP bind address |
| `--max-delta <float>` | gate default `0.5` | override `maxDeltaPerStep` |
| `--box <float>` | gate default `6.0` | override `hoverBoxRadius` |
| `--log <path>` | none | JSONL log file |
| `--selftest` | — | smoke: load model, golden diff, gate pass-check, exit 0/1 |

## Protocol (verbatim spec)

Line-delimited UTF-8 JSON, one message per line ending `\n`.

Messages **IN** (bridge -> runner):

```json
{"t":"hello",...}
{"t":"ep_start","ep":<int>,"seed":<int>}
{"t":"obs","ep":<int>,"step":<int>,"obs":[15 floats],"pos_err":<float>}
{"t":"ep_end",...}
{"t":"run_end",...}
```

`ep_start` resets the gate. After each `obs` the runner replies:

```json
{"t":"act","a":[ax,ay,az],"cause":"<safety cause>"}
```

where `a` is the **safety-filtered** action. The runner computes
`telemetryAgeSec` itself (time since the previous `obs` line arrived; `0` for
the first obs of an episode after `ep_start`). `batteryPct` and `DepthGrid`
are null in loopback. After `ep_end`/`run_end` it just logs.

JSONL log line per obs:

```json
{"ep":0,"step":3,"raw":[...3 policy outputs...],"action":[...3 filtered...],"cause":"pass","telemetry_age":0.55}
```

plus `{"ev":"ep_start",...}` on episode boundaries and `{"ev":"client_lost"}`
on disconnect.

### Selftest (evidence from this machine)

```
=== pilot selftest ===
model=../export/sac_actor_hover_v7.onnx backend=auto
BACKEND onnx: opened sac_actor_hover_v7.onnx
GOLDEN onnx: rows=200 max_abs_diff=0.000e+00 (worst row -1) tol=1e-05 -> PASS
GATE smoke: cause=pass action=[0.0, 0.0, 0.0] -> PASS
=== selftest PASS ===
```

Golden data: `src/main/resources/golden_obs.json` — first 100 rows of
`data/ccw_data_merged.npz['obs']` + 100-row `default_rng(20261109)` sample,
with exact float32 ONNX actions (generated with
`~/.venvs/pilot/bin/python` + onnxruntime 1.31.1; cross-checked
against the npz `A` column: 2.1e-7).

## Backend status

| backend | status | evidence |
|---|---|---|
| `onnx` | **works** (the loopback backend) | `com.microsoft.onnxruntime:onnxruntime:1.19.2`, JVM desktop jar with bundled natives; selftest golden max_abs_diff = 0.000e+00 over 200 rows |
| `tflite` | **not available on desktop JVM** | No desktop-capable artifact found: tried `org.tensorflow:tensorflow-lite` **2.16.1** and **2.17.0** (Maven Central, `packaging=aar`, Android-only) and `com.google.ai.edge.litert:litert` **1.0.0 … 2.3.2** (Google Maven `dl.google.com`, `packaging=aar`, Android-only). The tflite file is fp32 plain; parity with PyTorch 4.5e-7 was proven separately with python tflite-runtime (not installed in the pilot venv, so golden covers ONNX only). `PilotModel` resolves `org.tensorflow.lite.Interpreter` via `Class.forName` (compile-time safe, no dep on the classpath); `--backend tflite` fails with a clear `UnsupportedOperationException` naming tried versions. |

## SafetyGate rules & priorities

`SafetyGate.filter(obs, prevAction, batteryPct, telemetryAgeSec, depthGrid)` —
pure logic, allocation-light, stateless except `returnNeeded` (cleared by
`reset()`; `prevAction` is caller-supplied, never cached). The candidate
policy action is an optional 6th parameter; when omitted it is read from
`obs[12..14]`. First match wins:

| # | cause | condition (defaults) | action |
|---|---|---|---|
| 1 | `telemetry_loss_hover` | `telemetryAgeSec > 2.0` | zeros (hover in place) |
| 2 | `battery_land` | `batteryPct <= 15` | `(0,0,+1)` forced descend |
| 3 | `battery_return_hover` | `batteryPct <= 30` (sets `returnNeeded=true`) | zeros |
| 4 | `geofence_pullin` | `norm(obs[0..2]) >= 7.5` | per-axis `clip(-obs[0..2], ±1.0)` |
| 5 | `hover_box_pullin` | `norm(obs[0..2]) >= 6.0` | per-axis `clip(-1.0·obs[0..2], ±1.0)` |
| 6 | `obstacle_veto` | `depthGrid.nearestDistanceM < 1.0` **and** `dot(policy, nearestUnitXY₃) > 0` (only when moving **toward** it) | zeros |
| 7 | `pass` (`\|rate_limited` `\|speed_capped`) | otherwise | policy action, per-axis rate-limited vs `prevAction` to `±0.5`/step, then clamped to `±1.0` |

Priorities proven by tests: telemetry loss beats battery & geofence; battery
beats geofence (land beats return); geofence beats hover box; hover box beats
obstacle veto. Boundary equalities: `>=` triggers box/geofence, `>` for
telemetry, `<=` for battery, `<` for depth distance.

## Files

```
settings.gradle.kts · build.gradle.kts · gradle.properties · README.md
src/main/kotlin/miniai/pilot/SafetyGate.kt       — SafetyConfig, DepthGrid, Decision, SafetyGate
src/main/kotlin/miniai/pilot/PilotModel.kt       — ONNX backend + reflective (desktop-absent) TFLite path
src/main/kotlin/miniai/pilot/LoopbackRunner.kt   — TCP server main (miniai.pilot.LoopbackRunnerKt)
src/main/resources/golden_obs.json               — 200 golden obs + exact ONNX actions
src/test/kotlin/miniai/pilot/SafetyGateTest.kt   — 37 JUnit5 tests
```
