# S6 — Phone-side pilot: parity, safety layer, sim loopback

Date: 2026-11-10 (run 18:32:06-18:42:17 CT). Verdict: **LOOPBACK_OK**.

## 1. Export + numeric parity (commits dabd3b0)
- SAC actor `hover_v7` exported to ONNX and LiteRT (fp32) via `tools/export_policy.py`.
- Replay of 20k recorded sim observations, PyTorch vs exported: **max abs diff 4.5e-7** (limit 1e-4) PASS.
- Kotlin runner `--selftest` on 200 golden obs: max abs diff 0.0 vs Python ONNX.
- Runtime on the phone-side path is ONNX Runtime Java (desktop LiteRT/TFLite jar does not exist; Android AAR only — documented).

## 2. Safety layer (commit a8cc447)
`pilot/phone` SafetyGate, phone-side, separate module: hover box 6 m, geofence 7.5 m, speed cap ±1 m/s, rate limit 0.5/step, telemetry-loss >2 s -> hover, battery 30/15 pct, depth-grid veto 1 m. Exact cause strings + priority chain. **37 JUnit tests, all green.**

## 3. Sim loopback (this run)
Kotlin pilot runner (sable, ONNX) <-> Blocks sim via `pilot/loopback/loopback_bridge.py` over TCP JSON-lines, fired as FPVS6Loop scheduled task on the Yoga. Identical episodes both passes (4 eps x 120 steps, seed=100+ep, FPV_DISTURB_EVERY=6 SCALE=0.4).

| episode | python mean_err | wire mean_err |
|---|---|---|
| 0 | 0.3380 | 0.3406 |
| 1 | 0.3417 | 0.3425 |
| 2 | 0.3221 | 0.3219 |
| 3 | 0.3241 | 0.3297 |

- python_mean_err = **0.3315** (0.335 m class reproduced: PASS)
- wire_mean_err = **0.3337** (within 0.15 of python: PASS)
- mean_abs_delta = **0.0143** (<= 0.15: PASS; SafetyGate rate limit did not bite — no `--max-delta 2.0` rerun needed)
- violations = **0**, terminated_any = **False** -> **LOOPBACK_OK**

Evidence in this folder (`eval_loopback_result.txt`, `eval_loopback.csv`); raw bridge log + runner JSONL banked on the S6 kanban card artifacts.

## 4. Rig bugs found + fixed (all aborted pre-sim, zero wasted sim time)
- `5be4386` register_s6.ps1: PowerShell `-replace` is case-insensitive; it clobbered the mixed-case `-RunnerHost` parameter name inside the task XML. Fixed with `-creplace` for the all-caps placeholder.
- `bac4b5a` s6_loop.ps1: bridge python inherited cwd = system32 -> PermissionError writing `projectairsim_client.log`. Added `-WorkingDirectory`.
- `70be3da` s6_loop.ps1: projectairsim resolves `sim_config/scene_basic_drone.jsonc` relative to cwd; bridge cwd set to `D:\fpv\ProjectAirSim\client\python\example_user_scripts` (same cwd the proven eval rig uses).

## 5. Scope status vs card DONE criteria
- Parity report: this file + reports/ — PASS
- Safety unit tests: 37/37 green — PASS
- Loopback hover in sim equals Python result (0.335 m class): 0.3337 vs 0.3315 — PASS

NOT pushed to the public origin yet: these commits embed tailscale IPs and local paths; the scrub-before-push scan (per repo policy) must run first.
