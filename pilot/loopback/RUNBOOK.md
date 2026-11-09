# S6 Loopback Runbook — sim bridge ↔ Kotlin pilot runner end-to-end test

Actors:
- **sable** (this box, tailscale `<SABLE_TAILSCALE_IP>`): runs the Kotlin runner as a **TCP server** on `0.0.0.0:9931`.
- **Yoga** (`user@<YOGA_TAILSCALE_IP>`, Windows): runs Blocks sim + `loopback_bridge.py` via scheduled task `FPVS6Loop`; the bridge **dials IN** to sable over tailscale.

Files in this directory (deploy targets in parentheses):

| file | deploys to | role |
|---|---|---|
| `loopback_bridge.py` | `D:\fpv\ops\` | python sim driver (PASS 1 python / PASS 2 wire) |
| `golden_obs.json` | `D:\fpv\ops\` | needed by `--dry-run` (sits next to the bridge); sim mode does not use it |
| `s6_loop.ps1` | `D:\fpv\ops\` | scheduled-task rig: collision guard → Blocks → bridge → teardown |
| `register_s6.ps1` | `D:\fpv\ops\` | registers one-shot task `FPVS6Loop` (InteractiveToken, same pattern as FPVProbeB) |

## 0. Local validation already performed (recipe + evidence)

```bash
# from this loopback/ dir, with the real runner:
(cd ~/projects/mini-ai-pilot/pilot/phone && \
  ./build/install/pilot-phone/bin/pilot-phone --listen 127.0.0.1:9937 \
  --log <this dir>/validation_runner_log.jsonl &)   # background it
~/.venvs/pilot/bin/python -m py_compile loopback_bridge.py        # -> ok
~/.venvs/pilot/bin/python loopback_bridge.py --dry-run \
  --runner-host 127.0.0.1 --runner-port 9937 --out-dir ./validation_out/
```

Observed (2026-11-09): `DRYRUN_OK 100 0`, exit 0. Runner log shows exactly
`hello → ep_start(100) → 50×obs/act → ep_end → ep_start(101) → 50×obs/act → ep_end → run_end`
with 100 `"cause"` records. Also verified the fail-path contract: sim mode with
no reachable runner / missing deps still writes all 3 result files with
`verdict=LOOPBACK_FAIL` + `reason=` line and exits 0. `s6_loop.ps1` and
`register_s6.ps1` are CRLF + structurally balanced (braces/parens/quotes); the
live PowerShell-parser check over ssh could NOT be run (Yoga went offline mid
validation — rerun `register_s6.ps1` step below as first real test if paranoid,
a parse error shows up as an immediate s6_loop_out.txt absence / PS error in task_log).

## 1. Pre-flight (Yoga must be awake)

```bash
tailscale ping -c 2 <YOGA_TAILSCALE_IP>
ssh -o ConnectTimeout=8 user@<YOGA_TAILSCALE_IP> 'echo UP'
```

## 2. Start the runner on sable (keep it alive for the whole run)

```bash
mkdir -p ~/fpv_ops_s6
cd ~/projects/mini-ai-pilot/pilot/phone
nohup ./build/install/pilot-phone/bin/pilot-phone --listen 0.0.0.0:9931 \
  --log ~/fpv_ops_s6/runner_$(date +%Y%m%d).jsonl > ~/fpv_ops_s6/runner.out 2>&1 &
sleep 5; head -1 ~/fpv_ops_s6/runner.out   # expect: pilot-loopback listening on 0.0.0.0:9931 backend=onnx
```

## 3. Deploy files to the Yoga

```bash
cd <this dir>
scp loopback_bridge.py golden_obs.json s6_loop.ps1 register_s6.ps1 \
  user@<YOGA_TAILSCALE_IP>:D:/fpv/ops/
```

## 4. Tailscale reachability test (Yoga → runner)

```bash
ssh user@<YOGA_TAILSCALE_IP> 'powershell -NoProfile "Test-NetConnection <SABLE_TAILSCALE_IP> -Port 9931 | Select-Object -Property TcpTestSucceeded"'
```

Must be `True`. If not: Windows firewall on Yoga must allow inbound TCP 9931 on
the Tailscale interface (or outbound isn't the issue — the runner binds 0.0.0.0
already; check the runner is up with `ss -ltnp | grep 9931` on sable).

## 5. Register + fire the task

```bash
# fires ~1 min after registration by default; or pass -StartAt 'yyyy-MM-ddTHH:mm:ss'
ssh user@<YOGA_TAILSCALE_IP> "powershell -NoProfile -ExecutionPolicy Bypass -File D:\fpv\ops\register_s6.ps1 -RunnerHost <SABLE_TAILSCALE_IP>"
```

Output: `REGISTERED FPVS6Loop boundary=... runner=<SABLE_TAILSCALE_IP>` + `schtasks /query` echo.

## 6. Monitor (poll every ~60 s; expect 10–25 min total)

```bash
ssh user@<YOGA_TAILSCALE_IP> 'powershell -NoProfile "Get-Content D:\fpv\ops\s6_loop_out.txt -Tail 20"'
ssh user@<YOGA_TAILSCALE_IP> 'powershell -NoProfile "Get-Content D:\fpv\ops\s6_bridge.log -Tail 20"'
ssh user@<YOGA_TAILSCALE_IP> 'powershell -NoProfile "Get-Content D:\fpv\ops\s6_py_stderr.txt -Tail 20"'
tail -f ~/fpv_ops_s6/runner_*.jsonl        # sable side: obs/act records stream here
```

Sentinels in `s6_loop_out.txt`: `SKIPPED_BUSY` (collision guard tripped —
wait/retry), `PORT_TIMEOUT_8989_300S` (Blocks failed to come up), then
`RESULT_TAIL:` + the one-line verdict + `TASK_DONE`.
In `s6_bridge.log`: `PYTHON_PASS_DONE` then `WIREEP0..3` lines then `VERDICT=`.
Bridge stderr file catches pre-main crashes.

## 7. Collect results

```bash
scp user@<YOGA_TAILSCALE_IP>:D:/fpv/ops/eval_loopback_result.txt \
    user@<YOGA_TAILSCALE_IP>:D:/fpv/ops/eval_loopback.csv \
    user@<YOGA_TAILSCALE_IP>:D:/fpv/ops/s6_bridge.log <archive dir>/
```

Result line format: `python_mean_err=X wire_mean_err=Y mean_abs_delta=Z
violations=V terminated_any=BOOL verdict=...` — OK iff no termination and
`mean_abs_delta<=0.15` and `violations<=1`; MARGINAL iff delta≤0.30; else FAIL
(with `reason=` on line 2 for link failures). `mean_abs_delta` is the mean
per-axis |wire − python| action difference over steps aligned per episode
(wire actions are SafetyGate-filtered, so small non-zero deltas are expected;
rate-limiting `|rate_limited` causes are visible in the runner JSONL log).
Reference: python PASS mean should land ≈0.33 m (hover_v7 protocol reproduced:
step_dt 0.5, max_steps 120, disturb every 6 / scale 0.4, seeds 100..103,
SAC cpu deterministic).

## 8. Teardown

```bash
ssh user@<YOGA_TAILSCALE_IP> 'powershell -NoProfile "schtasks /delete /tn FPVS6Loop /f"'
ssh user@<YOGA_TAILSCALE_IP> 'powershell -NoProfile "Get-Process Blocks* -EA SilentlyContinue"'  # should be empty (task's finally kills it)
# kill the sable runner:
pkill -f 'pilot-phone.*9931'   # or kill the tracked pid
```

## Failure-mode quick table

| symptom | cause / action |
|---|---|
| `SKIPPED_BUSY` | another FPV task or stray Blocks running — clear and re-fire |
| `LINK_FAIL` in bridge log | runner down / firewall on 9931 — redo steps 2+4 |
| `PORT_TIMEOUT_8989_300S` | Blocks cold-start slow/crashed — check `D:\fpv\sim\blocks` logs, re-fire |
| `verdict=LOOPBACK_FAIL reason=WIRE_INCOMPLETE ...` | runner died mid-run — check runner stdout `~/fpv_ops_s6/runner.out` |
| `violations=N` large | act timeouts — runner too slow or network stall; check `telemetry_age` in runner JSONL (age >2 s triggers `telemetry_loss_hover` zero-actions) |
