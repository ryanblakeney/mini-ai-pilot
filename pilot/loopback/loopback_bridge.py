#!/usr/bin/env python3
"""S6 loopback bridge -- sim-side driver for the Kotlin pilot-phone runner.

Runs on the Windows sim laptop (Yoga) under D:\\fpv\\venv312 python.

Modes
  --dry-run  stdlib-only protocol validation against a running runner (NO
             gym / projectairsim / stable_baselines3 imports at all). Replays
             the first 100 rows of golden_obs.json as 2 episodes x 50 steps,
             expects exactly one act line per obs, validates every reply,
             prints "DRYRUN_OK <n_acts> <n_errors>" and exits 0 iff all valid.
  default    full sim eval on the Yoga (Blocks must already be up, port 8989):
             PASS 1  python baseline eval -- reproduces the hover_v7 eval
                     protocol EXACTLY (step_dt=0.5, max_steps=120,
                     FPV_DISTURB_EVERY=6, FPV_DISTURB_SCALE=0.4, 4 episodes,
                     seeds 100+ep, SAC.load(policy_final, device cpu),
                     predict(obs, deterministic=True)).
             PASS 2  wire eval -- same env, same seeds, but the action comes
                     from the Kotlin runner over TCP (safety-filtered action).
             Then writes eval_loopback_result.txt / eval_loopback.csv /
             s6_bridge.log into --out-dir and os._exit(0).

Protocol (pilot/phone/README.md): line-delimited UTF-8 JSON over TCP, runner
is the SERVER, this bridge dials IN across tailscale. Runner replies after
each obs with {"t":"act","a":[3 floats in [-1,1]],"cause":"..."}.

Robustness contract: the python pass result must survive ANY wire failure.
Result files are written even on partial/aborted wire passes (verdict FAIL
plus a reason= line). Nothing is ever left half-flushed: os._exit at the very
end (Windows python teardown lesson from rl_eval.py).
"""
import argparse
import json
import math
import os
import socket
import sys
import time
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# hover_v7 eval protocol constants (fixed -- do not tune)
DISTURB_EVERY = 6
DISTURB_SCALE = 0.4
STEP_DT = 0.5

RUN_ID = "s6-" + time.strftime("%Y%m%d-%H%M%S")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="S6 phone-pilot loopback bridge")
    p.add_argument("--runner-host", default=None,
                   help="runner TCP host (required unless --dry-run)")
    p.add_argument("--runner-port", type=int, default=9931)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=100)
    p.add_argument("--act-timeout", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--policy-dir", default=r"D:\fpv\rl\runs\hover_v7")
    p.add_argument("--out-dir", default=r"D:\fpv\ops")
    p.add_argument("--golden", default=None,
                   help="golden_obs.json path (default: next to this script)")
    p.add_argument("--dry-run", action="store_true",
                   help="protocol-only replay of golden_obs.json, no sim")
    return p.parse_args(argv)


# --------------------------------------------------------------- wire helpers

def json_dumps(obj):
    return json.dumps(obj, separators=(",", ":"))


def send_line(sock, obj):
    sock.sendall((json_dumps(obj) + "\n").encode("utf-8"))


def tcp_connect(host, port, log, tries=10, delay=3.0):
    """Dial the runner with retries. Returns a connected socket or None."""
    last = None
    for i in range(1, tries + 1):
        try:
            s = socket.create_connection((host, port), timeout=15.0)
            try:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            log(f"TCP connected to {host}:{port} on attempt {i}")
            return s
        except OSError as e:
            last = e
            log(f"TCP connect {i}/{tries} to {host}:{port} failed: {e}")
            time.sleep(delay)
    log(f"TCP connect FAILED after {tries} attempts: {last}")
    return None


def validate_act(obj):
    """Return an error string if the reply is not a valid act message, else None."""
    if not isinstance(obj, dict):
        return "reply_not_object"
    if obj.get("t") != "act":
        return f"bad_t={obj.get('t')!r}"
    a = obj.get("a")
    if not isinstance(a, list) or len(a) != 3:
        return "bad_a_len"
    for x in a:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return "a_non_numeric"
        if not (-1.0 - 1e-9 <= float(x) <= 1.0 + 1e-9):
            return f"a_out_of_box={x}"
    cause = obj.get("cause")
    if not isinstance(cause, str) or not cause:
        return "missing_cause"
    return None


def read_act(rfile, sock, timeout):
    """Read one act line. Returns (status, action_or_None, err):
    status in: ok | timeout | eof | closed | bad_json."""
    sock.settimeout(timeout)
    try:
        line = rfile.readline()
    except socket.timeout:
        return ("timeout", None, f"no act reply within {timeout}s")
    except OSError as e:
        return ("closed", None, f"socket error: {e}")
    if line == "":
        return ("eof", None, "runner closed the connection")
    try:
        obj = json.loads(line)
    except ValueError as e:
        return ("bad_json", None, f"unparseable line: {line[:120]!r} ({e})")
    err = validate_act(obj)
    if err is not None:
        return ("bad_json", None, err)
    return ("ok", [float(x) for x in obj["a"]], None)


# --------------------------------------------------------------------- dry run

def run_dry(args):
    """Protocol-only validation. No numpy, no gym, no sb3, no sim."""
    golden_path = args.golden or os.path.join(SCRIPT_DIR, "golden_obs.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    rows = golden["obs"][:100]
    if len(rows) != 100:
        print(f"DRYRUN_FAIL golden has only {len(rows)} obs rows", flush=True)
        return 1
    eps = [rows[:50], rows[50:100]]
    total = 100

    sock = tcp_connect(args.runner_host, args.runner_port,
                       lambda m: print(m, flush=True))
    if sock is None:
        print("DRYRUN_FAIL 0 0 LINK_FAIL", flush=True)
        return 1
    rfile = sock.makefile("r", encoding="utf-8")

    n_acts = 0
    n_errors = 0
    try:
        send_line(sock, {"t": "hello", "run_id": RUN_ID + "-dry",
                         "episodes": 2, "max_steps": 50, "mode": "dryrun"})
        for ep_i, ep_rows in enumerate(eps):
            send_line(sock, {"t": "ep_start", "ep": ep_i,
                             "seed": args.seed_base + ep_i})
            errs = []
            for step_i, row in enumerate(ep_rows):
                pos_err = math.sqrt(sum(float(v) * float(v) for v in row[:3]))
                send_line(sock, {"t": "obs", "ep": ep_i, "step": step_i,
                                 "obs": [float(v) for v in row],
                                 "pos_err": pos_err})
                status, act, err = read_act(rfile, sock, max(args.act_timeout, 10.0))
                if status == "ok":
                    n_acts += 1
                    errs.append(pos_err)
                else:
                    n_errors += 1
                    print(f"DRYRUN_ERR ep={ep_i} step={step_i} {status}: {err}",
                          flush=True)
                    if status in ("eof", "closed"):
                        print("DRYRUN_FAIL connection lost mid-run", flush=True)
                        return 1
            send_line(sock, {"t": "ep_end", "ep": ep_i, "steps": len(ep_rows),
                             "mean_err": (sum(errs) / len(errs)) if errs else 0.0,
                             "max_err": max(errs) if errs else 0.0,
                             "final_err": errs[-1] if errs else 0.0,
                             "terminated": False})
        send_line(sock, {"t": "run_end", "python_mean_err": -1.0,
                         "wire_mean_err": -1.0, "violations": n_errors})
        time.sleep(0.2)  # let run_end land in the runner log before we bail
    finally:
        try:
            rfile.close()
            sock.close()
        except OSError:
            pass

    if n_errors == 0 and n_acts == total:
        print(f"DRYRUN_OK {n_acts} {n_errors}", flush=True)
        return 0
    print(f"DRYRUN_FAIL {n_acts} {n_errors}", flush=True)
    return 1


# -------------------------------------------------------------------- sim mode

class Log:
    """Timestamped append-log to out-dir/s6_bridge.log + stdout, flushed per line."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        try:
            self.f.write(line + "\n")
            self.f.flush()
        except OSError:
            pass

    def close(self):
        try:
            self.f.close()
        except OSError:
            pass


def episode_stats(errs):
    return (sum(errs) / len(errs), max(errs), errs[-1]) if errs else (0.0, 0.0, 0.0)


def mean_abs_delta(py_acts, wire_acts):
    """Mean |a_wire - a_python| over steps aligned per episode."""
    diffs = []
    for pa, wa in zip(py_acts, wire_acts):
        for a_p, a_w in zip(pa, wa):
            diffs.append(sum(abs(p - w) for p, w in zip(a_p, a_w)) / 3.0)
    return (sum(diffs) / len(diffs)) if diffs else float("nan")


def verdict_of(terminated_any, delta, violations, wire_complete):
    if not wire_complete:
        return "LOOPBACK_FAIL"
    if terminated_any:
        return "LOOPBACK_FAIL"
    if delta <= 0.15 and violations <= 1:
        return "LOOPBACK_OK"
    if delta <= 0.30:
        return "LOOPBACK_MARGINAL"
    return "LOOPBACK_FAIL"


def write_outputs(out_dir, st, log):
    """Write result txt + csv. Never raises: each file guarded separately."""
    f4 = lambda x: f"{x:.4f}" if x == x else "-1.0000"  # nan -> -1.0000
    result_path = os.path.join(out_dir, "eval_loopback_result.txt")
    try:
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(
                f"python_mean_err={f4(st['python_mean'])} "
                f"wire_mean_err={f4(st['wire_mean'])} "
                f"mean_abs_delta={f4(st['delta'])} "
                f"violations={st['violations']} "
                f"terminated_any={st['terminated_any']} "
                f"verdict={verdict_of(st['terminated_any'], st['delta'], st['violations'], st['wire_complete'])}\n"
            )
            if st["reason"]:
                f.write(f"reason={st['reason']}\n")
        log(f"WROTE {result_path}")
    except Exception as e:
        log(f"RESULT_WRITE_FAIL {e!r}")
    csv_path = os.path.join(out_dir, "eval_loopback.csv")
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("run,steps,mean_err,final_err,terminated\n")
            for tag in ("python", "wire"):
                for i, row in enumerate(st[f"{tag}_rows"]):
                    name, steps, mean_err, final_err, term = row
                    f.write(f"{name},{steps},{mean_err:.6f},{final_err:.6f},{term}\n")
            f.write(f"python_mean_err,,{st['python_mean']:.6f},,\n")
        log(f"WROTE {csv_path}")
    except Exception as e:
        log(f"CSV_WRITE_FAIL {e!r}")


def run_sim(args):
    os.makedirs(args.out_dir, exist_ok=True)
    log = Log(os.path.join(args.out_dir, "s6_bridge.log"))
    st: "dict[str, Any]" = {
        "python_rows": [], "wire_rows": [], "py_acts": [], "wire_acts": [],
        "python_mean": float("nan"), "wire_mean": float("nan"),
        "delta": float("nan"), "violations": 0, "terminated_any": False,
        "wire_complete": False, "reason": "",
    }
    env = None
    sock = None
    try:
        log(f"BRIDGE_BEGIN run_id={RUN_ID} runner={args.runner_host}:{args.runner_port} "
            f"episodes={args.episodes} max_steps={args.max_steps} act_timeout={args.act_timeout}")

        # --- protocol-fixed environment knobs BEFORE env construction
        os.environ["FPV_DISTURB_EVERY"] = str(DISTURB_EVERY)
        os.environ["FPV_DISTURB_SCALE"] = str(DISTURB_SCALE)
        rl_dir = os.environ.get("FPV_RL_DIR", r"D:\fpv\rl")
        if rl_dir not in sys.path:
            sys.path.insert(0, rl_dir)

        import numpy as np
        from stable_baselines3 import SAC
        from gym_env import BlocksDroneEnv

        env = BlocksDroneEnv(step_dt=STEP_DT, max_steps=args.max_steps)
        policy_path = os.path.join(args.policy_dir, "policy_final")
        model = SAC.load(policy_path, device="cpu")
        log(f"POLICY loaded {policy_path} (SAC, cpu)")

        # ---------------- PASS 1: python baseline (exact 0.335 protocol)
        py_ep_means = []
        for ep in range(args.episodes):
            seed = args.seed_base + ep
            obs, info = env.reset(seed=seed)
            errs, acts = [], []
            term = trunc = False
            while not (term or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                errs.append(float(info["pos_err"]))
                acts.append([float(x) for x in np.asarray(action, dtype="float64").tolist()])
            mean_err, max_err, final_err = episode_stats(errs)
            py_ep_means.append(mean_err)
            st["terminated_any"] = st["terminated_any"] or bool(term)
            st["python_rows"].append((f"python_ep{ep}", len(errs), mean_err, final_err, bool(term)))
            st["py_acts"].append(acts)
            log(f"PYEP{ep} steps={len(errs)} mean_err={mean_err:.4f} "
                f"max_err={max_err:.4f} final_err={final_err:.4f} terminated={bool(term)}")
        st["python_mean"] = sum(py_ep_means) / len(py_ep_means)
        log(f"PYTHON_PASS_DONE python_mean_err={st['python_mean']:.4f}")

        # ---------------- PASS 2: wire eval through the Kotlin runner
        sock = tcp_connect(args.runner_host, args.runner_port, log)
        if sock is None:
            st["reason"] = "LINK_FAIL could not reach runner"
            log("WIRE_PASS_ABORTED LINK_FAIL")
        else:
            try:
                rfile = sock.makefile("r", encoding="utf-8")
                send_line(sock, {"t": "hello", "run_id": RUN_ID,
                                 "episodes": args.episodes,
                                 "max_steps": args.max_steps,
                                 "disturb": {"every": DISTURB_EVERY,
                                             "scale": DISTURB_SCALE}})
                wire_ep_means = []
                wire_alive = True
                wire_stop = ""
                for ep in range(args.episodes):
                    if not wire_alive:
                        break
                    seed = args.seed_base + ep
                    obs, info = env.reset(seed=seed)
                    send_line(sock, {"t": "ep_start", "ep": ep, "seed": seed})
                    errs, acts = [], []
                    term = trunc = False
                    while not (term or trunc):
                        send_line(sock, {"t": "obs", "ep": ep, "step": len(errs),
                                         "obs": [float(x) for x in np.asarray(obs, dtype="float64").tolist()],
                                         "pos_err": float(info["pos_err"]) if errs else float(np.linalg.norm(np.asarray(obs[:3], dtype="float64")))})
                        status, act, err = read_act(rfile, sock, args.act_timeout)
                        if status == "timeout":
                            st["violations"] += 1
                            log(f"VIOLATION ep={ep} step={len(errs)} act_timeout -> zeros")
                            act = [0.0, 0.0, 0.0]
                        elif status in ("eof", "closed", "bad_json"):
                            wire_alive = False
                            wire_stop = f"{status}: {err}"
                            st["violations"] += 1
                            log(f"WIRE_ABORT ep={ep} step={len(errs)} {wire_stop}")
                            break
                        action = np.clip(np.asarray(act, dtype="float32"), -1.0, 1.0)
                        obs, r, term, trunc, info = env.step(action)
                        errs.append(float(info["pos_err"]))
                        acts.append([float(x) for x in action.tolist()])
                    mean_err, max_err, final_err = episode_stats(errs)
                    if errs:
                        wire_ep_means.append(mean_err)
                        st["wire_rows"].append((f"wire_ep{ep}", len(errs), mean_err, final_err, bool(term)))
                        st["wire_acts"].append(acts)
                        st["terminated_any"] = st["terminated_any"] or bool(term)
                        send_line(sock, {"t": "ep_end", "ep": ep, "steps": len(errs),
                                         "mean_err": mean_err, "max_err": max_err,
                                         "final_err": final_err, "terminated": bool(term)})
                    log(f"WIREEP{ep} steps={len(errs)} mean_err={mean_err:.4f} "
                        f"max_err={max_err:.4f} final_err={final_err:.4f} terminated={bool(term)}")
                st["wire_complete"] = wire_alive and len(wire_ep_means) == args.episodes
                if not st["wire_complete"] and not st["reason"]:
                    st["reason"] = f"WIRE_INCOMPLETE {wire_stop}" if wire_stop else "WIRE_INCOMPLETE"
                if wire_alive:
                    send_line(sock, {"t": "run_end",
                                     "python_mean_err": st["python_mean"],
                                     "wire_mean_err": (sum(wire_ep_means) / len(wire_ep_means)) if wire_ep_means else -1.0,
                                     "violations": st["violations"]})
                if wire_ep_means:
                    st["wire_mean"] = sum(wire_ep_means) / len(wire_ep_means)
                if st["wire_complete"]:
                    st["delta"] = mean_abs_delta(st["py_acts"], st["wire_acts"])
                else:
                    st["py_acts"], st["wire_acts"] = [], []  # partial => delta meaningless
                try:
                    rfile.close()
                except OSError:
                    pass
            except OSError as e:
                st["wire_complete"] = False
                st["reason"] = st["reason"] or f"WIRE_ERROR {e!r}"
                log(f"WIRE_PASS_FAILED {e!r}")
            log(f"WIRE_PASS_DONE complete={st['wire_complete']} "
                f"wire_mean_err={st['wire_mean']:.4f} violations={st['violations']}")
    except Exception as e:
        st["reason"] = st["reason"] or f"FATAL {type(e).__name__}: {e}"
        log(f"FATAL {type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())
    finally:
        try:
            if env is not None:
                env.close()
                log("ENV_CLOSED")
        except Exception as e:
            log(f"ENV_CLOSE_FAIL {e!r}")
        try:
            if sock is not None:
                sock.close()
        except OSError:
            pass

    v = verdict_of(st["terminated_any"], st["delta"], st["violations"], st["wire_complete"])
    log(f"VERDICT={v} python_mean_err={st['python_mean']:.4f} "
        f"wire_mean_err={st['wire_mean']:.4f} mean_abs_delta={st['delta']:.4f} "
        f"violations={st['violations']} terminated_any={st['terminated_any']}")
    write_outputs(args.out_dir, st, log)
    log("BRIDGE_END")
    log.close()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)  # Windows python teardown lesson (rl_eval.py)


def main():
    args = parse_args()
    if args.dry_run:
        sys.exit(run_dry(args))
    if not args.runner_host:
        print("error: --runner-host is required unless --dry-run", file=sys.stderr)
        sys.exit(2)
    run_sim(args)


if __name__ == "__main__":
    main()
