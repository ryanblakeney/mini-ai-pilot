"""Q3 stage-2 waypoint-reach evaluation (ProjectAirSim Blocks rig).

Launch ONLY via the FPVEval route (task_eval_v4.ps1 with FPV_WAY=1) inside a
console session with Blocks already up. Mirrors rl_eval.py structure/output
(eval_result.txt + eval.csv in the policy's dir) but scores DISTANCE-TO-
WAYPOINT instead of hover error.

Runs 4 policy episodes x 120 steps (deterministic predict, seeds 0..3;
disturbances come from the env via FPV_DISTURB_EVERY/FPV_DISTURB_SCALE), plus
scripted baselines on the SAME seeds: zero-action and P-controller
v_cmd = clip(-0.8 * wp_err, -1, +1) per axis (wp_err is obs[:3] by design).

Policy loaded from FPV_POLICY (path to policy_final.zip). Acceptance is baked
into the verdict line:
  WAYPOINT_OK iff policy_mean_final < 1.0 m
             AND policy_mean_final < p_baseline_mean_final
             AND policy_arrivals >= 3 (of 4)
             AND zero baseline had no drift-outs/landings.
os._exit at end (Win python teardown segfault lesson).
"""
import csv
import os
import sys
import time
import traceback

RL_DIR = r"D:\fpv\rl"
_default_run = os.environ.get("FPV_RUN", RL_DIR + r"\runs\waypoint_v1")
POLICY = os.environ.get("FPV_POLICY", os.path.join(_default_run, "policy_final.zip"))
OUT = r"D:\fpv\ops\eval_out.txt"
N_EPISODES = 4
MAX_STEPS_EVAL = 120
PB_K = 0.8  # same scripted-P baseline gain as the hover eval

sys.path.insert(0, RL_DIR)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with open(OUT, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


try:
    open(OUT, "w").close()
    log("EVAL_BEGIN")

    import numpy as np
    from stable_baselines3 import SAC

    from waypoint_env import WaypointDroneEnv

    base_env = WaypointDroneEnv(step_dt=0.5, max_steps=MAX_STEPS_EVAL)
    env = base_env
    if os.environ.get("FPV_RESID"):
        # waypoint_v2: policy acts through ResidualWaypointEnv (P + bounded
        # correction). Guarded import: module absence can NEVER touch
        # default (FPV_RESID-unset) runs. Baselines stay on the RAW env.
        try:
            from residual_waypoint_env import ResidualWaypointEnv
            env = ResidualWaypointEnv(base_env)
            log("FPV_RESID=1: policy env wrapped in ResidualWaypointEnv")
        except ImportError as exc:
            raise RuntimeError(
                "FPV_RESID set but residual_waypoint_env unavailable") from exc
    model = SAC.load(POLICY, env=env)
    log(f"POLICY={POLICY} ALGO=SAC")
    out_dir = os.path.dirname(POLICY)

    def run_episode(acting, seed, e=None):
        """One episode; returns list of (dist, arrived, failed) per step."""
        e = env if e is None else e
        obs, info = e.reset(seed=seed)
        recs = []
        term = trunc = False
        while not (term or trunc):
            obs, r, term, trunc, info = e.step(acting(obs))
            recs.append((info["dist"], info["arrived"], info["failed"]))
        return recs

    def policy_act(obs):
        a, _ = model.predict(obs, deterministic=True)
        return a

    def p_act(obs):
        return np.clip(-PB_K * np.asarray(obs[:3], dtype=np.float32), -1.0, 1.0)

    def zero_act(obs):
        return np.zeros(3, dtype=np.float32)

    rows = []
    zero_failed = False
    pol_arrivals = 0

    def run_tagged(prefix, acting, log_tag, e=None):
        global zero_failed, pol_arrivals  # module-level (try block = globals)
        ep_rows = []
        for ep in range(N_EPISODES):
            recs = run_episode(acting, ep, e)
            dists = [x[0] for x in recs]
            arrived = bool(recs[-1][1]) if recs else False
            failed = bool(recs[-1][2]) if recs else False
            if prefix == "zero_ep" and failed:
                zero_failed = True
            if prefix == "policy_ep" and arrived:
                pol_arrivals += 1
            log(f"{log_tag}{ep} steps={len(recs)} mean_dist={np.mean(dists):.2f} "
                f"final_dist={dists[-1]:.2f} arrived={arrived} failed={failed}")
            ep_rows.append([f"{prefix}{ep}", len(recs), float(np.mean(dists)),
                            float(dists[-1]), arrived])
        rows.extend(ep_rows)
        return ep_rows

    zero_rows = run_tagged("zero_ep", zero_act, "ZERO", base_env)
    pbase_rows = run_tagged("pbase_ep", p_act, "PBASE", base_env)
    pol_rows = run_tagged("policy_ep", policy_act, "EP")

    env.close()

    def agg(rs):
        return (float(np.mean([r[3] for r in rs])), float(np.mean([r[2] for r in rs])))

    pol_final, pol_dist = agg(pol_rows)
    pb_final, pb_dist = agg(pbase_rows)
    zero_final, _ = agg(zero_rows)
    ok = (pol_final < 1.0 and pol_final < pb_final
          and pol_arrivals >= 3 and not zero_failed)
    verdict = "WAYPOINT_OK" if ok else "WAYPOINT_NOT_LEARNED"
    log(f"POLICY_MEAN_FINAL={pol_final:.2f} P_BASELINE_MEAN_FINAL={pb_final:.2f} "
        f"ARRIVALS={pol_arrivals}/{N_EPISODES} ZERO_CLEAN={not zero_failed} "
        f"VERDICT={verdict}")

    with open(os.path.join(out_dir, "eval_result.txt"), "w") as f:
        f.write(f"verdict={verdict}\n"
                f"policy_mean_final={pol_final:.3f}\n"
                f"p_baseline_mean_final={pb_final:.3f}\n"
                f"zero_baseline_mean_final={zero_final:.3f}\n"
                f"policy_arrivals={pol_arrivals}\n"
                f"policy_mean_dist={pol_dist:.3f}\n"
                f"p_baseline_mean_dist={pb_dist:.3f}\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with open(os.path.join(out_dir, "eval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "steps", "mean_dist", "final_dist", "arrived"])
        w.writerows(rows)
    log("EVAL_DONE")
except Exception:
    with open(r"D:\fpv\ops\eval_err.txt", "w") as f:
        f.write(traceback.format_exc())
    log("EVAL_FAILED see eval_err.txt")
    print(traceback.format_exc(), flush=True)

sys.stdout.flush()
os._exit(0)
