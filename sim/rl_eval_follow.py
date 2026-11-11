"""S7 follow-a-target evaluation (ProjectAirSim Blocks rig).

Launch ONLY via the FPVEval route (task_eval_v4.ps1 FPV_WAY branch pointed at
this script) inside a console session with Blocks already up. Mirrors
rl_eval_waypoint.py / rl_eval_obstacle.py structure and house style (EP /
RESULT lines to the ops log + eval.csv + eval_result.txt, this time into the
FPV_RUN dir), scoring CENTER ERROR, loss time and collisions.

Runs 4 episodes x 120 steps per condition on SHARED seeds 0..3 (same seed =>
same target script AND same detector-noise stream -- the env draws latency,
DR params and the script from the seeded rng in a fixed order), three
conditions:
  policy   deterministic predict from FPV_POLICY (SAC.load, PPO fallback)
  zero     all-zero action baseline
  ibvs     ibvs_follow.IBVSFollow fed from the SAME obs box features
  ibvs_avoid  ibvs_follow.AvoidIBVSFollow (D16: IBVS + depth-grid forward
            veto) fed the SAME obs: box feats + obs[22:37] grid + obs[8] yaw
Verdict is baked in:
  FOLLOW_OK iff policy_mean_center < ibvs_mean_center
          AND policy_mean_center < zero_mean_center
          AND policy_collisions == 0
          AND policy_mean_loss <= ibvs_mean_loss
Per-visible-step reporting per spec risk 3: loss (frag) time is a first-class
column, not folded into center err. os._exit at end (Win teardown segfault).
Baseline-only mode: if no policy file exists at FPV_POLICY, the stable_baselines3
import and model load are skipped, only zero + ibvs + ibvs_avoid run,
verdict=BASELINE_RECORDED.
"""
import csv
import os
import sys
import time
import traceback

RL_DIR = r"D:\fpv\rl"
RUN_DIR = os.environ.get("FPV_RUN", RL_DIR + r"\runs\follow_v1")
POLICY = os.environ.get("FPV_POLICY", os.path.join(RUN_DIR, "policy_final.zip"))
baseline_only = (not os.path.exists(POLICY))
OUT = r"D:\fpv\ops\eval_out.txt"
N_EPISODES = 4
MAX_STEPS_EVAL = 120

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

    from follow_env import FollowDroneEnv
    from ibvs_follow import AvoidIBVSFollow, IBVSFollow

    env = FollowDroneEnv(step_dt=0.5, max_steps=MAX_STEPS_EVAL)
    n_act = int(env.action_space.shape[0])

    if not baseline_only:
        # policy load: SAC first (house algo), PPO fallback
        from stable_baselines3 import PPO, SAC
        try:
            model = SAC.load(POLICY, env=env)
            algo = "SAC"
        except Exception:
            model = PPO.load(POLICY, env=env)
            algo = "PPO"
        log(f"POLICY={POLICY} ALGO={algo} COLL_ROUTE="
            f"{getattr(env, 'collision_route', '?')} YAW_DIMS={n_act}")
    else:
        log(f"BASELINE_ONLY no policy at {POLICY} - measuring zero + ibvs "
            f"+ ibvs_avoid")

    def run_episode(acting, seed, e=None):
        """One episode; list of (center_err, dist, in_view, loss, collided)."""
        e = env if e is None else e
        obs, info = e.reset(seed=seed)
        recs = []
        term = trunc = False
        while not (term or trunc):
            obs, r, term, trunc, info = e.step(acting(obs))
            recs.append((info["center_err"], info["dist"],
                         bool(info["in_view"]), info["loss_steps"],
                         bool(info["collided"])))
        return recs

    def policy_act(obs):
        a, _ = model.predict(obs, deterministic=True)
        return a

    def zero_act(obs):
        return np.zeros(n_act, dtype=np.float32)

    ibvs = IBVSFollow()
    avoid = AvoidIBVSFollow()

    def ibvs_inputs(obs):
        # box features from the SAME obs the policy sees (D13 obs v2 layout):
        # [15]cx_err [16]cy_err [17]clip(log(est/desired)) [18]desired/15
        # [19]range_valid [20]conf [21]age/10, yaw=obs[8]. Reconstruct the
        # range estimate from the obs itself => strictly same-seed,
        # obs-faithful comparison with the learned policy (D14).
        feats = obs[15:22]
        cx_err, cy_err, log_r = float(feats[0]), float(feats[1]), float(feats[2])
        desired = float(feats[3]) * 15.0
        valid = bool(feats[4] > 0.5)
        conf, age10 = float(feats[5]), float(feats[6])
        box = None if conf <= 0.0 else {
            "cx": env._cam["cx0"] + cx_err * env._cam["cx0"],
            "cy": env._cam["cy0"] + cy_err * env._cam["cy0"],
            "age": age10 * 10.0}
        est = desired * float(np.exp(log_r))
        return box, est, valid, desired

    def ibvs_act(obs):
        box, est, valid, desired = ibvs_inputs(obs)
        a = ibvs.action(box, est, valid, desired, drone_yaw=float(obs[8]))
        return a[:n_act]

    def avoid_act(obs):
        # D16: same obs stream + the agent's own depth grid obs[22:37]
        box, est, valid, desired = ibvs_inputs(obs)
        a = avoid.action(box, est, valid, desired, drone_yaw=float(obs[8]),
                         grid=obs[22:37])
        return a[:n_act]

    rows = []

    def run_tagged(prefix, acting, log_tag, reset=None):
        ep_rows = []
        for ep in range(N_EPISODES):
            if reset is not None:
                reset()
            recs = run_episode(acting, ep)
            cents = [x[0] for x in recs]
            dists = [x[1] for x in recs]
            loss = recs[-1][3] if recs else MAX_STEPS_EVAL
            coll = any(x[4] for x in recs)
            kind = env._script["kind"] if env._script else "?"
            log(f"{log_tag}{ep} seed={ep} kind={kind} steps={len(recs)} "
                f"mean_center={np.mean(cents):.3f} final_center={cents[-1]:.3f} "
                f"mean_dist={np.mean(dists):.2f} loss_steps={loss} "
                f"collided={coll}")
            ep_rows.append([f"{prefix}{ep}", len(recs), float(np.mean(cents)),
                            float(cents[-1]), float(np.mean(dists)),
                            int(loss), coll])
        rows.extend(ep_rows)
        return ep_rows

    zero_rows = run_tagged("zero_ep", zero_act, "ZERO")
    ibvs_rows = run_tagged("ibvs_ep", ibvs_act, "IBVS", reset=ibvs.reset)
    avoid_rows = run_tagged("avoid_ep", avoid_act, "AVOID", reset=avoid.reset)
    pol_rows = [] if baseline_only else run_tagged("policy_ep", policy_act, "EP")

    env.close()

    def agg(rs):
        return (float(np.mean([r[2] for r in rs])),
                float(np.mean([r[5] for r in rs])),
                int(sum(bool(r[6]) for r in rs)))

    ibv_c, ibv_loss, ibv_coll = agg(ibvs_rows)
    av_c, av_loss, av_coll = agg(avoid_rows)
    zer_c, zer_loss, zer_coll = agg(zero_rows)
    pol_c = pol_loss = pol_coll = None  # unused in baseline-only mode
    if baseline_only:
        verdict = "BASELINE_RECORDED"
        log(f"POLICY_MEAN_CENTER=NA IBVS_MEAN_CENTER={ibv_c:.3f} "
            f"AVOID_MEAN_CENTER={av_c:.3f} "
            f"ZERO_MEAN_CENTER={zer_c:.3f} POLICY_LOSS=NA "
            f"IBVS_LOSS={ibv_loss} AVOID_LOSS={av_loss} ZERO_LOSS={zer_loss} "
            f"COLLISIONS=NA "
            f"IBVS_COLLISIONS={ibv_coll}/{N_EPISODES} "
            f"AVOID_COLLISIONS={av_coll}/{N_EPISODES} "
            f"ZERO_COLLISIONS={zer_coll}/{N_EPISODES} VERDICT={verdict}")
    else:
        pol_c, pol_loss, pol_coll = agg(pol_rows)
        ok = (pol_c < ibv_c and pol_c < zer_c and pol_coll == 0
              and pol_loss <= ibv_loss)
        verdict = "FOLLOW_OK" if ok else "FOLLOW_NOT_LEARNED"
        log(f"POLICY_MEAN_CENTER={pol_c:.3f} IBVS_MEAN_CENTER={ibv_c:.3f} "
            f"AVOID_MEAN_CENTER={av_c:.3f} "
            f"ZERO_MEAN_CENTER={zer_c:.3f} POLICY_LOSS={pol_loss} "
            f"IBVS_LOSS={ibv_loss} AVOID_LOSS={av_loss} ZERO_LOSS={zer_loss} "
            f"COLLISIONS={pol_coll}/{N_EPISODES} "
            f"IBVS_COLLISIONS={ibv_coll}/{N_EPISODES} "
            f"AVOID_COLLISIONS={av_coll}/{N_EPISODES} "
            f"ZERO_COLLISIONS={zer_coll}/{N_EPISODES} VERDICT={verdict}")

    with open(os.path.join(RUN_DIR, "eval_result.txt"), "w") as f:
        if baseline_only:
            f.write(f"verdict={verdict}\n"
                    f"ibvs_mean_center={ibv_c:.4f}\n"
                    f"ibvs_avoid_mean_center={av_c:.4f}\n"
                    f"zero_mean_center={zer_c:.4f}\n"
                    f"ibvs_loss_steps={ibv_loss}\n"
                    f"ibvs_avoid_loss_steps={av_loss}\n"
                    f"zero_loss_steps={zer_loss}\n"
                    f"ibvs_collisions={ibv_coll}\n"
                    f"ibvs_avoid_collisions={av_coll}\n"
                    f"zero_collisions={zer_coll}\n")
        else:
            f.write(f"verdict={verdict}\n"
                    f"policy_mean_center={pol_c:.4f}\n"
                    f"ibvs_mean_center={ibv_c:.4f}\n"
                    f"ibvs_avoid_mean_center={av_c:.4f}\n"
                    f"zero_mean_center={zer_c:.4f}\n"
                    f"policy_loss_steps={pol_loss}\n"
                    f"ibvs_loss_steps={ibv_loss}\n"
                    f"ibvs_avoid_loss_steps={av_loss}\n"
                    f"zero_loss_steps={zer_loss}\n"
                    f"policy_collisions={pol_coll}\n"
                    f"ibvs_collisions={ibv_coll}\n"
                    f"ibvs_avoid_collisions={av_coll}\n"
                    f"zero_collisions={zer_coll}\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with open(os.path.join(RUN_DIR, "eval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "steps", "mean_center", "final_center",
                    "mean_dist", "loss_steps", "collided"])
        w.writerows(rows)
    log("EVAL_DONE")
except Exception:
    with open(r"D:\fpv\ops\eval_err.txt", "w") as f:
        f.write(traceback.format_exc())
    log("EVAL_FAILED see eval_err.txt")
    print(traceback.format_exc(), flush=True)

sys.stdout.flush()
os._exit(0)
