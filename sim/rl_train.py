"""Q3 stage-1 PPO training: BlocksDroneEnv hover-under-disturbance, SB3 MlpPolicy.

FPV_WAY=1 (2026-11-08): stage-2 waypoint-reach -- WaypointDroneEnv + the proven
v7 SAC recipe, SuccessWatch rew_thresh=55/len_thresh=20. Unset = stage-1 hover,
byte-identical to before.

Launch ONLY via FPVTrain scheduled task (console session, Blocks already up).
cwd must be ...\\example_user_scripts (World scene path resolution).
Heartbeat file D:\\fpv\\ops\\train_status.txt is polled by overnight patrol.
StS lessons applied: CSV metrics + final policy save + os._exit at the end.
"""
import os
import sys
import time
import traceback

RL_DIR = r"D:\fpv\rl"
RUN_DIR = os.environ.get("FPV_RUN", RL_DIR + r"\runs\hover_v1")
STATUS = r"D:\fpv\ops\train_status.txt"
TOTAL_STEPS = int(os.environ.get("FPV_STEPS", "2000"))

sys.path.insert(0, RL_DIR)


def hb(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with open(STATUS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


try:
    os.makedirs(os.path.join(RUN_DIR, "ckpt"), exist_ok=True)
    open(STATUS, "w").close()
    hb(f"TRAIN_BEGIN steps={TOTAL_STEPS}")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor

    from gym_env import BlocksDroneEnv

    WAY = os.environ.get("FPV_WAY") == "1"
    if WAY:
        # stage 2 (2026-11-08): waypoint-reach, subclass of BlocksDroneEnv.
        from waypoint_env import WaypointDroneEnv
        env = Monitor(WaypointDroneEnv(step_dt=0.5, max_steps=100))
    else:
        env = Monitor(BlocksDroneEnv(step_dt=0.5, max_steps=100))
    logger = configure(RUN_DIR, ["csv", "stdout"])

    class Heartbeat(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 100 == 0:
                hb(f"STEP={self.num_timesteps}")
            return True

    class MindDump(BaseCallback):
        """the owner wants to watch the net being formed: dump policy weight tensors
        every 100 env steps so the lab poller can animate them live."""
        def __init__(self, dump_dir):
            super().__init__()
            self.dir = dump_dir
            os.makedirs(dump_dir, exist_ok=True)

        def _on_step(self):
            if self.num_timesteps % 100 == 0 and self.num_timesteps > 0:
                try:
                    import numpy as np
                    sd = self.model.policy.state_dict()
                    arrays = {k: v.detach().cpu().numpy() for k, v in sd.items()
                              if v.is_floating_point()}
                    arrays["_steps"] = np.array([self.num_timesteps])
                    np.savez_compressed(os.path.join(self.dir, "latest.npz"), **arrays)
                    if self.num_timesteps % 1000 == 0:
                        np.savez_compressed(
                            os.path.join(self.dir, "step_%06d.npz" % self.num_timesteps),
                            **arrays)
                except Exception:
                    # SAC state_dict quirks (or anything else) must never kill the run.
                    pass
            return True

    class SuccessWatch(BaseCallback):
        """v6: flag suspected success once from Monitor episode stats.
        Defensive by design: a callback bug must never kill the run.
        NOTE (probed on SB3 2.9.2): callbacks expose the vec env as
        self.training_env (a DummyVecEnv); Monitor episode stats live at
        self.training_env.envs[0]."""

        def __init__(self, rew_thresh=-45.0, len_thresh=80.0):
            super().__init__()
            # v6 defaults (-45.0 / 80.0); v7 reward v2 rescales episode reward
            # positive, so v7 passes rew_thresh=70.0, len_thresh=95.0.
            self.rew_thresh = rew_thresh
            self.len_thresh = len_thresh

        def _monitor(self):
            # SB3 BaseCallback exposes the vec env as self.training_env
            # (probed 2.9.2: self.env does not exist on callbacks).
            venv = getattr(self, "training_env", None) or getattr(self, "env", None)
            if venv is None:
                return None
            if hasattr(venv, "get_episode_rewards"):
                return venv
            for sub in list(getattr(venv, "envs", None) or []):
                if sub is not None and hasattr(sub, "get_episode_rewards"):
                    return sub
            return None

        def _on_step(self):
            try:
                if not getattr(self, "_fired", False):
                    mon = self._monitor()
                    if mon is not None:
                        rews = mon.get_episode_rewards()
                        lens = mon.get_episode_lengths()
                        if len(rews) >= 10 and len(lens) >= 10:
                            r10 = sum(rews[-10:]) / 10.0
                            l10 = sum(lens[-10:]) / 10.0
                            if r10 > self.rew_thresh and l10 >= self.len_thresh:
                                hb(f"SUCCESS_SUSPECT steps={self.num_timesteps} rew={r10:.1f} len={l10:.1f}")
                                setattr(self, "_fired", True)
            except Exception:
                pass
            return True

    V7 = os.environ.get("FPV_V7") == "1"
    V6 = os.environ.get("FPV_V6") == "1"
    V5 = os.environ.get("FPV_V5") == "1"
    if WAY:
        # stage 2 waypoint-reach: proven v7 SAC recipe verbatim; SuccessWatch
        # thresholds for reward v3 -- arrivals end episodes early, so a learned
        # policy shows rew ~60+ at mean len ~20 (see waypoint_env.py header).
        from stable_baselines3 import SAC
        model = SAC(
            "MlpPolicy", env, learning_starts=500, buffer_size=50000,
            batch_size=256, train_freq=1, gradient_steps=1, ent_coef="auto",
            gamma=0.98, policy_kwargs=dict(net_arch=[64, 64]),
            verbose=1, seed=0)
        cb = [Heartbeat(), MindDump(os.path.join(RUN_DIR, "mind")),
              CheckpointCallback(save_freq=2500, save_path=os.path.join(RUN_DIR, "ckpt"),
                                 name_prefix="ckpt_"),
              SuccessWatch(rew_thresh=55.0, len_thresh=20.0)]
    elif V7:
        # v7 (2026-11-08 review): reward v2 in gym_env.py removed the early-
        # termination exploit (survival reward + -10 terminal penalty); off-policy
        # SAC per reviewer recipe replays the rare near-hover states instead of
        # discarding them every rollout.
        from stable_baselines3 import SAC
        model = SAC(
            "MlpPolicy", env, learning_starts=500, buffer_size=50000,
            batch_size=256, train_freq=1, gradient_steps=1, ent_coef="auto",
            gamma=0.98, policy_kwargs=dict(net_arch=[64, 64]),
            verbose=1, seed=0)
        cb = [Heartbeat(), MindDump(os.path.join(RUN_DIR, "mind")),
              CheckpointCallback(save_freq=2500, save_path=os.path.join(RUN_DIR, "ckpt"),
                                 name_prefix="ckpt_"),
              SuccessWatch(rew_thresh=70.0, len_thresh=95.0)]
    elif V6:
        # v6 (2026-11-08): v5 metrics proved the unsquashed unit-variance Normal
        # policy never left init (std 1.000->1.017, entropy == standard normal,
        # approx_kl~0): ±2 m/s exploration flung the drone to the 8 m term box
        # every ~25 steps. Fix: squashed exploration in the [-1,1] action box,
        # init std 0.4, modest ent_coef, periodic checkpoints.
        # NOTE (SB3 2.9.2): no init_noise_std kwarg -> use log_std_init=log(0.4);
        # squash_output=True requires gSDE, so use_sde=True on the PPO itself
        # (OnPolicyAlgorithm already forwards use_sde to the policy — it must
        # NOT also appear in policy_kwargs).
        import math
        model = PPO(
            "MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=4,
            ent_coef=0.005, gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[64, 64], squash_output=True,
                               log_std_init=math.log(0.4)),
            use_sde=True,
            verbose=1, seed=0)
        cb = [Heartbeat(), MindDump(os.path.join(RUN_DIR, "mind")),
              CheckpointCallback(save_freq=2500, save_path=os.path.join(RUN_DIR, "ckpt"),
                                 name_prefix="ckpt_"),
              SuccessWatch()]
    elif V5:
        # v5 (owner: continue training, 2026-11-08): C-option = curriculum + capacity.
        # Disturbance ramps 0 -> full over first 35% of steps; bigger net, more
        # exploration; eval bar unchanged (full-disturb regime via FPVEvalV4).
        base_scale = env.unwrapped.disturb_scale
        base_every = env.unwrapped.disturb_every

        class Curriculum(BaseCallback):
            def _on_step(self):
                prog = min(1.0, self.num_timesteps / max(1.0, TOTAL_STEPS * 0.35))
                env.unwrapped.disturb_scale = base_scale * prog
                if self.num_timesteps % 400 == 0:
                    hb(f"CURR prog={prog:.2f} scale={base_scale * prog:.2f}")
                return True

        model = PPO(
            "MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=4, ent_coef=0.03,
            gamma=0.99, gae_lambda=0.95, learning_rate=2e-4,
            policy_kwargs=dict(net_arch=[128, 128]), verbose=1, seed=0)
        cb = [Heartbeat(), Curriculum(), MindDump(os.path.join(RUN_DIR, "mind"))]
    else:
        model = PPO(
            "MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=4, ent_coef=0.01,
            gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[64, 64]), verbose=1, seed=0)
        cb = [Heartbeat(), MindDump(os.path.join(RUN_DIR, "mind"))]
    model.set_logger(logger)

    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)

    model.save(os.path.join(RUN_DIR, "policy_final"))
    hb(f"TRAIN_DONE steps={model.num_timesteps}")
except Exception:
    with open(r"D:\fpv\ops\train_err.txt", "w") as f:
        f.write(traceback.format_exc())
    hb("TRAIN_FAILED see train_err.txt")
    print(traceback.format_exc(), flush=True)

sys.stdout.flush()
os._exit(0)  # windows proactor teardown segfaults otherwise (proven 00:45)
