#!/usr/bin/env python
"""Export the trained SAC hover_v7 deterministic actor to ONNX and LiteRT (TFLite fp32).

Run from workspace root:
    ~/.venvs/pilot/bin/python scripts/export_actor.py

Outputs:
    export/onnx/sac_actor_hover_v7.onnx
    export/tflite/sac_actor_hover_v7.tflite
    export/savedmodel/sac_actor_hover_v7/   (Keras SavedModel, intermediate)

Architecture (read from the checkpoint, NOT hardcoded assumptions):
    obs[N,15] -> Linear(15,64) -> ReLU -> Linear(64,64) -> ReLU
              -> Linear(64,3) [mu] -> tanh -> action[N,3]
Note: the checkpoint's actor latent uses ReLU (SB3 default), not Tanh.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path("~/.hermes/plans/fpv-vision/artifacts/2026-11-08_hover_v7/policy_final.zip")
DATA = ROOT / "data" / "ccw_data_merged.npz"

ONNX_DIR = ROOT / "export" / "onnx"
TFLITE_DIR = ROOT / "export" / "tflite"
SAVEDMODEL_DIR = ROOT / "export" / "savedmodel" / "sac_actor_hover_v7"
ONNX_PATH = ONNX_DIR / "sac_actor_hover_v7.onnx"
TFLITE_PATH = TFLITE_DIR / "sac_actor_hover_v7.tflite"

OBS_DIM, ACT_DIM, HIDDEN = 15, 3, 64


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class DeterministicActor(nn.Module):
    """Deterministic policy: obs -> tanh(mu). Weights copied from the SB3 Actor."""

    def __init__(self, sb3_actor_state_dict: dict):
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.mu = nn.Linear(HIDDEN, ACT_DIM)
        with torch.no_grad():
            self.fc1.weight.copy_(sb3_actor_state_dict["latent_pi.0.weight"])
            self.fc1.bias.copy_(sb3_actor_state_dict["latent_pi.0.bias"])
            self.fc2.weight.copy_(sb3_actor_state_dict["latent_pi.2.weight"])
            self.fc2.bias.copy_(sb3_actor_state_dict["latent_pi.2.bias"])
            self.mu.weight.copy_(sb3_actor_state_dict["mu.weight"])
            self.mu.bias.copy_(sb3_actor_state_dict["mu.bias"])
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.mu(x))


def verify_hand_built(module: DeterministicActor, model: SAC, obs: np.ndarray) -> float:
    """Hand-built forward vs model.predict(obs, deterministic=True) on len(obs) samples."""
    with torch.no_grad():
        hand = module(torch.from_numpy(obs)).numpy()
    ref = np.stack([model.predict(o, deterministic=True)[0] for o in obs])
    diff = float(np.abs(hand - ref).max())
    print(f"[verify] hand-built vs SAC.predict on {len(obs)} samples: max abs diff = {diff:.3e}")
    assert diff < 1e-5, f"hand-built actor does not match predict: {diff}"
    return diff


def export_onnx(module: DeterministicActor) -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        str(ONNX_PATH),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "N"}, "action": {0: "N"}},
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
    # torch >=2.9 dynamo exporter may write weights as external .onnx.data; inline them
    ext = Path(str(ONNX_PATH) + ".data")
    if ext.exists():
        m = onnx.load(str(ONNX_PATH), load_external_data=True)
        onnx.save(m, str(ONNX_PATH))
        ext.unlink()
        onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
        print("[onnx] inlined external data into single file")
    print(f"[onnx] wrote {ONNX_PATH} sha256={sha256(ONNX_PATH)}")

    # quick smoke test through onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(123)
    x = rng.uniform(-1, 1, (64, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        ref = module(torch.from_numpy(x)).numpy()
    got = sess.run(["action"], {"obs": x})[0]
    d = float(np.abs(ref - got).max())
    print(f"[onnx] smoke parity (64 rows) max abs diff = {d:.3e}")
    assert d < 1e-5


def export_savedmodel_keras(module: DeterministicActor) -> Path:
    """Route (a): rebuild identical arch in Keras, copy weights, save SavedModel."""
    import tensorflow as tf

    sd = module.state_dict()
    inp = tf.keras.Input(shape=(OBS_DIM,), name="obs", dtype=tf.float32)
    x = tf.keras.layers.Dense(HIDDEN, activation="relu", name="fc1")(inp)
    x = tf.keras.layers.Dense(HIDDEN, activation="relu", name="fc2")(x)
    x = tf.keras.layers.Dense(ACT_DIM, activation=None, name="mu")(x)
    out = tf.keras.layers.Activation("tanh", name="action")(x)
    keras_model = tf.keras.Model(inp, out, name="sac_actor_hover_v7")

    def t(name):  # torch tensor -> float32 numpy
        return sd[name].detach().numpy().astype(np.float32)

    # PyTorch Linear weight is [out, in]; Keras kernel is [in, out]
    keras_model.get_layer("fc1").set_weights([t("fc1.weight").T, t("fc1.bias")])
    keras_model.get_layer("fc2").set_weights([t("fc2.weight").T, t("fc2.bias")])
    keras_model.get_layer("mu").set_weights([t("mu.weight").T, t("mu.bias")])

    # exact-weight check before serialising
    rng = np.random.default_rng(321)
    x = rng.uniform(-1, 1, (64, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        ref = module(torch.from_numpy(x)).numpy()
    got = keras_model.predict(x, verbose=0)
    d = float(np.abs(ref - got).max())
    print(f"[keras] exact-weight parity (64 rows) max abs diff = {d:.3e}")
    assert d < 1e-6

    if SAVEDMODEL_DIR.exists():
        import shutil

        shutil.rmtree(SAVEDMODEL_DIR)
    keras_model.export(SAVEDMODEL_DIR)
    return SAVEDMODEL_DIR


def convert_tflite(savedmodel_dir: Path) -> None:
    TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))
    converter.optimizations = []  # float32, no optimization
    tflite_model = converter.convert()
    TFLITE_PATH.write_bytes(tflite_model)
    print(f"[tflite] wrote {TFLITE_PATH} ({len(tflite_model)} bytes) sha256={sha256(TFLITE_PATH)}")

    # quick smoke test with the interpreter
    interp = _make_interpreter()
    interp.resize_tensor_input(interp.get_input_details()[0]["index"], (64, OBS_DIM))
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    rng = np.random.default_rng(654)
    x = rng.uniform(-1, 1, (64, OBS_DIM)).astype(np.float32)
    interp.set_tensor(in_idx, x)
    interp.invoke()
    got = interp.get_tensor(out_idx)
    with torch.no_grad():
        ref = DeterministicActor(_sb3_state_dict())(torch.from_numpy(x)).numpy()
    d = float(np.abs(ref - got).max())
    print(f"[tflite] smoke parity (64 rows) max abs diff = {d:.3e}")


def _sb3_state_dict() -> dict:
    return SAC.load(str(CKPT), device="cpu").actor.state_dict()


def _make_interpreter():
    """Prefer standalone tflite-runtime, fall back to tf.lite. Keep fp32 (no fp16 delegates)."""
    try:
        import tflite_runtime.interpreter as tflite

        print("[tflite] runner: tflite-runtime")
        return tflite.Interpreter(model_path=str(TFLITE_PATH), num_threads=1)
    except ImportError:
        import tensorflow as tf

        print("[tflite] runner: tf.lite.Interpreter")
        return tf.lite.Interpreter(model_path=str(TFLITE_PATH), num_threads=1)


def main() -> int:
    model = SAC.load(str(CKPT), device="cpu")
    sd = model.actor.state_dict()
    print("[actor state_dict manifest]")
    for k, v in sd.items():
        print(f"  {k:20s} {tuple(v.shape)}")

    obs = np.load(str(DATA))["obs"][:50].astype(np.float32)
    module = DeterministicActor(sd)
    verify_hand_built(module, model, obs)

    export_onnx(module)
    sm_dir = export_savedmodel_keras(module)
    convert_tflite(sm_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
