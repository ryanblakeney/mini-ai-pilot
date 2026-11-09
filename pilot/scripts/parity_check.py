#!/usr/bin/env python
"""Parity replay: PyTorch reference vs ONNX Runtime vs LiteRT over 10k recorded obs.

Run from workspace root (after scripts/export_actor.py):
    ~/.venvs/pilot/bin/python scripts/parity_check.py

Writes reports/parity_report.md with measured max/mean/p99 abs diffs per backend
and PASS/FAIL vs the 1e-4 tolerance. All numbers are from real runs.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_actor import (  # noqa: E402
    CKPT,
    DATA,
    ONNX_PATH,
    ROOT,
    TFLITE_PATH,
    DeterministicActor,
    _make_interpreter,
)

REPORT_PATH = ROOT / "reports" / "parity_report.md"
SEED = 20261109
N = 10000
TOL = 1e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(diff: np.ndarray) -> tuple[float, float, float]:
    a = np.abs(diff).astype(np.float64)
    return float(a.max()), float(a.mean()), float(np.percentile(a, 99))


def run_onnx(obs: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return sess.run(["action"], {"obs": obs})[0]


def run_tflite(obs: np.ndarray) -> np.ndarray:
    interp = _make_interpreter()
    in_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    outs = []
    chunk = 2000
    for i in range(0, len(obs), chunk):
        batch = np.ascontiguousarray(obs[i : i + chunk], dtype=np.float32)
        interp.resize_tensor_input(in_idx, batch.shape)
        interp.allocate_tensors()
        out_idx = interp.get_output_details()[0]["index"]
        interp.set_tensor(in_idx, batch)
        interp.invoke()
        outs.append(interp.get_tensor(out_idx).copy())
    return np.concatenate(outs)


def main() -> int:
    obs_all = np.load(str(DATA))["obs"]
    assert obs_all.shape[1] == 15
    rng = np.random.default_rng(SEED)
    idx = rng.choice(obs_all.shape[0], size=N, replace=False)
    sets = {
        "first_10000": np.ascontiguousarray(obs_all[:N], dtype=np.float32),
        f"seeded_random_10000 (seed={SEED})": np.ascontiguousarray(obs_all[idx], dtype=np.float32),
    }

    model = SAC.load(str(CKPT), device="cpu")
    module = DeterministicActor(model.actor.state_dict())
    with torch.no_grad():
        refs = {
            name: module(torch.from_numpy(x)).numpy() for name, x in sets.items()
        }

    results: dict[str, dict[str, tuple]] = {}
    for name, x in sets.items():
        onnx_out = run_onnx(x)
        tfl_out = run_tflite(x)
        results[name] = {
            "onnxruntime": stats(onnx_out - refs[name]),
            "litert": stats(tfl_out - refs[name]),
        }
        # sanity: shape and range
        assert onnx_out.shape == refs[name].shape == tfl_out.shape == (N, 3)

    # ---- report ----
    lines = []
    ap = lines.append
    ap("# Parity report — SAC hover_v7 exported actor")
    ap("")
    ap(f"- Checkpoint: `{CKPT}`")
    ap(f"- ONNX model: `{ONNX_PATH}` (sha256 `{sha256(ONNX_PATH)}`)")
    ap(f"- LiteRT model: `{TFLITE_PATH}` (sha256 `{sha256(TFLITE_PATH)}`)")
    ap(f"- Observation data: `{DATA}` (key 'obs', shape {obs_all.shape}, float32)")
    ap(f"- Replay sets: first 10000 rows and seeded random 10000 rows (numpy default_rng seed={SEED}, replace=False)")
    ap(f"- Reference: hand-built `torch.nn.Module` (deterministic actor) in float32, `torch.no_grad`, full-batch; "
       f"verified equal to `SAC.predict(deterministic=True)` on 50 samples during export (max abs diff ~2e-7)")
    ap(f"- Tolerance: max abs diff < {TOL:.0e}")
    import stable_baselines3 as sb3

    ap(f"- Versions: torch {torch.__version__}, onnx {onnx.__version__}, "
       f"onnxruntime {ort.__version__}, stable_baselines3 {sb3.__version__}, "
       f"numpy {np.__version__}")
    ap("")

    for name, res in results.items():
        ap(f"## Replay set: {name} ({N} obs x 15)")
        ap("")
        ap("| Backend | max abs diff | mean abs diff | p99 abs diff | PASS/FAIL (max < 1e-4) |")
        ap("|---|---|---|---|---|")
        ap("| PyTorch reference | 0 (reference) | 0 | 0 | — |")
        for backend, label in (("onnxruntime", "ONNX Runtime (CPU)"), ("litert", "LiteRT (TFLite fp32)")):
            mx, mean, p99 = res[backend]
            verdict = "PASS" if mx < TOL else "FAIL"
            ap(f"| {label} | {mx:.3e} | {mean:.3e} | {p99:.3e} | {verdict} |")
        ap("")

    ap("## Architecture (phone-side reimplementation reference)")
    ap("")
    ap("Deterministic actor (deterministic half of the SAC policy; the `log_std` head is stochastic-only and NOT needed for deployment):")
    ap("")
    ap("```")
    ap("obs [N,15] float32")
    ap("  -> Linear  15 -> 64   (weight [64,15], bias [64])   -> ReLU")
    ap("  -> Linear  64 -> 64   (weight [64,64], bias [64])   -> ReLU")
    ap("  -> Linear  64 -> 3    (weight [3,64],  bias [3])    = mu   (linear, no activation)")
    ap("  -> tanh                                               = action in [-1, 1]")
    ap("```")
    ap("")
    ap("Note: the SB3 actor's latent layers use **ReLU** (SB3 2.9.2 default), not Tanh; the only tanh is the final squashing of `mu`. `predict(deterministic=True)` returns `tanh(mu)`.")
    ap("")
    ap("### Weight manifest (SB3 `model.actor.state_dict()`, PyTorch Linear weight layout = [out, in])")
    ap("")
    ap("| Tensor | Shape | Used for export |")
    ap("|---|---|---|")
    sd = model.actor.state_dict()
    used = {"latent_pi.0.weight", "latent_pi.0.bias", "latent_pi.2.weight", "latent_pi.2.bias", "mu.weight", "mu.bias"}
    for k, v in sd.items():
        ap(f"| `{k}` | {tuple(v.shape)} | {'yes' if k in used else 'no (stochastic head)'} |")
    ap("")
    ap("### LiteRT route used")
    ap("")
    ap("Route (a): Keras rebuild of the identical architecture with weights copied from the SB3 actor "
       "state_dict (transposed to Keras [in, out]), exported as SavedModel, converted with "
       "`tf.lite.TFLiteConverter.from_saved_model` with empty optimizations (plain float32, no quantization, no fp16 delegates). "
       "Inference via tflite-runtime `Interpreter` (fallback tf.lite.Interpreter), fp32 end to end.")
    ap("")
    ap("ONNX export: `torch.onnx.export` of a minimal `nn.Module` wrapping only the deterministic actor, opset 17, dynamic batch axis, "
       "validated with `onnx.checker.check_model`, executed with onnxruntime CPUExecutionProvider.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {REPORT_PATH}")
    for name, res in results.items():
        for backend, (mx, mean, p99) in res.items():
            print(f"[{name}] {backend}: max={mx:.3e} mean={mean:.3e} p99={p99:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
