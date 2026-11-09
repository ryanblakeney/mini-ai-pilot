# Parity report — SAC hover_v7 exported actor

- Checkpoint: `~/.hermes/plans/fpv-vision/artifacts/2026-11-08_hover_v7/policy_final.zip`
- ONNX model: `~/.hermes/kanban/boards/fpv-vision/workspaces/t_5584c011/export/onnx/sac_actor_hover_v7.onnx` (sha256 `33c4f9c1db4664bfb702ebc6cc8ee6382e60de5a329d8e0e7b7f796f36d56c39`)
- LiteRT model: `~/.hermes/kanban/boards/fpv-vision/workspaces/t_5584c011/export/tflite/sac_actor_hover_v7.tflite` (sha256 `c52662a1101046dcf26af49e86f455fc3df6332d5db214465cb27837bca1a985`)
- Observation data: `~/.hermes/kanban/boards/fpv-vision/workspaces/t_5584c011/data/ccw_data_merged.npz` (key 'obs', shape (65133, 15), float32)
- Replay sets: first 10000 rows and seeded random 10000 rows (numpy default_rng seed=20261109, replace=False)
- Reference: hand-built `torch.nn.Module` (deterministic actor) in float32, `torch.no_grad`, full-batch; verified equal to `SAC.predict(deterministic=True)` on 50 samples during export (max abs diff ~2e-7)
- Tolerance: max abs diff < 1e-04
- Versions: torch 2.15.0+cpu, onnx 1.23.2, onnxruntime 1.31.1, stable_baselines3 2.9.2, numpy 2.5.4

## Replay set: first_10000 (10000 obs x 15)

| Backend | max abs diff | mean abs diff | p99 abs diff | PASS/FAIL (max < 1e-4) |
|---|---|---|---|---|
| PyTorch reference | 0 (reference) | 0 | 0 | — |
| ONNX Runtime (CPU) | 2.682e-07 | 3.902e-08 | 1.490e-07 | PASS |
| LiteRT (TFLite fp32) | 4.470e-07 | 7.157e-08 | 2.384e-07 | PASS |

## Replay set: seeded_random_10000 (seed=20261109) (10000 obs x 15)

| Backend | max abs diff | mean abs diff | p99 abs diff | PASS/FAIL (max < 1e-4) |
|---|---|---|---|---|
| PyTorch reference | 0 (reference) | 0 | 0 | — |
| ONNX Runtime (CPU) | 2.980e-07 | 3.903e-08 | 1.416e-07 | PASS |
| LiteRT (TFLite fp32) | 4.470e-07 | 7.113e-08 | 2.384e-07 | PASS |

## Architecture (phone-side reimplementation reference)

Deterministic actor (deterministic half of the SAC policy; the `log_std` head is stochastic-only and NOT needed for deployment):

```
obs [N,15] float32
  -> Linear  15 -> 64   (weight [64,15], bias [64])   -> ReLU
  -> Linear  64 -> 64   (weight [64,64], bias [64])   -> ReLU
  -> Linear  64 -> 3    (weight [3,64],  bias [3])    = mu   (linear, no activation)
  -> tanh                                               = action in [-1, 1]
```

Note: the SB3 actor's latent layers use **ReLU** (SB3 2.9.2 default), not Tanh; the only tanh is the final squashing of `mu`. `predict(deterministic=True)` returns `tanh(mu)`.

### Weight manifest (SB3 `model.actor.state_dict()`, PyTorch Linear weight layout = [out, in])

| Tensor | Shape | Used for export |
|---|---|---|
| `latent_pi.0.weight` | (64, 15) | yes |
| `latent_pi.0.bias` | (64,) | yes |
| `latent_pi.2.weight` | (64, 64) | yes |
| `latent_pi.2.bias` | (64,) | yes |
| `mu.weight` | (3, 64) | yes |
| `mu.bias` | (3,) | yes |
| `log_std.weight` | (3, 64) | no (stochastic head) |
| `log_std.bias` | (3,) | no (stochastic head) |

### LiteRT route used

Route (a): Keras rebuild of the identical architecture with weights copied from the SB3 actor state_dict (transposed to Keras [in, out]), exported as SavedModel, converted with `tf.lite.TFLiteConverter.from_saved_model` with empty optimizations (plain float32, no quantization, no fp16 delegates). Inference via tflite-runtime `Interpreter` (fallback tf.lite.Interpreter), fp32 end to end.

ONNX export: `torch.onnx.export` of a minimal `nn.Module` wrapping only the deterministic actor, opset 17, dynamic batch axis, validated with `onnx.checker.check_model`, executed with onnxruntime CPUExecutionProvider.
