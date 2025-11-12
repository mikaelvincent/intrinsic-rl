# Intrinsic Reward Integration (RIDE + R-IAC + Gating)

Research codebase for intrinsic-motivation RL that combines **RIDE** (impact) and **R-IAC** (region learning progress) with a **randomness-aware gate**. It includes a PPO backbone, pluggable intrinsic modules (vanilla, ICM, RND, RIDE, RIAC, proposed), multi-seed evaluation, and plotting tools.

---

## Quickstart

### 1) Environment

- **Python:** 3.10 (as pinned in `pyproject.toml`)
- **OS:** Linux/macOS/Windows. For MuJoCo headless on Linux set `MUJOCO_GL=egl`.

> Tip: Use a fresh venv.
```bash
python -m venv .venv && source .venv/bin/activate
```

### 2) Install

From the repo root:

```bash
# Core library (editable)
pip install -e .

# Optional dev/test tools
pip install -e ".[dev]"

# Optional: enable TensorBoard logging support
pip install -e ".[tb]"

# Optional env extras:
# - Box2D envs (BipedalWalker, CarRacing)
pip install -e ".[box2d]"

# - MuJoCo envs (Ant, HalfCheetah, Humanoid)
pip install -e ".[mujoco]"
```

> Why extras? Keeping heavy environment backends optional makes the base install fast
> and avoids pulling large native wheels unless you actually need those tasks.

> GPU is optional. For MuJoCo control on Linux servers:

```bash
export MUJOCO_GL=egl
```

### 3) Run a tiny smoke training (10k steps)

**Vanilla PPO on MountainCar:**

```bash
irl-train --config code/configs/mountaincar_vanilla.yaml --total-steps 10000
```

**Proposed method (RIDE + R-IAC + gate) on MountainCar:**

```bash
irl-train --config code/configs/mountaincar_ride.yaml --method proposed --total-steps 10000
```

**BipedalWalker baselines (short configs):**

```bash
irl-train --config code/configs/bipedal_rnd.yaml --total-steps 10000
irl-train --config code/configs/bipedal_riac.yaml --total-steps 10000
irl-train --config code/configs/bipedal_ride.yaml --total-steps 10000
```

**MuJoCo (Ant/HalfCheetah/Humanoid) — proposed:**

```bash
# Ensure MUJOCO_GL and that you've installed the extras:
# pip install -e ".[mujoco]"
export MUJOCO_GL=egl   # Linux headless
irl-train --config code/configs/mujoco/ant_proposed.yaml        --total-steps 100000
irl-train --config code/configs/mujoco/halfcheetah_proposed.yaml --total-steps 100000
irl-train --config code/configs/mujoco/humanoid_proposed.yaml   --total-steps 100000
```

### 4) Resume training safely

The trainer stores a **config hash** in checkpoints and **refuses to resume** if the current config doesn’t match.

```bash
irl-train --config code/configs/mountaincar_ride.yaml --total-steps 200000 --resume --run-dir runs/<your-run-dir>
```

**What gets restored on `--resume`:**

* **Policy & value weights** (exact network parameters).
* **PPO optimizer states (Adam)** for both policy and value — momentum/EMA is preserved for stable continuation.
* **Intrinsic module state** (if present and method matches), including any per-module RMS and region/gating statistics.
* **Global intrinsic RMS** (when the selected module is *not* normalized internally).
* **Observation normalizer** (for vector observations).
* **Counters/metadata** (current `step`, update counter) so training continues until the requested `--total-steps` (no extra offset).

**Safety & portability:**

* A **config-hash mismatch** aborts resume to prevent accidental cross-run continuation (e.g., different `env.id`, `method`, or PPO settings).
* On resume, optimizer tensors are moved to the active device automatically (CPU↔GPU) before training continues.

> Start fresh in the same directory by omitting `--resume` (default) or passing `--no-resume`. This will **not** load the latest checkpoint.

### 5) Evaluate a checkpoint (deterministic, no intrinsic)

```bash
irl-eval --env MountainCar-v0 \
         --ckpt runs/.../checkpoints/ckpt_step_100000.pt \
         --episodes 10
```

### 6) Plot learning curves

```bash
# Aggregate one group
irl-plot curves --runs "runs/proposed__MountainCar-v0__seed*" \
                --metric reward_total_mean \
                --smooth 5 \
                --out results/mc_proposed_curve.png

# Overlay groups
irl-plot overlay \
  --group "runs/proposed__BipedalWalker*" \
  --group "runs/ride__BipedalWalker*,runs/rnd__BipedalWalker*" \
  --labels "Proposed" --labels "RIDE+RND" \
  --metric reward_total_mean --smooth 5 \
  --out results/walker_overlay.png
```

### 7) Multi-seed sweep & stats

```bash
# Evaluate latest checkpoints, aggregate CSVs
irl-sweep eval-many --runs "runs/proposed__BipedalWalker*" --out results/summary.csv

# Non-parametric comparison (Mann–Whitney U, bootstrap CIs)
irl-sweep stats \
  --summary-raw results/summary_raw.csv \
  --env BipedalWalker-v3 \
  --method-a proposed --method-b ride \
  --metric mean_return --boot 5000
```

---

## Configuration

Configs live in `code/configs/` and are standard YAML. The loader validates types and invariants before training starts.

**Top-level keys**

* `seed`, `device`, `method`: run reproducibility and which intrinsic module to use (`vanilla`, `icm`, `rnd`, `ride`, `riac`, `proposed`).
* `env`: Gymnasium environment settings (ID, vectorized env count, frame-skip, domain randomization, discrete CarRacing toggle).
* `ppo`: PPO hyperparameters (steps per update, minibatches, epochs, learning rate, clip settings, KL knobs).
* `intrinsic`: Intrinsic reward coefficients and region settings. Gating thresholds live under `intrinsic.gate`.
* `logging`: CSV/TensorBoard cadence and checkpoint interval.
* `evaluation`: Periodic evaluation episodes without intrinsic rewards.
* `adaptation`: Optional scheduler that anneals intrinsic scaling based on policy entropy.

**Minibatch divisibility rule**

`ppo.minibatches` must evenly divide either `ppo.steps_per_update` or `ppo.steps_per_update * env.vec_envs`. The loader enforces this and explains how to adjust the values when it fails.

**Intrinsic normalization contract**

Each intrinsic module advertises whether it normalizes rewards internally via `outputs_normalized`.

* When `outputs_normalized=False`, the trainer applies a global running RMS so different modules share a consistent reward scale.
* When `outputs_normalized=True`, the module already returns normalized rewards (for example, the proposed method and RIAC use per-component RMS). In that case the trainer skips the global RMS and only applies clipping/scaling.

You can override any field from the CLI, e.g. `irl-train --config ... --method ride --device cuda:0`.

---

## Project layout

```
code/irl/         # library (trainer, models, intrinsic modules, utils)
code/configs/     # ready-to-run YAMLs (Box2D + MuJoCo)
code/tests/       # unit & integration tests (pytest)
```

Artifacts per run:

```
runs/<method>__<env>__seed<k>__<timestamp>/
  checkpoints/ckpt_step_*.pt
  logs/scalars.csv
  tb/              # TensorBoard (if enabled)
  diagnostics/     # RIAC/Proposed region dumps (if enabled)
```

---

## Troubleshooting

* **MuJoCo headless:** `export MUJOCO_GL=egl` on Linux. On macOS/Windows, headless EGL isn’t required; ensure a valid GL.
* **Box2D install:** On Windows, the `Box2D` wheel is included in `.[box2d]`. On Linux/macOS, `gymnasium[box2d]` is used.
* **CUDA not detected:** The trainer auto-falls back to CPU and prints a notice.
* **Minibatch divisibility:** Either `ppo.steps_per_update` or `ppo.steps_per_update * env.vec_envs` must be divisible by `ppo.minibatches` (strictly validated).

---

## Testing

```bash
pytest -q
```

---

## Citation / Use

This project is a research playground for intrinsic-motivation RL. Feel free to adapt it for your experiments, cite the repository when publishing results, and share improvements via pull requests.
