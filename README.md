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
  --group "runs/proposed__BipedalWalker-v3__seed*" \
  --group "runs/ride__BipedalWalker-v3__seed*,runs/rnd__BipedalWalker-v3__seed*" \
  --labels "Proposed" --labels "RIDE+RND" \
  --metric reward_total_mean --smooth 5 \
  --out results/walker_overlay.png
```

### 7) Multi-seed sweep & stats

```bash
# Evaluate latest checkpoints across multiple methods for one env and aggregate CSVs
irl-sweep eval-many \
  --runs "runs/vanilla__BipedalWalker-v3__seed*" \
  --runs "runs/icm__BipedalWalker-v3__seed*" \
  --runs "runs/rnd__BipedalWalker-v3__seed*" \
  --runs "runs/ride__BipedalWalker-v3__seed*" \
  --runs "runs/riac__BipedalWalker-v3__seed*" \
  --runs "runs/proposed__BipedalWalker-v3__seed*" \
  --out results/summary.csv

# Non-parametric comparison (Mann–Whitney U, bootstrap CIs)
irl-sweep stats \
  --summary-raw results/summary_raw.csv \
  --env BipedalWalker-v3 \
  --method-a proposed --method-b ride \
  --metric mean_return --boot 5000
```

On Windows `cmd.exe`, wildcards like `runs\*__BipedalWalker-v3__seed*` may sometimes expand into multiple arguments even when quoted.
The `eval-many` command is tolerant of this: any extra trailing arguments after the options are treated as additional run patterns.
For maximum clarity, prefer the “one `--runs` per method” style shown above.

---

## Configuration guide

All configs are plain YAML files validated on load. Ready-to-run examples live under
`code/configs/` (see the MuJoCo variants in `code/configs/mujoco/`).

**Top-level keys**

* `seed`, `device`, `method`: global reproducibility knobs and which intrinsic module to use (`vanilla`, `icm`, `rnd`, `ride`, `riac`, `proposed`).
* `env`: Gymnasium environment id, vectorized env count, frame skip, optional domain randomization, and the CarRacing discrete-action toggle.
* `ppo`: rollout length, minibatch count, update epochs, optimizer settings, and optional KL guards.
* `intrinsic`: shared hyperparameters for intrinsic modules. The gate thresholds (`intrinsic.gate.*`) only affect the proposed method.
* `adaptation`: entropy-aware scaling schedule for intrinsic weight `eta`.
* `evaluation`: cadence and episode count for periodic deterministic evaluation.
* `logging`: CSV/TensorBoard cadence plus checkpoint interval.

**Validation hints**

* `ppo.steps_per_update` must be divisible by `ppo.minibatches`. When training with vector envs, the product `ppo.steps_per_update * env.vec_envs` must also divide evenly by the minibatch count.
* Intrinsic clip (`intrinsic.r_clip`) and method-specific coefficients must stay positive; the loader raises clear errors when a setting would violate training assumptions.

**Intrinsic normalization contract**

* Modules that normalize internally (RIAC, proposed, and any module setting `outputs_normalized=True`) expose already-scaled rewards. The trainer trusts this flag and only applies clipping and the global `intrinsic.eta` multiplier.
* Modules that emit raw magnitudes (e.g., vanilla intrinsic off, RIDE without gating) rely on the trainer’s global `RunningRMS` scaler. The state is checkpointed alongside the policy so resumed runs pick up identical intrinsic scales.

Override any top-level field from the CLI, for example `--method proposed`, `--env BipedalWalker-v3`, or `--device cuda:0`.

---

## Project layout

```text
code/irl/         # library (trainer, models, intrinsic modules, utils)
code/configs/     # ready-to-run YAMLs (Box2D + MuJoCo)
code/tests/       # unit & integration tests (pytest)
```

Artifacts per run:

```text
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
* **Windows wildcards with `irl-sweep eval-many`:**

  * If `runs\*__Env__seed*` expands into multiple arguments, they are all accepted and interpreted as run patterns.
  * For maximum control, pass one glob per method via repeated `--runs` flags as in the examples above.

---

## Testing

```bash
pytest -q
```

---

## Citation / Use

This research scaffold supports experiments and comparisons across intrinsic-motivation methods. Contributions and extensions are welcome via issues or pull requests.
