# Intrinsic RL: Impact, Learning Progress, and Unified Intrinsic Rewards

## 1. Repository overview

This repository contains a research codebase for training PPO agents with intrinsic motivation signals, with a focus on:

- **Multiple intrinsic methods**: vanilla (no intrinsic), ICM, RND, RIDE (impact), RIAC (learning progress), and a Proposed method that combines impact and learning progress with optional region gating and internal normalization.
- **End-to-end training pipeline**: environment management, PPO advantage and update code, intrinsic module factory, logging, checkpointing, and resume support.
- **Experiment tooling**: ready-to-run YAML configs, experiment sweep utilities, deterministic evaluation, aggregation, and plotting for learning curves and summary statistics.

All Python sources live under `code/irl/`. Default experiment configs live under `code/configs/`.


---

## 2. System requirements

### 2.1 Operating systems

Recommended:

- **Primary**: Ubuntu 22.04 LTS (Linux)  
  - Strongly recommended for MuJoCo-based tasks (Ant/HalfCheetah/Humanoid) and GPU training.
- **Also supported (non-MuJoCo tasks)**:
  - macOS 12+ with a recent Python 3.10/3.11 install.
  - Windows 10/11 with Python 3.10/3.11.
  - For MuJoCo on Windows, a **Linux VM or WSL2** is recommended.

### 2.2 Hardware

Baseline **CPU-only** (good for development, unit tests, and smaller tasks such as MountainCar/BipedalWalker/CarRacing):

- 4+ vCPU
- 16 GB RAM
- 10–20 GB free disk space

Recommended **GPU training** (MuJoCo tasks, large sweeps):

- NVIDIA RTX-class GPU with **≥ 8 GB VRAM** (e.g., RTX 3060 or better)
- CUDA-capable driver matching your chosen PyTorch wheel
- 8+ vCPU
- 32 GB RAM
- 50+ GB free disk space

### 2.3 Python and tooling

- **Python**: 3.10 or 3.11 (project requires `>=3.10,<3.12`).
- **Build tools** (Linux VM/local):

  ```bash
  sudo apt update
  sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential
  ```

Adjust `python3.11` to your Python minor version if needed.

---

## 3. Environment setup

You can follow the same high-level steps for a **local machine** or a **Linux VM**. For a GPU-enabled VM, ensure the cloud provider or hypervisor has configured NVIDIA drivers and CUDA; PyTorch will then detect the GPU.

Assume the repository has been cloned as:

```bash
git clone <your-clone-url> Intrinsic-Reward-Integration-Based-on-State-Representation-Novelty-and-Competence-Progress
cd Intrinsic-Reward-Integration-Based-on-State-Representation-Novelty-and-Competence-Progress/code
```

All commands below should be run from the `code/` directory.

### 3.1 Create and activate a virtual environment

#### Ubuntu / other Linux

```bash
cd Intrinsic-Reward-Integration-Based-on-State-Representation-Novelty-and-Competence-Progress/code

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

#### macOS (Homebrew Python)

```bash
cd Intrinsic-Reward-Integration-Based-on-State-Representation-Novelty-and-Competence-Progress/code

brew install python@3.11   # if not already installed
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

#### Windows (PowerShell)

```powershell
cd Intrinsic-Reward-Integration-Based-on-State-Representation-Novelty-and-Competence-Progress\code

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

### 3.2 Install PyTorch

Install a PyTorch build appropriate for your platform and whether you want CPU or CUDA. Examples:

* **CPU-only (works everywhere)**:

  ```bash
  pip install "torch>=2.1"
  ```

* **CUDA example (Linux; adjust per your CUDA/toolkit setup)**:

  ```bash
  # Example for CUDA 12.1; adapt as needed:
  pip install --index-url https://download.pytorch.org/whl/cu121 torch
  ```

If you need a different CUDA version, follow the installation matrix from the official PyTorch website and then return here.

You can verify after installation:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda available?', torch.cuda.is_available())"
```

### 3.3 Install the project and core dependencies

From `code/`:

```bash
# Base install (CPU environments, trainer, evaluator, plots)
pip install -e .
```

Optional extras:

* **Developer/test tools** (pytest, type checking, formatting):

  ```bash
  pip install -e ".[dev]"
  ```

* **TensorBoard logging**:

  ```bash
  pip install -e ".[tb]"
  ```

* **Box2D environments** (BipedalWalker, CarRacing; pulls in `pygame`):

  ```bash
  pip install -e ".[box2d]"
  ```

* **MuJoCo environments** (Ant, HalfCheetah, Humanoid; Linux recommended):

  ```bash
  pip install -e ".[mujoco]"
  ```

You can combine extras, e.g.:

```bash
pip install -e ".[dev,tb,box2d,mujoco]"
```

### 3.4 Notes specific to VMs

* For **CPU-only VMs**, the instructions are identical to a local machine.

* For **GPU VMs**:

  * Ensure that drivers and CUDA are available at the host level (cloud provider image or admin setup).
  * Install a CUDA-enabled PyTorch wheel in the virtual environment as shown above.
  * Inside the venv, verify:

    ```bash
    python -c "import torch; assert torch.cuda.is_available(); print('CUDA ready')"
    ```

* For **headless Linux + MuJoCo**:

  * No manual `MUJOCO_GL` export is required; the trainer will set `MUJOCO_GL=egl` automatically for MuJoCo tasks on Linux when needed.

---

## 4. Pre-run verification

Before running any main experiments, verify that the installation and tooling are correct.

### 4.1 Verify imports and versions

From `code/` and with the virtual environment active:

```bash
python -c "import sys; print(sys.version)"
python -c "import gymnasium as gym; print('gymnasium ok, version', gym.__version__)"
python -c "import irl; print('irl version:', irl.__version__)"
```

Expected:

* No exceptions.
* Python version line, gymnasium version, and `irl version: ...`.

If you see `ModuleNotFoundError: irl`, check that you installed with `pip install -e .` and that you are in the `code/` directory.

### 4.2 Run the test suite (recommended)

With dev extras installed:

```bash
pip install -e ".[dev]"
pytest -q
```

Expected:

* Tests complete successfully.
* A non-zero number of tests are run; there should be no unexpected failures.
  (Some tests may be skipped if optional packages such as TensorBoard are not installed; skip messages are acceptable.)

If tests fail:

* Check that your Python version is within the supported range.
* Ensure you are using a fresh virtual environment and that the project is installed in editable mode.

### 4.3 Smoke training run (vanilla PPO, CPU)

This checks the end-to-end training path: environments, GAE, PPO update, logging, and checkpointing.

From `code/`:

```bash
python -m irl.train train \
  --method vanilla \
  --env MountainCar-v0 \
  --total-steps 64 \
  --device cpu
```

Expected:

* The command exits without errors.

* At the end it prints something like:

  ```text
  [green]Training finished[/green]
  Run dir: runs/vanilla__MountainCar-v0__seed1__...
  ```

* In that `run dir`:

  ```bash
  ls runs/vanilla__MountainCar-v0__seed1__*/logs
  ls runs/vanilla__MountainCar-v0__seed1__*/checkpoints
  ```

  * `logs/scalars.csv` exists and contains at least a header and one row.
  * `checkpoints/ckpt_latest.pt` exists.

If `scalars.csv` is missing:

* Confirm that you are in the `code/` directory.
* Ensure the training run finished without error messages.
* Check that your disk has free space.

### 4.4 Optional GPU verification

If you intend to run MuJoCo or large experiments on GPU:

```bash
python -c "import torch; print('cuda available?', torch.cuda.is_available())"
```

You should see `cuda available? True`. If not, verify your driver and CUDA setup and reinstall PyTorch with a CUDA build.

---

## 5. Running experiments

All commands below assume you are in the `code/` directory with the virtual environment active.

There are two main entry points:

* **Single-run training**: `python -m irl.train ...`
* **Suite orchestration (train/eval/plots)**: `python -m irl.experiments ...`

### 5.1 Single-run training (irl.train)

General pattern:

```bash
python -m irl.train train \
  --config <path-to-config.yaml> \
  --total-steps <total_env_steps> \
  --device <cpu-or-cuda> \
  [--resume] \
  [--run-dir <custom-run-dir>]
```

If `--config` is omitted, a default config is used and you can drive training via `--method` and `--env`.

#### 5.1.1 Examples: small, CPU-friendly tasks

**MountainCar (all methods; CPU)**

Example: Proposed intrinsic method with default config:

```bash
python -m irl.train train \
  --config configs/mountaincar_proposed.yaml \
  --total-steps 150000 \
  --device cpu
```

Other MountainCar variants are available:

* `configs/mountaincar_vanilla.yaml`
* `configs/mountaincar_icm.yaml`
* `configs/mountaincar_rnd.yaml`
* `configs/mountaincar_ride.yaml`
* `configs/mountaincar_riac.yaml`
* `configs/mountaincar_proposed_global_rms.yaml`
* `configs/mountaincar_proposed_impact_only.yaml`
* `configs/mountaincar_proposed_lp_only.yaml`
* `configs/mountaincar_proposed_nogate.yaml`

You can swap the config path to run baselines and ablations.

**BipedalWalker (requires Box2D extra)**

Install Box2D extra first:

```bash
pip install -e ".[box2d]"
```

Then, for the Proposed method:

```bash
python -m irl.train train \
  --config configs/bipedal_proposed.yaml \
  --total-steps 300000 \
  --device cpu
```

Available Bipedal configs include:

* `configs/bipedal_vanilla.yaml`
* `configs/bipedal_icm.yaml`
* `configs/bipedal_rnd.yaml`
* `configs/bipedal_ride.yaml`
* `configs/bipedal_riac.yaml`
* `configs/bipedal_proposed.yaml`

**CarRacing (requires Box2D extra)**

Install as above, then:

```bash
python -m irl.train train \
  --config configs/carracing_proposed.yaml \
  --total-steps 500000 \
  --device cpu
```

Other CarRacing configs:

* `configs/carracing_vanilla.yaml`
* `configs/carracing_icm.yaml`
* `configs/carracing_rnd.yaml`
* `configs/carracing_ride.yaml`
* `configs/carracing_riac.yaml`
* `configs/carracing_proposed_global_rms.yaml`
* `configs/carracing_proposed_impact_only.yaml`
* `configs/carracing_proposed_lp_only.yaml`
* `configs/carracing_proposed_nogate.yaml`

#### 5.1.2 Examples: MuJoCo tasks (Linux + GPU recommended)

Install MuJoCo extra:

```bash
pip install -e ".[mujoco]"
```

Then, for example, **Humanoid with Proposed intrinsic**:

```bash
python -m irl.train train \
  --config configs/mujoco/humanoid_proposed.yaml \
  --total-steps 2000000 \
  --device cuda:0 \
  --resume
```

Other MuJoCo configs follow the pattern:

* Ant: `configs/mujoco/ant_{vanilla,icm,rnd,ride,riac,proposed}.yaml`
* HalfCheetah: `configs/mujoco/halfcheetah_{vanilla,icm,rnd,ride,riac,proposed}.yaml`
* Humanoid ablations:
  `configs/mujoco/humanoid_proposed_global_rms.yaml`,
  `configs/mujoco/humanoid_proposed_impact_only.yaml`,
  `configs/mujoco/humanoid_proposed_lp_only.yaml`,
  `configs/mujoco/humanoid_proposed_nogate.yaml`.

Use `--resume` to continue long runs from the latest checkpoint; the trainer verifies configuration hashes to prevent mismatched resumes.

### 5.2 Suite training, evaluation, and plotting (irl.experiments)

The `irl.experiments` module orchestrates:

* **Training** all configs in a directory subtree (`train`).
* **Evaluating** the latest checkpoints (`eval`).
* **Generating overlay plots** from logged scalars (`plots`).
* A **full pipeline** that runs all three (`full`).

#### 5.2.1 Training a config suite

Example: train all MountainCar configs (CPU):

```bash
python -m irl.experiments train \
  --configs-dir configs \
  --include "mountaincar_*.yaml" \
  --total-steps 150000 \
  --runs-root runs_suite \
  --seed 1 \
  --device cpu \
  --resume
```

Example: train all CarRacing configs (requires Box2D):

```bash
python -m irl.experiments train \
  --configs-dir configs \
  --include "carracing_*.yaml" \
  --total-steps 500000 \
  --runs-root runs_suite \
  --seed 1 \
  --device cpu \
  --resume
```

Example: train all MuJoCo configs (Linux + GPU):

```bash
python -m irl.experiments train \
  --configs-dir configs/mujoco \
  --include "*.yaml" \
  --total-steps 1000000 \
  --runs-root runs_suite_mujoco \
  --seed 1 \
  --device cuda:0 \
  --resume
```

* `runs_root` is where per-run directories will be created.
* `--resume` makes the suite skip runs that have already reached `total_steps` and resume partially completed runs.

#### 5.2.2 Evaluating a suite

After training, evaluate each run’s latest checkpoint deterministically (no intrinsic rewards are used for evaluation):

```bash
python -m irl.experiments eval \
  --runs-root runs_suite \
  --results-dir results_suite \
  --episodes 5 \
  --device cpu
```

Outputs:

* `results_suite/summary_raw.csv`: per-run, per-seed statistics.
* `results_suite/summary.csv`: aggregated statistics (mean/std across seeds, per method/environment).

#### 5.2.3 Generating overlay plots from a suite

```bash
python -m irl.experiments plots \
  --runs-root runs_suite \
  --results-dir results_suite \
  --metric reward_total_mean \
  --smooth 5 \
  --shade
```

Outputs:

* Plots in `results_suite/plots/`, named like
  `EnvName__overlay_reward_total_mean.png`.

#### 5.2.4 Full pipeline in one command

For a complete **train → eval → plots** pipeline (e.g., all MountainCar configs):

```bash
python -m irl.experiments full \
  --configs-dir configs \
  --include "mountaincar_*.yaml" \
  --total-steps 150000 \
  --runs-root runs_suite_mc \
  --seed 1 \
  --device cpu \
  --episodes 5 \
  --results-dir results_suite_mc \
  --metric reward_total_mean \
  --smooth 5 \
  --shade \
  --resume
```

Use similar `--include` patterns for Bipedal, CarRacing, and MuJoCo configs.

---

## 6. Evaluating trained policies (single runs)

For individual checkpoints (outside the suite workflow), use `irl.eval`.

Example:

```bash
python -m irl.eval eval \
  --env MountainCar-v0 \
  --ckpt runs/vanilla__MountainCar-v0__seed1__*/checkpoints/ckpt_latest.pt \
  --episodes 10 \
  --device cpu \
  --out results/mc_vanilla_eval.json
```

This:

* Runs deterministic evaluation using greedy (mode) actions.
* Prints per-episode returns and lengths.
* Writes a JSON summary to the `--out` path if provided.

---

## 7. Plotting and post-processing

For more ad-hoc plotting and statistics, use `irl.plot` and `irl.sweep`.

### 7.1 Learning curves and overlays (irl.plot)

Aggregate a single method’s learning curve:

```bash
python -m irl.plot curves \
  --runs "runs/proposed__MountainCar-v0__seed1__*" \
  --metric reward_total_mean \
  --smooth 5 \
  --shade \
  --out results/mc_proposed_curve.png
```

Overlay multiple methods:

```bash
python -m irl.plot overlay \
  --group "runs/proposed__MountainCar-v0__*" \
  --group "runs/ride__MountainCar-v0__*,runs/rnd__MountainCar-v0__*" \
  --labels "Proposed" \
  --labels "RIDE+RND" \
  --metric reward_total_mean \
  --smooth 5 \
  --shade \
  --out results/mc_overlay.png
```

Bar chart from a suite summary:

```bash
python -m irl.plot bars \
  --summary results_suite/summary.csv \
  --env MountainCar-v0 \
  --out results_suite/bars_mc.png
```

### 7.2 Multi-seed evaluation and statistics (irl.sweep)

Evaluate many checkpoints and write summary CSVs:

```bash
python -m irl.sweep eval-many \
  --runs "runs/proposed__MountainCar-v0__*" \
  --runs "runs/ride__MountainCar-v0__*" \
  --episodes 10 \
  --device cpu \
  --out results/mc_summary.csv
```

This generates:

* `results/summary_raw.csv`
* `results/mc_summary.csv` (aggregated)

Compare two methods statistically (Mann–Whitney U + bootstrap CIs):

```bash
python -m irl.sweep stats \
  --summary-raw results/summary_raw.csv \
  --env MountainCar-v0 \
  --method-a proposed \
  --method-b ride \
  --metric mean_return \
  --boot 5000 \
  --alt two-sided
```

---

## 8. Reproducibility notes

* The trainer calls `seed_everything(cfg.seed, deterministic=cfg.exp.deterministic)` at startup:

  * Seeds **Python `random`**, **NumPy**, and **PyTorch**.
  * If `cfg.exp.deterministic` is `True` (default), it requests deterministic PyTorch algorithms where supported.
* Checkpointing:

  * Each checkpoint stores a **configuration hash**.
  * When resuming with `--resume`, the trainer compares the current config hash with the one in the checkpoint. A mismatch causes a clear error instead of silently resuming with incompatible settings.
* Vector/matrix shapes and time-major layout are enforced in the trainer and GAE implementation; tests cover image and vector pipelines to guard against shape regressions.

To reproduce reported experiments:

1. Use the provided YAML configs under `configs/` and `configs/mujoco/`.
2. Fix seeds explicitly (e.g. `seed: 1`, `seed: 2`, …) in configs or via suite `--seed` arguments.
3. Run with `irl.experiments full` for each environment group as described in §5.2.4.
4. Archive:

   * Config files used.
   * Resulting `summary.csv`, `summary_raw.csv`.
   * Plots under `results_*/plots`.

---

## 9. Troubleshooting

**`ModuleNotFoundError: irl`**

* Ensure you are in `code/` and have run `pip install -e .` in the active virtual environment.

**`torch.cuda.is_available()` is `False` but you expect GPU**

* Confirm that CUDA drivers and toolkit are installed on the host.
* Reinstall PyTorch with a CUDA-enabled wheel matching your system.

**Box2D/CarRacing errors**

* Ensure you installed the Box2D extra:

  ```bash
  pip install -e ".[box2d]"
  ```

* If `pygame` issues appear, reinstall or upgrade it (it is pulled by the extra).

**MuJoCo environment creation fails**

* Ensure:

  ```bash
  pip install -e ".[mujoco]"
  ```

* Use Ubuntu 22.04 or a similar Linux; MuJoCo support on other platforms is more brittle.

* For headless servers, no manual `MUJOCO_GL` configuration is required; the trainer sets `MUJOCO_GL=egl` automatically for MuJoCo envs on Linux.

**`pytest` failures**

* Verify dev extras: `pip install -e ".[dev]"`.
* Ensure all commands are run from `code/`.
* If only specific tests fail (e.g., related to MuJoCo) and you are not using MuJoCo, check whether the corresponding extras are installed; otherwise, keep the core CPU pipeline and configs as your reference.

**No `scalars.csv` or checkpoints after a training run**

* Check that the run completed without exceptions.
* Ensure the `logging` section in your config has reasonable intervals (e.g. `csv_interval: 1000`, `checkpoint_interval: 100000`).
* Confirm that `run_dir` exists and that you have write permissions.

---

## 10. Quick start (minimal recipe)

```bash
# 1. Go to code directory
cd Intrinsic-Reward-Integration-Based-on-State-Representation-Novelty-and-Competence-Progress/code

# 2. Create and activate venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. Install PyTorch (CPU) and project
pip install "torch>=2.1"
pip install -e .

# 4. Run a short vanilla PPO training run on MountainCar
python -m irl.train train \
  --method vanilla \
  --env MountainCar-v0 \
  --total-steps 64 \
  --device cpu

# 5. Inspect outputs
ls runs/vanilla__MountainCar-v0__seed1__*/logs
ls runs/vanilla__MountainCar-v0__seed1__*/checkpoints
```

Once these steps succeed, the environment is ready for full-size experiments (MountainCar/Bipedal/CarRacing/MuJoCo), multi-seed sweeps, evaluation, and plotting for use in your paper.
