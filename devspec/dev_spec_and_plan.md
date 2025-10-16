# Intrinsic Reward Integration — Development Specification & Implementation Plan
**Repository:** `Intrinsic-Reward-Integration-Based-on-State-Representation-Novelty-and-Competence-Progress`  

---

## 0) Executive Summary

This document specifies, end‑to‑end, how to implement, evaluate, and operate a research codebase that integrates **impact‑driven intrinsic motivation (RIDE)** with **region‑based learning progress (R‑IAC)** to handle **partially random** RL environments. It also defines baselines (Vanilla PPO, ICM, RND, RIDE, R‑IAC), environments (MountainCar, BipedalWalker, CarRacing, Ant, HalfCheetah, Humanoid), metrics, data formats, CLI, configuration, and an exhaustive sprint plan.

> **Design Pillars**
> 1. **Exploration that ignores irreducible noise.** Reward agent‑induced changes and regions showing measurable model improvement; suppress stochastic traps.
> 2. **Reproducible science.** Fixed seeds, deterministic configs, strong logging, multi‑seed statistics.
> 3. **Modularity.** Pluggable intrinsic modules, shared PPO training loop.
> 4. **Scalability.** Parallel envs, resumable checkpoints, container & headless Mujoco support.

---

## 1) System Overview & Goals

### 1.1 Goals
* Implement a unified intrinsic reward:
  * **Impact:** embedding change per step (RIDE‑like) with episodic de‑duplication.
  * **Learning Progress:** region‑wise decrease in forward‑model error (R‑IAC‑like).
  * **Partial Randomness Filter:** down‑weight regions with persistently high error and negligible progress.
* Provide **baseline** algorithms for fair comparison.
* Support **six environments** spanning discrete/continuous control with optional domain randomization.
* Deliver **robust evaluation**: learning curves, seed variance, sample efficiency, coverage proxies.
* Ensure **reproducibility** and **cost‑aware runs** (CPU/GPU guidance, resume, logging).

### 1.2 Scope
* **Included:** PPO‑based training, intrinsic modules (ICM, RND, RIDE, R‑IAC, Proposed), Gymnasium environments (Box2D, MuJoCo), full experiment harness.
* **Excluded (by design):** Real‑robot deployment; hierarchical or multi‑task RL; Minigrid (mentioned in notes but **not** in target environment set).

---

## 2) Functional Requirements

1. **Training Orchestrator**
   * Start/stop training for a method–environment pair, single or multiple seeds.
   * Parallel rollouts; PPO updates; intrinsic computation on‑the‑fly.
   * Periodic evaluation episodes **without** intrinsic rewards.

2. **Intrinsic Modules (Pluggable)**
   * `vanilla` (no intrinsic).
   * `icm` (inverse/forward error).
   * `rnd` (predictor vs fixed target MSE).
   * `ride` (embedding Δ with episodic visitation penalty).
   * `riac` (region learning progress).
   * `proposed` (RIDE + R‑IAC + randomness filter + adaptive weighting).

3. **Environment Layer**
   * Gymnasium wrappers for normalization, frame‑skip (where applicable), domain randomization toggles.
   * Headless rendering support; deterministic seed routing.

4. **Configuration & CLI**
   * YAML config per run; CLI commands for train/eval/plot/reproduce.
   * All hyperparameters declared in config; no hidden defaults.

5. **Logging & Checkpointing**
   * TensorBoard + CSV for metrics; JSON summaries at key milestones.
   * Periodic checkpoints; resumable runs; artifact hashing of configs.

6. **Evaluation & Reporting**
   * Multi‑seed aggregation (mean ± std).
   * Plots: returns, success rate, sample efficiency, intrinsic trends, forward‑model error.
   * CSV/JSON exports for tables; optional bootstrapped CIs.

---

## 3) Non‑Functional Requirements

* **Reproducibility:** Fixed seeds for env & nets; pinned packages; config immutability per run.
* **Performance:** Vectorized envs (8–32); batched networks; GPU optional (required for CarRacing image encoder).
* **Maintainability:** Type hints; unit tests; clear module boundaries; docs in this spec.
* **Reliability:** Graceful resume after pre‑emption; checkpoint integrity checks.
* **Portability:** Linux (Ubuntu 22.04+), Python 3.10; container recipe; headless EGL for MuJoCo.
* **Observability:** TensorBoard scalars/hparams; structured logs; failure diagnostics.

---

## 4) System Architecture

### 4.1 Components

```

+-------------------+         +-----------------------+
|   CLI (typer)     |  --->   |  Config Loader (YAML) |
+-------------------+         +-----------------------+
|                               |
v                               v
+-------------------+         +-----------------------+
|  Trainer          | <-----> |  Env Manager          |
|  (PPO Loop)       |         |  (vectorized, wraps)  |
+-------------------+         +-----------------------+
|      ^                           |
v      |                           v
+-------------------+         +-----------------------+
| Intrinsic Module  | <-----> |  Replay/Batch Buffer  |
| (pluggable)       |         | (for ICM/RND/Proposed)|
+-------------------+         +-----------------------+
|
v
+-------------------+         +-----------------------+
|  Logging          | <-----> |  Checkpoint Manager   |
|  (TB, CSV, JSON)  |         |  (resume, hash cfg)   |
+-------------------+         +-----------------------+

```

### 4.2 Data Flow (per PPO update)
1. **Collect** N steps from vectorized envs (e.g., 2048) with current policy.
2. **Compute intrinsic** per transition according to selected module; store `(s, a, r_ext, r_int, s')`.
3. **Compute advantages** on **total reward** `r_total = r_ext + η * r_int` (η per env).
4. **Optimize** policy/value (PPO), and **update** module networks (forward/inverse/predictor).
5. **Log** metrics; **checkpoint** per interval; **evaluate** every K steps without intrinsic.

### 4.3 External Dependencies
* Python 3.10
* PyTorch `>=2.1`
* Stable‑Baselines3 `>=2.3`
* Gymnasium `>=0.29` with extras:
  * `gymnasium[box2d]` (BipedalWalker, CarRacing)
  * `gymnasium[mujoco]` (Ant, HalfCheetah, Humanoid)
* mujoco (python) `>=3.0` (EGL headless support)
* numpy, pandas, matplotlib, tensorboard
* scikit‑learn (KD‑tree, quantiles for analysis)
* tyro/typer (CLI), pyyaml, tqdm

**Install (CPU baseline):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.1" "gymnasium[box2d,mujoco]>=0.29" "stable-baselines3>=2.3" mujoco numpy pandas matplotlib tensorboard scikit-learn typer pyyaml tqdm
```
**Headless rendering (MuJoCo):**
`export MUJOCO_GL=egl`

---

## 5) Detailed Methodologies

### 5.1 PPO Backbone (All Methods)
* Separate policy/value nets; MLP (256, 256, ReLU).
* Continuous actions: Gaussian with state‑indep. log‑std; Discrete: categorical.
* Hyperparams (defaults):
  * LR `3e-4`, γ `0.99`, λ (GAE) `0.95`, clip ε `0.2`, steps per update `2048`,
  * minibatches `32`, epochs `3–10` (default 10), entropy coef `0.0–0.01`.
* Vectorized envs: 8–32.

### 5.2 Common Preprocessing
* **Observation normalization:** running mean/std (for vector states). **Enabled by default across all tasks.**  
* **Image (CarRacing):** gray or RGB; 96×96; CNN encoder (3×32×8×8 stride 4, 32×64×4×4 stride 2) → FC 256.
* **CarRacing control mode:** **discrete by default** (5 actions: no‑op, steer left, steer right, gas, brake). Toggle with `env.discrete_actions: true|false` in config (set `false` for continuous controls).
* **Rewards:** optional clipping to `[-1, 1]` per env spec (MountainCar).

### 5.3 Baselines

#### 5.3.1 Vanilla PPO
* No intrinsic; ε schedule & entropy coef as config.

#### 5.3.2 ICM
* Embedding φ (MLP 256→256→128).
* Inverse: CE loss on `a_t | φ(s_t), φ(s_{t+1})`.
* Forward: MSE on `φ(s_{t+1}) | φ(s_t), a_t`.
* Intrinsic: forward MSE scaled by `η`.
* Loss weights inverse:forward = 1:1.

#### 5.3.3 RND
* Target: fixed MLP (512, 512).
* Predictor: same architecture, trainable.
* Intrinsic: `MSE(predictor(x), target(x))`, scaled by `η`.
* Running normalization of intrinsic by RMS.

#### 5.3.4 RIDE
* Embedding φ via forward/inverse training as in ICM (shared).
* **Impact reward:**
  \[
  r_{\text{impact}} = \frac{\|\phi(s_{t+1}) - \phi(s_t)\|_2}{1 + N_{\text{ep}}(b(\phi(s_{t+1})))}
  \]
  * `N_ep`: per‑episode count of **binned** embedding (see §5.5).
* Intrinsic: `η * α_impact * r_impact`.

#### 5.3.5 R‑IAC
* Partition latent space into regions `R_i` (KD‑tree style; split on max‑variance dim once capacity `m` exceeded).
* Forward model error per transition:
  \[
  e_t = \|\phi(s_{t+1}) - f_{\text{fw}}(\phi(s_t), a_t)\|_2^2
  \]
* Maintain **two EMAs** per region: `EMA_long`, `EMA_short` (β_long=0.995, β_short=0.9).
* **Learning Progress (LP):**  
  \[
  \text{LP}_i = \max\{0, \text{EMA}_{\text{long}} - \text{EMA}_{\text{short}}\}
  \]
  (positive when error is trending **down**).
* Intrinsic: `η * α_LP * LP_i`.

### 5.4 Proposed Method (Unified Intrinsic with Randomness Filter)

**Total intrinsic:**
\[
r_{\text{int}} = \alpha_{\text{impact}} \cdot \underbrace{r_{\text{impact}}}_{\text{RIDE-like}} \;+\; \alpha_{\text{LP}} \cdot \underbrace{\text{LP}_{i}}_{\text{R-IAC-like}} \quad \text{(when } \phi(s_t)\in R_i \text{)}
\]
**Total reward to PPO:** `r_total = r_ext + η · r_int`.

#### 5.4.1 Partial Randomness Filter (Region‑wise Gating)
Each region `R_i` tracks:
* `EMA_long`, `EMA_short`, `LP_i` (as above).
* A rolling **improvement score** `I_i = LP_i / (ε + EMA_long)`;  
* A **stochasticity score** `S_i = EMA_short / (ε + median_error_global)`.

**Gate rule:**  
If, for `K=5` consecutive refreshes (e.g., every 1k samples region‑local),
* `LP_i < τ_LP` **and** `S_i > τ_S` (high error with no improvement),
then mark `R_i` → **random/unlearnable** and set `gate_i=0` (suppress intrinsic).  
Else `gate_i=1`.  
Use hysteresis: require `LP_i > 2·τ_LP` for 2 refreshes to re‑enable.

**Gated intrinsic:** `r_int := gate_i * r_int`.

Default thresholds: `τ_LP = 0.01 * median_LP_global`, `τ_S = 2.0`.

#### 5.4.2 Adaptive Weighting (Simple Policy‑Aware Schedule)
Every `T_global = 50k` env steps:
* If **policy entropy** < `H_low` (e.g., 30% of initial) and **extrinsic return** has plateaued (<1% gain over last 3 evals):
  * Increase `α_LP ← 1.2·α_LP`, decrease `α_impact ← 0.9·α_impact` (cap within `[0.05, 2.0]`).
* If entropy high but progress low:
  * Increase `α_impact ← 1.1·α_impact` (more exploration).

#### 5.4.3 Normalization & Clipping
* Maintain running RMS for `r_impact` and `LP_i`; normalize each before combination.
* Clip combined `r_int` to `[-r_clip, r_clip]`, default `r_clip=5.0`.

### 5.5 Embedding Binning & Episodic Counts
* Compute coarse bin key:
  * `key = tuple(floor(φ / bin_size))` with `bin_size=0.25` (configurable).
* Per episode, `counts[key]++`. Reset counts on env reset.
* Used only in RIDE and Proposed (denominator term).

### 5.6 Region Partitioning
* **Structure:** Balanced binary KD‑tree over φ‑space.
* **Capacity:** `m=200` samples per region; `depth_max=12`.
* **Split:** When capacity exceeded:
  * Pick dim with largest variance; split at median of that dim.
* Maintain per‑region: sample count, last refresh time, EMA stats, gate flag, bounding box (for diagnostics).

### 5.7 Networks & Losses (Proposed)
* **Embedding φ:** MLP 256→256→128 (ReLU).  
* **Forward f_fw:** concat[φ(s), a] → 256→256→128 (ReLU); loss MSE.  
* **Inverse f_inv:** concat[φ(s), φ(s’)] → 256→256→|A| (CE for discrete, Gaussian NLL for continuous via mean/log‑std head).  
* Joint loss:
  \[
  \mathcal{L}_{\text{repr}} = \alpha_{\text{fw}} \cdot \text{MSE}_{\text{fw}} + \alpha_{\text{inv}} \cdot \text{CE/NLL}
  \]
  Defaults: `α_fw=α_inv=1.0`.
* Optimizer: Adam, LR `3e-4`, grad‑clip global‑norm 5.0.

### 5.8 Hyperparameters (Per Environment Defaults)

| Env            | Gym ID             | Steps (max) | Vec Envs | η (intrinsic) | Notes                                                      |
|----------------|--------------------|-------------|----------|----------------|------------------------------------------------------------|
| MountainCar    | MountainCar-v0     | 1e6         | 16       | 0.2            | Reward clip [-1,1]                                        |
| BipedalWalker  | BipedalWalker-v3   | 2e6         | 16       | 0.1            | Box2D; continuous                                         |
| CarRacing      | CarRacing-v3       | 3e6         | 16       | 0.05           | CNN encoder; **discrete controls by default**; frame‑skip 2 |
| Ant            | Ant-v5             | 5e6         | 16       | 0.05           | MuJoCo v5 tasks                                           |
| HalfCheetah    | HalfCheetah-v5     | 5e6         | 16       | 0.05           |                                                            |
| Humanoid       | Humanoid-v5        | 1e7         | 16       | 0.05           | Most compute‑intensive                                    |

---

## 6) Data Design

### 6.1 Directory Layout (per run)
```
runs/ <method>__<env>_*seed*<N>_*YYYYmmdd-HHMMSS/
config.yaml                  # immutable copy
checkpoints/
ckpt_step*<k>.pt
logs/
scalars.csv                # wide CSV with step, returns, etc.
tb/                        # TensorBoard event files
metrics.json               # summary at milestones
buffers/
(optional for ICM/RND/Proposed if off-policy buffer used)
diagnostics/
regions.jsonl              # one JSON per refresh snapshot
gates.csv                  # region gate status over time
plots/
learning_curve.png
```

### 6.2 Config Schema (YAML)
```yaml
seed: 1
device: "cuda:0"  # or "cpu"
method: "proposed" # vanilla|icm|rnd|ride|riac|proposed
env:
  id: "BipedalWalker-v3"        # gymnasium id (e.g., MountainCar-v0, CarRacing-v3, Ant-v5, HalfCheetah-v5, Humanoid-v5)
  vec_envs: 16
  frame_skip: 1
  domain_randomization: false
  discrete_actions: true        # default true for CarRacing; ignored for continuous-control envs
ppo:
  steps_per_update: 2048
  minibatches: 32
  epochs: 10
  learning_rate: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.0
intrinsic:
  eta: 0.1
  alpha_impact: 1.0
  alpha_lp: 0.5
  r_clip: 5.0
  bin_size: 0.25
  region_capacity: 200
  depth_max: 12
  ema_beta_long: 0.995
  ema_beta_short: 0.90
  gate:
    tau_lp_mult: 0.01   # multiply median LP
    tau_s: 2.0
    hysteresis_up_mult: 2.0
    min_consec_to_gate: 5
adaptation:
  enabled: true
  interval_steps: 50000
  entropy_low_frac: 0.3
evaluation:
  interval_steps: 50000
  episodes: 10
logging:
  tb: true
  csv_interval: 10000
  checkpoint_interval: 100000
```

### 6.3 Validation Rules

* Verify Gym spaces match policy heads (discrete vs continuous).
* Ensure `vec_envs * n_steps` divisible by `minibatches`.
* Confirm device visibility; fallback to CPU if CUDA not available.
* Deny run if config hash already used unless `--resume` is passed.

### 6.4 Data Types

* **Scalars CSV columns:** `step, ep_return, ep_len, eval_mean, eval_std, policy_entropy, value_loss, policy_loss, r_int_mean, r_int_impact_mean, r_int_lp_mean, forward_mse, gate_rate, ...`
* **regions.jsonl** per refresh:

  * `{ "time": step, "region_id": int, "count": int, "ema_short": float, "ema_long": float, "lp": float, "gate": 0|1, "bbox": [lo[], hi[]] }`

---

## 7) Evaluation & Testing

### 7.1 Metrics

* **Extrinsic Episode Return:** mean over trailing window; eval mean±std (10 eps).
* **Success/Completion Rate:** env‑specific (MountainCar solved %, Walker success).
* **Sample Efficiency:** steps to reach threshold (env‑specific).
* **Variance Across Seeds:** `mean ± std` across 5 seeds.
* **Intrinsic Trends:** `r_int_mean`, component means.
* **Forward Model Error:** EMA trajectories.
* **Coverage Proxy:** unique bins visited per episode (embedding bins).

### 7.2 Protocol

* Train until max steps or convergence.
* Evaluate every 50k steps; compute eval means.
* **Seeds:** {1,2,3,4,5} for each (method, env).

### 7.3 Statistical Tests

* At final checkpoints, perform **two‑sided Mann‑Whitney U** tests (α = **0.05**) between Proposed and each baseline using final returns across seeds.
* Report effect sizes and **95% bootstrap confidence intervals** (1,000 resamples) for mean returns.

### 7.4 Testing Strategy

* **Unit tests:** shapes, loss finite, gate logic, KD‑split invariants, hashing determinism.
* **Integration tests:** 10k‑step smoke runs on MountainCar for all methods; check non‑NaN metrics and checkpoint resume.
* **Determinism checks:** repeated run with same seed yields ≤1% metric drift.

---

## 8) Deployment & Runtime

### 8.1 Execution Commands

* **Train**

```bash
python -m irl.train --config configs/bipedal_proposed.yaml
```
* **Evaluate (checkpoint)**

```bash
python -m irl.eval --env BipedalWalker-v3 --ckpt runs/.../checkpoints/ckpt_step_1000000.pt --episodes 20
```
* **Multi‑seed sweep**

```bash
python -m irl.sweep --method proposed --env BipedalWalker-v3 --seeds 1 2 3 4 5
```
* **Plot**

```bash
python -m irl.plot --runs runs/proposed__BipedalWalker* --out results/walker_curves.png
```

### 8.2 Hardware Guidance

* **Default hardware assumption:** 16 vCPUs, 32 GiB RAM. **1×GPU** (T4/RTX 30xx/A100) available for **CarRacing** and **Humanoid**; CPU‑only acceptable for MountainCar, BipedalWalker, Ant, HalfCheetah.
* **CPU‑only:** MountainCar, BipedalWalker, HalfCheetah, Ant (acceptable with vectorization).
* **GPU advised:** CarRacing (CNN), Humanoid (throughput).
* **Headless:** `MUJOCO_GL=egl` on Linux servers.

**Expected wall‑clock per seed (approx., default budgets)**
* MountainCar: ~2 h (CPU)
* BipedalWalker: ~4 h (CPU)
* CarRacing: ~24 h (GPU)
* Ant: ~12 h (CPU)
* HalfCheetah: ~8 h (CPU)
* Humanoid: 24–48 h (CPU) or 12–24 h (GPU)

**Intervals**
* Checkpoint and evaluation intervals default to **50k steps**; acceptable across all environments under these budgets.

### 8.3 Containerization (reference)

```dockerfile
# Dockerfile (reference snippet)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 python3.10-venv python3-pip libgl1-mesa-glx libgl1-mesa-dri \
    libegl1-mesa && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=egl
RUN python3 -m pip install --upgrade pip
RUN pip install "torch>=2.1" "gymnasium[box2d,mujoco]>=0.29" "stable-baselines3>=2.3" mujoco numpy pandas matplotlib tensorboard scikit-learn typer pyyaml tqdm
WORKDIR /workspace
```

### 8.4 Automation

* Bash scripts for launching sweeps; auto‑shutdown on completion (cloud).
* Checkpoint upload to artifact store (optional).

---

## 9) Security, Privacy, Reliability

* **Security/Privacy:** No external datasets; no PII; artifacts are synthetic. Maintain dependency integrity via hashes and pinning.
* **Licensing:** Ensure MuJoCo and Gymnasium licenses are complied with (open‑source packages via pip).
* **Reliability:** Atomic checkpoint writes; verify file integrity; resume logic tested; periodic config & code hash logging for provenance.

---

## 10) Risk Register & Mitigations

| Risk                      | Impact               | Likelihood | Mitigation                                                       |
| ------------------------- | -------------------- | ---------- | ---------------------------------------------------------------- |
| MuJoCo headless issues    | Blocks CI/cloud runs | Med        | Use `MUJOCO_GL=egl`; install mesa‑egl; run smoke test in CI      |
| Intrinsic explosion       | Training instability | Med        | RMS normalization, clipping (`r_clip=5.0`), gradient clipping    |
| Region tree fragmentation | Memory/time overhead | Low        | Depth cap, capacity threshold, periodic pruning of empty regions |
| Non‑determinism           | Weak reproducibility | Med        | Fix seeds; disable nondet backends; log library versions         |
| CarRacing throughput      | Slow                 | Med        | Frame‑skip=2; reduced CNN; use GPU                               |

---

## 11) Pseudocode (Key Loops)

### 11.1 Training (Proposed)

```python
def train_proposed(cfg):
    rng = SeedEverything(cfg.seed)
    envs = make_vec_env(cfg.env, cfg.env.vec_envs, seed=rng.env)
    policy, value = build_policy_value(cfg)
    module = ProposedModule(cfg.intrinsic)  # φ, f_fw, f_inv, regions

    for step in range(0, cfg.max_steps, cfg.ppo.steps_per_update):
        batch = rollout(envs, policy, steps=cfg.ppo.steps_per_update)
        # compute intrinsic on-the-fly
        for tr in batch:
            phi_t = module.embed(tr.s)
            phi_tp1 = module.embed(tr.s_next)
            r_impact = module.impact(phi_t, phi_tp1, tr.a)  # with episodic bin counts
            r_lp, reg = module.learning_progress(phi_t, tr.a, phi_tp1)
            r_int = module.combine_and_gate(r_impact, r_lp, reg)  # gating + norm + clip
            tr.r_total = tr.r_ext + cfg.intrinsic.eta * r_int
            module.update_buffers(tr, phi_t, phi_tp1)  # for EMA / region refresh

        adv, v_targets = compute_gae(batch, value, gamma=cfg.ppo.gamma, lam=cfg.ppo.gae_lambda)
        ppo_update(policy, value, batch, adv, v_targets, cfg.ppo)

        module.update_representation(batch)  # optimize φ, f_fw, f_inv
        module.refresh_regions_if_needed()   # update EMAs, LP, gates; maybe split regions

        log_metrics(...)
        checkpoint_if_needed(...)
        if step % cfg.evaluation.interval_steps == 0:
            eval_metrics = evaluate(policy, env_id=cfg.env.id, episodes=cfg.evaluation.episodes)
            log_eval(eval_metrics)

        if cfg.adaptation.enabled and step % cfg.adaptation.interval_steps == 0:
            module.adapt_weights(policy_entropy=last_entropy, recent_eval=recent_eval_stats)
```

### 11.2 Region Refresh

```python
def refresh_region(R_i, samples):
    # Update EMAs with new forward errors
    for e in samples.errors:
        R_i.ema_long = beta_long * R_i.ema_long + (1-beta_long) * e
        R_i.ema_short = beta_short * R_i.ema_short + (1-beta_short) * e
    R_i.lp = max(0.0, R_i.ema_long - R_i.ema_short)

    # Gating
    if R_i.lp < tau_lp and (R_i.ema_short / (eps + median_error_global)) > tau_s:
        R_i.consec_bad += 1
    else:
        R_i.consec_bad = 0

    if R_i.gate == 1 and R_i.consec_bad >= min_consec_to_gate:
        R_i.gate = 0
    elif R_i.gate == 0 and R_i.lp > hysteresis_up_mult * tau_lp:
        R_i.gate = 1
```

---

## 12) Coding Standards & Repo Layout

```
irl/                         # package root
  __init__.py
  cfg/                       # default YAMLs
  envs/                      # wrappers, registries
  models/                    # policy, value, encoders
  intrinsic/                 # icm.py, rnd.py, ride.py, riac.py, proposed.py
  algo/                      # ppo.py, advantage.py
  data/                      # buffers, storage
  train.py                   # CLI entry
  eval.py
  sweep.py
  plot.py
tests/
  unit/
  integration/
configs/
  mountaincar_proposed.yaml
  bipedal_proposed.yaml
  ...
runs/                        # generated
```

* **Type hints, black/ruff formatting**, docstrings for public functions.
* **Tests** must run via `pytest -q` and complete <5 min for smoke suite.

---

## 13) Implementation Plan (Sprints)

> **Timeline dependency:** Each sprint builds on previous; within a sprint, tasks are **sequential**. All steps include inputs, outputs, tests, and completion criteria.

### Sprint 0 — Project Bootstrap (Goal: scaffold, envs, CI smoke)

**Steps**

1. Create package skeleton (see §12).
2. Add `pyproject.toml` with deps; set Python 3.10.
3. Implement config loader (YAML → dataclasses).
4. Implement Env Manager (vectorized; seeding; wrappers).
5. Add PPO backbone (policy/value MLPs; action heads; advantage/GAE; optimizer).
6. Add logging (TensorBoard, CSV) and checkpoint manager.
7. Add CLI (`train`, `eval`) with Typer.
   **Dependencies:** none.
   **Inputs:** N/A.
   **Outputs:** runnable `vanilla` PPO on MountainCar (10k steps).
   **Testing & Verification**

* Unit: config parse; env step; PPO step shapes.
* Integration: `python -m irl.train --config configs/mountaincar_vanilla.yaml` → produces logs, ckpt, non‑NaN metrics.
  **Completion Criteria:** Vanilla PPO runs for 10k steps; TB curves appear; one checkpoint produced.

### Sprint 1 — Baselines ICM & RND

**Steps**
1. Implement ICM module (φ, f_fw, f_inv; losses; intrinsic forward MSE).
2. Implement RND (target/predictor; intrinsic = MSE).
3. Add module factory & plug into PPO loop.
4. Add RMS normalization utility for intrinsic.

**Dependencies:**
Sprint 0.

**Inputs:**
configs for `icm`, `rnd`.

**Outputs:**
Runs on MountainCar, BipedalWalker (short smoke).

**Testing**
* Unit: loss finiteness; predictor learning curve decreases on fixed dataset.
* Integration: 50k‑step smoke runs; intrinsic > 0 and decays over time when revisiting.

**Completion Criteria:**
Stable learning; logs include `r_int_mean`, module losses.

### Sprint 2 — RIDE

**Steps**
1. Implement RIDE embedding reuse (share with ICM nets).
2. Implement episodic binning & counts (§5.5).
3. Compute impact reward and integrate scaling.
4. Add configuration knobs: `bin_size`, `alpha_impact`.

**Dependencies:**
Sprint 1.

**Inputs:**
RIDE config.

**Outputs:**
RIDE run on MountainCar, Walker.

**Testing**
* Unit: per‑episode counts reset; repeated state reduces reward.
* Integration: `r_int_impact_mean` positive early, decays on toggling.

**Completion Criteria:**
RIDE curves/logs sane; no crashes.

### Sprint 3 — R‑IAC (Regions & LP)

**Steps**
1. Implement KD‑tree region store with split logic.
2. Track per‑region EMA_short/long; compute LP.
3. Intrinsic = `α_LP * LP_i` (normalized).
4. Diagnostics export (`regions.jsonl`, `gates.csv` with all gates=1 initially).

**Dependencies:**
Sprint 2.

**Inputs:**
region capacity, depths, βs.

**Outputs:**
R‑IAC run on MountainCar, Walker.

**Testing**
* Unit: split triggers at capacity; EMAs monotone to inputs.
* Integration: LP decreases when model stabilizes; intrinsic shifts to new regions.

**Completion Criteria:**
Regions formed; LP logged; stable training.

### Sprint 4 — Proposed Method (Combine + Gate)

**Steps**
1. Combine RIDE + R‑IAC: `r_int = α_imp * r_impact + α_LP * LP_i`.
2. Implement gate metrics (I_i, S_i) and gating rule (§5.4.1).
3. Add hysteresis; metrics: gate_rate (% gated).
4. Normalize/clipping pipeline; module unit tests.

**Dependencies:**
Sprint 3.

**Inputs:**
thresholds; hysteresis; mins.

**Outputs:**
Proposed runs on MountainCar, Walker (100k steps).

**Testing**
* Unit: gate flips only under criteria; hysteresis respected.
* Integration: compare ungated vs gated; gated shows reduced reward in random subregions (simulate with injected noise env wrapper).

**Completion Criteria:** Proposed runs stable; gate_rate non‑zero under noise; logs clean.

### Sprint 5 — CNN Encoder & CarRacing

**Steps**
1. Implement CNN encoder (configurable) and image preprocess.
2. Integrate with ICM/RIDE/Proposed embeddings.
3. Add optional frame‑skip=2 wrapper.

**Dependencies:**
Sprint 4.

**Inputs:**
CarRacing config.

**Outputs:**
Short smoke run; GPU pathway validated.

**Testing**
* Unit: CNN output shape; gradient flow.
* Integration: 100k steps; TB shows learning; throughput acceptable.

**Completion Criteria:** CarRacing runs without OOM; intrinsic logged.

### Sprint 6 — MuJoCo Set (Ant, HalfCheetah, Humanoid)

**Steps**
1. Add MuJoCo env configs; confirm `MUJOCO_GL=egl`.
2. Validate action/obs adapters; reward scales.
3. Run smoke tests (100k steps) for baselines + proposed.

**Dependencies:**
Sprints 4–5.

**Inputs:**
mujoco installed.

**Outputs:**
Logs and ckpts for all three envs (short runs).

**Testing**
* Integration: steps/sec acceptable; no rendering crashes; metrics present.

**Completion Criteria:**
All three run headless; stable.

### Sprint 7 — Evaluation Harness & Plots

**Steps**
1. Implement evaluator: runs `episodes` without intrinsic; returns aggregated stats.
2. Multi‑seed sweeps & aggregation; export `summary.csv`.
3. Plotting scripts (learning curves, shaded std, bar charts).
4. Statistical tests (MWU; bootstrap optional).

**Dependencies:**
Sprint 6.

**Inputs:**
run glob patterns.

**Outputs:**
`results/*.png`, `summary.csv`.

**Testing**
* Unit: eval excludes intrinsic; deterministic with seed.
* Integration: produce plots for MountainCar & Walker across baselines.

**Completion Criteria:**
Reproducible plots; CSV summary complete.

### Sprint 8 — Robustness, Resume & Docs

**Steps**
1. Implement checkpoint resume (config hash check).
2. Fault‑tolerant writes; atomic tmp→final move.
3. Determinism checks; CI smoke tests.
4. Document configs; update examples; finalize this spec’s alignment.

**Dependencies:**
Sprint 7.

**Inputs:**
interrupted runs.

**Outputs:**
Successful resume; determinism report.

**Testing**
* Kill process mid‑run; resume from last ckpt and continue.
* Repeatability within tolerance.

**Completion Criteria:**
Resume works; determinism acceptable.

---

## 14) Completion Criteria (Project‑level)

* All six environments trained with all baselines + proposed for 5 seeds (full budget feasible).
* Plots and tables generated; proposed shows robustness in partially random settings (qualitative/quantitative).
* Code passes unit/integration tests; container build succeeds; README install & run steps proven.

---

## 15) Quickstart (One‑pager)

```bash
# 1) Setup
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.1" "gymnasium[box2d,mujoco]>=0.29" "stable-baselines3>=2.3" mujoco numpy pandas matplotlib tensorboard scikit-learn typer pyyaml tqdm
export MUJOCO_GL=egl  # for headless servers

# 2) Smoke test (Vanilla PPO on MountainCar)
python -m irl.train --config configs/mountaincar_vanilla.yaml

# 3) Proposed on BipedalWalker (short)
python -m irl.train --config configs/bipedal_proposed.yaml

# 4) Evaluate a checkpoint
python -m irl.eval --env BipedalWalker-v3 --ckpt runs/.../checkpoints/ckpt_step_1000000.pt --episodes 10

# 5) Multi-seed sweep + plots
python -m irl.sweep --method proposed --env BipedalWalker-v3 --seeds 1 2 3 4 5
python -m irl.plot --runs "runs/proposed__BipedalWalker*" --out results/walker.png
```
