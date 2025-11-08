# Configuration Guide (YAML)

This document explains the configuration schema, invariants, and shows runnable examples that match the trainer and tests.

> All keys are validated strictly. Unknown/missing keys raise a `ConfigError`.  
> Source of truth: `code/irl/cfg/schema.py` and `code/irl/cfg/loader.py`.

---

## 1) Top-level schema

```yaml
seed: 1                 # Global seed (env + nets)
device: "cpu"           # or "cuda:0"
method: "proposed"      # vanilla | icm | rnd | ride | riac | proposed

env:
  id: "MountainCar-v0"
  vec_envs: 16
  frame_skip: 1
  domain_randomization: false
  discrete_actions: true    # ignored for continuous-control envs (MuJoCo)

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
  eta: 0.1                 # global intrinsic scale η
  # RIDE
  alpha_impact: 1.0
  bin_size: 0.25
  # R-IAC
  alpha_lp: 0.5
  region_capacity: 200
  depth_max: 12
  ema_beta_long: 0.995
  ema_beta_short: 0.90
  # Clipping applied after normalization
  r_clip: 5.0
  # Proposed gating
  gate:
    tau_lp_mult: 0.01
    tau_s: 2.0
    hysteresis_up_mult: 2.0
    min_consec_to_gate: 5

adaptation:               # simple policy-aware schedule (disabled in smoke configs)
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

### Notes & invariants

* **Divisibility:** Either `ppo.steps_per_update` **or** `ppo.steps_per_update * env.vec_envs` must be divisible by `ppo.minibatches`. (Strictly enforced.)
* **Ranges:** `0 < gamma ≤ 1`, `0 ≤ gae_lambda ≤ 1`, `clip_range > 0`, `r_clip > 0`, `bin_size > 0`, `alpha_impact > 0`.
* **Method-specific use:**

  * `vanilla`: `intrinsic.eta` can be left `0.0` (ignored).
  * `icm`: requires actions and `next_obs`.
  * `rnd`: uses (next_)observations only.
  * `ride`: uses impact + optional episodic binning (trainer uses raw impact path by default).
  * `riac`: uses per-region learning progress (LP) normalized by a running RMS.
  * `proposed`: combines **normalized** impact and LP, then applies **region gating**.
* **Normalization paths:**

  * Some modules (RIAC, Proposed) set `outputs_normalized=True` and are normalized internally before clipping.
  * Others use a **global** `RunningRMS` in the trainer and then clip to `[-r_clip, r_clip]`.

---

## 2) Minimal runnable examples

### A) Vanilla PPO (MountainCar)

```yaml
# file: code/configs/mountaincar_vanilla.yaml
seed: 1
device: "cpu"
method: "vanilla"
env:
  id: "MountainCar-v0"
  vec_envs: 8
  frame_skip: 1
  domain_randomization: false
  discrete_actions: true
ppo:
  steps_per_update: 128
  minibatches: 32
  epochs: 4
  learning_rate: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.01
intrinsic:
  eta: 0.0
adaptation:
  enabled: false
evaluation:
  interval_steps: 50000
  episodes: 5
logging:
  tb: false
  csv_interval: 1000
  checkpoint_interval: 10000
```

Run:

```bash
irl-train --config code/configs/mountaincar_vanilla.yaml --total-steps 10000
```

---

### B) RIDE (BipedalWalker) — short smoke

```yaml
# file: code/configs/bipedal_ride.yaml
seed: 1
device: "cpu"
method: "ride"
env:
  id: "BipedalWalker-v3"
  vec_envs: 8
  frame_skip: 1
  domain_randomization: false
ppo:
  steps_per_update: 128
  minibatches: 32
  epochs: 4
  learning_rate: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.0
intrinsic:
  eta: 0.1
  alpha_impact: 1.0
  bin_size: 0.25
  r_clip: 5.0
adaptation:
  enabled: false
evaluation:
  interval_steps: 50000
  episodes: 5
logging:
  tb: false
  csv_interval: 1000
  checkpoint_interval: 10000
```

Run:

```bash
irl-train --config code/configs/bipedal_ride.yaml --total-steps 10000
```

---

### C) RIAC (BipedalWalker) — short smoke

```yaml
# file: code/configs/bipedal_riac.yaml
# (see repo for full file; shown here to highlight RIAC knobs)
method: "riac"
intrinsic:
  eta: 0.1
  alpha_lp: 0.5
  region_capacity: 200
  depth_max: 12
  ema_beta_long: 0.995
  ema_beta_short: 0.90
  r_clip: 5.0
```

Run:

```bash
irl-train --config code/configs/bipedal_riac.yaml --total-steps 10000
```

---

### D) Proposed (MuJoCo Ant/HalfCheetah/Humanoid)

```yaml
# file: code/configs/mujoco/ant_proposed.yaml
seed: 1
device: "cuda:0"     # use "cpu" if no GPU
method: "proposed"
env:
  id: "Ant-v5"
  vec_envs: 16
  frame_skip: 1
  domain_randomization: false
  discrete_actions: true  # ignored in continuous control; kept for schema parity
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
# proposed gating lives under intrinsic.gate with sensible defaults
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

Run (Linux headless):

```bash
export MUJOCO_GL=egl
irl-train --config code/configs/mujoco/ant_proposed.yaml --total-steps 100000
```

---

## 3) CLI overrides & patterns

* Override method, env, or device without editing YAML:

```bash
irl-train --config code/configs/mountaincar_vanilla.yaml --method ride --device cpu
```

* Choose a custom run directory (useful for resumes):

```bash
irl-train --config code/configs/bipedal_ride.yaml --run-dir runs/my_experiment --total-steps 10000
```

---

## 4) Gating (Proposed) at a glance

* Region store: KD-tree in φ-space (`capacity`, `depth_max`).
* EMAs per region: `ema_long`, `ema_short`; **LP = max(0, long − short)**.
* Gate OFF when **LP low** and **stochasticity high** for `min_consec_to_gate` refreshes.
* Hysteresis: require `hysteresis_up_mult × τ_LP` for two consecutive refreshes to re-enable.
* Thresholds:

  * `τ_LP = tau_lp_mult × median_LP_global`
  * `S_i = ema_short / (eps + median_error_global)` compared to `tau_s`.

You can inspect gating trends via the logged `gate_rate` metric (Proposed).

---

## 5) Common pitfalls

* **Divisibility check fails:** adjust `ppo.minibatches` or `ppo.steps_per_update` so that either `steps_per_update` or `steps_per_update × vec_envs` is divisible by `minibatches`.
* **CUDA requested but unavailable:** the trainer falls back to CPU and prints a warning.
* **CarRacing throughput:** prefer GPU and consider `env.frame_skip: 2` to speed up smoke tests.

---

## 6) Where things live

* Schema: `code/irl/cfg/schema.py`
* Loader & validation: `code/irl/cfg/loader.py`
* Ready-made configs: `code/configs/**/*.yaml`
* Trainer entry: `code/irl/trainer/loop.py`
