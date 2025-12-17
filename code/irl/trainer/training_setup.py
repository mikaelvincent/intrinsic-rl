from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam

from irl.cfg import Config, to_dict
from irl.envs import EnvManager
from irl.intrinsic import RunningRMS, create_intrinsic_module, is_intrinsic_method
from irl.models import PolicyNetwork, ValueNetwork
from irl.utils.checkpoint import CheckpointManager, compute_cfg_hash
from irl.utils.determinism import seed_everything
from irl.utils.loggers import (
    MetricLogger,
    log_domain_randomization,
    log_intrinsic_norm_hint,
    log_resume_intrinsic_warning,
    log_resume_loaded,
    log_resume_no_checkpoint,
    log_resume_optimizer_warning,
    log_resume_state_restored,
)
from irl.utils.spaces import is_image_space

from .build import default_run_dir, ensure_mujoco_gl, single_spaces
from .obs_norm import RunningObsNorm
from .runtime_utils import _move_optimizer_state_to_device


@dataclass
class PPOOptimizers:
    policy: Adam
    value: Adam


@dataclass
class IntrinsicContext:
    module: Optional[Any]
    method: str
    eta: float
    use_intrinsic: bool
    norm_mode: str
    outputs_normalized: Optional[bool]
    rms: RunningRMS


@dataclass
class TrainingSession:
    run_dir: Path
    device: torch.device

    env: Any
    obs_space: Any
    act_space: Any
    is_image: bool
    num_envs: int

    policy: PolicyNetwork
    value: ValueNetwork
    optimizers: PPOOptimizers

    intrinsic: IntrinsicContext
    obs_norm: Optional[RunningObsNorm]

    ckpt: CheckpointManager
    metric_logger: MetricLogger

    obs: Any
    global_step: int
    update_idx: int


def _resolve_deterministic_flag(cfg: Config) -> bool:
    deterministic = False
    try:
        exp_cfg = getattr(cfg, "exp", None)
        if exp_cfg is not None:
            deterministic = bool(getattr(exp_cfg, "deterministic", False))
    except Exception:
        deterministic = False
    return deterministic


def _init_run_dir_and_ckpt(cfg: Config, run_dir: Optional[Path]) -> Tuple[Path, CheckpointManager]:
    resolved = Path(run_dir) if run_dir is not None else default_run_dir(cfg)
    ckpt = CheckpointManager(
        resolved,
        interval_steps=cfg.logging.checkpoint_interval,
        max_to_keep=getattr(cfg.logging, "checkpoint_max_to_keep", None),
    )
    return resolved, ckpt


def _maybe_load_resume_payload(
    cfg: Config, ckpt: CheckpointManager, resume: bool
) -> Tuple[Optional[dict], int]:
    resume_payload: Optional[dict] = None
    resume_step: int = 0

    if resume:
        try:
            payload_cpu, step_cpu = ckpt.load_latest(map_location="cpu")
            current_hash = compute_cfg_hash(to_dict(cfg))
            stored_hash = payload_cpu.get("cfg_hash")
            if stored_hash is None:
                stored_hash = compute_cfg_hash(payload_cpu.get("cfg", {}) or {})
            if str(stored_hash) != str(current_hash):
                raise RuntimeError(
                    "Config hash mismatch when resuming:\n"
                    f"  checkpoint: {stored_hash}\n"
                    f"  current   : {current_hash}\n"
                    "Refuse to resume with a different configuration. "
                    "Supply a matching config or run with --no-resume."
                )
            resume_payload = payload_cpu
            resume_step = int(step_cpu)
            log_resume_loaded(resume_step, ckpt.latest_path)
        except FileNotFoundError:
            log_resume_no_checkpoint()
        except Exception:
            raise

    return resume_payload, resume_step


def _build_env(cfg: Config, *, logger) -> Any:
    manager = EnvManager(
        env_id=cfg.env.id,
        num_envs=cfg.env.vec_envs,
        seed=cfg.seed,
        frame_skip=cfg.env.frame_skip,
        domain_randomization=cfg.env.domain_randomization,
        discrete_actions=cfg.env.discrete_actions,
        car_action_set=cfg.env.car_discrete_action_set,
        render_mode=None,
        async_vector=bool(getattr(cfg.env, "async_vector", False)),
        make_kwargs=None,
    )
    env = manager.make()
    if int(cfg.env.vec_envs) > 1:
        logger.info(
            "Vector env mode: %s (num_envs=%d) for env_id=%s",
            "Async" if bool(getattr(cfg.env, "async_vector", False)) else "Sync",
            int(cfg.env.vec_envs),
            cfg.env.id,
        )
    return env


def _build_intrinsic(
    cfg: Config,
    *,
    obs_space: Any,
    act_space: Any,
    device: torch.device,
    logger,
) -> Tuple[Optional[Any], bool, str, Optional[bool], str, float, RunningRMS]:
    method_l = str(cfg.method).lower()
    eta = float(cfg.intrinsic.eta)
    use_intrinsic = is_intrinsic_method(method_l) and eta > 0.0
    intrinsic_module = None
    intrinsic_norm_mode = "none"
    intrinsic_outputs_normalized_flag: Optional[bool] = None

    checkpoint_include_points = True
    try:
        checkpoint_include_points = bool(getattr(cfg.intrinsic, "checkpoint_include_points", True))
    except Exception:
        checkpoint_include_points = True

    if is_intrinsic_method(method_l):
        fail_on_intrinsic_error = bool(getattr(cfg.intrinsic, "fail_on_error", True))
        try:
            intrinsic_module = create_intrinsic_module(
                method_l,
                obs_space,
                act_space,
                device=device,
                bin_size=float(cfg.intrinsic.bin_size),
                alpha_impact=float(cfg.intrinsic.alpha_impact),
                alpha_lp=float(cfg.intrinsic.alpha_lp),
                region_capacity=int(cfg.intrinsic.region_capacity),
                depth_max=int(cfg.intrinsic.depth_max),
                ema_beta_long=float(cfg.intrinsic.ema_beta_long),
                ema_beta_short=float(cfg.intrinsic.ema_beta_short),
                gate_tau_lp_mult=float(cfg.intrinsic.gate.tau_lp_mult),
                gate_tau_s=float(cfg.intrinsic.gate.tau_s),
                gate_hysteresis_up_mult=float(cfg.intrinsic.gate.hysteresis_up_mult),
                gate_min_consec_to_gate=int(cfg.intrinsic.gate.min_consec_to_gate),
                gate_min_regions_for_gating=int(cfg.intrinsic.gate.min_regions_for_gating),
                normalize_inside=bool(cfg.intrinsic.normalize_inside),
                gating_enabled=bool(cfg.intrinsic.gate.enabled),
                checkpoint_include_points=bool(checkpoint_include_points),
            )
            if intrinsic_module is not None and hasattr(intrinsic_module, "checkpoint_include_points"):
                try:
                    setattr(
                        intrinsic_module,
                        "checkpoint_include_points",
                        bool(checkpoint_include_points),
                    )
                except Exception:
                    pass

            if not use_intrinsic:
                logger.warning(
                    "Method %r selected but intrinsic.eta=%.3g; intrinsic rewards disabled (eta=0).",
                    method_l,
                    eta,
                )
        except Exception as exc:
            logger.error("Failed to create intrinsic module %r (%s).", method_l, exc)
            if fail_on_intrinsic_error:
                raise RuntimeError(
                    "Intrinsic module construction failed for method "
                    f"{method_l!r}. Set intrinsic.fail_on_error=False to continue without "
                    "intrinsic rewards."
                ) from exc
            logger.error("intrinsic.fail_on_error is False; continuing without intrinsic rewards.")
            intrinsic_module = None
            use_intrinsic = False

    if intrinsic_module is not None:
        try:
            intrinsic_outputs_normalized_flag = bool(
                getattr(intrinsic_module, "outputs_normalized", False)
            )
        except Exception:
            intrinsic_outputs_normalized_flag = None

        if intrinsic_outputs_normalized_flag is True:
            intrinsic_norm_mode = "module_rms"
        elif intrinsic_outputs_normalized_flag is False:
            intrinsic_norm_mode = "trainer_rms"
        else:
            intrinsic_norm_mode = "unknown"
    else:
        intrinsic_norm_mode = "none"
        intrinsic_outputs_normalized_flag = None

    int_rms = RunningRMS(beta=0.99, eps=1e-8)
    return (
        intrinsic_module,
        use_intrinsic,
        intrinsic_norm_mode,
        intrinsic_outputs_normalized_flag,
        method_l,
        eta,
        int_rms,
    )


def _reset_env(env: Any, seed: int | None) -> tuple[Any, Any]:
    if seed is None:
        return env.reset()

    B = int(getattr(env, "num_envs", 1))
    if B <= 1:
        try:
            return env.reset(seed=int(seed))
        except (TypeError, ValueError):
            return env.reset()

    seeds = [int(seed) + i for i in range(B)]
    try:
        return env.reset(seed=seeds)
    except (TypeError, ValueError):
        try:
            return env.reset(seed=int(seed))
        except (TypeError, ValueError):
            return env.reset()


def _log_reset_diagnostics(
    *,
    env: Any,
    intrinsic_module: Optional[Any],
    method_l: str,
    intrinsic_outputs_normalized_flag: Optional[bool],
    seed: int | None,
) -> Any:
    printed_dr_hint = False
    printed_intr_norm_hint = False

    obs, info = _reset_env(env, seed)

    try:
        if isinstance(info, dict) and ("dr_applied" in info) and not printed_dr_hint:
            diag = info.get("dr_applied")
            msg = ""
            if isinstance(diag, dict):
                mj = int(diag.get("mujoco", 0))
                b2 = int(diag.get("box2d", 0))
                msg = f"mujoco={mj}, box2d={b2}"
            elif isinstance(diag, (list, tuple)):
                mj = 0
                b2 = 0
                n = 0
                for d in diag:
                    if isinstance(d, dict):
                        mj += int(d.get("mujoco", 0))
                        b2 += int(d.get("box2d", 0))
                        n += 1
                msg = f"mujoco={mj}, box2d={b2} (across {n} envs)"
            else:
                msg = str(diag)
            log_domain_randomization(msg)
            printed_dr_hint = True
    except Exception:
        pass

    if intrinsic_module is not None and not printed_intr_norm_hint:
        try:
            outputs_norm_flag = (
                intrinsic_outputs_normalized_flag
                if intrinsic_outputs_normalized_flag is not None
                else bool(getattr(intrinsic_module, "outputs_normalized", False))
            )
            log_intrinsic_norm_hint(method_l, bool(outputs_norm_flag))
            printed_intr_norm_hint = True
        except Exception:
            pass

    return obs


def _restore_from_checkpoint(
    *,
    resume_payload: Optional[dict],
    resume_step: int,
    policy: PolicyNetwork,
    value: ValueNetwork,
    pol_opt: Adam,
    val_opt: Adam,
    intrinsic_module: Optional[Any],
    method_l: str,
    int_rms: RunningRMS,
    obs_norm: Optional[RunningObsNorm],
    is_image: bool,
    device: torch.device,
    logger,
) -> Tuple[int, int]:
    global_step = 0
    update_idx = 0

    if resume_payload is None:
        return global_step, update_idx

    try:
        policy.load_state_dict(resume_payload["policy"])
        value.load_state_dict(resume_payload["value"])
    except Exception:
        logger.warning("Could not load policy/value weights from checkpoint; using fresh init.")

    try:
        int_rms.load_state_dict(resume_payload.get("intrinsic_norm", {}))
    except Exception:
        pass

    try:
        if not is_image and resume_payload.get("obs_norm") is not None and obs_norm is not None:
            on = resume_payload["obs_norm"]
            import numpy as _np

            obs_norm.count = float(on.get("count", obs_norm.count))
            obs_norm.mean = _np.asarray(on.get("mean", obs_norm.mean), dtype=_np.float64)
            obs_norm.var = _np.asarray(on.get("var", obs_norm.var), dtype=_np.float64)
    except Exception:
        pass

    try:
        intr = resume_payload.get("intrinsic")
        if intrinsic_module is not None and isinstance(intr, dict) and intr.get("method") == method_l:
            sd = intr.get("state_dict", None)
            extra_state = intr.get("extra_state", None)

            if isinstance(sd, dict):
                res = intrinsic_module.load_state_dict(sd, strict=False)
                missing = [k for k in res.missing_keys if k != "_extra_state"]
                unexpected = [k for k in res.unexpected_keys if k != "_extra_state"]
                if missing or unexpected:
                    logger.warning(
                        "Intrinsic state mismatch on resume for method=%s (missing=%s, unexpected=%s).",
                        method_l,
                        missing,
                        unexpected,
                    )

            if extra_state is None:
                extra_state = resume_payload.get("intrinsic_extra_state", None)
                if extra_state is None:
                    extra_state = resume_payload.get("intrinsic_state", None)

            if (
                extra_state is not None
                and hasattr(intrinsic_module, "set_extra_state")
                and callable(getattr(intrinsic_module, "set_extra_state"))
                and not (isinstance(sd, dict) and "_extra_state" in sd)
            ):
                intrinsic_module.set_extra_state(extra_state)

            intrinsic_module.to(device)
    except Exception:
        log_resume_intrinsic_warning(method_l)

    try:
        opt_payload = resume_payload.get("optimizers", {})
        pol_state = opt_payload.get("policy", None)
        val_state = opt_payload.get("value", None)
        if pol_state is not None:
            pol_opt.load_state_dict(pol_state)
            _move_optimizer_state_to_device(pol_opt, next(policy.parameters()).device)
        if val_state is not None:
            val_opt.load_state_dict(val_state)
            _move_optimizer_state_to_device(val_opt, next(value.parameters()).device)
    except Exception:
        log_resume_optimizer_warning()

    global_step = int(resume_step)
    try:
        update_idx = int((resume_payload.get("meta") or {}).get("updates", update_idx))
    except Exception:
        update_idx = update_idx

    log_resume_state_restored(global_step)
    return global_step, update_idx


def build_training_session(
    cfg: Config,
    *,
    device: torch.device,
    run_dir: Optional[Path],
    resume: bool,
    logger,
) -> TrainingSession:
    deterministic = _resolve_deterministic_flag(cfg)
    seed_everything(int(cfg.seed), deterministic=deterministic)

    try:
        ensure_mujoco_gl(cfg.env.id)
    except Exception:
        pass

    run_dir_resolved, ckpt = _init_run_dir_and_ckpt(cfg, run_dir)
    resume_payload, resume_step = _maybe_load_resume_payload(cfg, ckpt, resume)

    env = _build_env(cfg, logger=logger)
    obs_space, act_space = single_spaces(env)

    is_image = bool(is_image_space(obs_space))

    policy = PolicyNetwork(obs_space, act_space).to(device)
    value = ValueNetwork(obs_space).to(device)

    pol_opt = Adam(policy.parameters(), lr=float(cfg.ppo.learning_rate))
    val_opt = Adam(value.parameters(), lr=float(cfg.ppo.learning_rate))

    (
        intrinsic_module,
        use_intrinsic,
        intrinsic_norm_mode,
        intrinsic_outputs_normalized_flag,
        method_l,
        eta,
        int_rms,
    ) = _build_intrinsic(
        cfg,
        obs_space=obs_space,
        act_space=act_space,
        device=device,
        logger=logger,
    )

    ml = MetricLogger(run_dir_resolved, cfg.logging)
    ml.log_hparams(to_dict(cfg))

    obs = _log_reset_diagnostics(
        env=env,
        intrinsic_module=intrinsic_module,
        method_l=method_l,
        intrinsic_outputs_normalized_flag=intrinsic_outputs_normalized_flag,
        seed=int(cfg.seed),
    )

    B = int(getattr(env, "num_envs", 1))

    obs_norm = None if is_image else RunningObsNorm(shape=int(obs_space.shape[0]))

    global_step, update_idx = _restore_from_checkpoint(
        resume_payload=resume_payload,
        resume_step=resume_step,
        policy=policy,
        value=value,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=intrinsic_module,
        method_l=method_l,
        int_rms=int_rms,
        obs_norm=obs_norm,
        is_image=is_image,
        device=device,
        logger=logger,
    )

    return TrainingSession(
        run_dir=run_dir_resolved,
        device=device,
        env=env,
        obs_space=obs_space,
        act_space=act_space,
        is_image=is_image,
        num_envs=B,
        policy=policy,
        value=value,
        optimizers=PPOOptimizers(policy=pol_opt, value=val_opt),
        intrinsic=IntrinsicContext(
            module=intrinsic_module,
            method=method_l,
            eta=float(eta),
            use_intrinsic=bool(use_intrinsic),
            norm_mode=str(intrinsic_norm_mode),
            outputs_normalized=intrinsic_outputs_normalized_flag,
            rms=int_rms,
        ),
        obs_norm=obs_norm,
        ckpt=ckpt,
        metric_logger=ml,
        obs=obs,
        global_step=int(global_step),
        update_idx=int(update_idx),
    )
