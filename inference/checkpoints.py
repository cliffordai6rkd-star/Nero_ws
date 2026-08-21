from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any, Mapping


class CheckpointError(RuntimeError):
    pass


_ARITHMETIC_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ARITHMETIC_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_arithmetic_eval(expression: Any) -> int | float:
    """Evaluate numeric checkpoint interpolation without executing Python code."""
    try:
        parsed = ast.parse(str(expression), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid arithmetic expression: {expression!r}") from exc

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("arithmetic expression constants must be numeric")
            return value
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC_BINARY_OPERATORS:
            return _ARITHMETIC_BINARY_OPERATORS[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITHMETIC_UNARY_OPERATORS:
            return _ARITHMETIC_UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError(
            "arithmetic expression may contain only numbers and +, -, *, /, //, %, **"
        )

    result = evaluate(parsed)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError("arithmetic expression must produce a number")
    return result


def _register_checkpoint_resolvers(omega_conf: Any) -> None:
    omega_conf.register_new_resolver(
        "eval", _safe_arithmetic_eval, replace=True
    )


def restore_checkpoint_model(
    path: str | Path,
    device: str,
    *,
    use_ema: bool,
    kind: str,
    pinn_mode: str = "wrench_gru",
    model_overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Instantiate a model solely from checkpoint cfg and restore its weights."""
    try:
        import torch
    except ImportError as exc:
        raise CheckpointError("checkpoint inference requires torch") from exc

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CheckpointError(f"{kind} checkpoint does not exist: {checkpoint_path}")
    with checkpoint_path.open("rb") as stream:
        try:
            import dill

            payload = torch.load(
                stream,
                map_location="cpu",
                pickle_module=dill,
                weights_only=False,
            )
        except ImportError:
            payload = torch.load(stream, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"{kind} checkpoint root must be a mapping")
    cfg = payload.get("cfg", payload.get("config"))
    if cfg is None:
        raise CheckpointError(f"{kind} checkpoint has no embedded cfg/config")
    model_cfg = cfg.get("policy") if isinstance(cfg, Mapping) else None
    if model_cfg is None and isinstance(cfg, Mapping):
        model_cfg = cfg.get("model")
    target = model_cfg.get("_target_") if isinstance(model_cfg, Mapping) else None
    if target is not None:
        try:
            import hydra
            from omegaconf import OmegaConf, open_dict
        except ImportError as exc:
            raise CheckpointError(
                f"{kind} Hydra checkpoint requires hydra-core and omegaconf"
            ) from exc
        _register_checkpoint_resolvers(OmegaConf)
        cfg = OmegaConf.create(cfg)
        model_cfg = OmegaConf.select(cfg, "policy")
        if model_cfg is None:
            model_cfg = OmegaConf.select(cfg, "model")
        missing = object()
        for key, value in (model_overrides or {}).items():
            update_key = key
            # Older checkpoints put the DINO path at policy level, while the
            # pure visual Transformer policy keeps it on its image encoder.
            if (
                key == "dino_model_name_or_path"
                and OmegaConf.select(model_cfg, key, default=missing) is missing
                and OmegaConf.select(
                    model_cfg,
                    "obs_encoder.pretrained_model_name_or_path",
                    default=missing,
                )
                is not missing
            ):
                update_key = "obs_encoder.pretrained_model_name_or_path"
            if OmegaConf.select(model_cfg, update_key, default=missing) is missing:
                raise CheckpointError(
                    f"{kind} checkpoint model has no override field {key!r}"
                )
            OmegaConf.update(model_cfg, update_key, value, merge=False)
        scheduler_cfg = OmegaConf.select(model_cfg, "noise_scheduler")
        scheduler_target = OmegaConf.select(model_cfg, "noise_scheduler._target_")
        if scheduler_cfg is not None and isinstance(scheduler_target, str):
            incompatible_keys = (
                ("set_alpha_to_one",)
                if scheduler_target.endswith("DDPMScheduler")
                else (("variance_type",) if scheduler_target.endswith("DDIMScheduler") else ())
            )
            for key in incompatible_keys:
                if key in scheduler_cfg:
                    with open_dict(scheduler_cfg):
                        del scheduler_cfg[key]
        model = hydra.utils.instantiate(model_cfg)
    elif kind.upper() == "PINN":
        try:
            if pinn_mode == "wrench_gru":
                from model.pinn_model.model_v1 import WrenchSequenceGRUV1

                model_type = WrenchSequenceGRUV1
            elif pinn_mode == "world_model_v3":
                from model.pinn_model.model_v3 import DeterministicWorldModelV3

                model_type = DeterministicWorldModelV3
            elif pinn_mode == "world_model_v4":
                from model.pinn_model.model_v4 import DeterministicWorldModelV4

                model_type = DeterministicWorldModelV4
            elif pinn_mode == "world_model_v5":
                from model.pinn_model.model_v5 import StateToStateFlowWorldModelV5

                model_type = StateToStateFlowWorldModelV5
            elif pinn_mode in {
                "contact_world_model",
                "contact_world_model_opd",
                "contact_wm",
                "contact_wm_opd",
            }:
                from model.pinn_model.torque_world_model import TorqueWorldModel

                model_type = TorqueWorldModel
            else:
                raise CheckpointError(
                    "pinn_mode must be 'wrench_gru', 'world_model_v3', "
                    "'world_model_v4', 'world_model_v5', "
                    "'contact_world_model', or 'contact_world_model_opd', "
                    f"got {pinn_mode!r}"
                )
        except ImportError as exc:
            raise CheckpointError(
                "PINN checkpoint uses a native PINN model, but the PINN package "
                "is not installed (pip install -e /mnt/code/lcx/PINN)"
            ) from exc
        model = model_type(dict(cfg))
    else:
        raise CheckpointError(
            f"{kind} checkpoint cfg must contain self-describing policy/model._target_"
        )

    if kind.upper() == "PINN" and pinn_mode in {
        "contact_world_model",
        "contact_world_model_opd",
        "contact_wm",
        "contact_wm_opd",
    }:
        contact_contract = bool(getattr(model, "q_tau_contact_contract", False))
        contact_contract = contact_contract or (
            str(getattr(model, "state_contract", "")).lower() == "q_tau_contact"
        )
        if not contact_contract:
            raise CheckpointError(
                "contact world-model inference requires checkpoint model.state_contract="
                "'q_tau_contact'"
            )

    state_dicts = payload.get("state_dicts")
    if isinstance(state_dicts, Mapping):
        candidates = (
            ("ema_model", "model_ema", "model")
            if use_ema
            else ("model", "model_ema", "ema_model")
        )
        state = next((state_dicts[key] for key in candidates if key in state_dicts), None)
    else:
        state = payload.get(
            "model_state_dict",
            payload.get(
                "state_dict",
                payload.get("model_ema", payload.get("model")),
            ),
        )
    if not isinstance(state, Mapping):
        raise CheckpointError(f"{kind} checkpoint contains no model state dictionary")
    model.load_state_dict(state, strict=True)
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            try:
                checkpoint_config = OmegaConf.to_container(cfg, resolve=True)
            except Exception:
                # Training-only resolvers such as ${now:...} need not be
                # registered in deployment just to preserve checkpoint metadata.
                checkpoint_config = OmegaConf.to_container(cfg, resolve=False)
        else:
            checkpoint_config = dict(cfg)
    except ImportError:
        checkpoint_config = dict(cfg)
    model._inference_checkpoint_config = checkpoint_config
    model._inference_normalizer = payload.get("normalizer")
    model.to(torch.device(device))
    model.eval()
    return model


def call_pinn(
    model: Any,
    inputs: Mapping[str, Any],
    recurrent_state: Any = None,
) -> tuple[Any, Any]:
    """Apply the small, explicit runtime contract expected from a PINN checkpoint."""
    if hasattr(model, "forward_step"):
        output = model.forward_step(inputs, recurrent_state)
    elif hasattr(model, "predict_force"):
        output = model.predict_force(inputs)
    elif hasattr(model, "predict"):
        output = model.predict(inputs)
    else:
        output = model(inputs)
    if isinstance(output, Mapping):
        for key in ("f_ext", "wrench", "force", "force_target", "target_wrench"):
            if key in output:
                return output[key], output.get("recurrent_state", recurrent_state)
        if "wrench_pred" in output:
            return output["wrench_pred"], output.get("recurrent_state", recurrent_state)
        raise CheckpointError(
            "PINN mapping output must contain f_ext/wrench/force/force_target/target_wrench"
        )
    return output, recurrent_state
