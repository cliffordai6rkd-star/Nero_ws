from __future__ import annotations

import ast
import operator
import sys
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


def _prepare_optional_package_source(target: Any, *, kind: str) -> None:
    """Make sibling source checkouts visible before Hydra imports a target.

    Some deployments keep only bytecode in ``third_party/diffusion_policy``
    while the checkpoint embeds a target whose source lives in the sibling
    checkout ``../diffusion_policy``.  Python caches namespace/package paths,
    so merely prepending ``sys.path`` is insufficient when an earlier import
    already loaded the vendored package.  Update the loaded package paths in
    place and leave the vendored package as the fallback.
    """
    if str(kind).upper() != "DP" or not isinstance(target, str):
        return
    if not target.startswith("diffusion_policy."):
        return
    candidates = (
        Path(__file__).resolve().parents[2] / "diffusion_policy",
        Path("/mnt/code/lcx/diffusion_policy"),
        Path("/home/rei/mnt/code/lcx/diffusion_policy"),
    )
    for root in candidates:
        package_root = root / "diffusion_policy"
        if not package_root.is_dir():
            continue
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        # If the package is already imported, prefer the source checkout for
        # subsequent submodule resolution without deleting live class objects.
        package = sys.modules.get("diffusion_policy")
        if package is not None:
            package_path = getattr(package, "__path__", None)
            if package_path is not None:
                source_text = str(package_root)
                if source_text not in package_path:
                    try:
                        package_path.insert(0, source_text)
                    except (AttributeError, TypeError):
                        # Namespace packages expose ``_NamespacePath`` rather
                        # than a mutable list.  Replacing it with an ordered
                        # list keeps the vendored path as a fallback while
                        # making the source checkout win resolution.
                        package.__path__ = [source_text, *list(package_path)]
        break


def _prepare_pinn_source() -> None:
    """Expose the sibling PINN checkout when it is not installed as a wheel."""

    candidates = (
        Path(__file__).resolve().parents[2] / "PINN",
        Path("/mnt/code/lcx/PINN"),
        Path("/home/rei/mnt/code/lcx/PINN"),
    )
    for root in candidates:
        if not (root / "model" / "pinn_model").is_dir():
            continue
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        package = sys.modules.get("model")
        package_path = getattr(package, "__path__", None)
        if package_path is not None:
            source_text = str(root / "model")
            if source_text not in package_path:
                try:
                    package_path.insert(0, source_text)
                except (AttributeError, TypeError):
                    package.__path__ = [source_text, *list(package_path)]
        return


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
    pinn_mode = str(pinn_mode).strip().lower().replace("-", "_")
    # Keep old mode strings source-compatible while making ContactWorldModel
    # v2 the only WM implementation that can be restored.
    wm_aliases = {
        "world_model": "contact_world_model",
        "world_model_v3": "contact_world_model",
        "world_model_v4": "contact_world_model",
        "world_model_v5": "contact_world_model",
        "swm": "contact_world_model",
        "torque_world_model": "contact_world_model",
        "torque_wm": "contact_world_model",
        "swm_opd": "contact_world_model_opd",
        "torque_world_model_opd": "contact_world_model_opd",
        "torque_wm_opd": "contact_world_model_opd",
        "contact_wm": "contact_world_model",
        "contact_wm_opd": "contact_world_model_opd",
    }
    pinn_mode = wm_aliases.get(pinn_mode, pinn_mode)
    try:
        import torch
    except ImportError as exc:
        raise CheckpointError("checkpoint inference requires torch") from exc

    checkpoint_path = Path(path).expanduser().resolve()
    # Native LeRobot policies are exported as a directory rather than a
    # torch-pickled file.  Keep the optional dependency isolated in the
    # high-level adapter and return that adapter as the self-describing model
    # object expected by the existing pipeline/check command.
    if str(kind).upper() == "DP":
        from inference.policies.lerobotdp import (
            LeRobotDiffusionPolicy,
            is_lerobot_checkpoint,
        )

        if is_lerobot_checkpoint(checkpoint_path):
            inference_steps = None
            if model_overrides is not None:
                raw_steps = model_overrides.get("num_inference_steps")
                if raw_steps is not None:
                    inference_steps = int(raw_steps)
            return LeRobotDiffusionPolicy.from_pretrained(
                checkpoint_path,
                device=device,
                num_inference_steps=inference_steps,
            )
    if not checkpoint_path.is_file():
        raise CheckpointError(f"{kind} checkpoint does not exist: {checkpoint_path}")
    try:
        import dill
    except ImportError:
        dill = None
    try:
        # Open a fresh stream for the fallback path.  ``torch.load`` may have
        # consumed bytes before raising a missing-class ImportError, so the
        # same file handle cannot safely be reused.
        with checkpoint_path.open("rb") as stream:
            if dill is None:
                payload = torch.load(stream, map_location="cpu", weights_only=False)
            else:
                payload = torch.load(
                    stream,
                    map_location="cpu",
                    pickle_module=dill,
                    weights_only=False,
                )
    except ModuleNotFoundError as exc:
        raise CheckpointError(
            f"{kind} checkpoint deserialization requires missing module "
            f"{exc.name!r}"
        ) from exc
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
        _prepare_optional_package_source(target, kind=kind)
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
        _prepare_pinn_source()
        try:
            if pinn_mode == "wrench_gru":
                from model.pinn_model.model_v1 import WrenchSequenceGRUV1

                model_type = WrenchSequenceGRUV1
            elif pinn_mode in {
                "contact_world_model",
                "contact_world_model_opd",
                "contact_wm",
                "contact_wm_opd",
            }:
                from model.pinn_model.contact_world_model import ContactWorldModel

                model_type = ContactWorldModel
            else:
                raise CheckpointError(
                    "pinn_mode must be 'wrench_gru' or a Contact World Model mode, "
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
        expected_version = getattr(model, "MODEL_VERSION", "contact_world_model_v2")
        if payload.get("model_version") != expected_version:
            raise CheckpointError(
                "WM checkpoint is not a canonical ContactWorldModel v2 checkpoint: "
                f"model_version={payload.get('model_version')!r}, "
                f"expected={expected_version!r}"
            )
        configured_inputs = tuple(getattr(model, "inputs", ()))
        required_inputs = ("q", "dq", "delta_q", "tau")
        if configured_inputs != required_inputs:
            raise CheckpointError(
                "ContactWorldModel inference requires model.inputs="
                f"{list(required_inputs)}, got {list(configured_inputs)}"
            )
        if int(getattr(model, "joint_dim", 7)) != 7 or int(
            getattr(model, "action_dim", 7)
        ) != 7:
            raise CheckpointError(
                "ContactWorldModel inference requires joint_dim=7 and action_dim=7"
            )

    state_dicts = payload.get("state_dicts")
    if isinstance(state_dicts, Mapping):
        # The native PINN trainer stores EMA weights under ``model`` and the
        # non-EMA copy under ``model_raw``.  Other checkpoint families use
        # explicit ``model_ema``/``ema_model`` names, so keep those aliases as
        # fallbacks while making ``use_ema=False`` meaningful for PINN files.
        if kind.upper() == "PINN":
            candidates = (
                ("model", "model_ema", "ema_model", "model_raw")
                if use_ema
                else ("model_raw", "model", "model_ema", "ema_model")
            )
        else:
            candidates = (
                ("ema_model", "model_ema", "model")
                if use_ema
                else ("model", "model_ema", "ema_model")
            )
        state = next(
            (
                state_dicts[key]
                for key in candidates
                if isinstance(state_dicts.get(key), Mapping)
            ),
            None,
        )
    else:
        if kind.upper() == "PINN":
            candidates = (
                ("model", "model_ema", "ema_model", "model_raw")
                if use_ema
                else ("model_raw", "model", "model_ema", "ema_model")
            )
        else:
            candidates = (
                ("model_ema", "ema_model", "model_state_dict", "state_dict", "model")
                if use_ema
                else ("model_state_dict", "state_dict", "model", "model_ema", "ema_model")
            )
        state = next(
            (
                payload[key]
                for key in candidates
                if isinstance(payload.get(key), Mapping)
            ),
            None,
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
