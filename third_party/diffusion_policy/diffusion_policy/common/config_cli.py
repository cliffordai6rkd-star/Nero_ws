import os
import pathlib
from typing import Iterable, Optional, Sequence, Tuple


_CONFIG_REF_OPTIONS = ("-c", "--config")
_LEGACY_CONFIG_OPTIONS = ("--config-name", "-cn", "--config-dir", "-cd")


def split_config_reference(
    config_ref: str,
    default_config_dir: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    config_ref = os.path.expanduser(config_ref)
    config_path = pathlib.Path(config_ref)
    config_name = config_path.name
    if config_name in ("", ".", ".."):
        raise ValueError(f"Invalid config path: {config_ref}")

    config_dir = None
    if str(config_path.parent) != ".":
        config_dir = str(config_path.parent)
    elif default_config_dir is not None:
        config_dir = str(default_config_dir)

    return config_dir, config_name


def rewrite_config_reference_argv(
    argv: Sequence[str],
    default_config_dir: Optional[str] = None,
) -> list[str]:
    if len(argv) <= 1:
        return list(argv)

    args = list(argv)
    config_idx = None
    config_ref = None
    consumed = 0

    for idx, arg in enumerate(args[1:], start=1):
        if arg in _CONFIG_REF_OPTIONS:
            if idx + 1 >= len(args):
                raise SystemExit("Missing path after -c/--config")
            config_idx = idx
            config_ref, extra_consumed = _consume_config_ref_tokens(args, idx + 1)
            consumed = 1 + extra_consumed
            break
        for option in _CONFIG_REF_OPTIONS:
            prefix = option + "="
            if arg.startswith(prefix):
                config_idx = idx
                inline_value = arg[len(prefix):]
                config_ref, extra_consumed = _consume_config_ref_tokens(
                    args,
                    idx + 1,
                    initial_value=inline_value,
                )
                consumed = 1 + extra_consumed
                break
        if config_idx is not None:
            break

    if config_idx is None:
        return args

    if _has_cli_option(args[1:], _LEGACY_CONFIG_OPTIONS):
        raise SystemExit(
            "Do not combine -c/--config with --config-dir/--config-name."
        )

    config_dir, config_name = split_config_reference(
        config_ref=config_ref,
        default_config_dir=default_config_dir,
    )

    replacement = [f"--config-name={config_name}"]
    if config_dir is not None:
        replacement.insert(0, f"--config-dir={config_dir}")

    return args[:config_idx] + replacement + args[config_idx + consumed :]


def _consume_config_ref_tokens(
    args: Sequence[str],
    start_idx: int,
    initial_value: Optional[str] = None,
) -> Tuple[str, int]:
    tokens = []
    consumed = 0

    if initial_value is not None:
        tokens.append(initial_value)

    idx = start_idx
    while idx < len(args):
        arg = args[idx]
        if tokens and _is_cli_boundary(arg):
            break
        tokens.append(arg)
        consumed += 1
        idx += 1

    config_ref = " ".join(token for token in tokens if token)
    if not config_ref:
        raise SystemExit("Missing path after -c/--config")

    return config_ref, consumed


def _is_cli_boundary(arg: str) -> bool:
    if arg.startswith('-'):
        return True

    # Hydra overrides always advertise themselves with an operator.
    return '=' in arg or arg.startswith('+') or arg.startswith('~')


def _has_cli_option(args: Iterable[str], options: Sequence[str]) -> bool:
    for arg in args:
        for option in options:
            if arg == option or arg.startswith(option + "="):
                return True
    return False
