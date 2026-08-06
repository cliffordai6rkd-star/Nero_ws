"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
python train.py -c diffusion_policy/config/train_zarr/train_hirol_fr3_unet_abs_jp.yaml

GPU Selection:
# Single GPU training - specify device
python train.py --config-name=train_hirol_fr3_unet_abs_jp_ee_state.yaml training.device=cuda:0
python train.py --config-name=train_hirol_fr3_unet_abs_jp_ee_state.yaml training.device=cuda:1

# Multi-GPU training preparation
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --config-name=train_hirol_fr3_unet_abs_jp_ee_state.yaml training.device=cuda:0

# CPU training (for debugging)
python train.py --config-name=train_hirol_fr3_unet_abs_jp_ee_state.yaml training.device=cpu

Examples:
# Train on specific GPU with custom output directory
python train.py --config-dir=. --config-name=train_hirol_fr3_unet_abs_jp_ee_state.yaml \
    training.device=cuda:1 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_gpu1'

# Multi-GPU server deployment
CUDA_VISIBLE_DEVICES=2,3 python train.py --config-dir=. --config-name=train_hirol_fr3_unet_abs_jp_ee_state.yaml \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_multi_gpu'

Notes:
- Use CUDA_VISIBLE_DEVICES to control which GPUs are visible to PyTorch
- training.device=cuda:0 refers to the first visible GPU (after CUDA_VISIBLE_DEVICES filtering)
- For multi-GPU training, set CUDA_VISIBLE_DEVICES first, then use cuda:0 as primary device
- Check GPU availability with: nvidia-smi
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)   

import hydra
from omegaconf import OmegaConf
import pathlib
import os
import tempfile
from hydra.core.hydra_config import HydraConfig
from diffusion_policy.common.config_cli import rewrite_config_reference_argv
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


REPO_ROOT = pathlib.Path(__file__).parent.resolve()
BCPOLICY_PACKAGE_ROOT = REPO_ROOT.joinpath("BCpolicy")
DEFAULT_CONFIG_DIR = REPO_ROOT.joinpath("diffusion_policy", "config")
_TEMP_CONFIG_DIR = None

if BCPOLICY_PACKAGE_ROOT.is_dir():
    bcpolicy_path = str(BCPOLICY_PACKAGE_ROOT)
    if bcpolicy_path not in sys.path:
        sys.path.insert(0, bcpolicy_path)

if DEFAULT_CONFIG_DIR.is_dir():
    os.environ.setdefault("DIFFUSION_POLICY_CONFIG_DIR", str(DEFAULT_CONFIG_DIR))


def _infer_resume_output_dir(cfg: OmegaConf):
    if not OmegaConf.select(cfg, "training.resume", default=False):
        return None

    hydra_cfg = HydraConfig.get()
    config_sources = getattr(hydra_cfg.runtime, "config_sources", [])
    for source in config_sources:
        if getattr(source, "schema", None) != "file":
            continue

        source_path = getattr(source, "path", None)
        if not source_path:
            continue

        config_dir = pathlib.Path(source_path)
        if not config_dir.is_absolute():
            config_dir = pathlib.Path(hydra_cfg.runtime.cwd).joinpath(config_dir)
        if config_dir.name != ".hydra":
            continue

        run_dir = config_dir.parent.resolve()
        if run_dir.joinpath("checkpoints", "latest.ckpt").is_file():
            return str(run_dir)

    return None


def _get_file_config_dirs():
    hydra_cfg = HydraConfig.get()
    result = []
    for source in getattr(hydra_cfg.runtime, "config_sources", []):
        if getattr(source, "schema", None) != "file":
            continue

        source_path = getattr(source, "path", None)
        if not source_path:
            continue

        config_dir = pathlib.Path(source_path)
        if not config_dir.is_absolute():
            config_dir = pathlib.Path(hydra_cfg.runtime.cwd).joinpath(config_dir)
        result.append(config_dir)
    return result


def _load_task_config_from_ref(task_ref: str, config_dirs=None):
    task_ref = os.path.expanduser(task_ref)
    if not task_ref:
        raise ValueError("Explicit task config cannot be empty.")
    task_path = pathlib.Path(task_ref)
    if task_path.suffix == "":
        task_path = task_path.with_suffix(".yaml")

    candidates = []
    if task_path.is_absolute():
        candidates.append(task_path)
        candidates.append(DEFAULT_CONFIG_DIR.joinpath(str(task_path).lstrip("/")))
    else:
        if config_dirs is None:
            config_dirs = _get_file_config_dirs()
        for config_dir in config_dirs:
            candidates.append(config_dir.joinpath(task_path))
        candidates.append(DEFAULT_CONFIG_DIR.joinpath(str(task_path).lstrip("/")))
        candidates.append(REPO_ROOT.joinpath(task_path))

    for candidate in candidates:
        if candidate.is_file():
            return OmegaConf.load(candidate)

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find explicit task config {task_ref!r}. Searched:\n{searched}"
    )


def _resolve_explicit_task_config(cfg: OmegaConf):
    task_ref = OmegaConf.select(cfg, "task", default=None)
    if not isinstance(task_ref, str):
        return cfg

    cfg.task = _load_task_config_from_ref(task_ref)
    return cfg


def _find_cli_option_value(args, option):
    for idx, arg in enumerate(args):
        if arg == option and idx + 1 < len(args):
            return args[idx + 1], idx, False
        prefix = option + "="
        if arg.startswith(prefix):
            return arg[len(prefix):], idx, True
    return None, None, None


def _replace_cli_option_value(args, option, value):
    old_value, idx, inline = _find_cli_option_value(args, option)
    if idx is None:
        return args + [f"{option}={value}"]
    args = list(args)
    if inline:
        args[idx] = f"{option}={value}"
    else:
        args[idx + 1] = value
    return args


def _materialize_explicit_task_config_argv(args):
    global _TEMP_CONFIG_DIR

    config_dir, _, _ = _find_cli_option_value(args, "--config-dir")
    config_name, _, _ = _find_cli_option_value(args, "--config-name")
    if config_dir is None or config_name is None:
        return args

    config_dir_path = pathlib.Path(config_dir)
    if not config_dir_path.is_absolute():
        config_dir_path = REPO_ROOT.joinpath(config_dir_path)
    config_path = config_dir_path.joinpath(config_name)
    if not config_path.is_file():
        return args

    cfg = OmegaConf.load(config_path)
    task_ref = OmegaConf.select(cfg, "task", default=None)
    if not isinstance(task_ref, str):
        return args

    task_cfg = _load_task_config_from_ref(
        task_ref,
        config_dirs=[config_path.parent, DEFAULT_CONFIG_DIR],
    )
    cfg.task = task_cfg

    _TEMP_CONFIG_DIR = tempfile.TemporaryDirectory(prefix="diffusion_policy_cfg_")
    temp_config_dir = pathlib.Path(_TEMP_CONFIG_DIR.name)
    temp_config_path = temp_config_dir.joinpath(config_path.name)
    OmegaConf.save(config=cfg, f=temp_config_path, resolve=False)

    args = _replace_cli_option_value(args, "--config-dir", str(temp_config_dir))
    args = _replace_cli_option_value(args, "--config-name", temp_config_path.name)
    return args


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    cfg = _resolve_explicit_task_config(cfg)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    
    cls = hydra.utils.get_class(cfg._target_)
    resume_output_dir = _infer_resume_output_dir(cfg)
    if resume_output_dir is not None:
        print(f"Using resume output_dir: {resume_output_dir}")
    workspace: BaseWorkspace = cls(cfg, output_dir=resume_output_dir)
    workspace.run()

if __name__ == "__main__":
    default_config_dir = DEFAULT_CONFIG_DIR
    sys.argv = rewrite_config_reference_argv(
        sys.argv,
        default_config_dir=str(default_config_dir),
    )
    sys.argv = _materialize_explicit_task_config_argv(sys.argv)
    main()
