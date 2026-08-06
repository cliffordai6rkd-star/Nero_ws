from __future__ import annotations

import pytest

from inference.checkpoints import _register_checkpoint_resolvers, _safe_arithmetic_eval


def test_safe_arithmetic_eval_resolves_current_dp_horizon() -> None:
    assert _safe_arithmetic_eval("8*8") == 64
    assert _safe_arithmetic_eval("(8 + 2) // 2") == 5


@pytest.mark.parametrize(
    "expression",
    (
        "__import__('os').system('echo unsafe')",
        "open('/tmp/unsafe', 'w')",
        "value.attribute",
        "[8, 8]",
    ),
)
def test_safe_arithmetic_eval_rejects_python_execution(expression: str) -> None:
    with pytest.raises(ValueError, match="arithmetic expression"):
        _safe_arithmetic_eval(expression)


def test_checkpoint_eval_resolver_resolves_nested_dp_config() -> None:
    omegaconf = pytest.importorskip("omegaconf")
    _register_checkpoint_resolvers(omegaconf.OmegaConf)
    config = omegaconf.OmegaConf.create(
        {
            "action_horizon": 8,
            "action_chunk_steps": 8,
            "horizon": "${eval:'${action_horizon}*${action_chunk_steps}'}",
            "policy": {"horizon": "${horizon}"},
        }
    )
    omegaconf.OmegaConf.resolve(config)
    assert config.horizon == 64
    assert config.policy.horizon == 64
