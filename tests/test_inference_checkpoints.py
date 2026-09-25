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


@pytest.mark.parametrize('family', ['deterministic', 'legacy_deterministic', 'carswm'])
@pytest.mark.parametrize('use_ema', [True, False])
@pytest.mark.parametrize('stride', [1, 2])
def test_world_model_checkpoint_restore_and_visualization(tmp_path, family, use_ema, stride):
    import copy
    import numpy as np
    torch = pytest.importorskip('torch')
    from inference.checkpoints import _prepare_pinn_source, restore_checkpoint_model
    _prepare_pinn_source()
    from model.pinn_model.contact_world_model import ContactWorldModel
    from model.pinn_model.deterministic_world_model import DeterministicRobotStateWorldModel
    from scripts.visualize_wm_lerobotv3 import _sample_history

    cfg = {
        'dataloader': {'state_history_horizon': 6, 'prediction_horizon': 4,
                       'action_condition_horizon': 3, 'high_fps': 100, 'expert_fps': 25,
                       'dq_source': 'hardware', 'normalize_mode': 'gaussian',
                       'normalize_lowdim_keys': ['q', 'dq', 'delta_q', 'tau', 'action']},
        'model': {'inputs': ['q', 'dq', 'delta_q', 'tau'], 'outputs': ['q', 'tau'],
                  'hidden_dim': 8, 'attention_heads': 2, 'state_layers': 1,
                  'action_layers': 1, 'decoder_layers': 1, 'dropout': 0.0},
        'train': {'downsample': stride},
    }
    cls = ContactWorldModel if family == 'carswm' else DeterministicRobotStateWorldModel
    ema = cls(cfg).eval()
    raw = copy.deepcopy(ema)
    with torch.no_grad():
        next(raw.parameters()).add_(0.5)
    contract_key = 'deterministic_wm_contract' if family == 'legacy_deterministic' else 'carswm_contract'
    keys = ('q', 'dq', 'delta_q', 'tau', 'action')
    normalizer = {'normalize_mode': 'gaussian', 'normalize_lowdim_keys': list(keys),
                  'stats': {key: {'mean': torch.ones(7), 'std': torch.full((7,), 2.)} for key in keys}}
    payload = {'config': cfg, 'model_version': ema.MODEL_VERSION,
               contract_key: ema.checkpoint_contract(), 'model': ema.state_dict(),
               'model_raw': raw.state_dict(), 'normalizer': normalizer}
    path = tmp_path / 'wm.pt'
    torch.save(payload, path)
    restored = restore_checkpoint_model(path, 'cpu', use_ema=use_ema,
                                        kind='PINN', pinn_mode='contact_world_model')
    assert isinstance(restored, cls) and not restored.training
    expected = ema if use_ema else raw
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert restored._inference_normalizer_obj is not None
    inputs = {key: torch.randn(1, 6, 7) for key in keys[:-1]}
    inputs['action'] = torch.randn(1, 3, 7)
    sampled = restored.sample(inputs, num_samples=2, steps=1, solver='euler')
    assert sampled['q_pred'].shape == (1, 2, 4, 7)
    assert sampled['contact_probability'].shape == (1, 2, 4, 3)
    if family != 'carswm':
        torch.testing.assert_close(sampled['q_pred'][:, 0], sampled['q_pred'][:, 1], rtol=0, atol=0)
    history = {key: inputs[key][0].numpy() for key in keys[:-1]}
    predictions, noise, _ = _sample_history(restored, history, inputs['action'][0].numpy(),
                                             samples=2, steps=1, solver='euler',
                                             required_outputs=('q', 'tau'))
    assert predictions['q'].shape == (2, 4, 7)
    assert np.isfinite(predictions['q']).all()
    if family != 'carswm':
        assert noise is None
        normalized = {key: restored._inference_normalizer_obj.gaussian_normalize(key, value)
                      for key, value in inputs.items()}
        predicted = restored.predict(normalized)['q_pred']
        physical = restored._inference_normalizer_obj.gaussian_denormalize('q', predicted)
        np.testing.assert_allclose(predictions['q'][0], physical[0].numpy(), atol=1e-6)
        from pathlib import Path
        from inference.pi0_wm.wm import WMAdapter
        import model.pinn_model.deterministic_world_model as deterministic_module
        pinn_root = Path(deterministic_module.__file__).resolve().parents[2]
        adapter = WMAdapter({'pinn_root': str(pinn_root), 'checkpoint': str(path),
                             'device': 'cpu', 'use_ema': True, 'num_samples': 2}, 100, 25)
        physical_pi0 = adapter.infer((history, inputs['action'][0].numpy()))
        assert physical_pi0['q'].shape == (2, 4, 7)
        np.testing.assert_array_equal(physical_pi0['q'][0], physical_pi0['q'][1])

    # Exercise the public replay entry point with recorded action-token timing.
    from scripts.visualize_wm_lerobotv3 import sample, _action_condition
    arrays = {key: np.zeros((40, 7)) for key in keys}
    arrays['action'][:, 0] = np.arange(40)
    arrays['timestamp'] = np.arange(40) * 1e7
    arrays['action_index'] = np.arange(40) // 4
    np.testing.assert_array_equal(_action_condition(restored, arrays['action'], 5,
                                                    arrays['action_index'])[:, 0], [8, 12, 16])
    replay, _, _ = sample(restored, arrays, start=5, samples=2, steps=1)
    assert replay.shape == (2, 4, 7)
    payload[contract_key] = {}
    torch.save(payload, path)
    from inference.checkpoints import CheckpointError
    with pytest.raises(CheckpointError, match='contract validation failed'):
        restore_checkpoint_model(path, 'cpu', use_ema=use_ema,
                                 kind='PINN', pinn_mode='contact_world_model')
