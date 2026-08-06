from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.diffusion.context_transformer_for_diffusion import (
    ContextTransformerForDiffusion,
)
from diffusion_policy.model.vision.contact_curriculum import (
    ContactAwareImageMasker,
    ContactDetector,
    MaskProbabilityScheduler,
)
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import (
    ForceAwareDiffusionTransformerPolicy,
    mean_pose_chunk,
    normalize_pose_quaternions,
)


class FakeDinoBackbone(nn.Module):
    def __init__(self, hidden_size=16, num_register_tokens=2, patch_count=6):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_register_tokens=num_register_tokens,
        )
        self.patch_count = patch_count
        self.projection = nn.Linear(3, hidden_size)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        pooled = pixel_values.mean(dim=(-2, -1))
        cls = self.projection(pooled).unsqueeze(1)
        registers = torch.zeros(
            pixel_values.shape[0],
            self.config.num_register_tokens,
            self.config.hidden_size,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        patches = self.projection(pooled).unsqueeze(1).expand(
            -1, self.patch_count, -1
        )
        return SimpleNamespace(
            last_hidden_state=torch.cat([cls, registers, patches], dim=1)
        )


def make_obs_encoder(n_emb=16, temporal_encoder="gru"):
    return ForceAwareObsEncoder(
        pretrained_model_name_or_path="unused",
        n_emb=n_emb,
        n_head=4,
        max_obs_steps=2,
        wrench_dim=6,
        wrench_history_steps=8,
        force_temporal_encoder=temporal_encoder,
        force_encoder_layers=1,
        dropout=0.0,
        freeze_backbone=True,
        backbone=FakeDinoBackbone(hidden_size=12),
    )


def make_normalizer():
    normalizer = LinearNormalizer()
    normalizer["wrench_ext"] = SingleFieldLinearNormalizer.create_fit(
        torch.linspace(-20.0, 20.0, 600).reshape(100, 6),
        last_n_dims=1,
    )
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
        torch.linspace(-2.0, 2.0, 700).reshape(100, 7),
        last_n_dims=1,
    )
    return normalizer


def make_policy(mask_scope="full_context"):
    encoder = make_obs_encoder()
    expert = ContextTransformerForDiffusion(
        input_dim=7,
        output_dim=7,
        horizon=8,
        n_emb=16,
        n_head=4,
        n_layer=1,
        n_cond_layers=1,
        max_context_tokens=16,
        p_drop_attn=0.0,
    )
    scheduler = DDIMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )
    policy = ForceAwareDiffusionTransformerPolicy(
        shape_meta={
            "obs": {
                "wrist": {"shape": [3, 16, 16], "type": "rgb"},
                "wrench_ext": {"shape": [8, 6], "type": "low_dim"},
            },
            "action": {"shape": [7]},
        },
        noise_scheduler=scheduler,
        horizon=8,
        n_action_steps=7,
        n_obs_steps=2,
        image_key="wrist",
        wrench_key="wrench_ext",
        num_inference_steps=2,
        n_emb=16,
        contact_threshold=5.0,
        image_mask_scope=mask_scope,
        mask_schedule={
            "schedule_type": "cosine",
            "start_step": 0,
            "end_step": 100,
            "start_probability": 1.0,
            "end_probability": 0.0,
        },
        obs_encoder=encoder,
        action_expert=expert,
    )
    policy.set_normalizer(make_normalizer())
    return policy


def test_contact_detector_uses_physical_wrench_units():
    wrench = torch.zeros(2, 2, 8, 6)
    wrench[0, -1, -2, 0] = 5.1
    wrench[1, -1, -2, 0] = 4.9
    contact = ContactDetector(5.0, history_reducer="max")(wrench)
    assert contact.tolist() == [[False, True], [False, False]]


@pytest.mark.parametrize("schedule_type", ["linear", "cosine", "exponential"])
def test_mask_scheduler_has_exact_endpoints(schedule_type):
    scheduler = MaskProbabilityScheduler(
        schedule_type=schedule_type,
        start_step=10,
        end_step=110,
        start_probability=0.9,
        end_probability=0.1,
    )
    assert scheduler.probability(0) == pytest.approx(0.9)
    assert scheduler.probability(10) == pytest.approx(0.9)
    assert 0.1 < scheduler.probability(60) < 0.9
    assert scheduler.probability(110) == pytest.approx(0.1)
    assert scheduler.probability(1000) == pytest.approx(0.1)


def test_piecewise_scheduler_and_mask_scopes():
    scheduler = MaskProbabilityScheduler(
        schedule_type="piecewise",
        piecewise_steps=[0, 10, 20],
        piecewise_probabilities=[1.0, 0.5, 0.0],
    )
    assert scheduler.probability(5) == pytest.approx(0.75)
    contact = torch.tensor([[False, True], [True, False]])
    current = ContactAwareImageMasker("current_observation")(
        contact, probability=1.0
    )
    full = ContactAwareImageMasker("full_context")(contact, probability=1.0)
    assert current.tolist() == [[False, True], [False, False]]
    assert full.tolist() == [[True, True], [False, False]]


@pytest.mark.parametrize("temporal_encoder", ["gru", "lstm", "transformer", "none"])
def test_force_encoder_returns_eight_tokens_per_observation(temporal_encoder):
    encoder = make_obs_encoder(temporal_encoder=temporal_encoder)
    encoder.train()
    context = encoder(
        image=torch.rand(2, 2, 3, 16, 16),
        wrench_history=torch.randn(2, 2, 8, 6),
        image_token_mask=torch.tensor([[False, True], [False, False]]),
    )
    assert context.shape == (2, 16, 16)
    assert not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())
    context.sum().backward()
    assert all(parameter.grad is None for parameter in encoder.backbone.parameters())
    assert encoder.wrench_projection.weight.grad is not None


def test_context_transformer_output_shape():
    model = ContextTransformerForDiffusion(
        input_dim=7,
        output_dim=7,
        horizon=8,
        n_emb=16,
        n_head=4,
        n_layer=1,
        max_context_tokens=16,
    )
    output = model(
        sample=torch.randn(3, 8, 7),
        timestep=torch.tensor([1, 2, 3]),
        context=torch.randn(3, 16, 16),
    )
    assert output.shape == (3, 8, 7)


def test_policy_loss_masks_training_contact_and_backpropagates():
    policy = make_policy()
    policy.train()
    batch = {
        "obs": {
            "wrist": torch.rand(2, 2, 3, 16, 16),
            "wrench_ext": torch.zeros(2, 2, 8, 6),
        },
        "action": torch.randn(2, 8, 7),
    }
    batch["obs"]["wrench_ext"][0, -1, -1, 0] = 10.0
    loss = policy.compute_loss(batch, optimizer_step=0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert policy.last_curriculum_metrics["curriculum/contact_fraction"] == pytest.approx(0.25)
    assert policy.last_curriculum_metrics["curriculum/masked_image_fraction"] == pytest.approx(0.5)
    loss.backward()
    assert policy.model.output_head.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in policy.obs_encoder.backbone.parameters()
    )


def test_policy_masks_normalized_wrench_before_contact():
    policy = make_policy()
    raw_wrench = torch.full((1, 2, 8, 6), 1.0)
    raw_wrench[:, 1, -1, 0] = 10.0
    expected_normalized = policy.normalizer["wrench_ext"].normalize(raw_wrench)
    captured = {}
    original_forward = policy.obs_encoder.forward

    def capture_forward(image, wrench_history, image_token_mask=None):
        captured["wrench_history"] = wrench_history.detach().clone()
        return original_forward(
            image=image,
            wrench_history=wrench_history,
            image_token_mask=image_token_mask,
        )

    policy.obs_encoder.forward = capture_forward
    policy._encode_context(
        {
            "wrist": torch.rand(1, 2, 3, 16, 16),
            "wrench_ext": raw_wrench,
        },
        apply_curriculum_mask=False,
    )

    gated_wrench = captured["wrench_history"]
    assert torch.count_nonzero(gated_wrench[:, 0]) == 0
    assert torch.allclose(gated_wrench[:, 1], expected_normalized[:, 1])


def test_policy_prediction_is_unmasked_and_returns_pose_target():
    policy = make_policy()
    policy.eval()
    obs = {
        "wrist": torch.rand(2, 2, 3, 16, 16),
        "wrench_ext": torch.zeros(2, 2, 8, 6),
    }
    obs["wrench_ext"][:, -1, -1, 0] = 10.0
    result = policy.predict_action(obs, generator=torch.Generator().manual_seed(1))
    assert result["action_pred"].shape == (2, 8, 7)
    assert result["action"].shape == (2, 7, 7)
    assert result["action_target"].shape == (2, 7)
    assert torch.allclose(
        torch.linalg.vector_norm(result["action_target"][:, 3:7], dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    assert policy.last_curriculum_metrics["curriculum/masked_image_fraction"] == 0.0


def test_pose_mean_sign_aligns_quaternions():
    action = torch.tensor(
        [[
            [0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 4.0, 6.0, 0.0, 0.0, 0.0, -1.0],
        ]]
    )
    target = mean_pose_chunk(action)
    assert torch.allclose(target[0, :3], torch.tensor([1.0, 3.0, 5.0]))
    assert torch.allclose(target[0, 3:], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_pose_quaternion_normalization_has_identity_fallback():
    action = torch.zeros(1, 2, 7)
    action[0, 1, 3:7] = torch.tensor([0.0, 0.0, 0.0, 2.0])
    normalized = normalize_pose_quaternions(action)
    assert torch.allclose(
        normalized[..., 3:7],
        torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]),
    )
