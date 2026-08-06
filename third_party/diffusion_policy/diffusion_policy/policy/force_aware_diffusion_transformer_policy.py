from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pose_util import (
    normalize_quaternion_xyzw,
    relative_pose_to_absolute_pose,
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
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def mean_pose_chunk(action: torch.Tensor, quaternion_eps: float = 1e-8) -> torch.Tensor:
    """Average xyz arithmetically and xyzw quaternions on a common hemisphere."""
    if action.ndim != 3:
        raise ValueError(f"action must be [B, T, D], got {tuple(action.shape)}")
    if action.shape[1] < 1:
        raise ValueError("cannot aggregate an empty action chunk")
    if action.shape[-1] != 7:
        return action.mean(dim=1)

    position = action[..., :3].mean(dim=1)
    quaternion = action[..., 3:7]
    reference = quaternion[:, :1]
    signs = torch.where(
        (quaternion * reference).sum(dim=-1, keepdim=True) < 0,
        -torch.ones((), dtype=quaternion.dtype, device=quaternion.device),
        torch.ones((), dtype=quaternion.dtype, device=quaternion.device),
    )
    aligned = quaternion * signs
    quaternion_mean = aligned.mean(dim=1)
    norm = torch.linalg.vector_norm(quaternion_mean, dim=-1, keepdim=True)
    fallback = F.normalize(reference[:, 0], dim=-1, eps=quaternion_eps)
    quaternion_mean = torch.where(
        norm > quaternion_eps,
        quaternion_mean / norm.clamp_min(quaternion_eps),
        fallback,
    )
    return torch.cat([position, quaternion_mean], dim=-1)


def normalize_pose_quaternions(action: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if action.shape[-1] != 7:
        return action
    normalized = normalize_quaternion_xyzw(action[..., 3:7], eps=eps)
    return torch.cat([action[..., :3], normalized], dim=-1)


class ForceAwareDiffusionTransformerPolicy(BaseImagePolicy):
    """Image-wrench Diffusion Transformer for contact-rich manipulation."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: SchedulerMixin,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        action_start_index: Optional[int] = None,
        image_key: str = "wrist",
        wrench_key: str = "wrench_ext",
        relative_pose_actions: bool = False,
        action_reference_key: str = "action_reference",
        num_inference_steps: Optional[int] = None,
        dino_model_name_or_path: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        dino_local_files_only: bool = True,
        freeze_dino_backbone: bool = True,
        wrench_history_steps: int = 8,
        n_emb: int = 256,
        n_head: int = 4,
        n_layer: int = 8,
        n_cond_layers: int = 1,
        force_temporal_encoder: str = "gru",
        force_encoder_layers: int = 1,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        contact_threshold: float = 5.0,
        contact_force_dims: Sequence[int] = (0, 1, 2),
        contact_history_reducer: str = "max",
        image_mask_scope: str = "full_context",
        mask_schedule: Optional[dict] = None,
        obs_encoder: Optional[ForceAwareObsEncoder] = None,
        action_expert: Optional[nn.Module] = None,
        **scheduler_step_kwargs,
    ) -> None:
        super().__init__()
        if set(shape_meta.get("obs", {})) != {image_key, wrench_key}:
            raise ValueError(
                "Force-aware DP shape_meta must contain exactly the configured image "
                f"and wrench keys ({image_key!r}, {wrench_key!r})."
            )
        image_meta = shape_meta["obs"][image_key]
        wrench_meta = shape_meta["obs"][wrench_key]
        if image_meta.get("type") != "rgb":
            raise ValueError(f"{image_key!r} must be an rgb observation")
        wrench_shape = tuple(wrench_meta["shape"])
        if wrench_shape != (wrench_history_steps, 6):
            raise ValueError(
                f"{wrench_key!r} must have shape ({wrench_history_steps}, 6), "
                f"got {wrench_shape}"
            )
        action_shape = tuple(shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise ValueError("action shape must be one-dimensional per timestep")
        if action_start_index is None:
            action_start_index = n_obs_steps - 1
        if not 0 <= action_start_index < horizon:
            raise ValueError("action_start_index must be in [0, horizon)")
        if not 1 <= n_action_steps <= horizon - action_start_index:
            raise ValueError(
                "n_action_steps must fit between action_start_index and horizon"
            )

        action_dim = action_shape[0]
        if relative_pose_actions and action_dim != 7:
            raise ValueError(
                "relative_pose_actions requires action shape [7] in xyz + xyzw format"
            )
        if not action_reference_key:
            raise ValueError("action_reference_key must not be empty")
        if obs_encoder is None:
            obs_encoder = ForceAwareObsEncoder(
                pretrained_model_name_or_path=dino_model_name_or_path,
                n_emb=n_emb,
                n_head=n_head,
                max_obs_steps=n_obs_steps,
                wrench_dim=wrench_shape[-1],
                wrench_history_steps=wrench_history_steps,
                force_temporal_encoder=force_temporal_encoder,
                force_encoder_layers=force_encoder_layers,
                dropout=p_drop_attn,
                freeze_backbone=freeze_dino_backbone,
                local_files_only=dino_local_files_only,
            )
        context_tokens = n_obs_steps * obs_encoder.context_tokens_per_observation
        if action_expert is None:
            action_expert = ContextTransformerForDiffusion(
                input_dim=action_dim,
                output_dim=action_dim,
                horizon=horizon,
                n_emb=n_emb,
                n_head=n_head,
                n_layer=n_layer,
                n_cond_layers=n_cond_layers,
                max_context_tokens=context_tokens,
                p_drop_emb=p_drop_emb,
                p_drop_attn=p_drop_attn,
                causal_attn=causal_attn,
            )

        self.obs_encoder = obs_encoder
        self.model = action_expert
        self.noise_scheduler = noise_scheduler
        self.contact_detector = ContactDetector(
            threshold=contact_threshold,
            force_dims=contact_force_dims,
            history_reducer=contact_history_reducer,
        )
        self.image_masker = ContactAwareImageMasker(scope=image_mask_scope)
        self.mask_scheduler = MaskProbabilityScheduler(**(mask_schedule or {}))
        self.normalizer = LinearNormalizer()

        self.horizon = int(horizon)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.action_start_index = int(action_start_index)
        self.action_dim = int(action_dim)
        self.image_key = image_key
        self.wrench_key = wrench_key
        self.relative_pose_actions = bool(relative_pose_actions)
        self.action_reference_key = str(action_reference_key)
        self.scheduler_step_kwargs = scheduler_step_kwargs
        self.optimizer_step = 0
        self.last_curriculum_metrics: Dict[str, float] = {}
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(noise_scheduler.config.num_train_timesteps)
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_optimizer_step(self, optimizer_step: int) -> None:
        self.optimizer_step = int(optimizer_step)

    def _validate_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        missing = {self.image_key, self.wrench_key} - set(obs_dict)
        if missing:
            raise KeyError(f"missing force-aware observations: {sorted(missing)}")
        image = obs_dict[self.image_key][:, : self.n_obs_steps]
        raw_wrench = obs_dict[self.wrench_key][:, : self.n_obs_steps]
        if image.shape[1] != self.n_obs_steps or raw_wrench.shape[1] != self.n_obs_steps:
            raise ValueError(f"policy requires {self.n_obs_steps} observation steps")
        return image, raw_wrench

    def _encode_context(
        self,
        obs_dict: Dict[str, torch.Tensor],
        apply_curriculum_mask: bool,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        image, raw_wrench = self._validate_obs(obs_dict)
        normalized_wrench = self.normalizer[self.wrench_key].normalize(raw_wrench)
        contact = self.contact_detector(raw_wrench)
        # Contact detection intentionally uses the raw wrench in physical units.
        # Once normalized, hide the complete wrench history for each observation
        # that has not crossed the contact threshold. This keeps pre-contact
        # behavior image-conditioned without changing the force encoder shape.
        normalized_wrench = normalized_wrench * contact.to(
            dtype=normalized_wrench.dtype
        )[..., None, None]
        probability = self.mask_scheduler.probability(self.optimizer_step)
        image_mask = self.image_masker(
            contact,
            probability=probability,
            enabled=apply_curriculum_mask,
            generator=generator,
        )
        self.last_curriculum_metrics = {
            "curriculum/mask_probability": probability,
            "curriculum/contact_fraction": contact.float().mean().item(),
            "curriculum/masked_image_fraction": image_mask.float().mean().item(),
            "curriculum/optimizer_step": float(self.optimizer_step),
        }
        return self.obs_encoder(
            image=image,
            wrench_history=normalized_wrench,
            image_token_mask=image_mask,
        )

    def conditional_sample(
        self,
        shape: Tuple[int, int, int],
        context: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            shape,
            device=context.device,
            dtype=context.dtype,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(
            self.num_inference_steps,
            device=context.device,
        )
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, timestep, context)
            trajectory = self.noise_scheduler.step(
                prediction,
                timestep,
                trajectory,
                generator=generator,
                **self.scheduler_step_kwargs,
            ).prev_sample
        return trajectory

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action conditioning is not implemented")
        context = self._encode_context(obs_dict, apply_curriculum_mask=False)
        batch_size = context.shape[0]
        normalized_prediction = self.conditional_sample(
            (batch_size, self.horizon, self.action_dim),
            context=context,
            generator=generator,
        )
        model_action_prediction = self.normalizer["action"].unnormalize(
            normalized_prediction
        )
        model_action_prediction = normalize_pose_quaternions(model_action_prediction)
        action_prediction = model_action_prediction
        if self.relative_pose_actions:
            if self.action_reference_key not in obs_dict:
                raise KeyError(
                    f"relative pose prediction requires obs[{self.action_reference_key!r}]"
                )
            action_reference = obs_dict[self.action_reference_key]
            if action_reference.ndim == 3:
                action_reference = action_reference[:, -1]
            if action_reference.shape != (batch_size, 7):
                raise ValueError(
                    f"obs[{self.action_reference_key!r}] must have shape [B,7] "
                    f"or [B,T,7], got {tuple(obs_dict[self.action_reference_key].shape)}"
                )
            action_prediction = relative_pose_to_absolute_pose(
                model_action_prediction, action_reference
            )
        start = self.action_start_index
        end = start + self.n_action_steps
        selected_action = action_prediction[:, start:end]
        if selected_action.shape[1] == 0:
            raise RuntimeError("selected action chunk is empty")
        return {
            "action": selected_action,
            "action_pred": action_prediction,
            "model_action_pred": model_action_prediction,
            "action_target": mean_pose_chunk(selected_action),
        }

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer_step: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if "valid_mask" in batch:
            raise NotImplementedError("valid_mask loss weighting is not implemented")
        if optimizer_step is not None:
            self.set_optimizer_step(optimizer_step)
        action = batch["action"]
        if action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"action must be [B, {self.horizon}, {self.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        normalized_action = self.normalizer["action"].normalize(action)
        context = self._encode_context(
            batch["obs"],
            apply_curriculum_mask=self.training,
            generator=generator,
        )
        noise = torch.randn(
            normalized_action.shape,
            device=normalized_action.device,
            dtype=normalized_action.dtype,
            generator=generator,
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (normalized_action.shape[0],),
            device=normalized_action.device,
            generator=generator,
        ).long()
        noisy_action = self.noise_scheduler.add_noise(
            normalized_action,
            noise,
            timesteps,
        )
        prediction = self.model(noisy_action, timesteps, context)
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = normalized_action
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                normalized_action,
                noise,
                timesteps,
            )
        else:
            raise ValueError(f"unsupported prediction type {prediction_type!r}")
        return F.mse_loss(prediction, target)

    def get_optimizer(
        self,
        learning_rate: float,
        action_expert_weight_decay: float = 1e-3,
        fusion_weight_decay: float = 1e-6,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        action_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        fusion_parameters = [
            parameter for parameter in self.obs_encoder.parameters() if parameter.requires_grad
        ]
        return torch.optim.AdamW(
            [
                {"params": action_parameters, "weight_decay": action_expert_weight_decay},
                {"params": fusion_parameters, "weight_decay": fusion_weight_decay},
            ],
            lr=learning_rate,
            betas=betas,
        )
