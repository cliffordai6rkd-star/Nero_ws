from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


class ContextTransformerForDiffusion(ModuleAttrMixin):
    """Denoise an action sequence while attending to arbitrary context tokens."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_emb: int = 256,
        n_head: int = 4,
        n_layer: int = 8,
        n_cond_layers: int = 1,
        max_context_tokens: int = 64,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
    ) -> None:
        super().__init__()
        if n_emb % n_head != 0:
            raise ValueError("n_emb must be divisible by n_head")
        if horizon < 1 or max_context_tokens < 1:
            raise ValueError("horizon and max_context_tokens must be positive")

        self.horizon = int(horizon)
        self.max_context_tokens = int(max_context_tokens)
        self.n_emb = int(n_emb)

        self.input_embedding = nn.Linear(input_dim, n_emb)
        self.action_position = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.context_position = nn.Parameter(
            torch.zeros(1, max_context_tokens + 1, n_emb)
        )
        self.time_embedding = SinusoidalPosEmb(n_emb)
        self.embedding_dropout = nn.Dropout(p_drop_emb)

        if n_cond_layers > 0:
            condition_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.condition_encoder = nn.TransformerEncoder(
                condition_layer,
                num_layers=n_cond_layers,
            )
        else:
            self.condition_encoder = nn.Identity()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.output_norm = nn.LayerNorm(n_emb)
        self.output_head = nn.Linear(n_emb, output_dim)

        if causal_attn:
            mask = torch.full((horizon, horizon), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("action_mask", mask, persistent=False)
        else:
            self.action_mask = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        context: torch.Tensor,
    ) -> torch.Tensor:
        if sample.ndim != 3:
            raise ValueError(f"sample must be [B, H, Da], got {tuple(sample.shape)}")
        if context.ndim != 3:
            raise ValueError(f"context must be [B, S, D], got {tuple(context.shape)}")
        batch_size, action_steps = sample.shape[:2]
        if action_steps > self.horizon:
            raise ValueError(f"action steps {action_steps} exceed horizon {self.horizon}")
        if context.shape[0] != batch_size or context.shape[2] != self.n_emb:
            raise ValueError("context batch/embedding dimensions do not match the model")
        if context.shape[1] > self.max_context_tokens:
            raise ValueError(
                f"context has {context.shape[1]} tokens, maximum is {self.max_context_tokens}"
            )

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        else:
            timestep = timestep.to(sample.device)
        timestep = timestep.expand(batch_size)

        time_token = self.time_embedding(timestep).unsqueeze(1)
        memory = torch.cat([time_token, context], dim=1)
        memory = self.embedding_dropout(
            memory + self.context_position[:, : memory.shape[1]]
        )
        memory = self.condition_encoder(memory)

        action_tokens = self.input_embedding(sample)
        action_tokens = self.embedding_dropout(
            action_tokens + self.action_position[:, :action_steps]
        )
        decoded = self.decoder(
            tgt=action_tokens,
            memory=memory,
            tgt_mask=(
                None
                if self.action_mask is None
                else self.action_mask[:action_steps, :action_steps]
            ),
        )
        return self.output_head(self.output_norm(decoded))

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
