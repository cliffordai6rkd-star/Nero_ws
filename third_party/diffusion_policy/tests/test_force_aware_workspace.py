import unittest

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    TrainForceAwareDiffusionWorkspace,
    optimizer_updates_per_epoch,
)


class _FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.seen_optimizer_steps = []
        self.last_curriculum_metrics = {}

    def compute_loss(self, batch, optimizer_step):
        self.seen_optimizer_steps.append(optimizer_step)
        self.last_curriculum_metrics = {
            "curriculum/optimizer_step": float(optimizer_step)
        }
        return (self.weight * batch["value"]).square().mean()

    def set_optimizer_step(self, optimizer_step):
        self.optimizer_step = optimizer_step


class _Logger:
    def __init__(self):
        self.records = []

    def log(self, value, **kwargs):
        self.records.append(dict(value))


class ForceAwareWorkspaceStepTest(unittest.TestCase):
    def test_optimizer_update_count(self):
        self.assertEqual(optimizer_updates_per_epoch(0, 4), 0)
        self.assertEqual(optimizer_updates_per_epoch(5, 2), 3)

    def test_accumulation_advances_curriculum_only_after_optimizer_step(self):
        workspace = TrainForceAwareDiffusionWorkspace.__new__(
            TrainForceAwareDiffusionWorkspace
        )
        workspace.model = _FakePolicy()
        workspace.optimizer = torch.optim.SGD(workspace.model.parameters(), lr=0.1)
        workspace.global_step = 0
        workspace.optimizer_step = 0
        workspace.epoch = 0

        cfg = OmegaConf.create(
            {
                "training": {
                    "max_train_steps": None,
                    "gradient_accumulate_every": 2,
                    "max_grad_norm": None,
                    "tqdm_interval_sec": 0.01,
                }
            }
        )
        dataloader = [
            {"value": torch.tensor([float(index + 1)])}
            for index in range(5)
        ]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            workspace.optimizer, lambda _: 1.0
        )
        wandb_logger = _Logger()
        json_logger = _Logger()

        workspace._train_epoch(
            cfg=cfg,
            dataloader=dataloader,
            device=torch.device("cpu"),
            lr_scheduler=scheduler,
            ema=None,
            wandb_run=wandb_logger,
            json_logger=json_logger,
        )

        self.assertEqual(workspace.global_step, 5)
        self.assertEqual(workspace.optimizer_step, 3)
        self.assertEqual(workspace.model.seen_optimizer_steps, [0, 0, 1, 1, 2])
        self.assertEqual(len(wandb_logger.records), 3)


if __name__ == "__main__":
    unittest.main()
