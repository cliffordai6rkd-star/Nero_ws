if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import json
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy, time
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.memory_budget import (
    build_memory_limited_dataloader_kwargs,
    set_process_memory_limit,
)
import logging as log

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def _freeze_obs_encoder(self):
        self.model.obs_encoder.eval()
        self.model.obs_encoder.requires_grad_(False)

    def _collect_model_stats(self):
        policy_total_params = sum(p.numel() for p in self.model.parameters())
        policy_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        obs_encoder_total_params = sum(p.numel() for p in self.model.obs_encoder.parameters())
        obs_encoder_trainable_params = sum(
            p.numel() for p in self.model.obs_encoder.parameters() if p.requires_grad
        )
        diffusion_total_params = sum(p.numel() for p in self.model.model.parameters())
        diffusion_trainable_params = sum(
            p.numel() for p in self.model.model.parameters() if p.requires_grad
        )
        return {
            'policy_total_params': int(policy_total_params),
            'policy_trainable_params': int(policy_trainable_params),
            'obs_encoder_total_params': int(obs_encoder_total_params),
            'obs_encoder_trainable_params': int(obs_encoder_trainable_params),
            'diffusion_total_params': int(diffusion_total_params),
            'diffusion_trainable_params': int(diffusion_trainable_params),
        }

    def _save_model_stats(self, model_stats):
        stats_path = pathlib.Path(self.output_dir).joinpath('model_stats.json')
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open('w', encoding='utf-8') as f:
            json.dump(model_stats, f, indent=2, sort_keys=True)
        return stats_path

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        max_ram_gb = OmegaConf.select(cfg, "training.max_ram_gb")
        memory_reserve_gb = OmegaConf.select(cfg, "training.memory_reserve_gb", default=2.0)
        enforce_process_ram_limit = bool(
            OmegaConf.select(cfg, "training.enforce_process_ram_limit", default=False)
        )

        if enforce_process_ram_limit and max_ram_gb is not None and str(cfg.training.device).startswith("cuda"):
            log.warning(
                "Skipping RLIMIT_AS RAM cap on CUDA training because it breaks cuDNN/CUDA initialization. "
                "Using dataloader/dataset memory budgeting only."
            )
        elif enforce_process_ram_limit and max_ram_gb is not None:
            set_process_memory_limit(max_ram_gb)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                log.info(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset_kwargs = dict()
        memory_aware_dataset_targets = {
            "diffusion_policy.dataset.hirol_dataset.HirolDataset",
            "diffusion_policy.dataset.hirol_lerobot_v3_dataset.HirolLeRobotV3Dataset",
        }
        if cfg.task.dataset.get("_target_") in memory_aware_dataset_targets:
            dataset_kwargs["memory_limit_gb"] = max_ram_gb
            dataset_kwargs["memory_reserve_gb"] = memory_reserve_gb
        dataset = hydra.utils.instantiate(cfg.task.dataset, **dataset_kwargs)
        assert isinstance(dataset, BaseImageDataset)
        shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
        train_dataloader_kwargs = build_memory_limited_dataloader_kwargs(
            dataloader_cfg=OmegaConf.to_container(cfg.dataloader, resolve=True),
            shape_meta=shape_meta,
            n_obs_steps=cfg.dataset_obs_steps,
            action_horizon=cfg.horizon,
            memory_limit_gb=max_ram_gb,
            memory_reserve_gb=memory_reserve_gb,
            loader_name="train",
        )
        train_dataloader = DataLoader(dataset, **train_dataloader_kwargs)
        normalizer = dataset.get_normalizer(**cfg.get("normalizer", {}))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader_kwargs = build_memory_limited_dataloader_kwargs(
            dataloader_cfg=OmegaConf.to_container(cfg.val_dataloader, resolve=True),
            shape_meta=shape_meta,
            n_obs_steps=cfg.dataset_obs_steps,
            action_horizon=cfg.horizon,
            memory_limit_gb=max_ram_gb,
            memory_reserve_gb=memory_reserve_gb,
            loader_name="val",
        )
        val_dataloader = DataLoader(val_dataset, **val_dataloader_kwargs)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        self.update_wandb_output_dir()

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        if cfg.training.freeze_encoder:
            self._freeze_obs_encoder()

        model_stats = self._collect_model_stats()
        model_stats['freeze_encoder'] = bool(cfg.training.freeze_encoder)
        stats_path = self._save_model_stats(model_stats)
        log.info('Saved model stats to %s', stats_path)
        log.info(
            'Policy params: total=%s trainable=%s',
            f"{model_stats['policy_total_params']:,}",
            f"{model_stats['policy_trainable_params']:,}",
        )
        log.info(
            'Obs encoder params: total=%s trainable=%s',
            f"{model_stats['obs_encoder_total_params']:,}",
            f"{model_stats['obs_encoder_trainable_params']:,}",
        )
        log.info(
            'Diffusion params: total=%s trainable=%s',
            f"{model_stats['diffusion_total_params']:,}",
            f"{model_stats['diffusion_trainable_params']:,}",
        )

        # device transfer
        device = torch.device(cfg.training.device)
        log.info(f"DEBUG: cfg.training.device = {cfg.training.device}")
        log.info(f"DEBUG: device = {device}")
        log.info(f"DEBUG: Available CUDA devices: {torch.cuda.device_count()}")
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        debug_print_steps = max(1, int(OmegaConf.select(cfg, "training.debug_print_steps", default=100)))
        wandb_rgb_steps = OmegaConf.select(cfg, "training.wandb_rgb_steps", default=None)
        if wandb_rgb_steps is not None:
            wandb_rgb_steps = max(1, int(wandb_rgb_steps))

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self._freeze_obs_encoder()

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    loading_batch_start = time.perf_counter()
                    for batch_idx, batch in enumerate(tepoch):
                        batch_loading = time.perf_counter() - loading_batch_start
                        # if batch_idx % debug_print_steps == 0:
                        #     log.info(f'batch loading time: {batch_loading}')
                        iteration_start_time = time.perf_counter()
                        # 总时间计时开始
                        batch_start_time = time.perf_counter()
                        
                        # wandb log rgb input
                        if wandb_rgb_steps is not None and batch_idx % wandb_rgb_steps == 0:
                            batch_size = len(batch)
                            rand_id = np.random.randint(batch_size)
                            obs_data = batch['obs']
                            for obs_key, cur_obs_data in obs_data.items():
                                # Check if this is actually image data (3 or 4 dimensions)
                                if len(cur_obs_data.shape) >= 4:
                                    log.info(f'wanddb logging {obs_key}')
                                    img = cur_obs_data[rand_id, 0]
                                    wandb_run.log({f"input for {obs_key}": wandb.Image(img, caption=f"tensor chw [0,1] in {self.epoch}_{batch_idx}")})
                                    

                        # device transfer
                        start = time.perf_counter()
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        data_batch_time = time.perf_counter() - start
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # GPU内存使用监控
                        memory_allocated = 0
                        memory_reserved = 0
                        if batch_idx % 10 == 0:  # 每10个batch检查一次
                            torch.cuda.synchronize()
                            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB

                        # 完整的前向传播和损失计算
                        torch.cuda.synchronize()
                        forward_start = time.perf_counter()
                        raw_loss = self.model.compute_loss(batch)
                        torch.cuda.synchronize()
                        forward_time = time.perf_counter() - forward_start

                        # 反向传播时间
                        backward_start = time.perf_counter()
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        backward_time = time.perf_counter() - backward_start

                        # 优化器步骤时间
                        optimizer_time = 0
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            opt_start = time.perf_counter()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            optimizer_time = time.perf_counter() - opt_start

                        # update ema
                        ema_step_time = 0
                        if cfg.training.use_ema:
                            start = time.perf_counter()
                            ema.step(self.model)
                            ema_step_time = time.perf_counter() - start

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        
                        # 总时间
                        total_batch_time = time.perf_counter() - batch_start_time
                        # 其他未归类时间
                        accounted_time = data_batch_time + forward_time + backward_time + optimizer_time + ema_step_time
                        other_time = total_batch_time - accounted_time

                        # 详细的时间日志
                        detailed_log = {
                            'data_loading': f"{data_batch_time:.4f}",
                            'forward': f"{forward_time:.4f}",
                            'backward': f"{backward_time:.4f}",
                            'optimizer': f"{optimizer_time:.4f}",
                            'ema': f"{ema_step_time:.4f}",
                            'other': f"{other_time:.4f}",
                            'total': f"{total_batch_time:.4f}",
                            'loss': f"{raw_loss_cpu:.3f}"
                        }

                        # 添加内存信息
                        if batch_idx % debug_print_steps == 0:
                            detailed_log['mem_alloc'] = f"{memory_allocated:.1f}GB"
                            detailed_log['mem_reserved'] = f"{memory_reserved:.1f}GB"

                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        
                        # 每个step都打印详细分析用于debug
                        # if batch_idx % debug_print_steps == 0:
                        #     log.info(f"\n=== Batch {batch_idx} Detailed Timing Analysis ===")
                        #     log.info(f"Data Loading:     {data_batch_time*1000:.2f}ms ({data_batch_time/total_batch_time*100:.1f}%)")
                        #     log.info(f"Forward Pass:     {forward_time*1000:.2f}ms ({forward_time/total_batch_time*100:.1f}%)")
                        #     log.info(f"Backward Pass:    {backward_time*1000:.2f}ms ({backward_time/total_batch_time*100:.1f}%)")
                        #     log.info(f"Optimizer Step:   {optimizer_time*1000:.2f}ms ({optimizer_time/total_batch_time*100:.1f}%)")
                        #     log.info(f"EMA Update:       {ema_step_time*1000:.2f}ms ({ema_step_time/total_batch_time*100:.1f}%)")
                        #     log.info(f"Other/Overhead:   {other_time*1000:.2f}ms ({other_time/total_batch_time*100:.1f}%)")
                        #     log.info(f"Total Batch Time: {total_batch_time*1000:.2f}ms")
                        #     log.info(f"GPU Memory:       {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
                        #     log.info("=" * 60)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                        real_batch_used_time = time.perf_counter() - batch_start_time
                        torch.cuda.synchronize()  # 确保GPU操作完成
                        sync_time = time.perf_counter() - batch_start_time
                        total_iteration_time = time.perf_counter() - iteration_start_time
                        # if batch_idx % debug_print_steps == 0:
                        #     log.info(f'batch time: {real_batch_used_time:.3f}s, with sync: {sync_time:.3f}s, total iteration: {total_iteration_time:.3f}s for batch {batch_idx}')
                        loading_batch_start = time.perf_counter()
                        
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        # Always persist the final state after the training loop so resumed
        # runs also land a recoverable checkpoint even when checkpoint cadence
        # does not align with the last epoch.
        if cfg.checkpoint.save_last_ckpt:
            log.info(
                "Saving final checkpoint after training loop "
                f"(epoch={self.epoch}, global_step={self.global_step})"
            )
            self.save_checkpoint(use_thread=False)
        if cfg.checkpoint.save_last_snapshot:
            self.save_snapshot()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
