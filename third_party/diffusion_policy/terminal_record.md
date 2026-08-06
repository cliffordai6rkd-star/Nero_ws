# Training

## 
- choose the specific training yaml
- check the task yaml and change the corresponding dataset path

## single seed GPU trainning
```bash
python train.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_unet_abs_jp.yaml dataset_path=/home/zyx/dataset/dp/fr3/0920/water_pouring_1_step_0_skip_abs_jps.zarr training.seed=42 training.device=cuda:6 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_water_pouring_unet_abs_jp'

python train.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_unet_abs_jp.yaml dataset_path=/home/zyx/dataset/dp/fr3/0915/block_stacking_1_step_0_skip_abs_jps.zarr training.seed=42 training.device=cuda:7 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_block_stacking_unet_abs_jp'

python train.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_unet_abs_jp.yaml dataset_path=/home/zyx/dataset/dp/fr3/0920/solid_transfer_1_step_0_skip_abs_jps.zarr training.seed=42 training.device=cuda:7 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_solid_transfer_unet_abs_jp_ds'

python train.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_unet_abs_jp.yaml dataset_path=/home/zyx/dataset/dp/fr3/1022/pick_n_place_1_step_2_skip_abs_jps.zarr training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_pick_n_place_unet_abs_jp_ds'

python train.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_unet_abs_jp.yaml dataset_path=/home/zyx/dataset/dp/fr3/1022/block_stacking_1_step_3_skip_abs_jps.zarr training.seed=42 training.device=cuda:1 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_block_stacking_unet_abs_jp_ds'

python train.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_unet_abs_jp.yaml dataset_path=/home/zyx/dataset/dp/fr3/1020/insert_tube_1_step_2_skip_abs_jps.zarr training.seed=42 training.device=cuda:3 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_insert_tube_unet_abs_jp_ds'

python train.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_unet_jps_or_ee_state2euler.yaml dataset_path=/home/zyx/dataset/dp/fr3/1020/insert_tube_1_step_2_skip_jps2pose_euler.zarr training.seed=42 training.device=cuda:4 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_insert_tube_unet_jps2pose_euler'
python train.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_unet_jps_or_ee_state2euler.yaml dataset_path=/home/zyx/dataset/dp/fr3/1020/insert_tube_1_step_2_skip_jps2delPose_euler.zarr training.seed=42 training.device=cuda:5 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_insert_tube_unet_jps2delPose_euler'
python train.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_unet_jps_or_ee_state2euler.yaml dataset_path=/home/zyx/dataset/dp/fr3/1020/insert_tube_1_step_2_skip_pose2delPose_euler.zarr training.seed=42 training.device=cuda:6 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_insert_tube_unet_pose2delPose_euler'
```

## multiple gpu:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # select GPUs to be managed by the ray cluster
ray start --head --num-gpus=4
python ray_train_multirun.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp --seeds=42,43,44,45 --monitor_key=test/mean_score -- multi_run.run_dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_pick_n_place_unet_abs_jp' multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_pick_n_place_unet_abs_jp'
```