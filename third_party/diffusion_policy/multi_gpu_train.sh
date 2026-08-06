export CUDA_VISIBLE_DEVICES=0,1,2,3  # select GPUs to be managed by the ray cluster
ray start --head --num-gpus=4
python ray_train_multirun.py --config-dir=./diffusion_policy/config --config-name=train_hirol_fr3_unet_abs_jp_ee_state --seeds=42,43,44,45 --monitor_key=test/mean_score -- multi_run.run_dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_pick_n_place_unet_abs_jp' multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_pick_n_place_unet_abs_jp'
