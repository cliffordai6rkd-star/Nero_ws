import unittest

from diffusion_policy.common.config_cli import (
    rewrite_config_reference_argv,
    split_config_reference,
)


class ConfigCliTest(unittest.TestCase):
    def test_split_config_reference_with_parent_dir(self):
        config_dir, config_name = split_config_reference(
            'diffusion_policy/config/train_zarr/train_hirol_fr3_unet_abs_jp.yaml'
        )

        self.assertEqual(config_dir, 'diffusion_policy/config/train_zarr')
        self.assertEqual(config_name, 'train_hirol_fr3_unet_abs_jp.yaml')

    def test_split_config_reference_uses_default_dir_for_bare_filename(self):
        config_dir, config_name = split_config_reference(
            'train_hirol_fr3_unet_abs_jp.yaml',
            default_config_dir='diffusion_policy/config',
        )

        self.assertEqual(config_dir, 'diffusion_policy/config')
        self.assertEqual(config_name, 'train_hirol_fr3_unet_abs_jp.yaml')

    def test_rewrite_config_reference_argv_replaces_short_flag(self):
        argv = rewrite_config_reference_argv(
            [
                'train.py',
                '-c',
                'diffusion_policy/config/train_lerobot_v3/train_hirol_fr3_unet_abs_jp_ee_state.yaml',
                'training.device=cuda:0',
            ]
        )

        self.assertEqual(
            argv,
            [
                'train.py',
                '--config-dir=diffusion_policy/config/train_lerobot_v3',
                '--config-name=train_hirol_fr3_unet_abs_jp_ee_state.yaml',
                'training.device=cuda:0',
            ],
        )

    def test_rewrite_config_reference_argv_joins_split_path_tokens(self):
        argv = rewrite_config_reference_argv(
            [
                'train.py',
                '-c',
                'diffusion_policy/config/train_lerobot_v3/train_',
                'hirol_fr3_pnp_cam_ee_joint_to_ee_unet.yaml',
                'training.device=cuda:0',
            ]
        )

        self.assertEqual(
            argv,
            [
                'train.py',
                '--config-dir=diffusion_policy/config/train_lerobot_v3',
                '--config-name=train_ hirol_fr3_pnp_cam_ee_joint_to_ee_unet.yaml',
                'training.device=cuda:0',
            ],
        )

    def test_rewrite_config_reference_argv_joins_inline_split_path_tokens(self):
        argv = rewrite_config_reference_argv(
            [
                'train.py',
                '--config=diffusion_policy/config/train_lerobot_v3/train_',
                'hirol_fr3_pnp_cam_ee_joint_to_ee_unet.yaml',
                '--multirun',
            ]
        )

        self.assertEqual(
            argv,
            [
                'train.py',
                '--config-dir=diffusion_policy/config/train_lerobot_v3',
                '--config-name=train_ hirol_fr3_pnp_cam_ee_joint_to_ee_unet.yaml',
                '--multirun',
            ],
        )

    def test_rewrite_config_reference_argv_rejects_mixed_flags(self):
        with self.assertRaises(SystemExit):
            rewrite_config_reference_argv(
                [
                    'train.py',
                    '-c',
                    'diffusion_policy/config/train_zarr/train_hirol_fr3_unet_abs_jp.yaml',
                    '--config-name=train_hirol_fr3_unet_abs_jp.yaml',
                ]
            )


if __name__ == '__main__':
    unittest.main()
