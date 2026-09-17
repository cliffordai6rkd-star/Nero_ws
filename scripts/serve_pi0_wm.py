#!/usr/bin/env python3
"""Run inside the actual openpi training environment; preserve its transforms."""
from pathlib import Path
import argparse
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='pi0_wm YAML with verified interface mapping')
    parser.add_argument('--train-config', required=True, help='ACTUAL registered LoRA TrainConfig name')
    parser.add_argument('--checkpoint', required=True, help='trained checkpoint directory including assets/')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    import yaml
    from openpi.training import config as train_configs
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    logging.basicConfig(level=logging.INFO)
    interface = yaml.safe_load(Path(args.config).read_text())['pi0']['interface']
    if interface['training_config'] != args.train_config:
        raise ValueError('client interface must name the actual training configuration')
    train = train_configs.get_config(args.train_config)
    variants = (str(getattr(train.model, 'paligemma_variant', '')),
                str(getattr(train.model, 'action_expert_variant', '')))
    if not any('lora' in value.lower() for value in variants):
        raise ValueError('selected training config is not a LoRA model')
    data = train.data.create(train.assets_dirs, train.model)
    if tuple(data.action_sequence_keys) not in (('action',), ('action.ee_pose',)):
        raise ValueError('training action_sequence_keys must read action.ee_pose or its v2.1 action alias')
    from openpi.transforms import RepackTransform, flatten_dict
    repacks = [t for t in data.repack_transforms.inputs if isinstance(t, RepackTransform)]
    if len(repacks) != 1:
        raise ValueError('expected one inspectable training RepackTransform; verify this custom training interface')
    mapping = flatten_dict(repacks[0].structure)
    if mapping.get(interface['state_key']) not in ('observation.ee_pose', 'observation.state'):
        raise ValueError('training state mapping is not EE pose / converted EE state alias')
    if mapping.get('actions') != data.action_sequence_keys[0]:
        raise ValueError('training action mapping disagrees with sequence column')
    for name, key in interface['images'].items():
        if mapping.get(key) != f'observation.images.{name}':
            raise ValueError(f'training camera mapping disagrees with interface: {name} -> {key}')
    if any('aloha' in type(t).__name__.lower() for t in (*data.data_transforms.inputs, *data.data_transforms.outputs)):
        raise ValueError('ALOHA joint/gripper transforms cannot be used for Nero EE poses')
    # Show the real training mappings and transforms, never select an ALOHA or
    # generic joint adapter on behalf of an unknown fine-tuned checkpoint.
    logging.info('ACTUAL training repack=%s inputs=%s outputs=%s asset_id=%s',
                 data.repack_transforms, data.data_transforms.inputs,
                 data.data_transforms.outputs, data.asset_id)
    logging.info('Declared deployment interface=%s; checkpoint=%s', interface, args.checkpoint)
    # Official loader restores normalization from checkpoint/assets/<asset_id>
    # and applies training data/model input and inverse output transforms.
    policy = policy_config.create_trained_policy(train, args.checkpoint)
    metadata = dict(policy.metadata)
    metadata['pi0_wm'] = interface
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
