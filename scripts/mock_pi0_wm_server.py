#!/usr/bin/env python3
"""Small wire-compatible mock server for the OFFICIAL openpi websocket client."""
from pathlib import Path
import argparse
import time
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--latency', type=float, default=0.04)
    args = parser.parse_args()
    from websockets.sync.server import serve
    from openpi_client import msgpack_numpy
    config = yaml.safe_load(Path(args.config).read_text())['pi0']
    def handle(socket):
        socket.send(msgpack_numpy.packb({'pi0_wm': config['interface']}))
        for message in socket:
            obs = msgpack_numpy.unpackb(message)
            value = obs
            for key in config['interface']['state_key'].split('/'):
                value = value[key]
            for key in config['interface']['images'].values():
                image = obs
                for part in key.split('/'):
                    image = image[part]
                assert image.dtype == np.uint8 and image.shape[-1] == 3
            time.sleep(args.latency)
            socket.send(msgpack_numpy.packb({'actions': np.tile(value, (50, 1))}))
    with serve(handle, '127.0.0.1', args.port, compression=None, max_size=None) as server:
        print(f'mock pi0 server ready on {args.port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
