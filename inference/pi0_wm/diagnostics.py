"""Print chunk snapshots off the hardware control thread."""
import json
import logging
from queue import SimpleQueue
from threading import Thread

import numpy as np

log = logging.getLogger(__name__)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'unsupported diagnostic value: {type(value)}')


class ChunkDiagnostics:
    def __enter__(self):
        self.queue = SimpleQueue()
        self.thread = Thread(target=self._run, name='chunk-diagnostics', daemon=True)
        self.thread.start()
        return self

    def emit(self, snapshot):
        self.queue.put(snapshot)

    def _run(self):
        while True:
            snapshot = self.queue.get()
            if snapshot is None:
                return
            log.info('PI0_CHUNK %s', json.dumps(snapshot, default=_json_value, ensure_ascii=False))

    def __exit__(self, *args):
        self.queue.put(None)
        self.thread.join(timeout=2)
