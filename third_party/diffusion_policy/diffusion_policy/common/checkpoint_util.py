from typing import Optional, Dict
import os

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: Optional[str] = None,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))

        os.makedirs(self.save_dir, exist_ok=True)
        
        # 
        if self.monitor_key is None:
            if ckpt_path in self.path_value_map:
                return ckpt_path

            if len(self.path_value_map) >= self.k:
                delete_path = next(iter(self.path_value_map))
                del self.path_value_map[delete_path]
                if os.path.exists(delete_path):
                    os.remove(delete_path)

            self.path_value_map[ckpt_path] = data.get('epoch', len(self.path_value_map))
            return ckpt_path

        value = data[self.monitor_key]
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
