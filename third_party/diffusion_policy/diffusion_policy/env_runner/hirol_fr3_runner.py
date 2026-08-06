from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from typing import Dict

# Runner is for simulation evaluation during policy training
class HirolFr3Runner(BaseImageRunner):
    def __init__(self,
            output_dir,
            test_start_seed=100000
        ):
        super().__init__(output_dir)
        self.test_start_seed = test_start_seed
    
    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
    
