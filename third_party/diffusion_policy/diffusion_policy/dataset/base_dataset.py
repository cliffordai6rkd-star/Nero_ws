from typing import Dict

import torch
import torch.nn
from diffusion_policy.model.common.normalizer import LinearNormalizer


# 父类只定义接口  为子类提供要求和规范
class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        # 父类提供通用低维数据验证集的通用空模板  在子类里详细定义  
        return BaseLowdimDataset()

    # 归一化
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    #“把这个数据集里的所有 action 都拿出来，作为一个 Tensor 返回。
    def get_all_actions(self) -> torch.Tensor:
        # 如果子类每没有定义就会rasie error
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseImageDataset':
        # return an empty dataset by default
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # 如果子类每没有定义就会rasie error
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()
