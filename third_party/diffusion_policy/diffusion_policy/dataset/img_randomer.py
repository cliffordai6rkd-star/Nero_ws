import torchvision.transforms as transforms
from PIL import Image


class Image_randomer():
    def __init__(self, config):

        self._config = config
          # 基于torch的数据增强
        self.resize = config.get("resize",False)
        self.resize_size = config.get("resize_size", (224,224))

        self.random_crop = config.get("random_crop", True)
        self.random_crop_size = config.get("random_crop_size", (200,200))

        # self.center_crop = config.get("center_crop", True)
        # self.center_crop_size = config.get("center_crop_size", (200,200))
        
        self.rotation = config.get("rotation", False)
        self.rotation_degree = config.get("rotation_degree", 0)

        self.color_jitter = config.get("color_jitter", False)
        self.color_jitter_params = config.get("color_jitter_params", {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1
        })
        self.transform = self.image_transform_builder()

    def image_transform_builder(self):
        transform_list = []
        if self.resize:
            # 加入 Resize(resize_size)
            transform_list.append(transforms.Resize(self.resize_size))  

        if self.random_crop:
            # 加入 RandomCrop(random_crop_size)
            transform_list.append(transforms.RandomCrop(self.random_crop_size))

        if self.rotation:
            # 加入 RandomRotation(rotation_degree)
            transform_list.append(transforms.RandomRotation(self.rotation_degree))
        if self.color_jitter:
            # 加入 ColorJitter(**color_jitter_params)
            transform_list.append(transforms.ColorJitter(**self.color_jitter_params))

        transform_list.append(transforms.ToTensor())  
        # 最后加入 ToTensor()
        return transforms.Compose(transform_list)



    def __call__(self, image):
        
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")
        image_tensor = self.transform(image)
        # print(image_tensor)
        return image_tensor
