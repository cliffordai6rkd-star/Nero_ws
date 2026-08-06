from transformers import pipeline
from transformers.image_utils import load_image
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

class DINOv3Encoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        self.freeze_encoder = config.get("train","freeze_encoder",True)
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        if self.freeze_encoder:
            for p in self.model.parameters:
                p.requires_grad = False

    
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        # x:[B,C,H,W]

        #优先用 HuggingFace 提供的全局特征
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feature = outputs.pooler_output
        else:    # 如果没有，就自己取 CLS token 当全局特征
            feature = outputs.last_hidden_state[:, 0] # CLS token:[B,N,D] N 为token数量 -> [B,D]

        if feature.ndim != 2:
            raise RuntimeError(f"Expected [B, D], got {feature.shape}")

        return feature
    


if __name__ == "__main__":
    url="http://images.cocodataset.org/val2017/000000039769.jpg",
    pretrained_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m"
    image = load_image(url)
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(pretrained_model_name)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    print(outputs.shape)