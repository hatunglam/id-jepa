import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoModelForDepthEstimation
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from transformers import AutoConfig, AutoModel, DPTForDepthEstimation, DPTImageProcessor

class DepthHead(nn.Module):
    def __init__(self):
        super().__init__()

        model_config = AutoConfig.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf")
        model = AutoModelForDepthEstimation.from_config(model_config)
        self.num_hidden_size = model_config.neck_hidden_sizes 
        self.neck = model.neck
        self.head = model.head
        self.head.activation2 = torch.nn.Sigmoid()

    def forward(self, x, max_depth, patch_height, patch_width):
        x = self.neck(x, patch_height=patch_height, patch_width=patch_width)
        x = self.head(x, patch_height=patch_height, patch_width=patch_width) 
        x = x * max_depth
        return x

def init_DINO_encoder(checkpoint_path: str | None = "checkpoints/ID_JEPA_base_42_1.0e-03-100.ckpt",
                      use_pretrained_head=False):
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["state_dict"]
        weights = {k.replace("image_encoder.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.")}

        # Load into a Dino model
        img_encoder_config = AutoConfig.from_pretrained("facebook/dinov2-base")
        img_encoder = AutoModel.from_config(img_encoder_config)
        img_encoder.load_state_dict(weights, strict=True)
    else:
        img_encoder = AutoModel.from_pretrained(
            "facebook/dinov2-base", trust_remote_code=False
        )

    if use_pretrained_head:
        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-base-384")
    else:
        # depth_estimator_config = AutoConfig.from_pretrained("Intel/dpt-beit-base-384")
        # depth_estimator = DPTForDepthEstimation(depth_estimator_config)
        depth_estimator = DepthHead()
    return img_encoder, depth_estimator

def init_DepthAnything_encoder(use_pretrained_head=False):
    img_encoder = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf").backbone

    if use_pretrained_head:
        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-base-384")
    else:
        # depth_estimator_config = AutoConfig.from_pretrained("Intel/dpt-beit-base-384")
        # depth_estimator = DPTForDepthEstimation(depth_estimator_config)
        depth_estimator = DepthHead()
    return img_encoder, depth_estimator

def init_model_encoder(config="dino-dpt",
                       checkpoint_path: str | None = "checkpoints/ID_JEPA_base_42_1.0e-03-100.ckpt",
                       use_pretrained_head=False):
    if use_pretrained_head:
        print("Using pretrained Depth Estimator")
    else:
        print("Using non-pretrained Depth Estimator")
    
    if config.lower() == "dino-dpt":
        return init_DINO_encoder(checkpoint_path=checkpoint_path,
                                 use_pretrained_head=use_pretrained_head)
    elif config.lower() == "depthanything-dpt":
        return init_DepthAnything_encoder(use_pretrained_head=use_pretrained_head)
    