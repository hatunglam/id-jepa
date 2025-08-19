import torch
import torch.nn
from transformers import AutoModel
from transformers import AutoModelForDepthEstimation
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from transformers import AutoConfig, AutoModel, DPTForDepthEstimation, DPTImageProcessor

def init_DINO_encoder(checkpoint_path: str | None = "checkpoints/ID_JEPA_base_42_1.0e-03-100.ckpt"):
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        weights = {k.replace("image_encoder.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.")}

        # Load into a Dino model
        img_encoder_config = AutoConfig.from_pretrained("facebook/dinov2-base")
        img_encoder = AutoModel.from_config(img_encoder_config)
        img_encoder.load_state_dict(weights, strict=False)
    else:
        img_encoder = AutoModel.from_pretrained(
            "facebook/dinov2-base", trust_remote_code=False
        )

    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-base-384")

    return img_encoder, depth_estimator

def init_DepthAnything_encoder():
    img_encoder = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf").backbone
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-base-384")
    return img_encoder, depth_estimator

def init_model_encoder(config="dino-dpt",
                       checkpoint_path: str | None = "checkpoints/ID_JEPA_base_42_1.0e-03-100.ckpt"):
    
    if config.lower() == "dino-dpt":
        return init_DINO_encoder(checkpoint_path=checkpoint_path)
    elif config.lower() == "depthanything-dpt":
        return init_DepthAnything_encoder()