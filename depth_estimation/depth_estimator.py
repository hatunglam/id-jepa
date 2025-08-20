import torch
import torch.nn as nn

class DepthEstimatorBase(nn.Module):
    def __init__(self,
                 image_encoder,
                 depth_estimator):
        super().__init__()
        self.image_encoder = image_encoder
        self.depth_estimator = depth_estimator
        self.output_attentions = False
        self.output_hidden_states = True

    def forward_base(self, pixel_values):
        output = self.image_encoder(pixel_values,
                                    output_hidden_states=self.output_hidden_states,
                                    output_attentions=self.output_attentions)
        
        num_neck_hidden_state = len(self.depth_estimator.num_hidden_size)
        hidden_states = output.hidden_states[-num_neck_hidden_state:]

        patch_height, patch_width = None, None
        if self.image_encoder.config is not None:
            _, _, height, width = pixel_values.shape
            patch_size = self.image_encoder.config.patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size

        # hidden_states = self.depth_estimator.neck(hidden_states, patch_height, patch_width)
        # predicted_depth = self.depth_estimator.head(hidden_states)
        predicted_depth = self.depth_estimator(x=hidden_states, max_depth=10.0,
                                               patch_height=patch_height, patch_width=patch_width)

        return predicted_depth
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import requests
    from init_encoder import init_model_encoder
    from transformers import AutoModelForDepthEstimation
    from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
    from transformers import AutoConfig, AutoModel, DPTForDepthEstimation, DPTImageProcessor

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-base-384")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values

    image_encoder, depth_estimator = init_model_encoder(config="depthanything-dpt")
    model = DepthEstimatorBase(image_encoder, depth_estimator)

    with torch.no_grad():
        predicted_depth = model(pixel_values)

    prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),  # (B, 1, H, W)
    size=image.size[::-1],         # (H, W)
    mode="bicubic",
    align_corners=False)

    # Convert to displayable image
    output = prediction.squeeze().cpu().detach().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    # Display
    depth.save("depth_output.png") 
