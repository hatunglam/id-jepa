from typing import Callable, Dict

from ...configs import get_image_experiment_config, get_image_model_config
from model_2.model_graydepth import IDJEPA
from ..vit import get_model_config as get_vit_config

experiment_config = get_image_experiment_config()
model_config = get_image_model_config()

MODEL_SIZES = {
    "nano": {"num_decoder_layers": 2},
    "tiny": {"num_decoder_layers": 4},
    "small": {"num_decoder_layers": 4},
    "base": {"num_decoder_layers": 8},
    "large": {"num_decoder_layers": 8},
    "huge": {"num_decoder_layers": 12},
    "gigantic": {"num_decoder_layers": 16},
}


def get_model_config(size: str) -> int:
    return MODEL_SIZES[size]["num_decoder_layers"]


def create_ijepa_model(vit_size: str) -> IDJEPA:
    embed_dim, enc_depth, num_heads = get_vit_config(size=vit_size)
    num_layers_decoder: int = get_model_config(size=vit_size)

    return IDJEPA(
        decoder_depth=num_layers_decoder,
        lr=experiment_config["LR"],
        weight_decay=experiment_config["WEIGHT_DECAY"],
        target_aspect_ratio=experiment_config["TARGET_ASPECT_RATIO"],
        target_scale_interval=experiment_config["TARGET_SCALE_INTERVAL"],
        context_aspect_ratio=experiment_config["CONTEXT_ASPECT_RATIO"],
        context_scale=experiment_config["CONTEXT_SCALE"],
        num_target_blocks=experiment_config["NUM_TARGET_BLOCKS"],
        m=experiment_config["M"],
        momentum_limits=experiment_config["MOMENTUM_LIMITS"],
        img_size=model_config["IMAGE_SIZE"],
        patch_size=model_config["PATCH_SIZE"],
        in_chans_rgb=model_config["IN_CHANS"],
        embed_dim=embed_dim,
        enc_depth=enc_depth,
        num_heads=num_heads,
        post_emb_norm=model_config["POST_EMBED_NORM"],
        post_enc_norm=model_config["POST_ENCODE_NORM"],
        layer_dropout=model_config["LAYER_DROPOUT"],
    )


def make_ijepa_builder(size: str) -> Callable[[], IDJEPA]:
    def builder() -> IDJEPA:
        return create_ijepa_model(vit_size=size)

    builder.__name__ = f"ijepa_{size}_builder"

    return builder


ijepa_model_builders: Dict[str, Callable[[], IDJEPA]] = {
    size: make_ijepa_builder(size) for size in MODEL_SIZES
}



