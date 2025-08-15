from transformers import AutoModel, AutoImageProcessor
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from transformers import Dinov2Config
from transformers.models.bit.image_processing_bit import BitImageProcessor

USE_VARIATIONAL_PREDICTOR = True
if USE_VARIATIONAL_PREDICTOR:
    print("Training Using Variational Predictor")
else:
    print("Training With Default Predictor")

pre_trained_img_model = AutoModel.from_pretrained(
    "facebook/dinov2-base",
    trust_remote_code=False,
)

config = pre_trained_img_model.config
img_model = Dinov2Model(config)

img_feature_extractor: BitImageProcessor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

depth_feature_extractor: BitImageProcessor = AutoImageProcessor.from_pretrained(
"facebook/dinov2-base"
)

img_feature_extractor.do_rescale = False
img_feature_extractor.do_center_crop = True
img_feature_extractor.do_resize = False
img_feature_extractor.do_normalize = True
img_feature_extractor.do_convert_rgb = True 

def image_preprocessor(image):
    processed_img = img_feature_extractor(images=image, return_tensors="pt")
    return processed_img["pixel_values"]

depth_feature_extractor.do_rescale = False
depth_feature_extractor.do_center_crop = True
depth_feature_extractor.do_resize = False
depth_feature_extractor.do_normalize = False
depth_feature_extractor.do_convert_rgb = False
depth_feature_extractor.image_std = [1, 1, 1]
depth_feature_extractor.image_mean = [0, 0, 0]

def depth_preprocessor(depth):
    processed_depth = depth_feature_extractor(images=depth, return_tensors="pt")
    depth_tensor = processed_depth["pixel_values"]
    depth_tensor = depth_tensor.repeat(1, 3, 1, 1)
    return depth_tensor


from new_idjepa.training import IDJEPA
from dataset.datamodule import RGBDDataModule
import gc
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from configs import (get_image_experiment_config,
                     get_image_runtime_config,
                     get_image_tracking_config,
                     get_image_model_config,
                     get_image_dataset_config,)

if __name__ == "__main__":
    experiment_config = get_image_experiment_config()
    runtime_config = get_image_runtime_config()
    tracking_config = get_image_tracking_config()
    model_config = get_image_model_config()
    dataset_config = get_image_dataset_config()
    # model_config["IMAGE_SIZE"] = tuple(model_config["IMAGE_SIZE"])
    
    MODEL_NAME = experiment_config["MODEL_NAME"]
    MODEL_SIZE = experiment_config["MODEL_SIZE"]
    SEED = experiment_config["SEED"]
    LR = experiment_config["LR"]
    MAX_EPOCHS = experiment_config["MAX_EPOCHS"]
    CHECKPOINT_DIR = tracking_config["CHECKPOINT_DIR"]

    torch.set_float32_matmul_precision(runtime_config["FLOAT32_MATMUL_PRECISION"])

    pl.seed_everything(SEED)

    model_id = f"{MODEL_SIZE}_{SEED}_{LR:.1e}-{MAX_EPOCHS}"

    jepa_model = IDJEPA(
        image_encoder=img_model,
        image_preprocessor=image_preprocessor,
        depth_encoder=pre_trained_img_model,
        depth_preprocessor=depth_preprocessor,
        decoder_depth=6,
        n_heads=8,
        predictor_embed_dim=768,
        post_enc_norm=False,
        mode="train",
        context_ratio_range=(0.85, 0.95),
        target_mask_range=(0.85, 0.95),
        freeze="depth",
        variational_predictor=USE_VARIATIONAL_PREDICTOR,
        lr=LR,
        weight_decay=0.05
    )

    datamodule = RGBDDataModule(dataset_config, experiment_config)
    print("RGBD datamodule loaded")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_DIR,
                                          filename=MODEL_NAME,
                                          monitor=tracking_config["CHECKPOINT_MONITOR"],
                                          mode=tracking_config["CHECKPOINT_MODE"],
                                          save_top_k=tracking_config["TOK_K_CHECKPOINTS"],)
    lr_monitor = LearningRateMonitor(logging_interval=tracking_config["LOGGING_INTERVAL"])

    # Logger
    logger = TensorBoardLogger(save_dir=tracking_config["LOG_DIR"], name=MODEL_NAME, version=model_id)

    # Trainer
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                         accelerator=runtime_config["ACCELERATOR"],
                         devices=runtime_config["DEVICES"],
                         precision=runtime_config["PRECISION"],
                         gradient_clip_val=experiment_config["GRADIENT_CLIP_VAL"],
                         callbacks=[checkpoint_callback, lr_monitor],
                         val_check_interval=tracking_config["VAL_CHECK_INTERVAL"],
                         logger=logger,)

    # Train
    trainer.fit(jepa_model, datamodule=datamodule)
    trainer.save_checkpoint(f"{CHECKPOINT_DIR}/{MODEL_NAME}_{model_id}.ckpt")






