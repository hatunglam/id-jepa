from transformers import AutoModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from transformers import AutoConfig
from transformers import AutoModelForDepthEstimation
from base_idjepa.training import IDJEPA
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
    
    MODEL_NAME = experiment_config["MODEL_NAME"]
    MODEL_SIZE = experiment_config["MODEL_SIZE"]
    SEED = experiment_config["SEED"]
    LR = experiment_config["LR"]
    MAX_EPOCHS = experiment_config["MAX_EPOCHS"]
    CHECKPOINT_DIR = tracking_config["CHECKPOINT_DIR"]

    torch.set_float32_matmul_precision(runtime_config["FLOAT32_MATMUL_PRECISION"])
    pl.seed_everything(SEED)

    model_id = f"{MODEL_SIZE}_{SEED}_{LR:.1e}-{MAX_EPOCHS}"

    if model_config["USE_PRETRAINED_ENCODER"]:
        print("Using Pretrained DINOv2 as student ...")
        image_encoder = AutoModel.from_pretrained("facebook/dinov2-base",
                                                  trust_remote_code=False,)
    else:
        print("Using Non-Pretrained DINOv2 as student ...")
        config = AutoConfig.from_pretrained("facebook/dinov2-base",
                                            trust_remote_code=False,)
        image_encoder = Dinov2Model(config)
    
    if model_config["TEACHER_MODEL_TYPE"].lower() == "depthanything":
        print("Using DepthAnything as Teacher...")
        depth_encoder = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf").backbone
    else:
        print("Using DINOv2 as Teacher...")
        depth_encoder = AutoModel.from_pretrained("facebook/dinov2-base",
                                                  trust_remote_code=False,)
    depth_encoder = depth_encoder.eval()

    jepa_model = IDJEPA(image_encoder=image_encoder,
                        depth_encoder=depth_encoder,
                        decoder_depth=8,
                        n_heads=12,
                        predictor_embed_dim=384,
                        post_enc_norm=False,
                        mode="train",
                        context_ratio_range=experiment_config["CONTEXT_SCALE"],
                        target_mask_range=experiment_config["TARGET_SCALE"],
                        lr=LR,
                        weight_decay=0.05)

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






