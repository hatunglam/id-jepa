from transformers import AutoModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from transformers import AutoConfig
from transformers import AutoModelForDepthEstimation
from depth_estimation.depth_estimator import init_encoder
from depth_estimation.depth_datamodule import DepthDataModule
from depth_estimation.depth_training import DepthEstimator
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

    model_id = "Depth_estimator"

    image_encoder, depth_estimator = init_encoder(use_checkpoint=True)
    
    model = DepthEstimator(image_encoder=image_encoder,
                           depth_estimator=depth_estimator,
                           lr=LR,)
    
    
    datamodule = DepthDataModule(dataset_config, experiment_config)

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
    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint(f"depth_estiamtor_final.ckpt")






