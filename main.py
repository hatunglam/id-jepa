import gc
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from configs import (
    get_image_experiment_config,
    get_image_runtime_config,
    get_image_tracking_config,
    get_image_model_config,
    get_image_dataset_config,
)

from model_2.model_builder import ijepa_model_builders
from data_loader import RGBDDataModule

# Load configs
experiment_config = get_image_experiment_config()
runtime_config = get_image_runtime_config()
tracking_config = get_image_tracking_config()
model_config = get_image_model_config()
dataset_config = get_image_dataset_config()

# Unpack configs
MODEL_NAME = experiment_config["MODEL_NAME"]
MODEL_SIZE = experiment_config["MODEL_SIZE"]
LR = experiment_config["LR"]
SEED = experiment_config["SEED"]
MAX_EPOCHS = experiment_config["MAX_EPOCHS"]
GRADIENT_CLIP_VAL = experiment_config["GRADIENT_CLIP_VAL"]

LOG_DIR = tracking_config["LOG_DIR"]
LOGGING_INTERVAL = tracking_config["LOGGING_INTERVAL"]
TOK_K_CHECKPOINTS = tracking_config["TOK_K_CHECKPOINTS"]
CHECKPOINT_DIR = tracking_config["CHECKPOINT_DIR"]
CHECKPOINT_MONITOR = tracking_config["CHECKPOINT_MONITOR"]
CHECKPOINT_MODE = tracking_config["CHECKPOINT_MODE"]
VAL_CHECK_INTERVAL = tracking_config["VAL_CHECK_INTERVAL"]

ACCELERATOR = runtime_config["ACCELERATOR"]
DEVICES = runtime_config["DEVICES"]
PRECISION = runtime_config["PRECISION"]
FLOAT32_MATMUL_PRECISION = runtime_config["FLOAT32_MATMUL_PRECISION"]

torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)

if __name__ == "__main__":
    pl.seed_everything(SEED)

    # Build model
    model_id = f"{MODEL_SIZE}_{SEED}_{LR:.1e}-{MAX_EPOCHS}"
    model = ijepa_model_builders[MODEL_SIZE]()
    print(f"Model built: {MODEL_NAME}_{model_id}")

    # Build datamodule
    datamodule = RGBDDataModule(
        dataset_dir=dataset_config["DATASET_DIR"],
        batch_size=experiment_config["BATCH_SIZE"],
        img_size=model_config["IMAGE_SIZE"],
    )
    print("RGBD datamodule loaded")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=MODEL_NAME,
        monitor=CHECKPOINT_MONITOR,
        mode=CHECKPOINT_MODE,
        save_top_k=TOK_K_CHECKPOINTS,
    )
    lr_monitor = LearningRateMonitor(logging_interval=LOGGING_INTERVAL)

    # Logger
    logger = TensorBoardLogger(save_dir=LOG_DIR, name=MODEL_NAME, version=model_id)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=VAL_CHECK_INTERVAL,
        logger=logger,
    )

    # Train
    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint(f"{CHECKPOINT_DIR}/{MODEL_NAME}_{model_id}.ckpt")

    # Test
    trainer.test(model, datamodule=datamodule)

    # Cleanup
    del datamodule
    gc.collect()
