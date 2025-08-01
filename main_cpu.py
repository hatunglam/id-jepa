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

# Override configs for minimal CPU test
MODEL_NAME = "test_cpu_run"
MODEL_SIZE = "nano"  # Use smallest model
LR = 1e-4
SEED = 42
MAX_EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_CLIP_VAL = 0.0

LOG_DIR = "./logs"
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_MONITOR = "val_loss"
CHECKPOINT_MODE = "min"
VAL_CHECK_INTERVAL = 1.0
LOGGING_INTERVAL = "epoch"
TOK_K_CHECKPOINTS = 1

ACCELERATOR = "cpu"
DEVICES = 1
PRECISION = 32

# Disable matmul precision tweak on CPU
# torch.set_float32_matmul_precision("high")  # Not used for CPU-only

if __name__ == "__main__":
    pl.seed_everything(SEED)

    # Build model
    model_id = f"{MODEL_SIZE}_{SEED}_{LR:.1e}-{MAX_EPOCHS}"
    model = ijepa_model_builders[MODEL_SIZE]()
    print(f"Model built: {MODEL_NAME}_{model_id}")

    # Build datamodule
    datamodule = RGBDDataModule(
        dataset_dir=dataset_config["DATASET_DIR"],
        batch_size=BATCH_SIZE,
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
        max_epochs=10,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=VAL_CHECK_INTERVAL,
        logger=logger,
        fast_dev_run=True,  # <--- Fast sanity check run
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test
    trainer.test(model, datamodule=datamodule)

    # Cleanup
    del datamodule
    gc.collect()
