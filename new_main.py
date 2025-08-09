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
from model_2.model_builder import ijepa_model_builders
from dataset import RGBDDataModule

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

    # Build model
    model_id = f"{MODEL_SIZE}_{SEED}_{LR:.1e}-{MAX_EPOCHS}"
    model = ijepa_model_builders[MODEL_SIZE]()
    print(f"Model built: {MODEL_NAME}_{model_id}")

    # Build datamodule
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
    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint(f"{CHECKPOINT_DIR}/{MODEL_NAME}_{model_id}.ckpt")

    # Test
    trainer.test(model, datamodule=datamodule)

    # Cleanup
    del datamodule
    gc.collect()
