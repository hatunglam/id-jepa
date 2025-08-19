import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .depth_dataset import DepthDataset

class DepthDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_config: dict,
                 experiment_config: dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.experiment_config = experiment_config

    def setup(self, stage=None):

        self.train_dataset = DepthDataset(data_dir=self.dataset_config["DATA_DIR"],
                                        mode="train",
                                        crop_size=self.dataset_config["CROP_SIZE"],
                                        teacher_model=self.dataset_config["TEACHER_MODEL"],
                                        max_depth=self.dataset_config["MAX_DEPTH"],)
        self.val_dataset = DepthDataset(data_dir=self.dataset_config["DATA_DIR"],
                                      mode="test",
                                      crop_size=self.dataset_config["CROP_SIZE"],
                                      teacher_model=self.dataset_config["TEACHER_MODEL"],
                                      max_depth=self.dataset_config["MAX_DEPTH"],)
        self.test_dataset = DepthDataset(data_dir=self.dataset_config["DATA_DIR"],
                                       mode="test",
                                       crop_size=self.dataset_config["CROP_SIZE"],
                                       teacher_model=self.dataset_config["TEACHER_MODEL"],
                                       max_depth=self.dataset_config["MAX_DEPTH"],)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.experiment_config["BATCH_SIZE"],
                          shuffle=True,
                          num_workers=self.experiment_config["NUM_WORKERS"],
                          pin_memory=self.experiment_config["PIN_MEMORY"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.experiment_config["BATCH_SIZE"],
                          shuffle=False,
                          num_workers=self.experiment_config["NUM_WORKERS"],
                          pin_memory=self.experiment_config["PIN_MEMORY"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.experiment_config["BATCH_SIZE"],
                          shuffle=False,
                          num_workers=self.experiment_config["NUM_WORKERS"],
                          pin_memory=self.experiment_config["PIN_MEMORY"])
