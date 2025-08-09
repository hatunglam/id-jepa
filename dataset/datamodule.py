import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .NYU_dataset import NYUDataset

class RGBDDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_config: dict,
                 experiment_config: dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.experiment_config = experiment_config

    def setup(self, stage=None):

        self.train_dataset = NYUDataset(data_dir=self.dataset_config["DATA_DIR"],
                                        mode="train",
                                        max_depth=self.dataset_config["MAX_DEPTH"],
                                        img_resize=self.dataset_config["IMG_RESIZE"],
                                        return_metric_depth=self.dataset_config["RETURN_METRIC_DEPTH"],
                                        apply_transforms=self.dataset_config["APPLY_TRANSFORMS"])
        self.val_dataset = NYUDataset(data_dir=self.dataset_config["DATA_DIR"],
                                      mode="test",
                                      max_depth=self.dataset_config["MAX_DEPTH"],
                                      img_resize=False,
                                      return_metric_depth=self.dataset_config["RETURN_METRIC_DEPTH"],
                                      apply_transforms=False)
        self.test_dataset = NYUDataset(data_dir=self.dataset_config["DATA_DIR"],
                                       mode="test",
                                       max_depth=self.dataset_config["MAX_DEPTH"],
                                       img_resize=False,
                                       return_metric_depth=self.dataset_config["RETURN_METRIC_DEPTH"],
                                       apply_transforms=False)

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
