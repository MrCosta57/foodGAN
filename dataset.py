from torchvision.datasets import Food101
import torchvision.transforms as transforms
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import config

class Food101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS_DATASET, val_ratio=config.VAL_DATASET_RATIO):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio=val_ratio

    def prepare_data(self):
        #Food101(self.data_dir, split="train", download=True)
        #Food101(self.data_dir, split="test", download=True)
        pass

    def setup(self, stage):
        if stage=="fit":
            self.full_train_set = Food101(root=self.data_dir, split="train",
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(config.IMG_SIZE),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                download=False,
            )

        if stage=="test":
            self.test_ds = Food101(root=self.data_dir, split="test",
                transform=transforms.Compose([
                    transforms.Resize(config.IMG_SIZE),
                    transforms.ToTensor()
                ]),
                download=False,
            )

        #TODO: predict


    def train_dataloader(self):
        return DataLoader(self.full_train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        #return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        pass

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)