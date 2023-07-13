from torch import optim, nn, Tensor
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
        Food101(self.data_dir, split="train", download=True)
        Food101(self.data_dir, split="test", download=True)

    def setup(self, stage):
        if stage=="fit":
            full_train_set = Food101(root=self.data_dir, split="train",
                transform=transforms.Compose([
                    #TODO: resize random crop
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                download=False,
            )

            #TODO: size should be 75,750 training images and 25,250 testing images.
            val_size=int(full_train_set.__len__*self.val_ratio)
            train_size=full_train_set.__len__-val_size
            self.train_ds, self.val_ds = random_split(full_train_set, [train_size, val_size])


        if stage=="test" or stage=="predict":
            self.test_ds = Food101(root=self.data_dir, split="test",
                #TODO: change maybe
                transform=transforms.ToTensor(),
                download=False,
            )


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)