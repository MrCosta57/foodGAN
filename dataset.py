from torchvision.datasets import Food101
import torchvision.transforms as transforms
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import tarfile, tempfile
import config

class Food101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS_DATASET, val_ratio=config.VAL_DATASET_RATIO):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio=val_ratio
        #self.temp_dir = tempfile.mkdtemp()

    def prepare_data(self):
        """ print("Search for dataset in ", config.DATA_DIR+config.FILE_NAME)
        print("Extract dataset in ", self.temp_dir)
        # Open the tar.gz file
        with tarfile.open(config.DATA_DIR+config.FILE_NAME, 'r:gz') as tar:
            # Extract the contents to the temporary environment
            tar.extractall(self.temp_dir)
        print("Extraction done!") """

        #Food101(self.data_dir, split="train", download=True)
        #Food101(self.data_dir, split="test", download=True)
        pass

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        if stage=="fit" or stage is None: #self.temp_dir
            full_train_set = Food101(root=self.data_dir, split="train",
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(config.IMG_SIZE),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                download=False,
            )
            """ transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            ) """
            
            val_size=int(full_train_set.__len__()*self.val_ratio)
            train_size=full_train_set.__len__()-val_size
            #print("Train len is: ", train_size)
            #print("Valid len is: ", val_size)
            self.train_ds, self.val_ds = random_split(full_train_set, [train_size, val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = Food101(root=self.data_dir, split="test",
                transform=transforms.Compose([
                    transforms.Resize(config.IMG_SIZE),
                    transforms.ToTensor()
                ]),
                download=False,
            )


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)