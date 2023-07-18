from torchvision.datasets import Food101
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import tarfile, tempfile
import config

class Custom_Food101():
    def __init__(self, data_dir=config.DATA_DIR, file_name=config.FILE_NAME, batch_size=config.BATCH_SIZE, is_compressed=True,
                 num_workers=config.NUM_WORKERS_DATASET, img_size=config.IMG_SIZE):

        self.data_dir = data_dir
        self.file_name = file_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size=img_size
        self.is_compressed = is_compressed
        if is_compressed:
            self.temp_dir = tempfile.mkdtemp()
        else:
            self.temp_dir = data_dir

    def prepare_data(self):
        if self.is_compressed:
            #print("Search for dataset in ", config.DATA_DIR+config.FILE_NAME)
            print("Extract dataset in ", self.temp_dir)
            # Open the tar.gz file
            with tarfile.open(self.data_dir+self.file_name, 'r:gz') as tar:
                # Extract the contents to the temporary environment
                tar.extractall(self.temp_dir)
            print("Extraction done!")


    def get_dataloaders(self):
        train_transform=transforms.Compose([
                transforms.RandomResizedCrop(config.IMG_SIZE),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(config.CHANNELS_IMG)], [0.5 for _ in range(config.CHANNELS_IMG)]
                )
        ])
        train=Food101(root=self.temp_dir, split="train", download=False, transform=train_transform)
        train_dl=DataLoader(train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


        test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)], [0.5 for _ in range(config.CHANNELS_IMG)])
        ])
        test=Food101(root=self.temp_dir, split="test", download=False, transform=test_transform)
        test_dl=DataLoader(test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        return train_dl, test_dl