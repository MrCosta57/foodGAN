from torchvision.datasets import Food101
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as transforms
import tarfile, tempfile
import config


class Custom_Food101:
    """
    Custom class for modelling the Food101 dataset.
    If `is_compressed` is True, the dataset provided in the `data_dir` is extracted to a temporary folder.
    If `is_compressed` is False, the dataset is downloaded to the `data_dir`.
    Note that `prepare_data()` must be called before `get_dataloaders()`.
    """

    def __init__(
        self,
        data_dir: str = config.DATA_DIR,
        file_name: str = config.FILE_NAME,
        batch_size: int = config.BATCH_SIZE,
        is_compressed: bool = True,
        num_workers: int = config.NUM_WORKERS_DATASET,
        img_size: int = config.IMG_SIZE,
    ):

        self.data_dir = data_dir
        self.file_name = file_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.is_compressed = is_compressed
        if is_compressed:
            self.temp_dir = tempfile.mkdtemp()
        else:
            self.temp_dir = data_dir

    def prepare_data(self):
        """
        Extracts the compressed dataset to a temporary folder. This method must be called before `get_dataloaders()` if the dataset is compressed.
        """
        if self.is_compressed:
            # print("Search for dataset in ", config.DATA_DIR+config.FILE_NAME)
            print("Extract dataset in ", self.temp_dir)
            # Open the tar.gz file
            with tarfile.open(self.data_dir + self.file_name, "r:gz") as tar:
                # Extract the contents to the temporary environment
                tar.extractall(self.temp_dir)
            print("Extraction done!")
        else:
            print("Download dataset in ", self.temp_dir)
            Food101(root=self.temp_dir, download=True)
            print("Download done!")

    def get_dataloader(self):
        """
        Returns the train and test dataloaders for the Food101 dataset.
        The train dataloader has random horizontal and vertical flips, random resized crops and normalization transformations.
        """
        train_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(config.CHANNELS_IMG)],
                    [0.5 for _ in range(config.CHANNELS_IMG)],
                ),
            ]
        )
        train = Food101(
            root=self.temp_dir, split="train", download=False, transform=train_transform
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(config.CHANNELS_IMG)],
                    [0.5 for _ in range(config.CHANNELS_IMG)],
                ),
            ]
        )
        test = Food101(
            root=self.temp_dir, split="test", download=False, transform=test_transform
        )

        concat = ConcatDataset([train, test])
        dl = DataLoader(
            concat,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return dl
