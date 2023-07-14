import lightning.pytorch as pl
from model import GAN
from dataset import Food101DataModule
from lightning.pytorch.loggers import TensorBoardLogger
import config

logger = TensorBoardLogger(config.LOG_DIR, name="food_model_v1")
trainer = pl.Trainer(
    logger=logger,
    accelerator="auto",
    min_epochs=1,
    max_epochs=3
)

dataset=Food101DataModule()
model=GAN()

trainer.fit(model, dataset)

#%load_ext tensorboard
#%tensorboard --logdir lightning_logs/
