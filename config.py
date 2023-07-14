import os, torch

LEARNING_RATE = 1e-4
B1=0.0
B2=0.99
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10.0
NUM_EPOCHS = 100
DEVICE= "cuda" if torch.cuda.is_available() else "cpu"

Z_DIM = 100
BATCH_SIZE = 64
IMG_SIZE = 64
CHANNELS_IMG = 3
GEN_EMBEDDING=100
NUM_CLASSES=101

FEATURES_DISC = 16
FEATURES_GEN = 16

NUM_WORKERS_DATASET=1 #int(os.cpu_count() / 2)
VAL_DATASET_RATIO=0.1
DATA_DIR="dataset/" #r"/content/drive/MyDrive/Colab Notebooks/ML/ML_project/dataset/"
FILE_NAME="food-101.tar.gz"
LOG_DIR="log/" #r"/content/drive/MyDrive/Colab Notebooks/ML/ML_project/logs/"