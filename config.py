import os

LEARNING_RATE = 1e-4
B1=0.0
B2=0.99
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10.0
NUM_EPOCHS = 100

Z_DIM = 100
BATCH_SIZE = 64
IMG_SIZE = 64
CHANNELS_IMG = 3
GEN_EMBEDDING=100
NUM_CLASSES=100

FEATURES_DISC = 16
FEATURES_GEN = 16

NUM_WORKERS_DATASET=1 #int(os.cpu_count() / 2)
VAL_DATASET_RATIO=0.1
DATA_DIR=r"/content/drive/MyDrive/Colab Notebooks/ML/ML_project/dataset/"
LOG_DIR=r"/content/drive/MyDrive/Colab Notebooks/ML/ML_project/logs/"