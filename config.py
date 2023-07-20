import os

# HYPERPARAMETERS
LEARNING_RATE = 1e-4
B1=0.0
B2=0.99
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10.0
NUM_EPOCHS = 10
Z_DIM = 100
BATCH_SIZE = 128
IMG_SIZE = 64
CHANNELS_IMG = 3
GEN_EMBEDDING=100
FEATURES_DISC = 64
FEATURES_GEN = 64

# CONFIGURATIONS
NUM_WORKERS_DATASET=int(os.cpu_count() / 2)
DATA_DIR=r"/content/drive/MyDrive/Colab Notebooks/ML/ML_project/dataset/"
FILE_NAME="food-101.tar.gz"
NUM_CLASSES=101
LOG_DIR=r"/content/drive/MyDrive/Colab Notebooks/ML/ML_project/logs/"
DEBUG_EVERY_ITER=50 #debug every 50 iterations of generator
PRECISION="32" #16-mixed or 16 precision leads to instability problems
NUM_DEVICES=1
CHECKPOINT_DIR=r"/content/drive/MyDrive/Colab Notebooks/ML/ML_project/checkpoints/"