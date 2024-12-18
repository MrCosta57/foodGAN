import os
import pathlib
import streamlit as st
import glob
import pandas as pd
import torch
from model import GAN

# VARIABLES
MODEL_DIR = "checkpoints"
LABELS_PATH = "dataset/labels.csv"


# UTILITIES
def get_num_devices() -> int:
    """
    Get the number of available devices
    Returns one if only CPU and more than one if GPU is available
    """
    return 1 + torch.cuda.device_count() if torch.cuda.is_available() else 1


def get_available_models() -> list[str]:
    """
    Returns the list of available models
    """
    ckpt_files = [
        os.path.splitext(os.path.basename(file))[0]
        for file in glob.glob(os.path.join(MODEL_DIR, "*.ckpt"))
    ]
    ckpt_files.sort(reverse=True)
    return ckpt_files


# LOAD DATA
@st.cache_data
def load_labels():
    df = pd.read_csv(LABELS_PATH, header=0)
    options = df["labels"].values
    return options


@st.cache_resource
def load_gan_model(model_name, device):
    if model_name is None:
        return None
    gan = GAN.load_model_from_ckp(model_name + ".ckpt")
    gan = gan.to(device)
    return gan


st.title(
    "WGAN-GP for food images generation\n :hamburger: :pizza: :meat_on_bone: :rice: :pancakes: :fried_egg:"
)
st.caption("Project realized by Giovanni Costa - 880892")


st.sidebar.header("Configuration")

# Device selection dropdown in the sidebar
device_option = st.sidebar.selectbox(
    "Select the device",
    options=range(get_num_devices()),
    format_func=lambda idx: (
        "CPU" if idx == 0 else f"GPU ({torch.cuda.get_device_name(idx-1)})"
    ),
)

# Model selection dropdown in the sidebar
model_options = st.sidebar.selectbox("Select the model", get_available_models())
label_list = load_labels()
index_label = st.sidebar.selectbox(
    "Select one food category",
    range(len(label_list)),
    format_func=lambda x: label_list[x],
)

# Load the model and tokenizer
device = torch.device(f"cuda:{device_option-1}" if device_option > 0 else "cpu")

# Load the model and tokenizer
model_path = None if not model_options else os.path.join(MODEL_DIR, model_options)
model = load_gan_model(model_path, device)

# Title generation button
if st.button("Generate") and model is not None:
    img_gen_text = st.text("Image generation...")
    img = model.generate(index_label)
    img_gen_text.text("Image generated!")
    st.image(img, caption=label_list[index_label], use_column_width=True)
