import streamlit as st
import glob
import pandas as pd
from model import GAN

# VARIABLES
checkpoints_path = "checkpoints/"
labels_path = "dataset/labels.csv"

# LOAD DATA
@st.cache_data
def load_labels():
    df=pd.read_csv(labels_path, header=0)
    options = df['labels'].values
    return options
@st.cache_data
def load_avilable_models():
    checkpoints=sorted(glob.glob(checkpoints_path+"*.ckpt"), reverse=True)
    options = [x.split('\\')[-1].split('.')[0] for x in checkpoints]
    return options
@st.cache_resource
def load_gan_model(model_name):
    gan=GAN.load_model_from_ckp(checkpoints_path+model_name+".ckpt")
    return gan

# GENERATE IMAGE
def can_generate():
    st.session_state.pressed=True
def generate(index):
    global gan
    img_gen_text = st.text('Image generation...')
    img=gan.generate(index)
    img_gen_text.text('Image generated!')
    return img


# MAIN
st.title("WGAN-GP for food images generation\n :hamburger: :pizza: :meat_on_bone: :rice: :pancakes: :fried_egg:")
st.caption('Project realized by Giovanni Costa - 880892')

label_list=load_labels()
model_list=load_avilable_models()

if len(model_list)==0:
    st.error(f'No GAN model found in directory "{checkpoints_path}"', icon="🚨")
index_model_name = st.sidebar.selectbox('Select one GAN model', range(len(model_list)), format_func=lambda x: model_list[x])
index_label = st.sidebar.selectbox('Select one food category', range(len(label_list)), format_func=lambda x: label_list[x])
gan=None
if len(model_list)!=0:
    gan=load_gan_model(model_list[index_model_name])

# Initialize session state (default to False)
if 'pressed' not in st.session_state:
    st.session_state['pressed'] = False

st.sidebar.button("Generate", key="gen_btn", on_click=can_generate)

# Generate image only if I press the button
if st.session_state.pressed and gan is not None:
    img=generate(index_label)
    st.image(img, caption=label_list[index_label], use_column_width=True)
    st.session_state.pressed=False