import streamlit as st
import pandas as pd
from model import GAN

checkpoints_path = "checkpoints/food101_wgan_gp_epochs_15.ckpt"
labels_path = "dataset/labels.csv"

@st.cache_data
def load_labels():
    df=pd.read_csv(labels_path, header=0)
    options = df['labels'].values
    return options

@st.cache_resource
def load_gan_model():
    gan=GAN.load_model_from_ckp(checkpoints_path)
    return gan

def can_generate():
    st.session_state.pressed=True

def generate(index):
    global gan
    img_gen_text = st.text('Image generation...')
    img=gan.generate(index)
    img_gen_text.text('Image generated!')
    return img



st.title("WGAN-GP for food images generation\n :hamburger: :pizza: :meat_on_bone: :rice: :pancakes: :fried_egg:")
st.caption('Project realized by Giovanni Costa - 880892')

options=load_labels()
gan=load_gan_model()
index = st.sidebar.selectbox('Select one food category', range(len(options)), format_func=lambda x: options[x])


if 'pressed' not in st.session_state:
    st.session_state['pressed'] = False
st.sidebar.button("Generate", key="gen_btn", on_click=can_generate)

if st.session_state.pressed:
    img=generate(index)
    st.image(img, caption=options[index], use_column_width=True)
    st.session_state.pressed=False