import streamlit as st
from PIL import Image

img1 = Image.open("images/metric1.png")
img2 = Image.open("images/metric2.png")

st.title('Model_1')
st.image(img1, width=1500)

st.title('Model_2')
st.image(img2, width=1500)