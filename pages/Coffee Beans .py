import torch
import torchvision
from torchvision import transforms as T
import streamlit as st
from PIL import Image
DEVICE = 'cuda'

trnsfrms1 = T.Compose([T.Resize((255, 255)),T.ToTensor()])
model = torch.jit.load('models/model1.pt')
model.to('cpu')
model.eval()
classes = ['Dark','Green','Light','Medium']
resize = T.Resize((224, 224))
st.title('Кофе')

images = st.file_uploader("загрузи изображение", type=["jpg", "jpeg", "png"])
if images is not None:
    if images.type not in ['image/jpeg','image/png']:
        st.error('не тот формат!')
    else:
        img = Image.open(images)
        img = trnsfrms1(img)
        img = img.unsqueeze(0)
        pred = model(img)
        pred = torch.max(pred,1)
        clas = classes[pred.indices]
        resize = Image.open(images)
        resize = resize.resize((400,300))
        st.image(resize)
        st.title(clas)