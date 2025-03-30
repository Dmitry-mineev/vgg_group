import torch
import torchvision
from torchvision import transforms as T
import streamlit as st
from PIL import Image
DEVICE = 'cpu'


trnsfrms1 = T.Compose([T.Resize((255, 255)),T.ToTensor()])
model = torch.jit.load('models/model2.pt')
model.to('cpu')
model.eval()
classes = ['Вишня',
 'Кофейное растение',
 'Огурец',
 'Лисий орех(Махана)',
 'Лимон',
 'Лимонное дерево',
 'Лимонно-грушевое пюре(баджра)',
 'Табак-растение',
 'Миндаль',
 'банан',
 'кардамон',
 'чили',
 'гвоздика',
 'кокос',
 'хлопок',
 'грамм',
 'джоуэр',
 'джут',
 'маис',
 'горчичное масло',
 'папайя',
 'ананас',
 'рис',
 'соевый боб',
 'сахарный тростник',
 'подсолнух',
 'чай',
 'помидор',
 'винья-радиати(Мунг)',
 'пшеница']
resize = T.Resize((224, 224))
st.title('Агрокультура')

images = st.file_uploader("загрузи изображение", type=["jpg", "jpeg", "png"])
if images is not None:
    if images.type not in ['image/jpeg','image/png']:
        st.error('не тот формат!')
    else:
        img = Image.open(images)
        img = trnsfrms1(img)
        img = img.unsqueeze(0)
        pred = model(img.to(DEVICE))
        pred = torch.max(pred,1)
        clas = classes[pred.indices]
        resize = Image.open(images)
        resize = resize.resize((400,300))
        st.image(resize)
        st.title(clas)