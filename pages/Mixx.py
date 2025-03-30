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
model2 = torch.jit.load('models/model2.pt')
model2.to('cpu')
model2.eval()

classes = ['Dark','Green','Light','Medium']
classes1 = ['Вишня',
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
st.title('Кто ты?')

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
        st.title(f"это кофе {clas}")

        pred2 = model2(img)
        pred2 = torch.max(pred2,1)
        clas1 = classes1[pred2.indices]
        resize1 = Image.open(images)
        resize1 = resize1.resize((400,300))
        st.image(resize1)
        st.title(f"...или {clas1}")