import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import requests
from io import BytesIO
import ssl
from urllib.request import urlretrieve
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Классификация изображений", layout="wide")
st.title("Классификация изображений двумя моделями")

# Отключение SSL-проверки для загрузки моделей (так вообще никто делать не рекомендует)
ssl._create_default_https_context = ssl._create_unverified_context

# ImageNet
@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        urlretrieve(url, "imagenet_classes.txt")
        with open("imagenet_classes.txt") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except Exception as e:
        st.error(f"Ошибка загрузки меток классов: {e}")
        return [f"class_{i}" for i in range(1000)]

labels = load_imagenet_labels()

# Загрузка предобученных моделей
@st.cache_resource
def load_models():
    resnet = models.resnet50(pretrained=True)
    efficientnet = models.efficientnet_b0(pretrained=True)
    resnet.eval()
    efficientnet.eval()
    return resnet, efficientnet

resnet, efficientnet = load_models()

# Загрузка по URL
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        st.error(f"Ошибка загрузки изображения по URL: {e}")
        return None

# Ограничение размера
def resize_image_for_display(image, max_size=600):
    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

# Обработка изображения
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def get_predictions(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    return probabilities

# Инициализация переменных
if 'images' not in st.session_state:
    st.session_state.images = []
    st.session_state.image_names = []
    st.session_state.performance_data = {
        'Image': [],
        'ResNet50_time': [],
        'EfficientNet_time': [],
        'Total_time': []
    }

# Интерфейс загрузки изображения
st.sidebar.header("Способы загрузки изображения")
option = st.sidebar.radio("Выберите способ загрузки:", 
                         ("Загрузить файлы", "Указать URL-ы"))

if option == "Загрузить файлы":
    uploaded_files = st.sidebar.file_uploader("Выберите файлы изображений", 
                                           type=["jpg", "jpeg", "png"],
                                           accept_multiple_files=True)
    if uploaded_files:
        st.session_state.images = []
        st.session_state.image_names = []
        st.session_state.performance_data = {
            'Image': [],
            'ResNet50_time': [],
            'EfficientNet_time': [],
            'Total_time': []
        }
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.images.append(image)
            st.session_state.image_names.append(uploaded_file.name)
else:
    urls = st.sidebar.text_area("Введите URL-ы изображений (по одному на строку):")
    if urls:
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]
        st.session_state.images = []
        st.session_state.image_names = []
        st.session_state.performance_data = {
            'Image': [],
            'ResNet50_time': [],
            'EfficientNet_time': [],
            'Total_time': []
        }
        for url in url_list:
            if url != "https://example.com/image.jpg":
                image = load_image_from_url(url)
                if image:
                    st.session_state.images.append(image)
                    st.session_state.image_names.append(os.path.basename(url))

# Если изображения загружены
if st.session_state.images:
    st.subheader(f"🔢 Обработано изображений: {len(st.session_state.images)}")
    
    for idx, (image, image_name) in enumerate(zip(st.session_state.images, st.session_state.image_names)):
        st.markdown(f"---\n### Изображение {idx+1}: {image_name}")
        
        # Отображаем изображение
        col_img, _ = st.columns([1, 2])
        with col_img:
            display_image = resize_image_for_display(image)
            st.image(display_image, caption=image_name, use_column_width=True)
        
        # Обработка моделей
        input_tensor = preprocess_image(image)
        
        col1, col2 = st.columns(2)
        
        # ResNet
        with col1:
            st.subheader("ResNet50")
            start_time = time.time()
            probs = get_predictions(resnet, input_tensor)
            resnet_time = (time.time() - start_time) * 1000
            
            top2_probs, top2_indices = torch.topk(probs, 2)
            
            st.write("**Топ-2 предсказания:**")
            for i in range(2):
                class_name = labels[top2_indices[i]].replace("_", " ").capitalize()
                st.write(f"🏆 **{class_name}**: {top2_probs[i].item()*100:.2f}%")
            
            st.write(f"⏱ **Время обработки**: {resnet_time:.2f} мс")
        
        # EfficientNet
        with col2:
            st.subheader("EfficientNet")
            start_time = time.time()
            probs = get_predictions(efficientnet, input_tensor)
            effnet_time = (time.time() - start_time) * 1000
            
            top2_probs, top2_indices = torch.topk(probs, 2)
            
            st.write("**Топ-2 предсказания:**")
            for i in range(2):
                class_name = labels[top2_indices[i]].replace("_", " ").capitalize()
                st.write(f"🏆 **{class_name}**: {top2_probs[i].item()*100:.2f}%")
            
            st.write(f"⏱ **Время обработки**: {effnet_time:.2f} мс")
        
        # Сохраняем метрики
        st.session_state.performance_data['Image'].append(f"Img {idx+1}")
        st.session_state.performance_data['ResNet50_time'].append(resnet_time)
        st.session_state.performance_data['EfficientNet_time'].append(effnet_time)
        st.session_state.performance_data['Total_time'].append(resnet_time + effnet_time)
    
    st.markdown("---")
    st.subheader("📊 Анализ производительности")
    
    df = pd.DataFrame(st.session_state.performance_data)
    
    # График времени выполнения
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, df['ResNet50_time'], width, label='ResNet50', color='#1f77b4')
    rects2 = ax.bar(x + width/2, df['EfficientNet_time'], width, label='EfficientNet', color='#ff7f0e')
    
    ax.set_xlabel('Изображения')
    ax.set_ylabel('Время (мс)')
    ax.set_title('Сравнение времени обработки')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Image'])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    st.pyplot(fig)
    
    # Сводная статистика
    st.subheader("📈 Сводные метрики")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Среднее время ResNet50", f"{df['ResNet50_time'].mean():.2f} мс")
    with col2:
        st.metric("Среднее время EfficientNet", f"{df['EfficientNet_time'].mean():.2f} мс")
    with col3:
        st.metric("Общее время обработки", f"{df['Total_time'].sum():.2f} мс")
    
    # Таблица с данными
    st.dataframe(df.set_index('Image'), use_container_width=True)

else:
    st.info("ℹ️ Пожалуйста, загрузите изображения через меню слева")

# Кнопка очистки результатов
if st.sidebar.button("🧹 Очистить все результаты"):
    st.session_state.images = []
    st.session_state.image_names = []
    st.session_state.performance_data = {
        'Image': [],
        'ResNet50_time': [],
        'EfficientNet_time': [],
        'Total_time': []
    }
    st.experimental_rerun()