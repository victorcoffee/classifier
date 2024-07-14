import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import models
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

st.title("Класифікатор зображень")

# Завантаження і підготовка даних Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test / 255.0
x_test = np.expand_dims(x_test, axis=-1)  # Якщо модель очікує (28, 28, 1)
y_test_categorical = to_categorical(y_test, num_classes=10)

# Вибір моделі
id_model = st.sidebar.selectbox("Виберіть модель:", ("CNN", "VGG16"))

if id_model == "CNN":
    model = models.load_model("fashion_mnist_cnn.h5")
    history = np.load("history_cnn.npy", allow_pickle=True).item()
    st.write("Модель CNN завантажена")

elif id_model == "VGG16":
    base_model = VGG16(weights=None, include_top=False, input_shape=(48, 48, 3))
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(10, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights("fashion_mnist_vgg16.h5")
    history = np.load("history_vgg16.npy", allow_pickle=True).item()
    st.write("Модель VGG16 завантажена")

# Завантаження зображення
uploaded_file = st.file_uploader("Виберіть зображення...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Обране зображення", use_column_width=True)

    # Перетворення зображення на чорно-біле або кольорове залежно від моделі
    if id_model == "CNN":
        image = image.convert("L")
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=-1)
    else:
        image = image.convert("RGB")  # Перетворення на кольорове зображення
        image = image.resize((48, 48))
        image_array = np.array(image)

    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Передбачення класу
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Список назв класів
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Отримання назви класу
    predicted_class_name = class_names[predicted_class[0]]
    st.write(f"Передбачений клас: {predicted_class_name}")

    # Чекбокс для відображення графіків
    show_plots = st.sidebar.checkbox("Показати графіки навчання")

    # Побудова графіків
    if show_plots:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(history["loss"], label="Train Loss")
        ax1.plot(history["val_loss"], label="Val Loss")
        ax1.legend()
        ax1.set_title("Втрати")

        ax2.plot(history["accuracy"], label="Train Accuracy")
        ax2.plot(history["val_accuracy"], label="Val Accuracy")
        ax2.legend()
        ax2.set_title("Точність")

        st.pyplot(fig)
