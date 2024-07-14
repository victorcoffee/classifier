import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Завантаження і підготовка даних Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.stack([x_train] * 3, axis=-1)
x_test = np.stack([x_test] * 3, axis=-1)
x_train = tf.image.resize(x_train, (48, 48))
x_test = tf.image.resize(x_test, (48, 48))

# Конвертація цільових значень в категоріальні мітки
y_train_categorical = to_categorical(y_train, num_classes=10)
y_test_categorical = to_categorical(y_test, num_classes=10)

# Завантаження моделі VGG16 без верхніх шарів
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(48, 48, 3))

# Додавання власних шарів
x = base_model.output
x = Flatten()(x)
predictions = Dense(10, activation="softmax")(x)

# Створення нової моделі
model = Model(inputs=base_model.input, outputs=predictions)

# Заморожування базових шарів VGG16
for layer in base_model.layers:
    layer.trainable = False

# Компіляція моделі
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Навчання моделі
history = model.fit(
    x_train,
    y_train_categorical,
    epochs=10,
    validation_data=(x_test, y_test_categorical),
)

# Збереження моделі і історії
model.save("fashion_mnist_vgg16.h5")
np.save("history_vgg16.npy", history.history)

# Побудова графіків
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.show()
