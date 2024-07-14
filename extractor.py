import os
from tensorflow.keras.datasets import fashion_mnist
from PIL import Image

# Створення директорії для збереження зображень
output_dir = "fashion_mnist_test_images"
os.makedirs(output_dir, exist_ok=True)

# Завантаження даних Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Збереження зображень на диск
for i, image in enumerate(x_test):
    img = Image.fromarray(image)
    img.save(os.path.join(output_dir, f"image_{i}.png"))

print(f"Saved {len(x_test)} images to {output_dir}")
