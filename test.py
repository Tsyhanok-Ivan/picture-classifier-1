import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import random

ds_test = tfds.load("cifar10", split="test", as_supervised=True)

ds_test_list = list(ds_test)

random.shuffle(ds_test_list)

random_images = ds_test_list[:10]

model = tf.keras.models.load_model("results/cifar10-model-v1.h5")


# Функція для препроцессингу зображень
def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (32, 32))
    return image, label


class_names = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

# Препроцесинг та класифікація зображень
for image, label in random_images:
    image, label = preprocess_image(image, label)

    # Додаємо розмірність пакету batch dimension
    image = tf.expand_dims(image, axis=0)

    # Класифікація зображення
    predictions = model.predict(image)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

    # Виведення зображення та назви класу
    plt.imshow(image[0])
    plt.title(f"Predicted class: {class_names[predicted_class]}")
    plt.show()
