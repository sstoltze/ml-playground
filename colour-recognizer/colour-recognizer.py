import tensorflow as tf
from tensorflow import keras
import numpy as np

print("TensorFlow:", tf.VERSION)
print("Keras:", keras.__version__)
print("Numpy:", np.version.full_version)

class_names = ["Black", "Blue", "Green", "Orange",
               "Red", "Violet", "White", "Brown", "Yellow"]


# Taken from https://www.tensorflow.org/guide/datasets
def image_parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized


def load_test_data():
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    dataset_directory = "./colour-datasets/"
    with open(dataset_directory + "images.txt", "r") as f:
        for line in f:
            (data, label) = line.rstrip("\n").split(" ")
            train_data.append(dataset_directory+data)
            train_labels.append(int(label))
    with open(dataset_directory + "timages.txt", "r") as f:
        for line in f:
            (data, label) = line.rstrip("\n").split(" ")
            test_data.append(dataset_directory + data)
            test_labels.append(int(label))
    train_data = tf.constant(train_data)
    train_labels = tf.constant(train_labels)
    test_data = tf.constant(test_data)
    test_labels = tf.constant(test_labels)
    print("data:", train_data.shape, " - labels:", train_labels.shape)
    print("map_fn:", tf.map_fn(image_parse_function, train_data).shape)
    return ((tf.map_fn(image_parse_function, train_data), train_labels),
            (tf.map_fn(image_parse_function, test_data),  test_labels))


def preprocess_data(train, test):
    # For fashion_mnist:
    return train / 255.0, test / 255.0


(train_images, train_labels), (test_images, test_labels) = load_test_data()
model = keras.Sequential([
    # Make a 1d array instead of 2d
    keras.layers.Flatten(input_shape=(28, 28, 3)),
    # 128 neuron layer
    keras.layers.Dense(128, activation=tf.nn.relu),
    # Softmax returns a probablility for each option
    keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
])

# Optimizer is how to update the model based on the result
model.compile(optimizer=tf.train.AdamOptimizer(),
              # Loss function for the model
              loss='sparse_categorical_crossentropy',
              # Uses correct / total predictions to measure success
              metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=5, steps_per_epoch=101)
