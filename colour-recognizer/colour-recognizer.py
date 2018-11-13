# Based on https://www.tensorflow.org/tutorials/keras/basic_classification
# Consider looking at https://medium.com/@nickbortolotti/what-is-your-favorite-tie-color-tensorflow-analysis-95c5f7554baa
import tensorflow as tf
from tensorflow import keras
import numpy as np

image_size = 56
class_names = ["Black", "Blue", "Green", "Orange",
               "Red", "Violet", "White", "Brown", "Yellow"]


# Taken from https://www.tensorflow.org/guide/datasets
def image_parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded,
                                           [image_size, image_size])
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
    return ((tf.map_fn(image_parse_function,
                       train_data,
                       # Required since return-type is different from input
                       dtype=tf.float32),
             train_labels),
            (tf.map_fn(image_parse_function,
                       test_data,
                       # Required since return-type is different from input
                       dtype=tf.float32),
             test_labels))


(train_images, train_labels), (test_images, test_labels) = load_test_data()

# The return dtype should be float32 with values between 0 and 1,
# hence we can define our model as in the tutorial.
# This leads to an accuracy of 0.1-0.2 on the test dataset
tutorial_model = keras.Sequential([
    # Make a 1d array instead of 2d
    keras.layers.Flatten(input_shape=(28, 28, 3)),
    # 128 neuron layer
    keras.layers.Dense(128, activation=tf.nn.relu),
    # Softmax returns a probablility for each option
    keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
])

# Try/catch to only train the model once while testing.
# Can be removed later
try:
    model
except NameError:
    print("Compiling model.")
    model = keras.Sequential([
        # Make a 1d array instead of 2d
        keras.layers.Flatten(input_shape=(image_size, image_size, 3)),
        # Neuron layer
        keras.layers.Dense(12*image_size, activation=tf.nn.relu),
        # Neuron layer
        keras.layers.Dense(12*image_size, activation=tf.nn.relu),
        # Neuron layer
        keras.layers.Dense(12*image_size, activation=tf.nn.relu),
        # Softmax returns a probablility for each option
        keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
    ])
    # Optimizer is how to update the model based on the result
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  # Loss function for the model
                  loss='sparse_categorical_crossentropy',
                  # Uses correct / total predictions to measure success
                  metrics=['accuracy'])

    # This step takes a while
    model.fit(train_images, train_labels, epochs=5, steps_per_epoch=30)

# Test the accuracy
print("Testing accuracy.")
test_loss, test_acc = model.evaluate(test_images, test_labels, steps=30)
print('Test accuracy:', test_acc)
