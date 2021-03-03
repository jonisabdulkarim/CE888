#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets

### Data Loading and Batching

# constant values
IMAGE_WIDTH = 256 # instead of 254, the original size
IMAGE_HEIGHT = 256
BATCH_SIZE = 32
SEED = 123
VALIDATION_SPLIT = 0.2

# import images for training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='./Training',
    color_mode='rgb',
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    batch_size=BATCH_SIZE
)
class_names = train_ds.class_names

# import images for validation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='./Training',
    color_mode='rgb',
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    batch_size=BATCH_SIZE
)

# import images for hold-out test
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='./Test',
    color_mode='rgb',
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE
)

### Visualise
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()


### Performance Optimisation

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)
data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

### Model Design

inputs = keras.Input(shape=input_shape)
#x = inputs
x = data_augmentation(inputs)


x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

previous_block_activation = x
for size in [8]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)

    x = layers.add([x, residual])
    previous_block_activation = x

x = layers.SeparableConv2D(8, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs, name="model_fire")

### Model Compilation, Building and Fitting

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    batch_size=BATCH_SIZE
)

model.save("./output_model")

### Model Evaluation

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

'''
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
'''

### Test Evaluation

test_loss, test_acc = model.evaluate(test_ds, batch_size=BATCH_SIZE)

print("Testing accuracy is: ", test_acc)
print("Testing loss is: ", test_loss)


### PLESASE WORK!!!
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
