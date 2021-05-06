# %% [code]
#!/usr/bin/env python
# coding: utf-8

#import os
#import datetime
#import multiprocessing
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16, Xception, InceptionResNetV2

# %% [code]
# constant values
IMAGE_SIZE = (254, 254)
BATCH_SIZE = 64  # [32, 64, 128], ideal: 64
SEED = 123
VALIDATION_SPLIT = 0.2
NO_OF_EPOCHS = 10 # initial, not total
FINE_TUNE_EPOCHS = 5
LEARNING_RATE = 1e-3
UPPER_UNITS = 64
LOWER_UNITS = 32
#FOLDER = "/kaggle/input/the-flame-dataset"
MODEL_CHOICE = 3
DATA_AUG = True

# %% [code]
# import images for training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #directory=FOLDER+'/Training/Training',
    directory="./Training",
    color_mode='rgb',
    image_size=IMAGE_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    batch_size=BATCH_SIZE
)
class_names = train_ds.class_names

# import images for validation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #directory=FOLDER+'/Training/Training',
    directory="./Training",
    color_mode='rgb',
    image_size=IMAGE_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    batch_size=BATCH_SIZE,
)

# import images for hold-out test
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #directory=FOLDER+'/Test/Test',
    directory="./Test",
    color_mode='rgb',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# %% [code]
# visualise
"""
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
"""

# %% [code]
# performance optimisation
# AUTOTUNE = tf.data.AUTOTUNE
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)  # changed from 32, AUTOTUNE, no cache, no shuffle=1000
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)  # 32, AUTOTUNE, no cache
# test_ds = test_ds.cache().prefetch

# %% [code]
# data augmentation TODO add more data aug?
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ], name="data_aug"
)

# %% [code]
# model inputs
input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
inputs = keras.Input(shape=input_shape)

if (DATA_AUG):
    x = data_augmentation(inputs)
else:
    x = inputs

if (MODEL_CHOICE == 1):
    x = tf.keras.applications.vgg16.preprocess_input(x)
elif (MODEL_CHOICE == 2):
    x = tf.keras.applications.xception.preprocess_input(x)
elif (MODEL_CHOICE == 3):
    x = tf.keras.applications.inception_resnet_v2.preprocess_input(x)
else:
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)

# %% [code] {"jupyter":{"outputs_hidden":true}}
"""
Adding new model for transfer learning, first get top base
"""
print("Creating model...")  # todo

if (MODEL_CHOICE == 1):
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
    )
elif (MODEL_CHOICE == 2): 
    base_model = Xception(
        include_top=False,
        weights="imagenet",
    )
elif (MODEL_CHOICE == 3):
    base_model = InceptionResNetV2(
        include_top=False,
        weights="imagenet",
    )

base_model.trainable = False    

print("Initial TF model built")
#base_model.summary()

# %% [code]
"""
Next, get last layer of TF model, and place new layers
"""
"""
if (MODEL_CHOICE == 1):
    last_layer = model.get_layer('block5_pool').output
else:
    last_layer = model.get_layer('avg_pool').output
"""

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Flatten()(x) # just added
x = layers.Dense(512, activation='relu', name='fc1')(x)
x = layers.Dense(256, activation='relu', name='fc2')(x)
x = layers.Dense(128, activation='relu', name='fc3')(x)
x = layers.Dense(64, activation='relu', name='fc4')(x) # Changed from 64-32, then 32-16
x = layers.Dense(32, activation='relu', name='fc5')(x)
output = layers.Dense(1, activation='sigmoid', name='output')(x)  # 2 classes

model = keras.Model(inputs, output)

print("Modified TF model built")
model.summary()

# %% [code]
# model compilation, building and fitting

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=0.1),  # lr: [def/1e-3, 1], eps: [def/1e-7, 0.1, 1], ideal: [lr: 1e-3, eps:0.1]
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# %% [code]
"""
Implement callback which stops early and saves the best model to file
"""
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(filepath='xception_model.h5', save_best_only=True),
]

# %% [code]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NO_OF_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    callbacks=callbacks
)

# model.save("./output_model")  # todo: doesn't seem necessary cuz of above cell?

# %% [code]
# obtain accuracy and loss from initial fitting

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(NO_OF_EPOCHS)

# %% [code]
# visualise training and validation loss
"""
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, acc, label='Training Accuracy')
plt.plot(history.epoch, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.epoch, loss, label='Training Loss')
plt.plot(history.epoch, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
"""

# %% [code] {"_kg_hide-input":true}
"""
new_model = tf.keras.models.load_model('/kaggle/working/xception_model.h5')
new_test_loss, new_test_acc = new_model.evaluate(test_ds, batch_size=BATCH_SIZE)

print("Testing accuracy is: ", new_test_acc)
print("Testing loss is: ", new_test_loss)

model = new_model
"""

# %% [code]
# test evaluation

test_loss, test_acc = model.evaluate(test_ds, batch_size=BATCH_SIZE)

print("Testing accuracy is: ", test_acc)
print("Testing loss is: ", test_loss)

# %% [code]
# fine-tuning
print("Number of layers in the base model: ", len(base_model.layers))

base_model.trainable = True

#fine_tune_at = 100
#for layer in base_model.layers[:100]:
#    layer.trainable = False

model.compile(loss="binary_crossentropy", 
              optimizer=tf.keras.optimizers.RMSprop(lr=(LEARNING_RATE/10)), 
              metrics = ["accuracy"])

model.summary()

print("Number of trainable layers in the whole model: ", len(model.trainable_variables))

# %% [code]
"""
Implement callback which stops early and saves the best model to file
"""
fine_callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(filepath='xception_model_tuned.h5', save_best_only=True),
]

# %% [code]
# fit fine-tuned model

total_epochs =  NO_OF_EPOCHS + FINE_TUNE_EPOCHS
current_epoch = history.epoch[-1]

history_fine = model.fit(train_ds,
                         validation_data=val_ds,
                         epochs=total_epochs,
                         batch_size=BATCH_SIZE,
                         verbose=2,
                         callbacks=fine_callbacks,
                         initial_epoch=current_epoch,
                         )

# %% [code]
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

# %% [code]
"""
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
#plt.ylim([0.8, 1])
plt.plot([NO_OF_EPOCHS-1,NO_OF_EPOCHS-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([NO_OF_EPOCHS-1,NO_OF_EPOCHS-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
"""


# %% [code]
loss, accuracy = model.evaluate(test_ds)
print('Test accuracy :', accuracy)

# %% [code]
"""
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
"""

# %% [code]
"""
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

Categories=['Fire','No_fire']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir=FOLDER+"/Training/Training/"
#path which contains all the categories of images

for i in Categories:
    print(f'loading... category : {i}')    
    path=os.path.join(datadir,i)
    
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
        
    print(f'loaded category:{i} successfully')
    
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
x=df.iloc[:,:-1] #input data
y=df.iloc[:,-1] #output data
"""
