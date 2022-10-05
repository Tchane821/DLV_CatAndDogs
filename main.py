"""
Main file.
Il contient le code de base à executé.
"""
import os
import random as pif
import shutil

from keras.dtensor import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt

path_images_src = "./data/train"

# Création de repertoire
workdir_path = "./data/workdir"
training_path = f"{workdir_path}/train"
validation_path = f"{workdir_path}/validation"
evaluate_path = f"{workdir_path}/evaluation"

dirs = [training_path, validation_path, evaluate_path]

if not os.path.exists(workdir_path):
    os.mkdir(workdir_path)
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)
            os.mkdir(f"{d}/cats")
            os.mkdir(f"{d}/dogs")

    images_train_path = os.listdir("./data/train")

    images_cats = images_train_path[:12500]
    images_dogs = images_train_path[-12500:]

    pif.shuffle(images_dogs)
    pif.shuffle(images_cats)

    images_cats_selected = images_cats[:2000]
    images_dogs_selected = images_dogs[:2000]

    train_images_cats = images_cats_selected[:1000]
    train_images_dogs = images_dogs_selected[:1000]

    eval_images_cats = images_cats_selected[1000:1500]
    eval_images_dogs = images_dogs_selected[1000:1500]

    valid_images_cats = images_cats_selected[-500:]
    valid_images_dogs = images_dogs_selected[-500:]

    # for step in ['train', 'eval', 'valid']:
    #     for group in ['cats', 'dogs']:
    #         for vars()[f"img_{step}"] in vars()[f"{step}_images_{group}"]:
    #             shutil.copy(f"{path_images_src}/{vars()['img_' + step]}", f"{training_path}/{group}")

    for img_train in train_images_cats:
        shutil.copy(f"{path_images_src}/{img_train}", f"{training_path}/cats")
    for img_train in train_images_dogs:
        shutil.copy(f"{path_images_src}/{img_train}", f"{training_path}/dogs")

    for img_train in eval_images_cats:
        shutil.copy(f"{path_images_src}/{img_train}", f"{evaluate_path}/cats")
    for img_train in eval_images_dogs:
        shutil.copy(f"{path_images_src}/{img_train}", f"{evaluate_path}/dogs")

    for img_train in valid_images_cats:
        shutil.copy(f"{path_images_src}/{img_train}", f"{validation_path}/cats")
    for img_train in eval_images_dogs:
        shutil.copy(f"{path_images_src}/{img_train}", f"{validation_path}/dogs")

# prétraitement
train_datagen = ImageDataGenerator(rescale=1 / 255)
test_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(
    training_path, target_size=(150, 150), batch_size=20, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_path, target_size=(150, 150), batch_size=20, class_mode='binary')

# modèle
model = models.Sequential()
model.add(layers.Conv2D(filters=10, kernel_size=(4, 4), padding='same', activation='relu', strides=2,
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(filters=20, kernel_size=(3, 3), padding='same', activation='relu', strides=2))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(units=200, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(units=150, activation='relu'))
model.add(layers.Dense(units=70, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['accuracy'])

history = model.fit(train_generator, steps_per_epoch=20, epochs=400,
                    validation_data=validation_generator, validation_steps=50)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
