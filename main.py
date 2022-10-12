"""
Main file.
Il contient le code de base à exécuter.
"""

"""
 ---------- LES IMPORTS ----------
"""

"""
 ---------- Organisation des répertoires de travail ----------
"""
# Création des dossiers de structure.
import os
import random as pif
import shutil
from keras.applications import VGG16
from keras.dtensor import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
if not os.path.exists("./data"):
    os.mkdir("./data")
if not os.path.exists("./data/savedmodels"):
    os.mkdir("./data/savedmodels")
if not os.path.exists("./data/allImages"):
    os.mkdir("./data/allImages")
    print("Veuillez déposer toutes les images de chat et de chien dans le dossier data/allImages,")
    print("puis relancez le script.")
    exit()

# Fichiers sources des images.
path_images_src = "./data/allImages"
path_savedmodels = "./data/savedmodels"

# Les différents répertoires à créer.
workdir_path = "./data/workdir"
train_path = f"{workdir_path}/train"
valid_path = f"{workdir_path}/valid"
eval_path = f"{workdir_path}/eval"

dirs = [train_path, valid_path, eval_path]

# S'il n'existe pas déjà, alors nous les créons.
if not os.path.exists(workdir_path):
    os.mkdir(workdir_path)
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)
            os.mkdir(f"{d}/cats")
            os.mkdir(f"{d}/dogs")
    # On récupère le nom de chaque image.
    images_train_path = os.listdir(path_images_src)
    # Nous savons qu'il y en a 25 000 et que 'os.listdir' trie par ordre alphabétique.
    # Donc les 12 500 premières sont les chats et les 12 500 suivantes sont les chiens.
    images_cats = images_train_path[:12500]
    images_dogs = images_train_path[-12500:]
    # Pour ne pas grader l'ordre alphabétique dans nos listes, nous les mélangeons.
    pif.shuffle(images_dogs)
    pif.shuffle(images_cats)
    # Parmi toutes les images de chien, nous en sélectionnons 2 000, idem pour les chats.
    images_cats_selected = images_cats[:2000]
    images_dogs_selected = images_dogs[:2000]
    # Les 1 000 premières serons celles pour l'entraînement du modèle.
    train_images_cats = images_cats_selected[:1000]
    train_images_dogs = images_dogs_selected[:1000]
    # Les 500 suivantes seront pour son évaluation.
    eval_images_cats = images_cats_selected[1000:1500]
    eval_images_dogs = images_dogs_selected[1000:1500]
    # Les 500 dernières seront pour la validation.
    valid_images_cats = images_cats_selected[-500:]
    valid_images_dogs = images_dogs_selected[-500:]
    # Nous réalisons ici la copie des images dans leur répertoire.
    for step in ['train', 'eval', 'valid']:
        for group in ['cats', 'dogs']:
            for image in vars()[f"{step}_images_{group}"]:
                shutil.copy(f"{path_images_src}/{image}",
                            f"{vars()[step + '_path']}/{group}")

"""
 ---------- PRE-TRAITEMENT DES DONNEES ----------
"""

# Nous créons un 'ImageDataGenerator' qui permet de transformer un pool d'image en un plus grand.
# Il va utiliser des transformations sur les images comme la rotation ou le zoom pour créer
# plus d'images qu'il n'en possède à l'origine.
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3
)
# Nous préciserons au générateur quel dossier utiliser ainsi que la taille des images souhaitée et enfin
# la taille du batch.
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=19,
    class_mode='binary')

# Ce 'ImageDataGenerator' permet également de normaliser les images donc nous en créons un aussi
# pour les images d'évaluation et de validation.
valid_datagen = ImageDataGenerator(rescale=1 / 255)

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(150, 150),
    batch_size=19,
    class_mode='binary')

eval_datagen = ImageDataGenerator(rescale=1 / 255)

eval_generator = valid_datagen.flow_from_directory(
    eval_path,
    target_size=(150, 150),
    batch_size=19,
    class_mode='binary')

"""
 ---------- LES MODELES ----------
"""

# Ensuite viennent nos deux modèles testés dans ce projet.

# Le premier ci-après est celui que nous avons réalisé et le suivant est celui qui utilise un réseau existant.
# Les deux modèles sont résumés dans la trace d'exécution.

# Notre modèle ----------------
model_own = models.Sequential(name="NotreModele")
model_own.add(layers.Conv2D(filters=32, kernel_size=(4, 4), padding='same', activation='relu', strides=2,
                            input_shape=(150, 150, 3)))
model_own.add(layers.MaxPooling2D(2, 2))
model_own.add(layers.Conv2D(filters=64, kernel_size=(3, 3),
              padding='same', activation='relu', strides=2))
model_own.add(layers.MaxPooling2D(2, 2))
model_own.add(layers.Conv2D(filters=128, kernel_size=(3, 3),
              padding='same', activation='relu', strides=2))
model_own.add(layers.MaxPooling2D(2, 2))
model_own.add(layers.Flatten())
model_own.add(layers.Dropout(0.5))
model_own.add(layers.Dense(units=512, activation='relu'))
model_own.add(layers.Dense(units=1, activation='sigmoid'))

model_own.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(
    learning_rate=1e-4), metrics=['accuracy'])

# Le modèle VGG ----------------
# Ici nous prenons comme base le réseau VGG16. Cela va nous permettre d'utiliser un réseau très performant
# que nous allons spécifier.
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

model_tun = models.Sequential(name="VGGModele")
model_tun.add(conv_base)
model_tun.add(layers.Flatten())
# Ici on spécifie que la sortie sera de 1 neurone pour 2 classes
model_tun.add(layers.Dense(256, activation='relu'))
model_tun.add(layers.Dense(1, activation='sigmoid'))

# Ici on va verrouiller les poids qui appartiennent a VGG pour n'entraîner que la spécification de sortie.
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model_tun.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(
    learning_rate=1e-4), metrics=['accuracy'])

mods = [model_own, model_tun]
# Pour nos deux modèles, nous allons les entraîner avec le même jeu de données et voir les résultats sur les courbes.
for m in mods:
    print("\n", end='')
    # Cette commande va afficher le résumé du modèle. Avec le nombre de poids, les couches, ect...
    m.summary()
    # Pour éviter d'entraîner les modèles à chaque run du script, nous les sauvegardons.
    if not os.path.exists(f"{path_savedmodels}/{m.name}.h5"):
        # L'entraînement à proprement parler.
        print(f"Entrainement du modèle {m.name} --------------------------")
        history = m.fit(train_generator,  # On utilise les générateurs d'image expliqués plus haut
                        steps_per_epoch=len(train_generator),
                        epochs=50,  # Le nombre d'itérations total avant de conserver le modèle
                        validation_data=valid_generator,
                        validation_steps=len(valid_generator)
                        )
        m.save(f"{path_savedmodels}/{m.name}.h5")

        # Ici nous créons des graphiques qui permettent de représenter la courbe d'apprentissage du modèle.
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.figure(f"1-{m.name}")
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(f"{path_savedmodels}/figs/1-{m.name}.png", format='png')

        plt.figure(f"2-{m.name}")
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(f"{path_savedmodels}/figs/2-{m.name}.png", format='png')

    else:
        # Dans cette partie nous évaluons le modèle chargé avec les données d'évaluation.
        print(f"Chargement du modèle {m.name} --------------------------")
        modele_load = models.load_model(f"{path_savedmodels}/{m.name}.h5")
        print("Evaluation du modèle:")
        metrics = modele_load.evaluate(
            eval_generator, steps=10, batch_size=19, return_dict=True)
        print(f"Loss: {metrics['loss']:.2f}")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print("\n\n", end='')

        # Nous affichons également les figures sauvegardées.
        plt.figure(f"1-{m.name}")
        fg1 = plt.imread(f"{path_savedmodels}/figs/1-{m.name}.png")
        plt.imshow(fg1)
        plt.axis('off')

        plt.figure(f"2-{m.name}")
        fg2 = plt.imread(f"{path_savedmodels}/figs/2-{m.name}.png")
        plt.imshow(fg2)
        plt.axis('off')

plt.show()
