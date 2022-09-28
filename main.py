"""
Main file.
Il contient le code de base à executé.
"""
import os
import random as pif
import shutil

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
