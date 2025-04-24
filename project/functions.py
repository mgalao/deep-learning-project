import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import models
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from functools import partial
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.models import load_model



def plot_batch(dataset, class_names, num_images, rows, cols):

    images, labels = next(iter(dataset))
    plt.figure(figsize=(20,20))

    for i in range(num_images):
        ax = plt.subplot(rows, cols, i + 1)
        image = images[i].numpy()
        label = labels[i].numpy()
        index = np.argmax(label)
        class_name = class_names[index]
        plt.imshow(image)
        plt.title(class_name, fontsize=12, color='crimson')
        plt.axis('off')
        
    plt.show()


def organize_split(image_base_path, base_output_dir, split_df, copy, split_name):
    for _, row in split_df.iterrows():
        src = Path(row["full_file_path"])
        dst = base_output_dir / split_name / str(row["family"]) / src.name

        os.makedirs(dst.parent, exist_ok=True)
        if copy==True:
            try:
                shutil.copy(src, dst)
            except FileNotFoundError:
                print(f"Not found: {src}")
        else:
            try:
                shutil.move(src, dst)
            except FileNotFoundError:
                print(f"Not found: {src}")
            


# Clean up the folders that are not needed
def cleanup_folders(base_path):
    keep_folders = {"train", "val", "test"}
    for item in base_path.iterdir():
        if item.is_dir() and item.name not in keep_folders:
            shutil.rmtree(item)


# Repoint image paths to the new train/val/test directories
def update_paths(df, split_name):
    df = df.copy()
    df["full_file_path"] = df.apply(
        lambda row: str(Path(f"../data/rare_species/{split_name}") / str(row["family"]) / Path(row["full_file_path"]).name),
        axis=1
    )
    return df
    

def plot_graph(title, xlabel, ylabel, counts):
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='midnightblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90) 
    plt.tight_layout()
    plt.show()


def build_sequential_model(list_of_layers):
    model = models.Sequential()
    for layer in list_of_layers: model.add(layer)
    return model


def plot_model_acc(num_epochs, train_acc, val_acc):
    x_axis = range(1,num_epochs+1)
    plt.plot(x_axis, train_acc, 'r', label='Training accuracy')
    plt.plot(x_axis, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_model_loss(num_epochs, train_loss, val_loss):
    x_axis = range(1,num_epochs+1)
    plt.plot(x_axis, train_loss, 'g', label='Training loss')
    plt.plot(x_axis, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def get_callbacks(checkpoint_file_path, metrics_file_path, lr_scheduler=None):
    os.makedirs(os.path.dirname(checkpoint_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)

    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_file_path,
        monitor='val_loss',
        verbose=0
    )
    callbacks.append(checkpoint_callback)

    metrics_callback = CSVLogger(metrics_file_path)
    callbacks.append(metrics_callback)

    if lr_scheduler is not None:
        lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
        callbacks.append(lr_scheduler_callback)

    return callbacks


def exp_decay_lr_scheduler(epoch, lr, factor = 0.95):
    lr *= factor
    return lr


def lr_scheduler(initial_lr, final_lr, n_epochs):
    factor = (final_lr / initial_lr) ** (1 / n_epochs)
    return partial(exp_decay_lr_scheduler, factor=factor)


def get_metric(dataset, model_name):
    print(f"\nEvaluating model: {model_name}")
    model = load_model(model_name)

    y_true = []
    y_pred = []

    for batch_x, batch_y in dataset:  # You were using train_ds, but use val_ds to match your validation metrics
        preds = model.predict(batch_x, verbose=0)
        y_true.append(np.argmax(batch_y.numpy(), axis=1))
        y_pred.append(np.argmax(preds, axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print(classification_report(y_true, y_pred, digits=4))
    print("Accuracy     :", accuracy_score(y_true, y_pred))
    print("F1 (macro)   :", f1_score(y_true, y_pred, average="macro"))
    print("Precision    :", precision_score(y_true, y_pred, average="macro"))
    print("Recall       :", recall_score(y_true, y_pred, average="macro"))


