import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
from tensorflow.keras import models
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from functools import partial
from pathlib import Path
import tensorflow as tf


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


def organize_split(image_base_path, base_output_dir, split_df, split_name):
    for _, row in split_df.iterrows():
        src = Path(row["full_file_path"])
        dst = base_output_dir / split_name / str(row["family"]) / src.name

        os.makedirs(dst.parent, exist_ok=True)
        try:
            shutil.copy(src, dst)
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
        lambda row: str(
            Path(f"../data/rare_species/{split_name}") / row["family"] / Path(row["full_file_path"]).name
        ),
        axis=1
    )
    return df

# Prepare the datasets (image and metadata) for model training
def build_ds_with_phylum(df, image_size=(224, 224), batch_size=32):
    image_paths = df["full_file_path"].values # get the full file paths
    # Convert the one-hot encoded columns to numpy arrays
    phylum_onehot = np.stack(df["phylum_onehot"].values)
    family_onehot = np.stack(df["family_onehot"].values)

    # Preprocess the images and metadata
    def process(image_path, phylum, family):
        image = tf.io.read_file(image_path) # read the image file
        image = tf.image.decode_jpeg(image, channels=3) # decode the image
        image = tf.image.resize(image, image_size) # resize the image
        image = tf.keras.applications.resnet50.preprocess_input(image) # preprocess the image
        return {"image_input": image, "phylum_input": phylum}, family

    # Create a TensorFlow dataset from the image paths and metadata
    ds = tf.data.Dataset.from_tensor_slices((image_paths, phylum_onehot, family_onehot)) # create a dataset from the image paths and metadata
    ds = ds.map(lambda x, y, z: process(x, y, z), num_parallel_calls=tf.data.AUTOTUNE) # map the process function to the dataset
    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE) # shuffle the dataset, batch it, and prefetch it for performance
    return ds


def build_ds_with_phylum_augmentation(
    df,
    image_size=(224, 224),
    batch_size=32,
    minority_class=None,
    normalization=True,
    oversampling=False,
    minority_weight=0.6,
    seed=42,
    train=False
):

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    # ---------------------
    # Basic processing step
    # ---------------------
    def process(image_path, phylum, family):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        return {"image_input": image, "phylum_input": phylum}, family

    # ---------------------
    # Augmentation pipeline
    # ---------------------
    aug_pipeline = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    def apply_augmenation(x, y):
        image = tf.image.rgb_to_grayscale(x["image_input"])
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = aug_pipeline(image)
        return {"image_input": image, "phylum_input": x["phylum_input"]}, y
    
    def apply_grey_scale(x, y):
        image = tf.image.rgb_to_grayscale(x["image_input"])
        image = tf.image.grayscale_to_rgb(image)
        return {"image_input": image, "phylum_input": x["phylum_input"]}, y

    # ---------------------
    # Dataset builder
    # ---------------------
    def make_raw_ds_from_df(sub_df):
        paths = sub_df["full_file_path"].values
        phylum = np.stack(sub_df["phylum_onehot"].values)
        family = np.stack(sub_df["family_onehot"].values)
        return tf.data.Dataset.from_tensor_slices((paths, phylum, family))

    # ---------------------
    # Main dataset construction
    # ---------------------
    if oversampling and minority_class is not None:
        minority_df = df[df["family"].isin(minority_class)]
        majority_df = df[~df["family"].isin(minority_class)]

        # Repeat raw datasets
        minority_raw_ds = make_raw_ds_from_df(minority_df).repeat()
        majority_raw_ds = make_raw_ds_from_df(majority_df)

        # Oversample dynamically using weighted sampling
        ds = tf.data.Dataset.sample_from_datasets(
            [majority_raw_ds, minority_raw_ds],
            weights=[1.0 - minority_weight, minority_weight],
            seed=seed
        )

        steps_per_epoch = int(len(majority_df) / ((1.0 - minority_weight) * batch_size))


    else:
        ds=make_raw_ds_from_df(df)
        steps_per_epoch=0     

    # Process, augment and normalize
    ds = ds.map(process, num_parallel_calls=2)
    if train:
        ds = ds.map(apply_augmenation, num_parallel_calls=2)
        ds = ds.shuffle(1000, seed=seed)

    else:
        ds = ds.map(apply_grey_scale, num_parallel_calls=2)
        

    if normalization:
        ds = ds.map(lambda x, y: (
            {"image_input": normalization_layer(x["image_input"]), "phylum_input": x["phylum_input"]}, y
        ), num_parallel_calls=2)

    # Shuffle, batch and prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)

    return ds, steps_per_epoch




'''

def build_ds_with_phylum_augmentation(
    df,
    image_size=(224, 224),
    batch_size=32,
    minority_class=None,
    normalization=True,
    oversampling=False,
    seed=42,
    train=False
):
    # --------------------
    # Oversampling before anything else
    # --------------------
    if oversampling and minority_class is not None:
        df_minority = df[df["family"].isin(minority_class)]
        df_augmented = pd.concat([df, df_minority], ignore_index=True)
    else:
        df_augmented = df

    image_paths = df_augmented["full_file_path"].values
    phylum_onehot = np.stack(df_augmented["phylum_onehot"].values)
    family_onehot = np.stack(df_augmented["family_onehot"].values)

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    # --------------------
    # Dataset creation
    # --------------------
    ds = tf.data.Dataset.from_tensor_slices((image_paths, phylum_onehot, family_onehot))

    def process(image_path, phylum, family):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = normalization_layer(image)
        return {"image_input": image, "phylum_input": phylum}, family

    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)

    # --------------------
    # Augmentation
    # --------------------
    aug_pipeline = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    def apply_augmentation(x, y):
        image = tf.image.rgb_to_grayscale(x["image_input"])
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = aug_pipeline(image)
        return {"image_input": image, "phylum_input": x["phylum_input"]}, y


    def apply_grayscale(x, y):
        image = tf.image.rgb_to_grayscale(x["image_input"])
        image = tf.image.grayscale_to_rgb(image)
        return {"image_input": image, "phylum_input": x["phylum_input"]}, y
    

    if train:
        ds = ds.map(apply_augmentation, num_parallel_calls=2)
    else:
        ds = ds.map(apply_grayscale, num_parallel_calls=2)

    # --------------------
    # Final shuffle, batch, prefetch
    # --------------------
    ds = ds.shuffle(1000, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    print('prepro done!!')

    return ds
'''

'''
def build_ds_with_phylum_augmentation(
    df,
    image_size=(224, 224),
    batch_size=32,
    minority_class=None,
    normalization=True,
    oversampling=False,
    seed=42
):
    image_paths = df["full_file_path"].values
    phylum_onehot = np.stack(df["phylum_onehot"].values)
    family_onehot = np.stack(df["family_onehot"].values)

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE) # shuffle the dataset, batch it, and prefetch it for performance

     # Apply oversampling before augmentation
    if oversampling and minority_class is not None:
        class_names = df["family"].unique().tolist()
        minority_indices = [class_names.index(name) for name in minority_class]

        def oversample_minority(inputs, labels):
            class_indices = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
            minority_indices_tf = tf.constant(minority_indices, dtype=tf.int32)

            is_minority = tf.reduce_any(
                tf.equal(tf.expand_dims(class_indices, axis=-1), minority_indices_tf), axis=-1
            )

            minority_inputs = tf.nest.map_structure(lambda t: tf.boolean_mask(t, is_minority), inputs)
            minority_labels = tf.boolean_mask(labels, is_minority)

            inputs_aug = tf.nest.map_structure(lambda a, b: tf.concat([a, b], axis=0), inputs, minority_inputs)
            labels_aug = tf.concat([labels, minority_labels], axis=0)

            indices = tf.range(tf.shape(labels_aug)[0])
            shuffled = tf.random.shuffle(indices)

            inputs_aug = tf.nest.map_structure(lambda t: tf.gather(t, shuffled), inputs_aug)
            labels_aug = tf.gather(labels_aug, shuffled)

            return inputs_aug, labels_aug

        ds = ds.map(oversample_minority, num_parallel_calls=tf.data.AUTOTUNE)

    def process(image_path, phylum, family):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        return {"image_input": image, "phylum_input": phylum}, family

    ds = tf.data.Dataset.from_tensor_slices((image_paths, phylum_onehot, family_onehot)) # create a dataset from the image paths and metadata
    ds = ds.map(lambda x, y, z: process(x, y, z), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE) # shuffle the dataset, batch it, and prefetch it for performance

    aug_pipeline = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    ])

    # Apply grayscale augmentation with rotations and flip
    def apply_grayscale(x, y):
        image = tf.image.rgb_to_grayscale(x["image_input"])
        image = tf.image.grayscale_to_rgb(image)  # keep 3 channels
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = aug_pipeline(image)
        return {"image_input": image, "phylum_input": x["phylum_input"]}, y
    
    ds = ds.map(apply_grayscale, num_parallel_calls=tf.data.AUTOTUNE)

    # Normalize after grayscale
    if normalization:
        ds = ds.map(lambda x, y: (
            {"image_input": normalization_layer(x["image_input"]), "phylum_input": x["phylum_input"]},
            y
        ), num_parallel_calls=tf.data.AUTOTUNE)

    #ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

'''



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


        if os.path.exists(metrics_file_path):
            try:
                with open(metrics_file_path, encoding="utf-8") as f:
                    f.read()
            except UnicodeDecodeError:
                print(f"⚠️ Log file {metrics_file_path} is not UTF-8 — deleting it to avoid crash.")
                os.remove(metrics_file_path)

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

