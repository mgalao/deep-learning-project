import os
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import image_dataset_from_directory
from keras.layers import RandomGrayscale
from keras_cv.layers import RandAugment, MixUp, CutMix  # Moved here correctly


class Preprocessor:
    def __init__(self, image_size=(224, 224), seed=42, batch_size=32):
        self.image_size = image_size
        self.seed = seed
        self.batch_size = batch_size

        # Rescale images to [0, 1] before feeding into the model
        self.base_normalize = keras.layers.Rescaling(1. / 255)

        # Dictionary of available augmentation strategies
        # Each entry is a name mapped to either a Sequential pipeline or a callable layer (like MixUp or CutMix)
        self.augmentations = {
            "none": keras.layers.Lambda(lambda x: x),  # No augmentation. Identity function

            "light": keras.Sequential([
                # Basic transformations: safe and useful for most datasets
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.05),
                keras.layers.RandomZoom(0.05),
            ]),

            "medium": keras.Sequential([
                # Adds light color and contrast variation
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomContrast(0.2),
            ]),

            "heavy": keras.Sequential([
                # Stronger augmentations for regularization
                keras.layers.RandomFlip("horizontal_and_vertical"),
                keras.layers.RandomRotation(0.15),
                keras.layers.RandomZoom(0.2),
                keras.layers.RandomContrast(0.3),
                RandomGrayscale(factor=1.0),  # applied always
                keras.layers.RandomBrightness(0.2),
                keras.layers.RandomTranslation(0.1, 0.1),
            ]),

            "grayscale": keras.Sequential([
                # Forces the model to rely on shape and texture instead of color
                RandomGrayscale(factor=1.0),
                keras.layers.RandomContrast(0.4),
            ]),

            "randaugment": RandAugment(
                # Applies N random powerful transformations per image
                value_range=(0, 255),  # Must match the range before normalization
                augmentations_per_image=2,
                magnitude=0.5,
                seed=seed
            ),

            "mixup": MixUp(
                # Combines two images and their labels using weighted averages
                alpha=0.2,
                seed=seed
            ),

            "cutmix": CutMix(
                # Replaces a region of one image with a patch from another image and mixes their labels
                alpha=1.0,
                seed=seed
            ),
        }

    def load_img(self, data_dir, label_mode="categorical", normalize=True, augment=None, cache=True, preprocessing_function=None, augment_prob=1.0):
        """
        Loads and preprocesses the image dataset.

        Parameters:
        - data_dir: path to image folder (subfolders = class names)
        - label_mode: "categorical" = one-hot encoding
        - normalize: whether to apply 1./255 rescaling
        - augment: name of the augmentation strategy to apply
        - cache: whether to use caching and prefetching for performance
        - augment_prob: float in [0,1] that controls probability of applying augmentation
        """

        # Load dataset from directory using TensorFlow utility
        dataset = image_dataset_from_directory(
            data_dir,
            image_size=self.image_size,
            label_mode=label_mode,
            batch_size=self.batch_size,
            shuffle=True,
            interpolation="bilinear"  # interpolation method defines how pixel values are estimated during this resizing. "bilinear" is smooth and fast, balances quality and speed
        )

        class_names = dataset.class_names

        # Normalize pixel values to [0, 1]
        if preprocessing_function is not None:
            dataset = dataset.map(lambda x, y: (preprocessing_function(x), y))
        elif normalize:
            normalization_layer = tf.keras.layers.Rescaling(1./255)
            dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

        # Apply selected augmentation strategy, if specified
        if augment:
            if augment not in self.augmentations:
                raise ValueError(f"Unknown augmentation strategy: {augment}")
            aug_layer = self.augmentations[augment]


            if augment_prob < 1.0:
                aug_layer = self.augmentation_with_probability(aug_layer, prob=augment_prob)

            # Apply the augmentation pipeline to the images
            dataset = dataset.map(lambda x, y: (aug_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Enable caching and prefetching for performance
        if cache:
            dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

        return dataset, class_names

    def augmentation_with_probability(self, aug_layer, prob):
        def apply(x):
            return tf.cond(
                tf.random.uniform([]) < prob,
                lambda: aug_layer(x),
                lambda: x
            )
        return keras.layers.Lambda(apply)
