import os
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import image_dataset_from_directory
from keras.layers import RandomGrayscale
from keras_cv.layers import RandAugment, MixUp, CutMix, RandomColorJitter 


class Preprocessor:
    def _init_(self, image_size=(224, 224), seed=42, batch_size=32):
        self.image_size = image_size
        self.seed = seed
        self.batch_size = batch_size

        # Dictionary of available augmentation strategies
        # Each entry is a name mapped to either a Sequential pipeline or a callable layer (like MixUp or CutMix)
        self.augmentations = {
            "none": keras.layers.Lambda(lambda x: x),  # No augmentation. Identity function

            "light": keras.Sequential([
                # Basic and safe transformations
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.05),
                keras.layers.RandomZoom(0.05),
                keras.layers.RandomContrast(0.1),
                RandomColorJitter(  # Combines brightness, contrast, saturation, hue
                    value_range=(0, 1), brightness_factor=0.05, contrast_factor=0.05, saturation_factor=0.05, hue_factor=0.01
                ),
                keras.layers.RandomSharpness(factor=0.2),
            ]),

            "medium": keras.Sequential([
                # Adds more geometric and color variation
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomTranslation(0.05, 0.05),
                RandomColorJitter(
                    value_range=(0, 1), brightness_factor=0.1, contrast_factor=0.15, saturation_factor=0.2, hue_factor=0.02
                ),
                keras.layers.RandomSharpness(factor=0.3),
            ]),

            "heavy": keras.Sequential([
                # Strong augmentations for robustness and generalization
                keras.layers.RandomFlip("horizontal_and_vertical"),
                keras.layers.RandomRotation(0.15),
                keras.layers.RandomZoom(0.2),
                keras.layers.RandomTranslation(0.1, 0.1),
                RandomColorJitter(
                    value_range=(0, 1), brightness_factor=0.2, contrast_factor=0.3, saturation_factor=0.3, hue_factor=0.05
                ),
                keras.layers.RandomSharpness(factor=0.4),
            ]),

            "grayscale": keras.Sequential([
                # Focus on shape/patterns instead of color
                self.random_grayscale_layer(1.0),
                keras.layers.RandomContrast(0.4),
            ]),

            "randaugment": RandAugment(
                value_range=(0, 255),
                augmentations_per_image=2,
                magnitude=0.5,
                seed=seed
            ),

            "mixup": MixUp(
                alpha=0.2,
                seed=seed
            ),

            "cutmix": CutMix(
                alpha=1.0,
                seed=seed
            ),
        }

    def load_img(self, data_dir, minority_class, label_mode="categorical", augment=None, cache=True, preprocessing_function=None, augment_prob=1.0):
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

        minority_indices = [class_names.index(fam) for fam in minority_class]

        class_names = dataset.class_names
        normalization_layer = tf.keras.layers.Rescaling(1./255)

        if preprocessing_function is not None:
            dataset = dataset.map(lambda x, y: (preprocessing_function(x), y))

        if augment:
            if augment in ["grayscale", "randaugment"]:
                aug_layer = self.augmentations[augment]
                if augment_prob < 1.0:
                    def augmentation_with_probability(aug_layer):
                        def apply(x):
                            return tf.cond(
                            tf.random.uniform([]) < augment_prob,
                            lambda: aug_layer(x),
                            lambda: x)
                        return keras.layers.Lambda(apply)
                    aug_layer = augmentation_with_probability(aug_layer)
                dataset = dataset.map(lambda x, y: (aug_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
                if preprocessing_function is None:
                    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
            else:
                if preprocessing_function is None:
                    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

                if augment not in self.augmentations:
                    raise ValueError(f"Unknown augmentation strategy: {augment}")
                
                aug_layer = self.augmentations[augment]

                # Handle MixUp and CutMix separately — they expect dict input and output
                if isinstance(aug_layer, (MixUp, CutMix)):
                    # Apply with probability
                    if augment_prob < 1.0:
                        def apply_mix(x, y):
                            def apply_aug():
                                result = aug_layer({"images": x, "labels": y})
                                return result["images"], result["labels"]
                            
                            def skip_aug():
                                return x, y

                            return tf.cond(
                                tf.random.uniform([]) < augment_prob,
                                true_fn=apply_aug,
                                false_fn=skip_aug
                            )
                    else:
                        def apply_mix(x, y):
                            result = aug_layer({"images": x, "labels": y})
                            return result["images"], result["labels"]

                    dataset = dataset.map(apply_mix, num_parallel_calls=tf.data.AUTOTUNE)
                else:
                    # Apply with probability if needed
                    if augment_prob < 1.0:

                        def augmentation_with_probability(aug_layer):
                            def apply(x):
                                return tf.cond(
                                tf.random.uniform([]) < augment_prob,
                                lambda: aug_layer(x),
                                lambda: x)
                            return keras.layers.Lambda(apply)

                        aug_layer = augmentation_with_probability(aug_layer)

                    # Standard augmentation map
                    dataset = dataset.map(lambda x, y: (aug_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            if preprocessing_function is None:
                dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

        # Enable caching and prefetching for performance
        if cache:
            dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

        return dataset, class_names

    
    def random_grayscale_layer(self, factor=1.0):
        return keras.layers.RandomGrayscale(factor=factor)
