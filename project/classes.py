import os
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import image_dataset_from_directory
from keras.layers import RandomGrayscale
from keras_cv.layers import RandAugment, MixUp, CutMix, RandomColorJitter 


class Preprocessor:
    def __init__(self, image_size=(224, 224), seed=42, batch_size=32):

        # Setting the image size and the batch size
        self.image_size = image_size
        self.seed = seed
        self.batch_size = batch_size

        # Dictionary of available augmentation strategies
        # Each entry is a name mapped to either a Sequential pipeline or a callable layer (like MixUp or CutMix)
        # We have different types of pipelines for augmentations to try with our dataset
        self.augmentations = {
            "none": keras.layers.Lambda(lambda x: x),  # No augmentation

            "light": keras.Sequential([
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
                # Acho que aqui podíamos adicionar outro tipo de augmentations com uma dada probabilidade
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
                # Mixes 2 images - alpha controls the parameter of the beta dsitribution 
                # where the coefficient for the linear combination is sampled
                # =0.2 is a default parameter, usually mixed images more similar to the one of the originals
                # Encorages generalization, reduces overfitting 
                alpha=0.2,
                seed=seed
            ),

            "cutmix": CutMix(
                # Cuts a part of one image and past it in another, and mixes the labels
                # Alpha is the parameter of the bata distribution that samples lambda, 
                # that defines the proportion of the image that is cut
                alpha=1.0,
                seed=seed
            ),
        }

    def load_img(self, data_dir, minority_class, label_mode="categorical", augment=None, cache=True, preprocessing_function=None, augment_prob=1.0, oversampling=False):
        
        """
        Parameters:
        - data_dir: path to image folder
        - minority_class: labels of the classes considered minority
        - label_mode: "categorical" = one-hot encoding
        - augment: name of the augmentation strategy to apply
        - cache: whether to use caching and prefetching for performance
        - preprocessing_function: if we are doing preprocessing for a pretrained model, we want to pass in this function 
        the preprocessing pipeline that is suitable for this model
        - augment_prob: float in [0,1] that controls probability of applying augmentation
        - oversampling: if we want to do oversampling
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

        # getting the class names - we know them because are the names of the folders
        class_names = dataset.class_names
        self.class_names = class_names

        # initializing the normalization layer
        normalization_layer = tf.keras.layers.Rescaling(1./255)

        # if we wanto to do oversampling of the minority classes:
        if oversampling:
            # minority_indices = [class_names.index(fam) for fam in minority_class]
            minority_indices = [self.class_names.index(name) for name in minority_class]

            # for images, labels in dataset:
            #    for i in range(len(labels)):
            #        label = tf.argmax(labels[i]).numpy()
            #        if label in minority_indices:
            #            print("aaa")

            # Function to oversample minority class samples
            def oversample_minority(image_batch, label_batch):

                # Get class indices from one-hot encoded labels
                class_indices = tf.cast(tf.argmax(label_batch, axis=-1), tf.int32)
                minority_indices_tf = tf.constant(minority_indices, dtype=tf.int32)

                # Boolean mask for minority class samples
                is_minority = tf.reduce_any(
                    tf.equal(tf.expand_dims(class_indices, axis=-1), minority_indices_tf), axis=-1
                )

                # Select minority samples
                minority_images = tf.boolean_mask(image_batch, is_minority)
                minority_labels = tf.boolean_mask(label_batch, is_minority)

                # Duplicate them (once, can duplicate more if needed)   
                image_batch_augmented = tf.concat([image_batch, minority_images], axis=0)
                label_batch_augmented = tf.concat([label_batch, minority_labels], axis=0)

                # Shuffle the augmented batch
                indices = tf.range(tf.shape(image_batch_augmented)[0])
                shuffled_indices = tf.random.shuffle(indices)
                image_batch_augmented = tf.gather(image_batch_augmented, shuffled_indices)
                label_batch_augmented = tf.gather(label_batch_augmented, shuffled_indices)


                return image_batch_augmented, label_batch_augmented

            # Apply the oversampling logic to the dataset
            dataset = dataset.map(oversample_minority, num_parallel_calls=tf.data.AUTOTUNE)
        

        if preprocessing_function is not None:
            dataset = dataset.map(lambda x, y: (preprocessing_function(x), y))

        if augment:

            # If we are applying augmentation methods that change color, we should do normalization
            # after applying the augmentation methods

            if augment in ["grayscale", "randaugment"]:
                aug_layer = self.augmentations[augment]
                # apply with probability
                if augment_prob < 1.0:
                    def augmentation_with_probability(aug_layer):
                        def apply(x):
                            return tf.cond(
                            tf.random.uniform([]) < augment_prob,
                            lambda: aug_layer(x),
                            lambda: x)
                        return keras.layers.Lambda(apply)
                    aug_layer = augmentation_with_probability(aug_layer)


                # applying the augmentation layer
                dataset = dataset.map(lambda x, y: (aug_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
                if preprocessing_function is None:
                    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
            
            # if the augmentation methods do not change the color of the images, 
            # than we do normalization before applying augmentation 
            else:
                if preprocessing_function is None:
                    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

                if augment not in self.augmentations:
                    raise ValueError(f"Unknown augmentation strategy: {augment}")
                
                aug_layer = self.augmentations[augment]

                # Handle MixUp and CutMix separately — they expect dict input and output 
                # and are the mixes between 2 images
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

                    # applying the augmentation layer
                    dataset = dataset.map(apply_mix, num_parallel_calls=tf.data.AUTOTUNE)
                
                # the other augmentations
                else:
                    # Apply with probability
                    if augment_prob < 1.0:

                        def augmentation_with_probability(aug_layer):
                            def apply(x):
                                return tf.cond(
                                tf.random.uniform([]) < augment_prob,
                                lambda: aug_layer(x),
                                lambda: x)
                            return keras.layers.Lambda(apply)

                        aug_layer = augmentation_with_probability(aug_layer)
                    
                    # applying the augmentation layer
                    dataset = dataset.map(lambda x, y: (aug_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            if preprocessing_function is None:
                dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

        # Enable caching and prefetching for performance
        if cache:
            dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

        return dataset, class_names

    
    # defining funtion for the grey scale layer
    def random_grayscale_layer(self, factor=1.0):
        return keras.layers.RandomGrayscale(factor=factor)