import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pathlib import Path
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import image_dataset_from_directory
from keras.layers import RandomGrayscale
from keras_cv.layers import RandAugment, MixUp, CutMix, RandomColorJitter
from tensorflow.keras.models import load_model
import glob
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


class Preprocessor:
    def __init__(self, image_size=(224, 224), seed=42, batch_size=32):

        # Setting the image size and the batch size
        self.image_size = image_size
        self.seed = seed
        self.batch_size = round(batch_size * 0.75)

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

            # "grayscale": keras.Sequential([
            #     # Acho que aqui podíamos adicionar outro tipo de augmentations com uma dada probabilidade
            #     keras.layers.RandomGrayscale(factor=1.0),
            #     keras.layers.RandomContrast(0.4),
            # ]),

            "grayscale": keras.Sequential([
                keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),        # Convert to (H, W, 1)
                keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),        # Convert back to (H, W, 3)
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
            # def oversample_minority(image_batch, label_batch):

            #     # Get class indices from one-hot encoded labels
            #     class_indices = tf.cast(tf.argmax(label_batch, axis=-1), tf.int32)
            #     minority_indices_tf = tf.constant(minority_indices, dtype=tf.int32)

            #     # Boolean mask for minority class samples
            #     is_minority = tf.reduce_any(
            #         tf.equal(tf.expand_dims(class_indices, axis=-1), minority_indices_tf), axis=-1
            #     )

            #     # Select minority samples
            #     minority_images = tf.boolean_mask(image_batch, is_minority)
            #     minority_labels = tf.boolean_mask(label_batch, is_minority)

            #     # Duplicate them (once, can duplicate more if needed)   
            #     image_batch_augmented = tf.concat([image_batch, minority_images], axis=0)
            #     label_batch_augmented = tf.concat([label_batch, minority_labels], axis=0)

            #     # Shuffle the augmented batch
            #     indices = tf.range(tf.shape(image_batch_augmented)[0])
            #     shuffled_indices = tf.random.shuffle(indices)
            #     image_batch_augmented = tf.gather(image_batch_augmented, shuffled_indices)
            #     label_batch_augmented = tf.gather(label_batch_augmented, shuffled_indices)


            #     return image_batch_augmented, label_batch_augmented


            def oversample_minority_fixed_size(image_batch, label_batch):
                # Target batch size
                target_size = round(self.batch_size / 0.75)

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

                # Augment: add minority samples to original batch
                image_batch_augmented = tf.concat([image_batch, minority_images], axis=0)
                label_batch_augmented = tf.concat([label_batch, minority_labels], axis=0)

                # Shuffle
                indices = tf.range(tf.shape(image_batch_augmented)[0])
                shuffled_indices = tf.random.shuffle(indices)
                image_batch_augmented = tf.gather(image_batch_augmented, shuffled_indices)
                label_batch_augmented = tf.gather(label_batch_augmented, shuffled_indices)

                current_size = tf.shape(image_batch_augmented)[0]

                # If too large → truncate
                def truncate():
                    return image_batch_augmented[:target_size], label_batch_augmented[:target_size]

                # If too small → resample to reach size
                def pad():
                    needed = target_size - current_size
                    rand_indices = tf.random.uniform([needed], minval=0, maxval=current_size, dtype=tf.int32)
                    extra_images = tf.gather(image_batch_augmented, rand_indices)
                    extra_labels = tf.gather(label_batch_augmented, rand_indices)
                    return tf.concat([image_batch_augmented, extra_images], axis=0), tf.concat([label_batch_augmented, extra_labels], axis=0)

                # Choose truncate or pad based on current size
                image_batch_final, label_batch_final = tf.cond(
                    current_size > target_size,
                    true_fn=truncate,
                    false_fn=pad
                )

                return image_batch_final, label_batch_final


            # Apply the oversampling logic to the dataset
            dataset = dataset.map(oversample_minority_fixed_size, num_parallel_calls=tf.data.AUTOTUNE)
        

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


class Experiment:
    def __init__(self, model, train_ds, val_ds, experiment_name, batch_size=32, image_size=(224, 224), log_path="experiment_log.csv"):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.log_path = Path(log_path)

        # Generate a unique timestamp for this experiment
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    class ExperimentLogger(Callback):
        def __init__(self, experiment_name, batch_size, image_size, train_ds, val_ds, log_path="experiment_log.csv"):
            super().__init__()
            self.experiment_name = experiment_name
            self.batch_size = batch_size
            self.image_size = image_size
            self.log_path = Path(log_path)

            # Store the datasets
            self.train_ds = train_ds
            self.val_ds = val_ds

        def _compute_f1(self, ds):
            y_true = np.concatenate([y.numpy() for _, y in ds])
            y_pred_probs = self.model.predict(ds)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_true, axis=1)
            
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            
            return f1_macro, f1_weighted

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            # Compute F1 score for training and validation datasets
            f1_train_macro, f1_train_weighted = self._compute_f1(self.train_ds)
            f1_val_macro, f1_val_weighted = self._compute_f1(self.val_ds)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            def r(val): return round(val, 4) if isinstance(val, float) else val

            results = {
                "experiment_name": self.experiment_name,
                "epoch": epoch + 1,
                "train_accuracy": r(logs.get("accuracy")),
                "val_accuracy": r(logs.get("val_accuracy")),
                "train_loss": r(logs.get("loss")),
                "val_loss": r(logs.get("val_loss")),
                "f1_train_macro": r(f1_train_macro),
                "f1_val_macro": r(f1_val_macro),
                "f1_train_weighted": r(f1_train_weighted),
                "f1_val_weighted": r(f1_val_weighted),
                "top5_train_accuracy": r(logs.get("top_k_categorical_accuracy")),
                "top5_val_accuracy": r(logs.get("val_top_k_categorical_accuracy")),
                "batch_size": self.batch_size,
                "image_size": f"{self.image_size[0]}x{self.image_size[1]}",
                "timestamp": timestamp
            }

            df = pd.DataFrame([results])

            if self.log_path.exists():
                df.to_csv(self.log_path, mode='a', header=False, index=True)
            else:
                df.to_csv(self.log_path, mode='w', index=True, header=True)

    def run_experiment(self, epochs=10, callbacks=None):
        # Find the most recent checkpoint for this experiment
        pattern = f"../project/best_model_{self.experiment_name}_*.keras"
        checkpoint_files = sorted(glob.glob(pattern))

        initial_epoch = 0

        if checkpoint_files and os.path.exists(self.log_path):
            latest_checkpoint = checkpoint_files[-1]  # last one (assuming sorted by name/timestamp)
            try:
                log_df = pd.read_csv(self.log_path)
                if "epoch" in log_df.columns and not log_df.empty:
                    initial_epoch = int(log_df["epoch"].max()) + 1
                    print(f"Resuming training from epoch {initial_epoch}")
                    self.model = load_model(latest_checkpoint)
            except Exception as e:
                print(f"Could not read log file or load model: {e}")
                print("Starting from scratch.")
        else:
            print("No checkpoint found, starting from scratch.")

        # Save to a new file (or overwrite last)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        new_checkpoint_path = f"project/best_model_{self.experiment_name}_{self.timestamp}.keras"

        # Define default callbacks
        default_callbacks = [
            self.ExperimentLogger(
                experiment_name=self.experiment_name,
                batch_size=self.batch_size,
                image_size=self.image_size,
                train_ds=self.train_ds,
                val_ds=self.val_ds
            ),
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint(new_checkpoint_path, save_best_only=True)
        ]

        if callbacks:
            all_callbacks = default_callbacks + callbacks
        else:
            all_callbacks = default_callbacks

        # Train
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=all_callbacks,
            verbose=1
        )

        return history