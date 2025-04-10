import os
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import image_dataset_from_directory
from keras.layers import RandomGrayscale
from keras_cv.layers import RandAugment, MixUp, CutMix  # Moved here correctly
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50


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
                self.random_grayscale_with_probability(prob=0.1), 
                keras.layers.RandomBrightness(0.2),
                keras.layers.RandomTranslation(0.1, 0.1),
            ]),

            "grayscale": keras.Sequential([
                # Forces the model to rely on shape and texture instead of color
                self.random_grayscale_with_probability(prob=1.0),
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

    def load_img(self, data_dir, label_mode="categorical", normalize=True, augment=None, shuffle=False, cache=True):
        """
        Loads and preprocesses the image dataset.

        Parameters:
        - data_dir: path to image folder (subfolders = class names)
        - label_mode: "categorical" = one-hot encoding
        - normalize: whether to apply 1./255 rescaling
        - augment: name of the augmentation strategy to apply
        - cache: whether to use caching and prefetching for performance
        """
        
        # Load dataset from directory using TensorFlow utility
        dataset = image_dataset_from_directory(
            data_dir,
            image_size=self.image_size,
            label_mode=label_mode,
            batch_size=self.batch_size,
            shuffle=shuffle,
            interpolation="bilinear"  # interpolation method defines how pixel values are estimated during this resizing. "bilinear" is mmooth and fast, balances quality and speed
        )

        class_names = dataset.class_names

        # Normalize pixel values to [0, 1]
        if normalize:
            dataset = dataset.map(lambda x, y: (self.base_normalize(x), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)

        # Apply selected augmentation strategy, if specified
        if augment:
            if augment not in self.augmentations:
                raise ValueError(f"Unknown augmentation strategy: {augment}")
            aug_layer = self.augmentations[augment]

            if isinstance(aug_layer, (MixUp, CutMix)):
                # Define a function that passes a single tuple to the layer
                def apply_mixup(x, y):
                    result = aug_layer({"images": x, "labels": y})
                    return result["images"], result["labels"]

                dataset = dataset.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)

            else:
                # Apply Sequential-like augmentation to images only
                dataset = dataset.map(lambda x, y: (aug_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Enable caching and prefetching for performance
        if cache:
            dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

        return dataset, class_names

    def random_grayscale_with_probability(self, prob=0.1, factor=1.0):
        """
        Applies grayscale with a given probability using tf.cond inside a Lambda layer.
        """
        grayscale_layer = RandomGrayscale(factor=factor)

        def apply(x):
            return tf.cond(
                tf.random.uniform([]) < prob,
                lambda: grayscale_layer(x),
                lambda: x
            )

        return keras.layers.Lambda(apply)


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
        def __init__(self, experiment_name, batch_size, image_size, log_path="experiment_log.csv"):
            super().__init__()
            self.experiment_name = experiment_name
            self.batch_size = batch_size
            self.image_size = image_size
            self.log_path = Path(log_path)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            history = self.model.history.history

            # Get the current timestamp in YYYY-MM-DD HH:MM:SS format
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Log metrics after every epoch
            results = {
                "experiment_name": self.experiment_name,
                "epoch": epoch + 1,  # Epoch starts from 0, but we want it to start from 1
                "train_accuracy": logs.get("accuracy", None),
                "val_accuracy": logs.get("val_accuracy", None),
                "train_loss": logs.get("loss", None),
                "val_loss": logs.get("val_loss", None),
                "top5_accuracy": logs.get("top_k_categorical_accuracy", None),
                "batch_size": self.batch_size,
                "image_size": f"{self.image_size[0]}x{self.image_size[1]}",
                "timestamp": timestamp  # Add the timestamp to the results
            }

            # Convert the results to DataFrame
            df = pd.DataFrame([results])

            # If the CSV file already exists, append to it with an index
            if self.log_path.exists():
                df.to_csv(self.log_path, mode='a', header=False, index=True)
            else:
                # If it doesn't exist, create it with the index column and the header
                df.to_csv(self.log_path, mode='w', index=True, header=True)

            print(f"✅ Logged results for epoch {epoch + 1} to {self.log_path} at {timestamp}")

    def run_experiment(self, epochs=10):
        # Set up dynamic checkpoint file name based on experiment and timestamp
        checkpoint_file = f"best_model_{self.experiment_name}_{self.timestamp}.h5"

        # Define callbacks
        experiment_logger = self.ExperimentLogger(
            experiment_name=self.experiment_name,
            batch_size=self.batch_size,
            image_size=self.image_size
        )

        callbacks = [
            experiment_logger,  # Log experiment results after each epoch
            EarlyStopping(patience=3, restore_best_weights=True),  # Early stopping callback
            ModelCheckpoint(checkpoint_file, save_best_only=True)  # Save best model based on validation performance
        ]

        # Train the model with callbacks
        history = self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=epochs, callbacks=callbacks)
        return history