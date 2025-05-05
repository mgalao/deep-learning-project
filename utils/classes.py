import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import image_dataset_from_directory
from keras_cv.layers import RandAugment, MixUp, CutMix, RandomColorJitter
from tensorflow.keras.models import load_model
import glob
import math
from keras.callbacks import Callback, ModelCheckpoint
import math



class ColorJitter(keras.layers.Layer):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, seed=None):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.seed = seed

    def jitter_fn(self, image):
        image = tf.image.random_brightness(image, max_delta=self.brightness)
        image = tf.image.random_contrast(image, lower=1 - self.contrast, upper=1 + self.contrast)
        image = tf.image.random_saturation(image, lower=1 - self.saturation, upper=1 + self.saturation)
        image = tf.image.random_hue(image, max_delta=self.hue)
        return image

    def call(self, x, training=True):
        if training:
            x = tf.map_fn(self.jitter_fn, x)
        return x

class CustomMixUp(tf.keras.layers.Layer):
    def __init__(self, alpha=0.4, seed=None):
        super().__init__()
        self.alpha = alpha
        self.seed = seed

    def call(self, inputs):
        x, y = inputs["images"], inputs["labels"]
        batch_size = tf.shape(x)[0]
        lam = np.random.beta(self.alpha, self.alpha)
        lam = tf.clip_by_value(lam, 0.2, 0.8)

        indices = tf.random.shuffle(tf.range(batch_size))
        x2 = tf.gather(x, indices)
        y2 = tf.gather(y, indices)

        x_mix = lam * x + (1 - lam) * x2
        y_mix = lam * y + (1 - lam) * y2

        return {
            "images": x_mix,
            "labels": y_mix,
            "lambda": lam,
            "indices": indices
        }
    
class Preprocessor:
    def __init__(self, image_size=(224, 224), seed=42, batch_size=32):

        # Setting the image size and the batch size
        self.image_size = image_size
        self.seed = seed
        self.batch_size = batch_size
        self.minority_indices = None 

        # Dictionary of available augmentation strategies
        # Each entry is a name mapped to either a Sequential pipeline or a callable layer (like MixUp or CutMix)
        # We have different types of pipelines for augmentations to try with our dataset
        self.augmentation_strategies = {
            "none": keras.layers.Lambda(lambda x: x),  # No augmentation

            "light": keras.Sequential([
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.05),
                keras.layers.RandomZoom(0.05),
                keras.layers.RandomContrast(0.1),
                ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
                keras.layers.RandomSharpness(factor=0.2),
            ]),

            "medium": keras.Sequential([
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomTranslation(0.05, 0.05),
                ColorJitter(brightness=0.1, contrast=0.15, saturation=0.2, hue=0.02),
                keras.layers.RandomSharpness(factor=0.3),
            ]),

            "heavy": keras.Sequential([
                keras.layers.RandomFlip("horizontal_and_vertical"),
                keras.layers.RandomRotation(0.15),
                keras.layers.RandomZoom(0.2),
                keras.layers.RandomTranslation(0.1, 0.1),
                ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.05),
                keras.layers.RandomSharpness(factor=0.4),
            ]),

            "grayscale": keras.Sequential([
                keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),        # Convert to (H, W, 1)
                keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),        # Convert back to (H, W, 3)
                keras.layers.RandomContrast(0.4),
            ]),

            "grayscale_plus": keras.Sequential([
                keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),        # Convert to (H, W, 1)
                keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),        # Convert back to (H, W, 3)
                keras.layers.RandomContrast(0.4),
                keras.layers.RandomSharpness(0.3),
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
            ]), 

            "randaugment": RandAugment(
                value_range=(0, 255),
                augmentations_per_image=2,
                magnitude=0.5,
                seed=seed
            ),

            "mixup": CustomMixUp(alpha=0.2),

            "geometric_transformations": keras.Sequential([
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomFlip("vertical"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomTranslation(0.1, 0.1),
            ]), 

            "color_lightening": keras.Sequential([
                ColorJitter(brightness=0.1, contrast=0.15, saturation=0.2, hue=0.02)
            ])}
        

    def _oversample_minority_fixed_size(self, images, labels):
        """
        Oversample minority-class examples to reach a fixed batch size.
        - images: tensor of shape [batch, H, W, C]
        - labels: one-hot tensor of shape [batch, num_classes]
        Returns:
            images_out, labels_out: both of shape [batch_size, ...]
        """
        target_size = tf.constant(self.batch_size, dtype=tf.int32)
        label_ids = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        minority_ids = tf.constant(self.minority_class_indices, dtype=tf.int32)

        is_minority = tf.reduce_any(
            tf.equal(tf.expand_dims(label_ids, axis=-1), minority_ids),
            axis=-1
        )

        minority_images = tf.boolean_mask(images, is_minority)
        minority_labels = tf.boolean_mask(labels, is_minority)
        majority_images = tf.boolean_mask(images, tf.logical_not(is_minority))
        majority_labels = tf.boolean_mask(labels, tf.logical_not(is_minority))

        augmented_images = tf.concat([images, minority_images], axis=0)
        augmented_labels = tf.concat([labels, minority_labels], axis=0)

        total_size = tf.shape(augmented_images)[0]
        indices = tf.range(total_size)
        shuffled = tf.random.shuffle(indices)
        augmented_images = tf.gather(augmented_images, shuffled)
        augmented_labels = tf.gather(augmented_labels, shuffled)

        def _truncate():
            idx = tf.random.shuffle(tf.range(tf.shape(augmented_images)[0]))[:target_size]
            return (tf.gather(augmented_images, idx),
                    tf.gather(augmented_labels, idx))

        def _pad():
            current = tf.shape(augmented_images)[0]
            needed = target_size - current
            maj_count = tf.shape(majority_images)[0]

            def _sample_idxs(count):
                # 1) shuffle [0,1,...,count-1]
                shuffled = tf.random.shuffle(tf.range(count))
                # 2) build an index vector [0,1,2,...,needed-1]
                full = tf.range(needed)
                # 3) map through modulo to “cycle” the shuffled list
                return tf.gather(shuffled, full % count)

            def _pad_from_majority():
                sel = _sample_idxs(maj_count)
                extras_img = tf.gather(majority_images, sel)
                extras_lbl = tf.gather(majority_labels, sel)
                return (
                    tf.concat([augmented_images, extras_img], axis=0),
                    tf.concat([augmented_labels, extras_lbl], axis=0),
                )

            def _pad_recycle():
                sel = _sample_idxs(current)
                extras_img = tf.gather(augmented_images, sel)
                extras_lbl = tf.gather(augmented_labels, sel)
                return (
                    tf.concat([augmented_images, extras_img], axis=0),
                    tf.concat([augmented_labels, extras_lbl], axis=0),
                )

            out_images, out_labels = tf.cond(
                maj_count > 0,
                _pad_from_majority,
                _pad_recycle
            )

            # final full‐batch shuffle
            final_idx = tf.random.shuffle(tf.range(tf.shape(out_images)[0]))
            return (
                tf.gather(out_images, final_idx),
                tf.gather(out_labels, final_idx),
            )

        # choose whether to truncate or pad to reach exactly batch_size
        image_batch_final, label_batch_final = tf.cond(
            tf.shape(augmented_images)[0] > target_size,
            true_fn=_truncate,
            false_fn=_pad
        )

        return image_batch_final, label_batch_final

    def load_img(
        self,
        data_dir,
        minority_class=None,
        label_mode="categorical",
        augment=None,
        cache=True,
        preprocessing_function=None,
        oversampling=False,
        shuffle=False
    ):
        """
        Load images from `data_dir`, apply optional oversampling, normalization, augmentations, caching.
        Returns: (tf.data.Dataset, class_names list)
        """
        if oversampling:
            load_batch = int(math.floor(self.batch_size * 0.75))
        else:
            load_batch = self.batch_size

        dataset = image_dataset_from_directory(
            data_dir,
            image_size=self.image_size,
            label_mode=label_mode,
            batch_size=load_batch,
            shuffle=shuffle,
            interpolation="bilinear"
        )

        norm_layer = tf.keras.layers.Rescaling(1.0 / 255)

        self.class_names = dataset.class_names

        if oversampling:
            self.minority_class_indices = [
                self.class_names.index(c) for c in minority_class
            ]
            dataset = dataset.map(
                self._oversample_minority_fixed_size,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        if preprocessing_function is not None:
            dataset = dataset.map(
                lambda x, y: (preprocessing_function(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            # STEP 1: Apply augmentation first
            if augment:
                aug_fn = self.augmentation_strategies.get(augment)
                if aug_fn is None:
                    raise ValueError(f"Unknown augmentation strategy: {augment}")

                if augment == "mixup":
                    dataset = dataset.map(
                        lambda x, y: (norm_layer(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
                    def apply_mix(x, y):
                        result = aug_fn({"images": x, "labels": y})
                        return result["images"], result["labels"]
                    
                    dataset = dataset.map(apply_mix, num_parallel_calls=tf.data.AUTOTUNE)
                else:
                    dataset = dataset.map(
                        lambda x, y: (aug_fn(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )

            if augment != "mixup":
                dataset = dataset.map(
                    lambda x, y: (norm_layer(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )

        if cache:
            dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

        return dataset, self.class_names


class Preprocessor_with_phylum:
    def __init__(self, image_size=(224, 224), seed=42, batch_size=32):
        # Store basic configuration
        self.image_size = image_size
        self.seed = seed
        self.batch_size = batch_size

        # Dictionary of augmentation strategies
        self.augmentations = {
            "none": keras.layers.Lambda(lambda x: x),  # No augmentation

            "light": keras.Sequential([
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.05),
                keras.layers.RandomZoom(0.05),
                keras.layers.RandomContrast(0.1),
                ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
                keras.layers.RandomSharpness(factor=0.2),
            ]),

            "medium": keras.Sequential([
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomTranslation(0.05, 0.05),
                ColorJitter(brightness=0.1, contrast=0.15, saturation=0.2, hue=0.02),
                keras.layers.RandomSharpness(factor=0.3),
            ]),

            "heavy": keras.Sequential([
                keras.layers.RandomFlip("horizontal_and_vertical"),
                keras.layers.RandomRotation(0.15),
                keras.layers.RandomZoom(0.2),
                keras.layers.RandomTranslation(0.1, 0.1),
                ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.05),
                keras.layers.RandomSharpness(factor=0.4),
            ]),

            "grayscale": keras.Sequential([
                keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),
                keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),
                keras.layers.RandomContrast(0.4),
            ]),

            "grayscale_plus": keras.Sequential([
                keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),
                keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),
                keras.layers.RandomContrast(0.4),
                keras.layers.RandomSharpness(0.3),
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
            ]),

            "randaugment": RandAugment(
                value_range=(0, 255),
                augmentations_per_image=2,
                magnitude=0.5,
                seed=seed
            ),

            "mixup": CustomMixUp(alpha=0.2),  # Special case, mixes images and labels

            "geometric_transformations": keras.Sequential([
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomFlip("vertical"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomTranslation(0.1, 0.1),
            ]),

            "color_lightening": keras.Sequential([
                ColorJitter(brightness=0.1, contrast=0.15, saturation=0.2, hue=0.02)
            ])
        }

    def _oversample_minority_fixed_size(self, inputs, labels):
        """
        Oversamples the minority class in a batch to help balance training.
        """
        images = inputs["image_input"]
        phylum = inputs["phylum_input"]

        # Determine batch size and extract class labels
        target_size = tf.constant(self.batch_size, dtype=tf.int32)
        label_ids = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        minority_ids = tf.constant(self.minority_class_indices, dtype=tf.int32)

        # Identify minority samples
        is_minority = tf.reduce_any(
            tf.equal(tf.expand_dims(label_ids, axis=-1), minority_ids), axis=-1
        )

        # Split minority and majority data
        minority_images = tf.boolean_mask(images, is_minority)
        minority_labels = tf.boolean_mask(labels, is_minority)
        minority_phylum = tf.boolean_mask(phylum, is_minority)

        majority_images = tf.boolean_mask(images, tf.logical_not(is_minority))
        majority_labels = tf.boolean_mask(labels, tf.logical_not(is_minority))
        majority_phylum = tf.boolean_mask(phylum, tf.logical_not(is_minority))

        # Concatenate original batch with extra minority samples
        augmented_images = tf.concat([images, minority_images], axis=0)
        augmented_labels = tf.concat([labels, minority_labels], axis=0)
        augmented_phylum = tf.concat([phylum, minority_phylum], axis=0)

        # Shuffle batch
        total_size = tf.shape(augmented_images)[0]
        indices = tf.random.shuffle(tf.range(total_size))
        augmented_images = tf.gather(augmented_images, indices)
        augmented_labels = tf.gather(augmented_labels, indices)
        augmented_phylum = tf.gather(augmented_phylum, indices)

        # Define truncate and pad paths
        def _truncate():
            idx = tf.random.shuffle(tf.range(tf.shape(augmented_images)[0]))[:target_size]
            return {
                "image_input": tf.gather(augmented_images, idx),
                "phylum_input": tf.gather(augmented_phylum, idx)
            }, tf.gather(augmented_labels, idx)

        def _pad():
            current = tf.shape(augmented_images)[0]
            needed = target_size - current
            maj_count = tf.shape(majority_images)[0]

            # Sample indices helper
            def _sample_idxs(count):
                full = tf.range(needed)
                return tf.gather(tf.random.shuffle(tf.range(count)), full % count)

            # Pad from majority class if possible
            def _pad_from_majority():
                sel = _sample_idxs(maj_count)
                return {
                    "image_input": tf.concat([augmented_images, tf.gather(majority_images, sel)], axis=0),
                    "phylum_input": tf.concat([augmented_phylum, tf.gather(majority_phylum, sel)], axis=0)
                }, tf.concat([augmented_labels, tf.gather(majority_labels, sel)], axis=0)

            # Otherwise recycle from current batch
            def _pad_recycle():
                sel = _sample_idxs(current)
                return {
                    "image_input": tf.concat([augmented_images, tf.gather(augmented_images, sel)], axis=0),
                    "phylum_input": tf.concat([augmented_phylum, tf.gather(augmented_phylum, sel)], axis=0)
                }, tf.concat([augmented_labels, tf.gather(augmented_labels, sel)], axis=0)

            return tf.cond(maj_count > 0, _pad_from_majority, _pad_recycle)

        # Decide whether to pad or truncate the batch
        return tf.cond(tf.shape(augmented_images)[0] > target_size, _truncate, _pad)

    def load_img(self, df, minority_class, family_encoder, phylum_encoder, shuffle=False, augment=None, cache=True, preprocessing_function=None, oversampling=False):
        """
        Loads images and labels, applies optional augmentation, normalization, and oversampling.
        """
        # Store encoders and corresponding class labels
        self.family_class_names = family_encoder.classes_.tolist()
        self.phylum_class_names = phylum_encoder.classes_.tolist()

        # Adjust batch size for oversampling mode
        batch_size = round(self.batch_size * 0.75) if oversampling else self.batch_size

        # Load raw data from DataFrame
        file_paths = df['full_file_path'].values
        family_onehot = np.stack(df["family_onehot"].values)
        phylum_onehot = np.stack(df["phylum_onehot"].values)

        # Build initial dataset
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, family_onehot, phylum_onehot))

        # Read and preprocess each image
        def _load_image(path, label, phylum):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            return {"image_input": image, "phylum_input": phylum}, label

        dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle dataset if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(file_paths), seed=self.seed, reshuffle_each_iteration=True)
            dataset = dataset.repeat() # Repeat only if shuffle is True -> training mode

        # Batch the dataset
        dataset = dataset.batch(batch_size)

        # Apply oversampling logic if enabled
        if oversampling:
            self.minority_class_indices = [
                self.family_class_names.index(c)
                for c in minority_class
                if c in self.family_class_names
            ]
            
            dataset = dataset.map(self._oversample_minority_fixed_size, num_parallel_calls=tf.data.AUTOTUNE)
            
        # Apply preprocessing function (if any)
        if preprocessing_function:
            dataset = dataset.map(lambda x, y: (
                {"image_input": preprocessing_function(x["image_input"]), "phylum_input": x["phylum_input"]},
                y
            ))
        else:
            # Apply augmentations if specified
            if augment:
                if augment not in self.augmentations:
                    raise ValueError(f"Unknown augmentation strategy: {augment}")
                aug_layer = self.augmentations[augment]

                if augment == "mixup":
                    # MixUp requires normalized input before augmentation
                    normalization_layer = tf.keras.layers.Rescaling(1./255)
                    dataset = dataset.map(lambda x, y: (
                        {"image_input": normalization_layer(x["image_input"]), "phylum_input": x["phylum_input"]}, y
                    ))

                    def apply_mixup(x, y):
                        mixed = aug_layer({
                            "images": x["image_input"],
                            "labels": y
                        })

                        lam = mixed["lambda"]
                        indices = mixed["indices"]

                        phylum = x["phylum_input"]
                        phylum_mix = lam * phylum + (1 - lam) * tf.gather(phylum, indices)

                        return {
                            "image_input": mixed["images"],
                            "phylum_input": phylum_mix
                        }, mixed["labels"]

                    dataset = dataset.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)

                else:
                    # Standard augmentation and normalization
                    dataset = dataset.map(lambda x, y: (
                        {
                            "image_input": aug_layer(x["image_input"]),
                            "phylum_input": x["phylum_input"]
                        }, y
                    ), num_parallel_calls=tf.data.AUTOTUNE)

        # Cache dataset for speed
        if cache:
            dataset = dataset.cache()

        # Prefetch right before final output
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset, self.family_class_names, self.phylum_class_names

class Experiment:
    def __init__(self, model, train_ds, val_ds, experiment_name, steps_per_epoch=None, batch_size=32, image_size=(224, 224), log_path="experiment_log.csv", resume=False, save_model=True):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.experiment_name = experiment_name
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.image_size = image_size
        self.log_path = Path(log_path)
        self.resume = resume
        self.save_model = save_model

        # Generate a timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Determine experiment ID
        if self.log_path.exists():
            log_df = pd.read_csv(self.log_path)
            if not resume:
                self.experiment_id = log_df["id"].max() + 1 if not log_df.empty else 0
            else:
                if experiment_name in log_df["experiment_name"].unique():
                    self.experiment_id = log_df[log_df["experiment_name"] == experiment_name]["id"].iloc[0]
                else:
                    self.experiment_id = log_df["id"].max() + 1 if not log_df.empty else 0
        else:
            self.experiment_id = 0

    class ExperimentLogger(Callback):
        def __init__(self, experiment_name, experiment_id, train_ds, val_ds, log_path="experiment_log.csv"):
            super().__init__()
            self.experiment_name = experiment_name
            self.experiment_id = experiment_id
            self.log_path = Path(log_path)
            self.train_ds = train_ds
            self.val_ds = val_ds

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            def r(val): return round(val, 4) if isinstance(val, float) else val
            
            results = {
                "id": self.experiment_id,
                "experiment_name": self.experiment_name,
                "epoch": epoch + 1,
                "train_accuracy": r(logs.get("accuracy")),
                "val_accuracy": r(logs.get("val_accuracy")),
                "train_loss": r(logs.get("loss")),
                "val_loss": r(logs.get("val_loss")),
                "f1_train_macro": r(logs.get("f1_macro")),
                "f1_val_macro": r(logs.get("val_f1_macro")),
                "f1_train_weighted": r(logs.get("f1_weighted")),
                "f1_val_weighted": r(logs.get("val_f1_weighted")),
                "top5_train_accuracy": r(logs.get("top5_accuracy")),
                "top5_val_accuracy": r(logs.get("val_top5_accuracy")),
                "timestamp": timestamp
            }

            df = pd.DataFrame([results])
            if self.log_path.exists():
                df.to_csv(self.log_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.log_path, mode='w', header=True, index=False)

    def run_experiment(self, epochs=10, callbacks=None):
        initial_epoch = 0
        checkpoint_path = None

        if self.resume:
            # Look for most recent checkpoint for this experiment
            pattern = f"../project/model_{self.experiment_name}_*.keras"
            checkpoint_files = sorted(glob.glob(pattern))

            if checkpoint_files:
                # Sort checkpoint files by timestamp extracted from filename
                checkpoint_files = sorted(
                    checkpoint_files,
                    key=lambda x: x.split("_")[-1].replace(".keras", "")
                )
                latest_checkpoint = checkpoint_files[-1]

                try:
                    log_df = pd.read_csv(self.log_path)

                    if "epoch" in log_df.columns and not log_df.empty:
                        # Filter logs for this experiment
                        experiment_logs = log_df[log_df["experiment_name"] == self.experiment_name].copy()

                        # Convert to datetime for accurate sorting
                        experiment_logs["timestamp"] = pd.to_datetime(experiment_logs["timestamp"])

                        if not experiment_logs.empty:
                            # Get latest run by timestamp
                            latest_entry = experiment_logs.sort_values("timestamp").iloc[-1]
                            initial_epoch = int(latest_entry["epoch"])
                            self.timestamp = latest_entry["timestamp"].strftime("%Y%m%d-%H%M%S")
                            self.experiment_id = int(latest_entry["id"])

                            print(f"Resuming training from epoch {initial_epoch} (timestamp {self.timestamp})")
                            self.model = load_model(latest_checkpoint)
                            checkpoint_path = latest_checkpoint
                        else:
                            print("No matching log entries found. Starting from scratch.")
                    else:
                        print("Log is empty or missing 'epoch' column. Starting from scratch.")
                except Exception as e:
                    print(f"Could not read log or load model: {e}")
                    print("Starting from scratch.")
            else:
                print("No checkpoint found, starting from scratch.")

        if not checkpoint_path:
            self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_path = f"../project/model_{self.experiment_name}_{self.timestamp}.keras"

        # Callbacks
        default_callbacks = [
            self.ExperimentLogger(
                experiment_name=self.experiment_name,
                experiment_id=self.experiment_id,
                train_ds=self.train_ds,
                val_ds=self.val_ds,
                log_path=self.log_path
            )
        ]

        if self.save_model:
            default_callbacks.append(
                ModelCheckpoint(checkpoint_path, save_best_only=True)
            )

        if callbacks:
            all_callbacks = default_callbacks + callbacks
        else:
            all_callbacks = default_callbacks

        # Training
        if self.steps_per_epoch: # If using phylum data, add steps_per_epoch
            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                steps_per_epoch=self.steps_per_epoch,
                epochs=epochs,
                initial_epoch=initial_epoch,
                callbacks=all_callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=epochs,
                initial_epoch=initial_epoch,
                callbacks=all_callbacks,
                verbose=1
            )

        return history
    

class ProgressiveUnfreeze(tf.keras.callbacks.Callback):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.layer_index = len(self.base_model.layers) - 1  

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0 and self.layer_index >= 0:  
            while self.layer_index >= 0:
                layer = self.base_model.layers[self.layer_index]
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
                    print(f"Epoch {epoch}: Layer now unfrozen {self.layer_index} ({layer.name})")
                    self.layer_index -= 1  
                    break
                else:
                    self.layer_index -= 1