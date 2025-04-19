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
        lam = tf.random.uniform((), minval=0.8, maxval=0.95)
        lam = tf.clip_by_value(lam, 0.3, 0.7)

        indices = tf.random.shuffle(tf.range(batch_size))
        x2 = tf.gather(x, indices)
        y2 = tf.gather(y, indices)

        x_mix = lam * x + (1 - lam) * x2
        y_mix = lam * y + (1 - lam) * y2

        return {"images": x_mix, "labels": y_mix}

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
        # Basic configuration
        self.image_size = image_size
        self.seed = seed
        self.batch_size = batch_size

        # Dictionary of supported augmentation strategies
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
    

    def load_img(self, df, minority_class, family_encoder, shuffle=False, augment=None, cache=True, preprocessing_function=None, oversampling=False):
        # Adjust batch size when oversampling is enabled
        if oversampling:
            batch_size = round(self.batch_size * 0.75)
        else:
            batch_size = self.batch_size

        # Extract file paths and family labels from DataFrame
        file_paths = df['full_file_path'].values
        family_onehot = np.stack(df["family_onehot"].values)
        phylum_onehot = np.stack(df["phylum_onehot"].values)

        # Create datasets
        image_label_ds = tf.data.Dataset.from_tensor_slices((file_paths, family_onehot))
        phylum_ds = tf.data.Dataset.from_tensor_slices(phylum_onehot)

        # Define a function to load and resize images from file paths
        def _load_image(file_path, label):
            image = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            return image, label

        # Apply image loading logic
        image_label_ds = image_label_ds.map(self._load_image, num_parallel_calls=tf.data.AUTOTUNE)

        # Zip inputs together: ((image, family), phylum)
        dataset = tf.data.Dataset.zip((image_label_ds, phylum_ds))

        # Format the dataset into model input format: {"image_input", "phylum_input"} → label
        dataset = dataset.map(
            lambda x, phylum: ({"image_input": x[0], "phylum_input": phylum}, x[1]),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Shuffle here (after pairing everything), so everything stays in sync
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(file_paths), seed=self.seed, reshuffle_each_iteration=True)

        # Continue with batching and prefetching
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Save class names for later reference
        class_names = family_encoder.classes_.tolist()
        self.class_names = class_names

        # Normalization
        normalization_layer = tf.keras.layers.Rescaling(1./255)

        # The rest of your code (oversampling, augmentations, etc.) goes below...
        # (You don’t need to change those parts)

        # Optional oversampling logic to balance minority classes in each batch
        if oversampling:
            self.minority_class_indices = [
                self.class_names.index(c) for c in minority_class
            ]
            dataset = dataset.map(
                self._oversample_minority_fixed_size,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Optional external preprocessing function
        if preprocessing_function is not None:
            dataset = dataset.map(lambda x, y: (
                {"image_input": preprocessing_function(x["image_input"]), "phylum_input": x["phylum_input"]},
                y
            ))
        else:
            # Apply augmentations
            if augment:
                if augment not in self.augmentations:
                    raise ValueError(f"Unknown augmentation strategy: {augment}")

                aug_layer = self.augmentations[augment]

                # Handle mixup/cutmix (needs special treatment)
                if augment=="mixup":
                    dataset = dataset.map(lambda x, y: (
                        {"image_input": normalization_layer(x["image_input"]), "phylum_input": x["phylum_input"]}, y
                    ))
                    def apply_mixup_with_phylum(x, y):
                        mix_result = aug_layer({"images": x["image_input"], "labels": y})
                        return {
                            "image_input": mix_result["images"],
                            "phylum_input": x["phylum_input"]  # phylum stays the same
                        }, mix_result["labels"]

                    dataset = dataset.map(apply_mixup_with_phylum, num_parallel_calls=tf.data.AUTOTUNE)
                else:
                    # Apply standard augmentations to image_input only
                    def apply_aug(x, y):
                        return {
                            "image_input": aug_layer(x["image_input"]),
                            "phylum_input": x["phylum_input"]
                        }, y

                    dataset = dataset.map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE)
            
            if augment != "mixup":
                dataset = dataset.map(lambda x, y: (
                    {"image_input": normalization_layer(x["image_input"]), "phylum_input": x["phylum_input"]}, y
                ))

        # Optionally cache the dataset in memory and prefetch for speed
        if cache:
            dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

        # Return dataset and class names
        return dataset, class_names


class Experiment:
    def __init__(self, model, train_ds, val_ds, experiment_name, batch_size=32, image_size=(224, 224), log_path="experiment_log.csv", resume=True, save_model=True):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.experiment_name = experiment_name
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
                "f1_val_weighted": r(logs.get("f1_val_weighted")),
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
                # Sort by timestamp in filename and pick the latest
                checkpoint_files = sorted(checkpoint_files, key=lambda x: x.split("_")[-1].replace(".keras", ""))
                latest_checkpoint = checkpoint_files[-1]

                try:
                    log_df = pd.read_csv(self.log_path)
                    if "epoch" in log_df.columns and not log_df.empty:
                        initial_epoch = int(log_df[log_df["experiment_name"] == self.experiment_name]["epoch"].max())
                        print(f"Resuming training from epoch {initial_epoch}")
                        self.model = load_model(latest_checkpoint)
                        checkpoint_path = latest_checkpoint
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
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=all_callbacks,
            verbose=1
        )

        return history
    


