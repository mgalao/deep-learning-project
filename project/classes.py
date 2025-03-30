import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self, data, image_size=(224, 224), seed=42):
        self.df = data.copy()
        self.image_size = image_size
        self.seed = seed

        self.label_encoder = LabelEncoder()
        self.df['label'] = self.label_encoder.fit_transform(self.df['family'])
        self.label_map = dict(zip(self.label_encoder.transform(self.label_encoder.classes_), self.label_encoder.classes_))

    def _load_image(self, filepath, label, augment=False):
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)

        if augment:
            image = tf.image.random_flip_left_right(image)

        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def create_dataset(self, dataframe, augment=False, batch_size=32, shuffle=True):
        filepaths = [os.path.join(self.image_root, path) for path in dataframe['filepath']]
        labels = dataframe['label'].values

        dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(filepaths), seed=self.seed)

        dataset = dataset.map(lambda x, y: self._load_image(x, y, augment), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset
