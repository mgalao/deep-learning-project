import os
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import image_dataset_from_directory
from keras.layers import RandAugment



class Preprocessor:
    def __init__(self, image_size=(224, 224), seed=42, batch_size=32):
        self.image_size = image_size
        self.seed = seed
        self.batch_size = batch_size

    def load_img(self, data_dir, value_range=(0,1), factor=0.5, num_ops=3, normalize=True, augment=False):
    
        dataset = image_dataset_from_directory(
            data_dir,
            image_size=self.image_size,
            label_mode="categorical",
            batch_size=self.batch_size,
            interpolation="bilinear",
            shuffle=True,
            seed=self.seed
        )

        class_names = dataset.class_names

        if normalize:
            normalization_layer = keras.layers.Rescaling(1./255)
            dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
         
        if augment:
            augmentation_layer = RandAugment(value_range=value_range, factor=factor, num_ops=num_ops, seed=self.seed)
            dataset = dataset.map(lambda x, y: (augmentation_layer(x), y))

        return dataset, class_names            