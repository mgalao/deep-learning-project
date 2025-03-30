import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


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
        src = image_base_path / row["full_file_path"]
        dst = base_output_dir / split_name / str(row["family"]) / src.name

        os.makedirs(dst.parent, exist_ok=True)
        try:
            shutil.move(src, dst)
        except FileNotFoundError:
            print(f"Not found: {src}")