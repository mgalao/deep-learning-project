import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import models
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from functools import partial
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model


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


def organize_split(image_base_path, base_output_dir, split_df, copy, split_name):
    for _, row in split_df.iterrows():
        src = Path(row["full_file_path"])
        dst = base_output_dir / split_name / str(row["family"]) / src.name

        os.makedirs(dst.parent, exist_ok=True)
        if copy==True:
            try:
                shutil.copy(src, dst)
            except FileNotFoundError:
                print(f"Not found: {src}")
        else:
            try:
                shutil.move(src, dst)
            except FileNotFoundError:
                print(f"Not found: {src}")
            


# Clean up the folders that are not needed
def cleanup_folders(base_path):
    keep_folders = {"train", "val", "test"}
    for item in base_path.iterdir():
        if item.is_dir() and item.name not in keep_folders:
            shutil.rmtree(item)


# Repoint image paths to the new train/val/test directories
def update_paths(df, split_name):
    df = df.copy()
    df["full_file_path"] = df.apply(
        lambda row: str(Path(f"../data/rare_species/{split_name}") / str(row["family"]) / Path(row["full_file_path"]).name),
        axis=1
    )
    return df
    

def plot_graph(title, xlabel, ylabel, counts):
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='midnightblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90) 
    plt.tight_layout()
    plt.show()


def build_sequential_model(list_of_layers):
    model = models.Sequential()
    for layer in list_of_layers: model.add(layer)
    return model


def plot_model_acc(num_epochs, train_acc, val_acc):
    x_axis = range(1,num_epochs+1)
    plt.plot(x_axis, train_acc, 'r', label='Training accuracy')
    plt.plot(x_axis, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_model_loss(num_epochs, train_loss, val_loss):
    x_axis = range(1,num_epochs+1)
    plt.plot(x_axis, train_loss, 'g', label='Training loss')
    plt.plot(x_axis, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def get_callbacks(checkpoint_file_path, metrics_file_path, lr_scheduler=None):
    os.makedirs(os.path.dirname(checkpoint_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)

    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_file_path,
        monitor='val_loss',
        verbose=0
    )
    callbacks.append(checkpoint_callback)

    metrics_callback = CSVLogger(metrics_file_path)
    callbacks.append(metrics_callback)

    if lr_scheduler is not None:
        lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
        callbacks.append(lr_scheduler_callback)

    return callbacks


def exp_decay_lr_scheduler(epoch, lr, factor = 0.95):
    lr *= factor
    return lr


def lr_scheduler(initial_lr, final_lr, n_epochs):
    factor = (final_lr / initial_lr) ** (1 / n_epochs)
    return partial(exp_decay_lr_scheduler, factor=factor)


def get_metric(dataset, model_name):
    print(f"\nEvaluating model: {model_name}")
    model = load_model(model_name)

    y_true = []
    y_pred = []

    for batch_x, batch_y in dataset:
        preds = model.predict(batch_x, verbose=0)
        y_true.append(np.argmax(batch_y.numpy(), axis=1))
        y_pred.append(np.argmax(preds, axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print(classification_report(y_true, y_pred, digits=4))
    print("Accuracy     :", accuracy_score(y_true, y_pred))
    print("F1 (macro)   :", f1_score(y_true, y_pred, average="macro"))
    print("Precision    :", precision_score(y_true, y_pred, average="macro"))
    print("Recall       :", recall_score(y_true, y_pred, average="macro"))

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(35, 30), fontsize=10, title="Confusion Matrix"):
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # Plot the heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm,
        cmap="Blues",
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"shrink": 0.5}
    )

    # Add titles and labels
    plt.title(title, fontsize=24, pad=20)
    plt.xlabel("Predicted Label", fontsize=20, labelpad=10)
    plt.ylabel("True Label", fontsize=20, labelpad=10)

    # Rotate and increase font size for labels
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Show the plot with tight layout
    plt.tight_layout()
    plt.show()


def show_correct_predictions_for_family(y_true, y_pred, pred_probs_all, test_images, class_names, family_name, num_images=6):
    # Find the index of the family
    matches = np.where(class_names == family_name)[0]
    if len(matches) == 0:
        print(f"Family '{family_name}' not found in class names.")
        return
    family_idx = matches[0]

    # Convert y_true to class labels
    y_true_labels = np.argmax(y_true, axis=1)

    # Get indices of correctly classified images for that family
    correct_indices = np.where((y_true_labels == y_pred) & (y_true_labels == family_idx))[0]

    if len(correct_indices) == 0:
        print(f"No correct predictions found for family '{family_name}'.")
        return

    # Get confidences
    confidences = np.max(pred_probs_all[correct_indices], axis=1)

    # Sort by descending confidence
    sorted_indices = np.argsort(confidences)[::-1]

    # Select top examples
    selected_indices = correct_indices[sorted_indices[:num_images]]

    # Prepare images
    images_to_show = []
    for idx in selected_indices:
        img = test_images[idx]
        true_label = y_true_labels[idx]
        pred_label = y_pred[idx]
        confidence = np.max(pred_probs_all[idx])
        images_to_show.append((img, true_label, pred_label, confidence))

    # Plot
    plt.figure(figsize=(16, 10))

    for i, (img, true_label, pred_label, confidence) in enumerate(images_to_show):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img.astype("uint8"))
        plt.axis('off')
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}\nConf: {confidence:.2f}",
                  color='green', fontsize=10)

    plt.suptitle(f"Most Confident Correct Predictions - {family_name}", fontsize=18)
    plt.tight_layout()
    plt.show()


def load_history(path):
    df = pd.read_csv(path)
    # Normalize column names
    df = df.rename(columns={
        'acc': 'accuracy',
        'val_acc': 'val_accuracy'
    })
    return df


def create_model_summary_df(files_dict):
    summary_list = []

    # Iterate through the files and extract the last epoch metrics
    for label, path in files_dict.items():
        df = load_history(path)
        last_epoch = df.iloc[-1]

        summary_list.append({
            "Model": label,
            "Train Accuracy": last_epoch.get("accuracy") or last_epoch.get("train_accuracy"),
            "Validation Accuracy": last_epoch.get("val_accuracy") or last_epoch.get("validation_accuracy"),
            "Train F1-Score": last_epoch.get("f1_score") or last_epoch.get("f1_train_macro"),
            "Validation F1-Score": last_epoch.get("val_f1_score") or last_epoch.get("f1_val_macro"),
            "Train Loss": last_epoch.get("loss") or last_epoch.get("train_loss"),
            "Validation Loss": last_epoch.get("val_loss") or last_epoch.get("validation_loss"),
        })

    # Create a DataFrame from the summary list
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.set_index("Model")
    summary_df = summary_df.sort_values(by="Validation F1-Score", ascending=False)
    
    return summary_df


def plot_model_metrics(summary_df, title_model=None):
    # Define the metrics to plot
    metrics = ['F1-Score', 'Accuracy', 'Loss']

    # Sort by Validation F1-Score descending
    summary_df = summary_df.sort_values(by='Validation F1-Score', ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(30, 12), sharey=True)  # Larger figure
    sns.set_style("whitegrid")

    model_names = summary_df.index.tolist()
    bar_width = 0.4
    y_pos = np.arange(len(model_names))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        train_vals = summary_df[f'Train {metric}']
        val_vals = summary_df[f'Validation {metric}']

        ax.barh(y_pos - bar_width/2, train_vals, height=bar_width, color='#31635A', label='Train')
        ax.barh(y_pos + bar_width/2, val_vals, height=bar_width, color='#C47C55', label='Validation')

        # Add value labels with 3 decimal places
        for j in range(len(model_names)):
            ax.text(train_vals.iloc[j] + 0.005, y_pos[j] - bar_width/2,
                    f'{train_vals.iloc[j]:.3f}', va='center', fontsize=14)
            ax.text(val_vals.iloc[j] + 0.005, y_pos[j] + bar_width/2,
                    f'{val_vals.iloc[j]:.3f}', va='center', fontsize=14)

        ax.set_title(metric, fontsize=22)
        ax.set_xlabel(metric, fontsize=18)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.invert_yaxis()

        if metric != 'Loss':
            ax.set_xlim(0, 1.1)

    plt.suptitle(f"{title_model} - Models Performance Comparison (Train vs Validation)", fontsize=26)
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_training_history(history_df, title_model=None):
    sns.set_style("whitegrid")

    # Reordered: F1-Score first
    metrics = {
        'F1-Score': ('f1_score', 'val_f1_score'),
        'Accuracy': ('accuracy', 'val_accuracy'),
        'Loss': ('loss', 'val_loss')
    }

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    for i, (title, (train_col, val_col)) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(history_df['epoch'], history_df[train_col], label='Train', color='#31635A', linewidth=2)
        ax.plot(history_df['epoch'], history_df[val_col], label='Validation', color='#C47C55', linewidth=2)

        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel(title, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)

        if title != 'Loss':
            ax.set_ylim(0, 1.05)

    plt.suptitle(f"{title_model} - Model Performance Metrics Over Epochs (Train vs Validation)", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()