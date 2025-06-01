import sys
from pathlib import Path

from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

# Add project src directory to path
sys.path.append(str((Path().resolve() / '../src').resolve()))

from config import DF2_CATEGORY_MAPPING

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def read_csv(file_path):
    return pd.read_csv(file_path, on_bad_lines='skip')

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def show_sample_images(df, class_column="articleType", n=5):
    unique_classes = df[class_column].unique()
    
    for cls in unique_classes:
        sample_rows = df[df[class_column] == cls].sample(n=min(n, len(df[df[class_column] == cls])), random_state=1)
        fig, axs = plt.subplots(1, len(sample_rows), figsize=(15, 3))
        fig.suptitle(cls, fontsize=16)
        
        for i, (_, row) in enumerate(sample_rows.iterrows()):
            img_path = row['image_path']
            try:
                img = Image.open(img_path)
                axs[i].imshow(img)
                axs[i].axis("off")
            except:
                axs[i].set_title("Not found")
                axs[i].axis("off")
        
        plt.tight_layout()
        plt.show()

def clean_deepfashion2(df):
    """
    Converts DeepFashion2 categories to unified major categories and extracts image id.
    """
    df = df.copy()
    df['id'] = df['path'].apply(lambda x: Path(x).stem)
    df['articleType'] = (
        df['category_name'].str.strip().str.lower()
        .map(DF2_CATEGORY_MAPPING).str.title()
    )
    return df

def encode_labels(df, column, new_column='encoded', return_encoder=False):
    """
    Encode string labels in a DataFrame column into integer codes.
    Adds the result as a new column (default: 'encoded').
    Returns the updated DataFrame and optionally the fitted encoder.
    """
    le = LabelEncoder()
    df[new_column] = le.fit_transform(df[column])
    if return_encoder:
        return df, le
    return df