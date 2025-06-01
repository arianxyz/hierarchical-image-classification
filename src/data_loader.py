import os
import sys
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

# Add project src directory to path
sys.path.append(str((Path().resolve() / '../src').resolve()))

from config import IMAGE_SIZE

def prepare_hierarchical_df(df, image_dir, available_categories, group_map):
    """
    Filter dataframe for selected categories, assign main group label (Clothing/Shoes/Bags), 
    and add full image path for each row. 
    Returns a tidy DataFrame for hierarchical classification training.
    """
    df = df[df['articleType'].isin(available_categories)].copy()
    df['group'] = df['articleType'].map(group_map)
    df['image_path'] = df['id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    df = df[['image_path', 'articleType', 'group']]
    return df

def load_images(
    df, 
    label_col, 
    image_col="image_path", 
    aug_map=None, 
    default_aug=None, 
    type_col="articleType"
):
    """
    Loads images and labels from a DataFrame using custom augmentations.

    Args:
        df (pd.DataFrame): Input DataFrame containing image paths and labels.
        label_col (str): Name of the column with label information.
        image_col (str): Name of the column with image file paths.
        aug_map (dict): Mapping from type_col values to torchvision transforms.
        default_aug (transform): Default transform if item not in aug_map.
        type_col (str): Column for determining which augmentation to use.

    Returns:
        images (list): List of transformed image tensors.
        labels (list): List of labels.
    """
    images = []
    labels = []

    for _, row in df.iterrows():
        try:
            img = Image.open(row[image_col]).convert("RGB")
            # Select augmentation
            if aug_map and row[type_col] in aug_map:
                aug = aug_map[row[type_col]]
            elif default_aug:
                aug = default_aug
            else:
                aug = transforms.ToTensor()
            img_tensor = aug(img)
            images.append(img_tensor)
            labels.append(row[label_col])
        except Exception as e:
            print(f"Error loading {row[image_col]}: {e}")
            continue

    return images, labels

def resize_tensor_list(tensor_list):
    """
    Resizes a list of image tensors to a specified size (IMAGE_SIZE).

    Args:
        tensor_list (list): List of image tensors.

    Returns:
        torch.Tensor: Stacked tensor of resized images.
    """
    resized = []
    for img in tensor_list:
        if img.shape[1:] != IMAGE_SIZE:
            img = transforms.Resize(IMAGE_SIZE)(img)
        resized.append(img)
    return torch.stack(resized).float()
