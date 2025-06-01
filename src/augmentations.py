import sys
from pathlib import Path

from torchvision import transforms

# Add project src directory to path
sys.path.append(str((Path().resolve() / '../src').resolve()))

from config import IMAGE_SIZE

def strong_aug_special_class():
    """
    Strong augmentation for special classes (e.g., Flats, Clutches)
    where there is high visual diversity or few samples.
    """
    return transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0)),
            transforms.RandomRotation(45),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(p=0.3)
        ], p=0.9),
        transforms.ToTensor()
    ])

def strong_augmentation():
    """
    Returns a torchvision.transforms.Compose object for strong augmentation.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomGrayscale(p=0.3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor()
    ])

def normal_augmentation():
    """
    Returns a torchvision.transforms.Compose object for normal augmentation.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
