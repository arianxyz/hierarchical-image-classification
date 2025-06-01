from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CLOTHING_CATEGORIES = ['Dresses', 'Skirts', 'Outerwear']
SHOES_CATEGORIES = ['Casual Shoes', 'Flats', 'Heels', 'Sports Shoes']
BAGS_CATEGORIES = ['Clutches', 'Handbags', 'Wallets']

NUM_CLASSES_CLOTHES = len(CLOTHING_CATEGORIES)
NUM_CLASSES_SHOES = len(SHOES_CATEGORIES)
NUM_CLASSES_BAGS = len(BAGS_CATEGORIES)


GROUP_MAP = {
    # Clothing
    'Dresses': 'Clothing',
    'Skirts': 'Clothing',
    'Outerwear': 'Clothing',
    # Shoes
    'Casual Shoes': 'Shoes',
    'Flats': 'Shoes',
    'Heels': 'Shoes',
    'Sports Shoes': 'Shoes',
    # Bags
    'Clutches': 'Bags',
    'Handbags': 'Bags',
    'Wallets': 'Bags'
}

# Mapping from DeepFashion2 fine categories to major class names
DF2_CATEGORY_MAPPING = {
    'short sleeve outwear': 'outerwear',
    'long sleeve outwear': 'outerwear',
    'short sleeve top': 'outerwear',
    'long sleeve top': 'outerwear',
    'trousers': 'outerwear',
    'shorts': 'outerwear',
    'short sleeve dress': 'dresses',
    'long sleeve dress': 'dresses',
    'sling dress': 'dresses',
    'vest dress': 'dresses',
    'sling': 'dresses',
    'skirt': 'skirts',
}