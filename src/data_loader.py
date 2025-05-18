from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import os


class FashionDataset(Dataset):
    def __init__(
            self, 
            device,
            df
        ):
        self.device = device
        self.df = df.copy()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB
        ])

        self.index_to_classes = {
            # Clothing
            0: 'long sleeve dress',
            1: 'short sleeve dress',
            2: 'sling dress',
            3: 'vest dress',
            4: 'skirt',
            5: 'long sleeve outwear',
            6: 'short sleeve outwear',

            # Shoes
            7: 'Flats',
            8: 'Boots',
            9: 'High Heels',

            # Bags
            10: 'Clutches',
            11: 'Shoulder Bags'
        }
        self.classes_to_index = {v: k for k, v in self.index_to_classes.items()}
    
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["filepath"]
        category = row["subcategory"]
        # group = row["group"]

        # Open and transform the image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Get encoded label
        label = torch.tensor(self.classes_to_index[category], dtype=torch.long)

        return image.to(self.device), label.to(self.device)
    


def main():
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # Load CSV and encode labels
    project_root = Path("..").resolve()
    CSV_PATH = os.path.join(project_root, "project/data/labels.csv")
    
    print(f"CSV_PATH: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    dataset = FashionDataset(device=device, df=df)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    idx = 111000
    image, label = dataset[idx]

    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    print(f"Label (int): {label.item()}")
    print('index_to_classes:', len(dataset.index_to_classes))


if __name__ == "__main__":
    main()
