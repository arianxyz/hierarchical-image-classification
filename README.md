# Transfer Learning-Based Hierarchical Image Classification for Second-Hand Women’s Fashion Items

This repository contains the source code, data preparation scripts, and experiments for the deep learning project: **Hierarchical Image Classification for Second-Hand Women's Fashion**, as applied to the Slowfashion platform.

## Project Summary

This project addresses the challenge of automated, fine-grained categorization of user-uploaded fashion item images in second-hand platforms. We built a modular hierarchical classification pipeline based on transfer learning (EfficientNet), leveraging public datasets and a custom taxonomy. The pipeline first classifies an image into a main group (Clothing, Shoes, Bags), then predicts the subcategory within that group (e.g., Dresses, Heels, Wallets). Our best models achieve high validation accuracy, and the results highlight both strengths and real-world challenges such as domain shift and class imbalance.

## Repository Structure

```bash
hierarchical-image-classification/
├── data/
│ ├── DeepFashion2/
│ ├── FashionData/
│ ├── test_samples/
│ └── df_hierarchical.csv
├── figures/
├── notebooks/
│ ├── data_preparation.ipynb
│ ├── 01_train_group_classifier.ipynb
│ ├── 02_train_clothing_classifier.ipynb
│ ├── 03_train_shoes_classifier.ipynb
│ ├── 04_train_bags_classifier.ipynb
│ └── 05_inference_pipeline.ipynb
├── results/
├── src/
│ ├── augmentations.py
│ ├── config.py
│ ├── data_loader.py
│ ├── losses.py
│ ├── model.py
│ ├── train_utils.py
│ └── utils.py
├── requirements.txt
├── README.md
```

## Getting Started

### 1. Requirements

- Python 3.8+
- torch (PyTorch) >=1.12
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- efficientnet_pytorch
- tqdm
- Pillow
- jupyter (for running notebooks)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

- Download [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) and [FashionDataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-small).
- Place the raw data folders inside the `data/` directory as described above.
- Run the notebook `notebooks/data_preparation.ipynb` to harmonize, clean, and split the data.
- This will generate the final CSV file in `data/df_hierarchical.csv` and organize the images as required.

> **Note**: Most code logic (data loading, augmentation, training, etc.) is provided as reusable modules in the src/ folder. These are imported and used within the Jupyter notebooks. You do not need to run the .py files directly.

### 3. Training

- All model training is performed via Jupyter Notebooks in the `notebooks/` directory. Each notebook is fully modular and self-contained.
  - **Train group classifier:**
    `notebooks/01_train_group_classifier.ipynb`
  - **Train Clothing subcategory classifier:**
    `notebooks/02_train_clothing_classifier.ipynb`
  - **Train Shoes subcategory classifier:**
    `notebooks/03_train_shoes_classifier.ipynb`
  - **Train Bags subcategory classifier:**
    `notebooks/04_train_bags_classifier.ipynb`

### 4. Inference

- Run hierarchical inference on new data via
  `notebooks/05_inference_pipeline.ipynb`
- This pipeline first classifies the group, then uses the correct specialist classifier for subcategory prediction.

## Results Summary

- **Group classifier:**
  Test accuracy: **99%**
- **Clothing classifier:**
  Validation accuracy: **72%** (challenges in Dresses vs Outerwear)
- **Shoes classifier:**
  Validation accuracy: **86%**
- **Bags classifier:**
  Validation accuracy: **91%**
- **End-to-end pipeline (manual test images):**
  Group accuracy: **67.5%**, Subcategory accuracy: **32.5%**

---

## Key Features

- **Modular code:** All parts are split into reusable modules and cleanly documented.
- **Custom taxonomy:** Designed for second-hand women's fashion use-case.
- **Transfer learning:** EfficientNetB0 backbone, fine-tuned per category.
- **Two-stage hierarchical classification:** Group classifier + specialist subcategory classifiers.
- **Advanced augmentation:** Strong augmentation for minority/heterogeneous classes.
- **Reproducibility:** All configs, splits, and models are versioned.
- **Visualization:** Confusion matrices, learning curves for all models.

---

## Contact

For questions, feedback, or collaborations, please contact [Arian Ghadirzadeh](mailto:arian.ghadirzadeh@gmail.com) or open an issue.

---

> **Note:**
> This project is a research prototype for academic purposes and is not affiliated with Slowfashion.se.

## Author

Arian Ghadirzadeh<br>
Deep Learning Course (TDIS22) – Project<br>
Jönköping University, School of Engineering (JTH)<br>
May 2025
