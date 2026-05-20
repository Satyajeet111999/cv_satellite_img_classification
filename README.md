# cv_satellite_img_classification

This repository demonstrates classical computer-vision feature engineering and machine learning for satellite image classification (no deep learning). The goal is to distinguish urban scenes from natural scenes using handcrafted features and classical classifiers.

## Problem Statement

Classify satellite images into two broad categories:
- Urban (e.g., Residential, Industrial, Highway)
- Natural (e.g., Forest, River, Pasture, SeaLake, Crops)

This project uses the EuroSAT dataset and intentionally avoids convolutional neural networks to illustrate traditional CV pipelines.

## Solution Overview

Pipeline steps:
- Preprocessing: load, convert to RGB, resize, and convert to grayscale
- Feature extraction: compute multiple handcrafted feature types (described below)
- Feature normalization: standard scaling before training
- Model training: train and evaluate classical ML classifiers

## Handcrafted Features Used

- Pixel-level features
	- RGB mean and standard deviation
	- Per-channel color histograms (32 bins)
- Texture features
	- GLCM properties: contrast, homogeneity, energy, correlation
	- Local Binary Patterns (LBP) histogram (uniform patterns)
- Gradient / structure features
	- Histogram of Oriented Gradients (HOG)
	- Edge density using Canny edge detector

These features capture color statistics, local texture, and structural edges that help separate built-up (urban) areas from natural landscapes.

## Classification Models

The notebook trains and compares several classical classifiers:

- Logistic Regression
	- Simple linear baseline; fast and interpretable.
- Random Forest
	- Ensemble tree-based model; robust to feature scaling and useful for feature importance.
- Support Vector Machine (SVM)
	- Effective with handcrafted descriptors; RBF kernel used for non-linear decision boundaries.

Evaluation metrics reported: accuracy, precision, recall, F1-score, and confusion matrix.

## How to Run

Open and run the notebook `cv_satellite_img_classification/satellite_img_classification.ipynb` to execute the full pipeline: dataset loading, feature extraction, training, evaluation, and visualization.

Required Python packages (examples):

- numpy
- opencv-python
- scikit-image
- scikit-learn
- matplotlib
- tqdm

Install with pip if needed:

pip install numpy opencv-python scikit-image scikit-learn matplotlib tqdm

## Results & Next Steps

The notebook prints classification reports and confusion matrices for each model and visualizes feature maps (LBP, edges, HOG) for sample images. Possible extensions:
- Increase dataset size or per-class limits
- Add cross-validation and hyperparameter search
- Compare with CNN-based baselines for reference

## License / Contact

This repository is for educational purposes. For questions or improvements, open an issue or contact the author.
