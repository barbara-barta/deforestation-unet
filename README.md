# Semantic segmentation of Sentinel-2 satellite imagery using (attention) U-Nets

<a target="_blank" href="https://datalumina.com/">
    <img src="https://img.shields.io/badge/Datalumina-Project%20Template-2856f7" alt="Datalumina Project" />
</a>

## Overview

This project recreates and explores 

The goal of this project is to reproduce the paper’s methodology for semantic segmentation of deforested areas in satellite imagery. The goal is to classify each pixel in a satellite image as forest or non-forest using both the U-Net and the attention U-net architecutre. The original paper was implemented in Tensorflow, and this reproduction is done in PyTorch. 
The project includes exploratory analysis of multispectral satellite data (including visualisation of RGB and near-infrared bands and computation of NDVI), as well as data preprocessing, feature engineering, model training, evaluation, and visualization of prediction results.

## Results
Include:
key metrics,
best model performance,
comparisons,
maybe one image/table.


## Visualisations

prediction examples,
loss curves,
confusion matrices,
Grad-CAM outputs,
segmentation masks.


## Dataset

The project uses a publicly available satellite imagery dataset containing paired satellite images and binary forest masks of the Amazon rainforest. The dataset consists of Sentinel-2 GeoTIFF imagery with a spatial resolution of 10 metres per pixel, and contains four bands: RGB a near-infrared (NIR) band. The train dataset contains 499 images, the validation dataset 100 images, and the test dataset 20 images.

The dataset pipeline includes:
1. loading multispectral GeoTIFF images using Rasterio,
2. min-max normalisation of image values,
3. conversion to PyTorch tensors,
4. creation of custom PyTorch Dataset and DataLoader classes for training, validation, and testing.

## Methodology 

Explain:
model architecture,
training strategy,
augmentations,
losses,
evaluation metrics.

The project implements semantic segmentation models from scratch in PyTorch, with a primary focus on the U-Net architecture and an Attention U-Net variant.
The baseline U-Net model consists of:
1. encoder blocks using convolutional layers and max pooling,
2. a bottleneck layer,
3. decoder blocks with transpose convolutions and skip connections.

An Attention U-Net extension was also implemented to improve the model’s ability to focus on relevant spatial regions during decoding. Attention gates were incorporated into the skip connections to suppress irrelevant background features and emphasise informative regions of the image.

Training was performed using binary semantic segmentation with:

- Binary Cross Entropy with Logits Loss,
- the Adam optimiser,
- deterministic training for reproducibility through fixed random seeds.

Model performance was evaluated using multiple segmentation metrics, including: precision, recall, F1-score, and Intersection over Union (IoU).
Experiments were repeated across multiple random seeds to evaluate training stability and reduce sensitivity to random initialisation. The metric means across the different experiment runs can be seen in the following table.
(ADD table)

## Experiments / Ablations


hyperparameter comparisons,
architecture variants,
failed experiments,
trade-offs.


## Future Work / Limitations

## Project Organization

```
├── LICENSE            <- Open-source license
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. 
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling (ADD remove this part)
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```

## Installation and usage

### 1. Clone the repository

```bash
git clone [https://github.com/barbara-barta/CIFAR10-classification.git](https://github.com/barbara-barta/deforestation-unet.git)
cd deforestation-unet
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run training

```bash
python train.py
```

### 5. Evaluate the model

```bash
python evaluate.py
```

### My notes

Note: Google Drive mounting is not supported when running Colab kernels inside VS Code.
For first-time data setup, open the notebook in browser-based Colab.


## References


## Author / Contact



