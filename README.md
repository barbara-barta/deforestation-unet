# Semantic segmentation of Sentinel-2 satellite imagery using (attention) U-Nets

<a target="_blank" href="https://datalumina.com/">
    <img src="https://img.shields.io/badge/Datalumina-Project%20Template-2856f7" alt="Datalumina Project" />
</a>

## Overview

This project recreates and explores the paper "An attention-based U-Net for detecting deforestation within satellite 
sensor imagery" by David John and Ce Zhang.

The goal of this project is to reproduce the paper’s methodology for semantic segmentation of deforested areas in satellite imagery, re-implementing the code in PyTorch. The aim is to classify each pixel in a satellite image as forest or non-forest using both the U-Net and the attention U-net architecutre. The project includes exploratory analysis of multispectral satellite data (including visualisation of RGB and near-infrared bands and computation of NDVI), as well as data preprocessing, feature engineering, model training, evaluation, and visualization of prediction results.

## Results

For both the vanilla U-net and the attention U-net, the metrics recorded are precision, recall, the F1 score, and the intersection over union (IoU). The models are trained using the 4-band Amazon dataset. Each model was trained 3 times with a different seed, and then evaluated on the test set. Below are the results for U-Net.

| Metric    | Test Score (Mean ± Std) |
| --------- | ------------------ |
| Precision | 0.9639 ± 0.0065    |
| Recall    | 0.9514 ± 0.0126    |
| F1 Score  | 0.9575 ± 0.0041    |
| IoU       | 0.9185 ± 0.0075    |

The following table shows the Attention U-Net results.

| Metric    | Test Score (Mean ± Std) |
| --------- | ------------------ |
| Precision | 0.9752 ± 0.0020    |
| Recall    | 0.9680 ± 0.0030    |
| F1 Score  | 0.9716 ± 0.0007    |
| IoU       | 0.9447 ± 0.0014    |

Considering the small sample size, we cannot say that the results are conclusive. However, it seems likely that the Attention U-Net performs better, given that in all metrics it surpasses the vanilla U-net by at least 1%. Most notable is the difference between the IoU's, with the Attention U-Net achieving a score that is grater by 2.6%.

## Visualisations

confusion matrices (ADD?)

The loss plot of the Attention U-net model shows convergence. Interestingly, the validation loss is smaller than the train loss. With the absence of stochasticity in the preprocessing pipeline, it seems that the validation set is simply more difficult.
<p align="center">
  <img width="700" alt="image" src="https://github.com/barbara-barta/deforestation-unet/blob/main/reports/figures/attn_loss_plot.png?raw=true" />
</p>
The models outputs match quite well with the actual masks,
<p align="center">
  <img width="800" alt="image" src="https://github.com/barbara-barta/deforestation-unet/blob/main/reports/figures/attn_unet_predictions.png?raw=true" />
</p>
especially when compared to the outputs of the vanilla U-Net, which are less detailed:
<p align="center">
  <img width="800" alt="image" src="https://github.com/barbara-barta/deforestation-unet/blob/main/reports/figures/unet_predictions.png?raw=true" />
</p>

The following table shows the number of parameters each model has, as well as the time it took to train the model per image.
| Network    | Number of Parameters (x10e6) | Train Time per image (s) |
| --------- | ------------------ | -----|
| Attention U-Net | 2.01   | 465 |
| U-Net | 31.03 | 650 |

Considering that the Attention U-net has 15x less parameters and trains in roughly 2/3 of the time it takes to train the baseline U-Net model, makes its higher performance even more impressive. It seems the attention mechanism is a valuable addition to the network.

## Dataset

The project uses a publicly available satellite imagery dataset containing paired satellite images and binary forest masks of the Amazon rainforest. The dataset consists of Sentinel-2 GeoTIFF imagery with a spatial resolution of 10 metres per pixel, and contains four bands: RGB a near-infrared (NIR) band. The train dataset contains 499 images, the validation dataset 100 images, and the test dataset 20 images.

The dataset pipeline includes:
1. loading multispectral GeoTIFF images using Rasterio,
2. min-max normalisation of image values,
3. conversion to PyTorch tensors,
4. creation of custom PyTorch Dataset and DataLoader classes for training, validation, and testing.

## Methodology 

The project implements semantic segmentation models from scratch in PyTorch, with a primary focus on the U-Net architecture and an Attention U-Net variant.
The baseline U-Net model consists of:
1. Four encoder blocks using convolutional layers and max pooling, with the number of layers in the blocks equal to 64, 128, 256, and 512, respectively
2. a bottleneck layer with 1024 filters,
3. decoder blocks with transpose convolutions and convolutions, with the number of layers in the blocks equal to 512, 256, 128, and 64, respectively
4. long skip connections between corresponding pairs of encoder and decoder blocks

An Attention U-Net extension was also implemented to improve the model’s ability to focus on relevant spatial regions during decoding. Attention gates were incorporated into the skip connections to suppress irrelevant background features and emphasise informative regions of the image. The number of filters in each convolutional layer was adjusted to 16,32,64,128 respectively. This was done to keep the number of parameters at a reasonable level, since this number increases through the introduction of attention gates. 

The Attention U-net architecture can be seen in the following image.

<p align="center">
  <img width="800" alt="image" src="https://github.com/barbara-barta/deforestation-unet/blob/main/reports/figures/Attention%20U-net%20architecture.png?raw=true" />
</p>

The attention gates, seen below, combine the corresponding encoder-phase vector with the output from a previous layer from the decoder phase. 
<p align="center">
  <img width="400" alt="image" src="https://github.com/barbara-barta/deforestation-unet/blob/main/reports/figures/attention%20mechanism.png?raw=true" />
</p>

For both the attention and vanilla U-net, the BCE loss was used with the Adam optimizer. The U-net model was trained on 20 epochs with a learning rate of 0.0001, and the Attention U-Net was trained on 60 epochs with a learning rate of 0.0005. No data augmentation was used. 

## Future Work / Limitations

An idea for future work is motivated by a common problem in climate monitoring using EO data: there is an abundance of unlabeled data gathered through various EO projects, such as the Copernicus Programme and the LANDSAT Program. However, labeled data is sparse. This presents a difficulty if we want to perform semantic segmentation on a region for which there is no labeled forest/non-forest data. One could use a model that was trained on a different region, but it is questionable how well that model would perform, given that forests in different geographical regions might look very different. 
One way to resolve this issue is to use contrastive learning, where a part of the model is pre-trained on a large, unlabeled, dataset. The loss criterion is chosen in such a way that the model learns features of the dataset which remain unchanged under variable circumstances, such as differnt lighting or orientation. The model is then fine-tuned on the specific task - in our case, semantic segmentation. A variant of this paradigm that was specifically designed using EO data is GLCNet. (ADD)

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

Technologies Used

The project was developed primarily in Python using PyTorch for deep learning and model training.

### Tools
- NumPy for numerical operations,
- Matplotlib for visualisation,
- Rasterio for reading GeoTIFF satellite imagery,
- GeoPandas for geospatial processing,
- Torchvision for data transformations,
- Google Colab for GPU-based experimentation and training.

### Datasets
- (ADD)
### Articles
- (ADD)
- (REMOVE THE ONES BELOW)
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, 2019 [https://arxiv.org/pdf/1610.02391](https://arxiv.org/pdf/1610.02391)
- Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers



## Author / Contact



