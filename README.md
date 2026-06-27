# Semantic segmentation of Sentinel-2 satellite imagery using (Attention) U-Net

## Overview

This project implements and evaluates U-Net and Attention U-Net models for binary semantic segmentation of deforestation in Sentinel-2 satellite imagery of the Amazon Rainforest.

The work is based on the paper An attention-based U-Net for detecting deforestation within satellite sensor imagery by David John and Ce Zhang. The original methodology was reimplemented in PyTorch Loghtning.

## Project Highlights

- Implemented U-Net and Attention U-Net architectures for 4-band Sentinel-2 image segmentation.
- Used PyTorch Lightning for structured training, validation, testing, checkpointing, and TensorBoard logging.
- Built a custom dataset pipeline for efficient loading and validation of multispectral GeoTIFF image/mask pairs using Rasterio.
- Performed EDA and found images with a cloud coverage higher than the 30% maximum claimed, as well as geographical overlap between the validation and train dataset.
- Added dataset validation checks for corrupt files, image-mask alignment, CRS consistency, dimensions, empty masks, constant-value images, and high-brightness/cloud-like artefacts.
- Evaluated models across multiple random seeds, reported mean ± standard deviation and discussed performance/complexity trade-offs.

## Results

For both the vanilla U-net and the attention U-net, the metrics recorded are precision, recall, the F1 score, and the intersection over union (IoU). The models are trained using the 4-band Amazon dataset. Each model was trained 3 times with a different seed, and then evaluated on the test set. Below are the results for the U-Net architecture.

| Metric    | Test Score (Mean ± Std) |
| --------- | ------------------  |
| Precision | 0.9704 ± 0.0179     |
| Recall    | 0.9529 ± 0.0316     |
| F1 Score  | 0.9612 ± 0.0074     |
| IoU       | 0.9254 ± 0.0138     |
| loss      | 0.1134 ± 0.0223     |

The following table shows the Attention U-Net results.

| Metric    | Test Score (Mean ± Std) |
| --------- | ------------------  |
| Precision | 0.9767 ± 0.0029     |
| Recall    | 0.9676 ± 0.0031     |
| F1 Score  | 0.9721 ± 0.0008     |
| IoU       | 0.9458 ± 0.0015     |
| loss      | 0.0730 ± 0.0011     |



Considering the small sample size, we cannot say that the results are conclusive. However, it seems likely that the Attention U-Net performs better, given that in all metrics but precision, it surpasses the vanilla U-net by at least 1%. Most notable is the difference between the IoU's, with the Attention U-Net achieving a score that is grater by 2%.

## Visualisations

The models outputs match quite well with the actual masks,
<p align="center">
  <img width="800" alt="image" src="https://github.com/barbara-barta/deforestation-unet/blob/main/reports/figures/attn_unet_predictions.png?raw=true" />
</p>
especially when compared to the outputs of the vanilla U-Net, which are less detailed:
<p align="center">
  <img width="800" alt="image" src="https://github.com/barbara-barta/deforestation-unet/blob/main/reports/figures/unet_predictions.png?raw=true" />
</p>

The following table shows the number of parameters each model has, as well as the time it took to train the model per image.
| Network    | Number of Parameters (x10e6) | Train Time per step (s) |
| --------- | ------------------ | -----|
| Attention U-Net | 2.01   | 465 |
| U-Net | 31.03 | 650 |

Considering that the Attention U-net has 15x less parameters and trains in roughly 2/3 of the time it takes to train the baseline U-Net model, makes its higher performance even more impressive. It seems the attention mechanism is a valuable addition to the network.

## Dataset

The project uses a publicly available satellite imagery dataset containing paired satellite images and binary forest masks of the Amazon rainforest. The dataset consists of Sentinel-2 GeoTIFF imagery with a spatial resolution of 10 metres per pixel, and contains four bands: RGB a near-infrared (NIR) band. The train dataset contains 499 images, the validation dataset 100 images, and the test dataset 20 images.

Before creating the dataset object and the dataloader, validations are performed to check that the data is not corrupt, that each image contains 4 bands, and each mask 1 band, that there are no empty values in the masks and images, that corresponding mask and image have equal bounding boxes, resolution, dimensions and coordinate reference system. The datatype for each mask and image is validated. A check is performed to see if there are any images or masks which are a constant value (such as all zero masks). The last validation identifies two images with heavy cloud coverage, even though authors claim that the dataset contains only images where cloud coverage is less than 30%. Consequently, we check all images for clouds by means of a threshold technique: for each pixel, the reflectance values in all 3 visible light bands are summed. When this quantity is higher than a certain threshold, the pixel is denoted as a "high brightness" pixel. Images with the highest number of "high brightness" pixels are plotted and inspected visually and validated. Boxplots are created with reflectance values in all 4 bands, and outliers are detected.
All images which do not pass the aforementioned validations are removed from the dataset.

Afterwards, the dataset pipeline is created. It includes:
1. loading multispectral GeoTIFF images using Rasterio,
2. min-max normalisation of image values,
3. conversion to PyTorch tensors,
4. creation of custom PyTorch Dataset and DataLoader classes for training, validation, and testing.

## Methodology 

The project implements semantic segmentation models from scratch in PyTorch Lightning: the U-Net architecture and an Attention U-Net.

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

Experiments were conducted in Google Colab using an NVIDIA A100 GPU with 80 GB of VRAM. Models were implemented in Python 3.13 using PyTorch 2.11.

## Future Work / Limitations

An idea for future work is motivated by a common problem in climate monitoring using EO data: there is an abundance of unlabeled data gathered through various EO projects, such as the Copernicus Programme and the LANDSAT Program. However, labeled data is sparse. This presents a difficulty if we want to perform semantic segmentation on a region for which there is no labeled forest/non-forest data. One could use a model that was trained on a different region, but it is questionable how well that model would perform, given that forests in different geographical regions might look very different. 
One way to resolve this issue is to use contrastive learning. A variant of this paradigm that was specifically designed using remote sensing imaging data is the **global style and local matching contrastive learning network (GLCNet)**. Using this method, both the global image-level representation and the local segment representations are learned.

## Project Organization

```
├── LICENSE            <- Open-source license
├── README.md          <- The top-level README for developers using this project
├── .gitignore          <- The git ignore file
├── .python-version     <- contains the Python version
││
├── notebooks          <- Jupyter notebooks
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── train.py            <- repeated-seed training script
│
│
├── predict.py          <- single-image prediction script
│
│
├── eval.py            <- checkpoint evaluation script
│
│
├── validate_dataset.py            <- standalone AM4 validation script
│
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── models.py               <- UNet, Attention UNet, Lightning training/validation/test logic
    │
    ├── data.py              <- transforms, `DeforestationDataset`, `AM4DataModule`
    │
    ├── plotting.py                <- NRG, NDVI, and prediction plotting helpers
    │
    ├── validation.py                <- `DatasetValidator` and centroid helper
    │
    └── utils.py                <-  path helpers, local/Colab detection, seeds, device helper
```


## Installation and usage

### 1. Clone the repository

```bash
git clone https://github.com/barbara-barta/deforestation-unet.git
cd deforestation-unet
```

### 2. Create a virtual environment (recommended)

```bash
conda env create -f environment.yml
conda activate deforestation
```

### 3. Run training

```bash
python train.py --model-name model_name --attention attention --num-epochs num_epochs --lr learning_rate --seed seed
```

### 4. Make predictions

```bash
python predict.py --checkpoint model_checkpoint --input input_file --output output_file
```

### 5. Evaluate the models

```bash
python eval.py --checkpoint model_checkpoint --split test
```


Note: Google Drive mounting is not supported when running Colab kernels inside VS Code.
For first-time data setup, open the notebook in browser-based Colab.

## References

### Tools
The project was developed primarily in Python using PyTorch for deep learning and model training.
- NumPy for numerical operations,
- Matplotlib for visualisation,
- Rasterio for reading GeoTIFF satellite imagery,
- GeoPandas for geospatial processing,
- Torchvision for data transformations,
- Google Colab for GPU-based experimentation and training.

### Datasets

This project uses Sentinel-2 satellite imagery from the Copernicus Programme for forest vs non-forest semantic segmentation.

Datasets used:
- Bragagnolo et al., Amazon and Atlantic Forest image datasets for semantic segmentation, [https://zenodo.org/records/4498086](https://zenodo.org/records/4498086)


### Articles
- John, David and Zhang, CE, *An attention-based U-Net for detecting deforestation within satellite sensor imagery*, 2022, [https://www.sciencedirect.com/science/article/pii/S0303243422000113#cited-by](https://www.sciencedirect.com/science/article/pii/S0303243422000113#cited-by)
- Li, Haifeng and Li, Yi and Zhang, Guo and Liu, Ruoyun and Huang, Haozhe and Zhu, Qing and Tao, Chao, *Global and Local Contrastive Self-Supervised
Learning for Semantic Segmentation of HR Remote
Sensing Images*, 2021 [https://arxiv.org/pdf/1610.02391](https://arxiv.org/abs/2106.10605v2)
- Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, 2018, [https://arxiv.org/abs/1804.03999](https://arxiv.org/abs/1804.03999)


## Author / Contact

This project was developed by Barbara Barta, an Applied Mathematics graduate specialising in machine learning, climate monitoring, and computer vision.

- GitHub: https://github.com/barbara-barta
- LinkedIn: https://linkedin.com/in/barbara-barta
- Email: barbara.barta.2@example.com

