import numpy as np
import matplotlib.pyplot as plt
import rasterio.plot as rplt
import seaborn as sns


def plot_nrg(src):
    # Read bands
    n = src.read(4)
    r = src.read(1)
    g = src.read(2)

    # Stack into NRG image
    nrg = np.dstack((n, r, g))

    # Normalize to 0-1
    nrg = nrg.astype(float) / 6000

    # Clip values
    nrg = np.clip(nrg, 0, 1)

    plt.figure(figsize=(7,7))
    plt.imshow(nrg)
    plt.title("NRG Image")
    plt.xlabel("Column #")
    plt.ylabel("Row #")
    plt.show()


def make_img_plots(src):
    red_band = src.read(1).astype('float32')
    green_band = src.read(2).astype('float32')
    nir_band = src.read(4).astype('float32')
    
    # NDVI
    denominator = nir_band + red_band
    ndvi = np.where(denominator == 0, 0, (nir_band - red_band) / denominator)
    ndvi = np.clip(ndvi, -1, 1)


    # Stack into NRG image, normlize and clip
    nrg = np.dstack((nir_band, red_band, green_band))
    nrg = nrg.astype(float) / 6000
    nrg = np.clip(nrg, 0, 1)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    # Plot RGB composite
    rplt.show(src, ax=axes[0])
    axes[0].set_title("RGB Composite")
    
    #Plot NIR band
    im = axes[1].imshow(nir_band, cmap='Reds')
    #rplt.show((src, 4), ax=axes[1], cmap='Reds')
    axes[1].set_title("NIR Band")
    plt.colorbar(im, ax=axes[1], shrink=0.7)
    
    # Plot NDVI
    #rplot.show(ndvi, ax=axes[2], cmap='RdYlGn', vmin=-1, vmax=1)
    im = axes[2].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[2].set_title("NDVI")
    plt.colorbar(im, ax=axes[2], shrink=0.7)

    #plot NRG
    im = axes[3].imshow(nrg, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[3].set_title("NRG Image")

    plt.tight_layout()
    plt.show()


def plot_precision(train_precisions, val_precisions):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=train_precisions, label="Train Precision", marker="o")
    sns.lineplot(data=val_precisions, label="Validation Precision", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Training and Validation Precision")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=train_losses, label="Train Loss", marker="o")
    sns.lineplot(data=val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
