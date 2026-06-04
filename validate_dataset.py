from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
import rasterio.plot as rplt
import geopandas as gpd

from src.config import (
    train_dataset_dir_imgs,
    test_dataset_dir_imgs,
    val_dataset_dir_imgs,
    train_dataset_dir_masks,
    test_dataset_dir_masks,
    val_dataset_dir_masks,
)


def centroid(bounds):
    return ((bounds.left + bounds.right) / 2,
            (bounds.top + bounds.bottom) / 2)


train_image_dict_am4 = {img.stem: img.name for img in train_dataset_dir_imgs.glob("*.tif")}
train_mask_dict_am4 = {lbl.stem: lbl.name for lbl in train_dataset_dir_masks.glob("*.tif")} 
test_image_dict_am4 = {img.stem: img.name for img in test_dataset_dir_imgs.glob("*.tif")}
test_mask_dict_am4 = {lbl.stem: lbl.name for lbl in test_dataset_dir_masks.glob("*.tif")} 
val_image_dict_am4 = {img.stem: img.name for img in val_dataset_dir_imgs.glob("*.tif")}
val_mask_dict_am4 = {lbl.stem: lbl.name for lbl in val_dataset_dir_masks.glob("*.tif")} 
brightness_scores = {}
images_to_remove = set()


def validate_metadata(img, msk, label):
    # check for corrupt files
    try:
        image = rio.open(img)
    except Exception as e:
        print(f"Error occurred while opening image for label {label}: {e}")
        return
    try:
        mask = rio.open(msk)
    except Exception as e:
        print(f"Error occurred while opening mask for label {label}: {e}")
        return
    
    with image, mask:
        try: 
            # check image and mask alignment
            assert image.bounds == mask.bounds
            # check crs matches
            assert image.crs == mask.crs
            # check for missing data
            assert image.nodata is None and mask.nodata is None
            # check that resolution matches
            assert image.res == mask.res
            #check that the dimensions match
            assert image.width == mask.width and image.height == mask.height
            #check that there are 4 bands in the image and 1 band in the mask
            assert image.count == 4 and mask.count == 1
            # check that the data types are correct (uint16 for image, uint8 for mask)
            assert image.dtypes[0] == 'uint16' and mask.dtypes[0] == 'uint8'
        except AssertionError:
            print(f"Validation failed for label {label}.")
            print(f"Image bounds: {image.bounds}, Mask bounds: {mask.bounds}")
            print(f"Image CRS: {image.crs}, Mask CRS: {mask.crs}")
            print(f"Image nodata: {image.nodata}, Mask nodata: {mask.nodata}")
            print(f"Image resolution: {image.res}, Mask resolution: {mask.res}")
            print(f"Image dimensions: ({image.width}, {image.height}), Mask dimensions: ({mask.width}, {mask.height})")
            print(f"Image bands: {image.count}, Mask bands: {mask.count}")
            print(f"Image data types: {image.dtypes}, Mask data types: {mask.dtypes}")
            images_to_remove.add(img)
            return
    return


def validate_image_mask(img, msk, label):  
    with rio.open(img) as image, rio.open(msk) as mask:
        #compute the number of pixels in the image that are above a certain brightness threshold (e.g. 1000) and save it to the brightness_scores dictionary
        brightness_threshold = 5000
        pixels = image.read([1,2,3])
        bright_pixels = np.sum((pixels[0] + pixels[1] + pixels[2]) > brightness_threshold)
        brightness_scores[image.name] = bright_pixels
        cent = centroid(image.bounds)
        # check that no image is completely empty (all pixel values are 0)
        try :
            assert np.any(pixels > 0)
        except AssertionError:
            print(f"Image for label {label} contains only zeros.")
            fig, axes = plt.subplots(1, 2, figsize=(10, 10))
            rplt.show(mask, ax=axes[0])
            axes[0].set_title("Mask")
            rplt.show(image, ax=axes[1])
            axes[1].set_title("Image")
            images_to_remove.add(img)
            return
        # check that no mask is completely empty (all pixel values are 0)
        try :
            assert np.any(mask.read(1) > 0)
        except AssertionError:
            print(f"Mask for label {label} contains only zeros.")
            axes = plt.subplots(1,2, figsize=(10, 10))[1]
            rplt.show(mask, ax=axes[0])
            axes[0].set_title("Mask")
            rplt.show(image, ax=axes[1])
            axes[1].set_title("Image")
            images_to_remove.add(img)
            return

    return cent


def run_validation(move_error_files=False):
    train_labels = []
    train_centroids = []
    test_labels = []
    val_labels = []
    test_centroids = []
    val_centroids = []

    print("Validating training images and masks...")
    for label, img_name in train_image_dict_am4.items():
        mask_name = train_mask_dict_am4[label]
        validate_metadata(train_dataset_dir_imgs/img_name, train_dataset_dir_masks/mask_name, label)
        cent = validate_image_mask(train_dataset_dir_imgs/img_name, train_dataset_dir_masks/mask_name, label)
        if cent is not None:
            train_labels.append(label)
            train_centroids.append(cent)

    print("Validating test images and masks...")
    for label, img_name in test_image_dict_am4.items():
        mask_name = test_mask_dict_am4[label]
        validate_metadata(test_dataset_dir_imgs/img_name, test_dataset_dir_masks/mask_name, label)
        cent = validate_image_mask(test_dataset_dir_imgs/img_name, test_dataset_dir_masks/mask_name, label)
        if cent is not None:
            test_labels.append(label)
            test_centroids.append(cent)

    print("Validating validation images and masks...")
    for label, img_name in val_image_dict_am4.items():
        mask_name = val_mask_dict_am4[label]
        validate_metadata(val_dataset_dir_imgs/img_name, val_dataset_dir_masks/mask_name, label)
        cent = validate_image_mask(val_dataset_dir_imgs/img_name, val_dataset_dir_masks/mask_name, label)
        if cent is not None:
            val_labels.append(label)
            val_centroids.append(cent)

    train_gdf = gpd.GeoDataFrame({"label": train_labels, "centroid": gpd.points_from_xy([c[0] for c in train_centroids], [c[1] for c in train_centroids])})
    test_gdf = gpd.GeoDataFrame({"label": test_labels, "centroid": gpd.points_from_xy([c[0] for c in test_centroids], [c[1] for c in test_centroids])})
    val_gdf = gpd.GeoDataFrame({"label": val_labels, "centroid": gpd.points_from_xy([c[0] for c in val_centroids], [c[1] for c in val_centroids])})
    train_gdf["buffer"] = train_gdf.centroid.buffer(5) # buffer for 5 meters around the centroid to check for overlap
    test_gdf["buffer"] = test_gdf.centroid.buffer(5)
    val_gdf["buffer"] = val_gdf.centroid.buffer(5)
    train_gdf.set_geometry("buffer", inplace=True)
    test_gdf.set_geometry("buffer", inplace=True)
    val_gdf.set_geometry("buffer", inplace=True)

    sindex = train_gdf.sindex

    possible_matches = val_gdf.geometry.apply(
        lambda geom: list(sindex.intersection(geom.bounds))
    )
    for idx, geom in enumerate(val_gdf.geometry):
        candidates = sindex.intersection(geom.bounds)
        for i in candidates:
            if geom.intersects(train_gdf.geometry[i]):
                print(f"Validation image {val_gdf.label[idx]} overlaps with training image {train_gdf.label[i]}.")

    image_paths = sorted(train_dataset_dir_imgs.glob("*.tif"))
    print(f"Found {len(image_paths)} training images.")

    bins = np.arange(0, 6001, 25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_counts = np.zeros(len(bins) - 1, dtype=np.int64)

    for color in ["Red", "Green", "Blue", "NIR"]:
        for img_path in image_paths:
            with rio.open(img_path) as src:
                if color == "Red":
                    band = src.read(1).astype(np.int32)
                elif color == "Green":
                    band = src.read(2).astype(np.int32)
                elif color == "Blue":
                    band = src.read(3).astype(np.int32)
                elif color == "NIR":
                    band = src.read(4).astype(np.int32)
            counts, _ = np.histogram(band, bins=bins)
            hist_counts += counts

        plt.figure(figsize=(10, 3))
        plt.bar(bin_centers, hist_counts, width=np.diff(bins), align="center", edgecolor="black")
        plt.xlabel(f"{color} band pixel value")
        plt.ylabel("Pixel count")
        plt.title(f"Histogram of {color.lower()} band values for AM4 training images")
        plt.xlim(bins[0], bins[-1])
        plt.tight_layout()
        plt.show()

    #find which image(s) have rgb values above 10000
    for img_path in image_paths:
        with rio.open(img_path) as src:
            red_band = src.read(1).astype(np.int32)
            green_band = src.read(2).astype(np.int32)
            blue_band = src.read(3).astype(np.int32)
            nir_band = src.read(4).astype(np.int32)
            if np.any(red_band > 10000):
                print(f"Image {img_path.name} has red values above 10000 at location {np.where(red_band > 10000)}.")
                images_to_remove.add(Path(img_path))
            if np.any(green_band > 10000):
                print(f"Image {img_path.name} has green values above 10000 at location {np.where(green_band > 10000)}.")
                images_to_remove.add(Path(img_path))
            if np.any(blue_band > 10000):
                print(f"Image {img_path.name} has blue values above 10000 at location {np.where(blue_band > 10000)}.")
                images_to_remove.add(Path(img_path))
            if np.any(nir_band > 10000):
                print(f"Image {img_path.name} has NIR values above 10000 at location {np.where(nir_band > 10000)}.")
                images_to_remove.add(Path(img_path))

    if move_error_files:
        #moving files with labels in images_to_remove to the error folder
        for image_path in images_to_remove:
            #create error folder if it doesn't exist
            error_folder = image_path.parent / "error"
            error_folder.mkdir(exist_ok=True)
            if image_path.exists():
                #move the image file to the error folder
                shutil.move(str(image_path), str(error_folder / image_path.name))
                print(f"Moved image file {image_path} to error folder.")


if __name__ == "__main__":
    run_validation(move_error_files=False)
