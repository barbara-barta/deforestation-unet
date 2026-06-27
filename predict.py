import argparse
from pathlib import Path

import numpy as np
import rasterio as rio
import torch

from src.data import MinMaxScale, ToTensor
from src.models import load_model_from_checkpoint
from src.plotting import plot_prediction_triplet
from src.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a deforestation mask for one raster image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["unet", "attn_unet", "attention_unet"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-output", type=Path, default=None)
    return parser.parse_args()


def read_image(path):
    with rio.open(path) as src:
        image = src.read().astype("float32")
        profile = src.profile.copy()

    image = MinMaxScale()(image)
    image = ToTensor()(image)
    return image, profile


def write_mask(mask, profile, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
    )

    with rio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)


def main():
    args = parse_args()

    device = get_device()

    model = load_model_from_checkpoint(
        args.model,
        args.checkpoint,
        num_inputs=4,
        num_outputs=1,
    ).to(device)
    model.eval()

    image, profile = read_image(args.input)
    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_batch)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > args.threshold).float()

    pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
    write_mask(pred_mask_np, profile, args.output)

    print(f"Saved prediction mask to {args.output}")

    if args.plot:
        plot_prediction_triplet(
            image=image,
            pred_mask=pred_mask_np,
            true_mask=None,
            save_path=args.plot_output,
        )


if __name__ == "__main__":
    main()
