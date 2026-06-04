import argparse
import numpy as np
import torch
import rasterio as rio

from src.dataset import train_transform
from src.model import UNet, AttentionUNet
from src.config import device


def load_model(checkpoint_path, attention=False):
    if attention:
        model = AttentionUNet(num_inputs=4,num_outputs=1).to(device)
    else:
        model = UNet(num_inputs=4,num_outputs=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict(model, input_path):
    with rio.open(input_path) as src:
        image = src.read().astype('float32')
        profile = src.profile.copy()

    image = train_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image)
        pred_mask = (pred > 0.).float().squeeze().cpu().numpy()

    return pred_mask, profile


def save_prediction(pred_mask, profile, output_path):
    profile.update(count=1, dtype="uint8")
    with rio.open(output_path, "w", **profile) as dst:
        dst.write(pred_mask.astype(np.uint8), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attention", action="store_true")
    args = parser.parse_args()

    model = load_model(args.checkpoint, attention=args.attention)
    pred_mask, profile = predict(model, args.input)
    save_prediction(pred_mask, profile, args.output)
    print(f"Saved prediction to {args.output}.")


if __name__ == "__main__":
    main()
