import argparse
from pathlib import Path
import torch

from src.config import base_dir, batch_size, num_workers
from src.dataset import make_am4_dataloaders
from src.model import UNet, AttentionUNet
from src.metrics import eval


def load_model(checkpoint_path, attention=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if attention:
        model = AttentionUNet(num_inputs=4, num_outputs=1).to(device)
    else:
        model = UNet(num_inputs=4, num_outputs=1).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")

    train_dataset_am4, test_dataset_am4, val_dataset_am4, train_dataloader_am4, test_dataloader_am4, val_dataloader_am4 = make_am4_dataloaders(
        base_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if args.split == "train":
        dataloader = train_dataloader_am4
    elif args.split == "val":
        dataloader = val_dataloader_am4
    else:
        dataloader = test_dataloader_am4

    model = load_model(
        checkpoint_path=args.checkpoint,
        attention=args.attention,
        device=device,
    )

    eval(model, dataloader)


if __name__ == "__main__":
    main()
