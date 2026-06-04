import argparse
from pathlib import Path
import torch
import torch.nn as nn

from src.config import CHECKPOINT_PATH, base_dir, batch_size, num_workers
from src.dataset import make_am4_dataloaders
from src.model import UNet, AttentionUNet
from src.utils import set_seed


def train_and_eval(model_name, model_hparams, optimizer_name, optimizer_hparams, num_epochs, train_dataloader, val_dataloader, attention, seed = 0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model = Path(CHECKPOINT_PATH) / f"{model_name}_{seed}.pt"
    if seed is not None:
        print(f"Setting seed equal to {seed}.")
        set_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if attention:
        model = AttentionUNet(**model_hparams).to(device)
    else:
        model = UNet(**model_hparams).to(device)

    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(),
        **optimizer_hparams
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    train_losses, train_precisions, train_recalls, train_f1s, train_ious, val_losses, val_precisions, val_recalls, val_f1s, val_ious = [], [], [], [], [], [], [], [], [], []

    for epoch in range(num_epochs):

        # training
        model.train()
        train_loss = 0.0
        train_precision = 0.0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for images, masks in train_dataloader:

            images = images.to(device)
            masks = masks.to(device).float()
            masks = masks.unsqueeze(1) if masks.ndim == 3 else masks

            optimizer.zero_grad()
            outputs = model(images)

            assert outputs.shape == masks.shape, (outputs.shape, masks.shape)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = (outputs > 0.).float()

            tp = (preds * masks).sum().item()
            fp = (preds * (1 - masks)).sum().item()
            fn = ((1 - preds) * masks).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

        train_loss /= len(train_dataloader.dataset)
        train_precision = total_tp / (total_tp + total_fp + 1e-8)
        train_recall = total_tp / (total_tp + total_fn + 1e-8)
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-8)
        train_iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)

        train_losses.append(train_loss)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        train_ious.append(train_iou)

        # validation
        model.eval()
        val_loss = 0.0
        val_precision = 0.0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        with torch.no_grad():
            for images, masks in val_dataloader:
                images = images.to(device)
                masks = masks.to(device).float()

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                preds = (outputs > 0.).float()

                tp = (preds * masks).sum().item()
                fp = (preds * (1 - masks)).sum().item()
                fn = ((1 - preds) * masks).sum().item()

                total_tp += tp
                total_fp += fp
                total_fn += fn

        val_loss /= len(val_dataloader.dataset)
        val_precision = total_tp / (total_tp + total_fp + 1e-8)
        val_recall = total_tp / (total_tp + total_fn + 1e-8)
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-8)
        val_iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)

        val_losses.append(val_loss)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_ious.append(val_iou)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train Precision: {train_precision:.4f}, "
            f"Val Precision: {val_precision:.4f}, "
            f"Train Recall: {train_recall:.4f}, "
            f"Val Recall: {val_recall:.4f}, "
            f"Train F1: {train_f1:.4f}, "
            f"Val F1: {val_f1:.4f}, "
            f"Train IoU: {train_iou:.4f}, "
            f"Val IoU: {val_iou:.4f}"
        )

        # checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), saved_model)
            print(f"Saved best model: {best_val_loss:.4f}")

    return model, [train_losses, train_precisions, train_recalls, train_f1s, train_ious], [val_losses, val_precisions, val_recalls, val_f1s, val_ious]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="unet_am4")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_dataset_am4, test_dataset_am4, val_dataset_am4, train_dataloader_am4, test_dataloader_am4, val_dataloader_am4 = make_am4_dataloaders(base_dir, batch_size=batch_size, num_workers=num_workers)

    model_am4, metrics_am4_train, metrics_am4_val = train_and_eval(
        model_name=args.model_name,
        model_hparams={"num_inputs": 4, "num_outputs": 1},
        optimizer_name="Adam",
        optimizer_hparams={"lr": args.lr},
        num_epochs=args.num_epochs,
        train_dataloader=train_dataloader_am4,
        val_dataloader=val_dataloader_am4,
        attention=args.attention,
        seed=args.seed,
    )

    torch.save(metrics_am4_train, Path(CHECKPOINT_PATH) / f"{args.model_name}_metrics_train_{args.seed}.pt")
    torch.save(metrics_am4_val, Path(CHECKPOINT_PATH) / f"{args.model_name}_metrics_val_{args.seed}.pt")


if __name__ == "__main__":
    main()
