import torch
import torch.nn as nn
from .config import device


def eval(model, dataloader):

  model.eval()
  criterion = nn.BCEWithLogitsLoss()
  total_tp = 0
  total_fp = 0
  total_fn = 0
  loss = 0.0

  with torch.no_grad():
      for images, masks in dataloader:

          images = images.to(device)
          masks = masks.to(device).float()
          outputs = model(images)
          batch_loss = criterion(outputs, masks)
          loss += batch_loss.item() * images.size(0)
          preds = (outputs > 0.).float()

          tp = (preds * masks).sum().item()
          fp = (preds * (1 - masks)).sum().item()
          fn = ((1 - preds) * masks).sum().item()

          total_tp += tp
          total_fp += fp
          total_fn += fn


  loss = loss / len(dataloader.dataset)
  precision = total_tp / (total_tp + total_fp + 1e-8)
  recall = total_tp / (total_tp + total_fn + 1e-8)
  f1 = 2 * precision * recall / (precision + recall + 1e-8)
  iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)

  print(
      f"Loss: {loss:.4f}, "
      f"Precision: {precision:.4f}, "
      f"Recall: {recall:.4f}, "
      f"F1: {f1:.4f}, "
      f"IoU: {iou:.4f}"
  )
  
  return loss, precision, recall, f1, iou
