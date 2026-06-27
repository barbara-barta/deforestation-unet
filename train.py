import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import AM4DataModule
from src.models import get_model
from src.utils import get_base_dir, local_run


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet or Attention UNet on AM4.")
    parser.add_argument("--model", choices=["unet", "attn_unet", "attention_unet"], default="unet")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("tb_logs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models"))
    parser.add_argument("--validate-prob", type=float, default=0.5)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-drive-cache", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    return parser.parse_args()


def default_epochs(model_name):
    if model_name == "unet":
        return 20
    return 60


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    base_dir = args.base_dir if args.base_dir is not None else get_base_dir()
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = base_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = args.log_dir
    if not log_dir.is_absolute():
        log_dir = base_dir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 4 if local_run() else 32

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if local_run() else 4

    max_epochs = args.max_epochs if args.max_epochs is not None else default_epochs(args.model)

    for seed in args.seeds:
        logger_name = "attn_unet" if args.model in {"attn_unet", "attention_unet"} else "unet"
        logger = TensorBoardLogger(
            save_dir=str(log_dir),
            name=logger_name,
            version=f"seed_{seed}",
        )

        pl.seed_everything(seed, workers=True)

        datamodule = AM4DataModule(
            base_dir=base_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=not args.no_pin_memory,
            validate_prob=args.validate_prob,
            download=not args.no_download,
            validate=not args.no_validate,
            use_drive_cache=not args.no_drive_cache,
        )

        model = get_model(
            args.model,
            num_inputs=4,
            num_outputs=1,
            lr=args.lr,
        )

        trainer = pl.Trainer(
            default_root_dir=checkpoint_dir,
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=max_epochs,
            log_every_n_steps=args.log_every_n_steps,
            fast_dev_run=args.fast_dev_run,
            logger=logger,
        )

        trainer.fit(model, datamodule=datamodule)
        datamodule.setup(stage="test")
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
