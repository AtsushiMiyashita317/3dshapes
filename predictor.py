import numpy as np
import h5py
import argparse

import torch
import torchvision as tv
import normflows as nf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from dataset import Dataset3DShapes
from module import PredictorModule

def main(args):
    # load dataset
    dataset = h5py.File('3dshapes.h5', 'r')
    print(dataset.keys())
    images = dataset['images'][:]
    labels = dataset['labels'][:]
    images = images.reshape(10, 10, 10, 8, 4, 15, 64, 64, 3)
    labels = labels.reshape(10, 10, 10, 8, 4, 15, 6)
    s = [slice(None)] * 6
    factor_dict = {
        'floor_hue': 0,
        'wall_hue': 1,
        'object_hue': 2,
        'scale': 3,
        'shape': 4,
        'orientation': 5
    }
    for factor in args.removed_factors:
        idx = factor_dict[factor]
        s[idx] = 0
    s = tuple(s)
    images = images[s]
    labels = labels[s]
    images = images.reshape(-1, 64, 64, 3)
    labels = labels.reshape(-1, 6)
    n_samples = images.shape[0]

    # データ分割（1:7）
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(n_samples / 8)
    val_idx, train_idx = indices[:split], indices[split:]

    # Set up model
    torch.manual_seed(0)

    # LightningModule化
    pl_model = PredictorModule()

    # DataLoader
    train_data = Dataset3DShapes(images=images, labels=labels, indices=train_idx)
    val_data = Dataset3DShapes(images=images, labels=labels, indices=val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # wandb logger
    wandb_logger = WandbLogger(project=args.project, name=args.run_name)

    # ModelCheckpointコールバック
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir if type(wandb_logger.experiment.dir) is str else args.ckpt_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_weights_only=False,
        verbose=True
    )

    # Trainer
    trainer = Trainer(
        max_steps=args.max_steps,
        num_nodes=args.num_nodes,
        devices=args.devices,
        strategy=args.strategy,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True
    )
    trainer.fit(pl_model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--removed_factors', type=str, nargs='+', default=[], help='Factors to remove from the dataset. Options: floor_hue, wall_hue, object_hue, scale, shape, orientation')
    
    parser.add_argument('--batch_size', type=int, default=4000, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=120000, help='Max training steps')
    parser.add_argument('--val_interval', type=int, default=50, help='Validation interval in steps')
    parser.add_argument('--sample_num', type=int, default=64, help='Number of generated images per validation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--project', type=str, default='3dshapes-predictor', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='predictor', help='wandb run name')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--strategy', type=str, default='auto', help='Distributed training strategy (ddp, ddp_spawn, etc)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training')
    parser.add_argument('--devices', type=str, default='auto', help='Number of devices (GPUs/CPUs) per node')
    args = parser.parse_args()
    main(args)