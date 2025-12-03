import numpy as np
import h5py
import argparse
import os

import torch
import torchvision as tv
import normflows as nf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from module import CLNFModule


class Dataset3DShapes(torch.utils.data.Dataset):
    def __init__(self, images, indices):
        self.images = images
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = self.images[self.indices[idx]]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img / 255.0
        return img


def main(args):
    # load dataset
    dataset = h5py.File('3dshapes.h5', 'r')
    print(dataset.keys())
    images = dataset['images'][:]
    n_samples = images.shape[0]

    # データ分割（8:2）
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(n_samples * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]
    
    # Set up model
    torch.manual_seed(0)
    
    # LightningModule化
    pl_model = CLNFModule(
        sample_num=args.sample_num,
        # ckpt_autoencoder=args.ckpt_autoencoder,
        ckpt_predictor=args.ckpt_predictor,
        # num_bases=args.num_bases
    )

    # DataLoader
    train_data = Dataset3DShapes(images, train_idx)
    val_data = Dataset3DShapes(images, val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # wandb logger
    wandb_logger = WandbLogger(project=args.project)

    # ModelCheckpointコールバック
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(wandb_logger.save_dir, args.ckpt_dir),
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
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True
    )
    trainer.fit(pl_model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('ckpt_autoencoder', type=str, help='Path to pretrained autoencoder checkpoint')
    parser.add_argument('ckpt_predictor', type=str, help='Path to pretrained predictor checkpoint')
    parser.add_argument('--num_bases', type=int, default=64, help='Number of bases')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=120000, help='Max training steps')
    parser.add_argument('--val_interval', type=int, default=50, help='Validation interval in steps')
    parser.add_argument('--sample_num', type=int, default=64, help='Number of generated images per validation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--project', type=str, default='3dshapes-clnf', help='wandb project name')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/clnf', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--strategy', type=str, default='auto', help='Distributed training strategy (ddp, ddp_spawn, etc)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training')
    parser.add_argument('--devices', type=str, default='auto', help='Number of devices (GPUs/CPUs) per node')
    args = parser.parse_args()
    main(args)
