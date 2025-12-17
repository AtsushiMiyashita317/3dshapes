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

    # データ分割（1:5）
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(n_samples / 6)
    val_idx, train_idx = indices[:split], indices[split:]
    
    # Set up model
    torch.manual_seed(0)
    
    # LightningModule化
    pl_model = CLNFModule(
        sample_num=args.sample_num,
        lr=args.lr,
        ckpt_predictor=args.ckpt_predictor,
        num_bases=args.num_bases,
        latent_dim=args.latent_dim,
        autoencoder_layers=args.autoencoder_layers,
        flow_layers=args.flow_layers,
        flow_hidden_dim=args.flow_hidden_dim,
        eps_p=args.eps_p,
        eps_q=args.eps_q,
        scale_map=args.scale_map,
    )

    # DataLoader
    train_data = Dataset3DShapes(images, train_idx)
    val_data = Dataset3DShapes(images, val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size // 5, shuffle=False, num_workers=args.num_workers)

    # wandb logger
    wandb_logger = WandbLogger(project=args.project)

    wandb_logger.log_hyperparams({
        "lr": args.lr,
        "num_bases": args.num_bases,
        "latent_dim": args.latent_dim,
        "autoencoder_layers": args.autoencoder_layers,
        "flow_layers": args.flow_layers,
        "flow_hidden_dim": args.flow_hidden_dim,
        "eps_p": args.eps_p,
        "eps_q": args.eps_q,
    })

    # ModelCheckpointコールバック
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir if type(wandb_logger.experiment.dir) is str else args.ckpt_dir,
        filename="best-{epoch:02d}-{val_nll:.4f}",
        monitor="val_nll",
        save_top_k=1,
        save_last=True,
        mode="min",
        save_weights_only=False,
        verbose=True
    )

    # 500エポックごとに保存するModelCheckpointコールバック
    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir if type(wandb_logger.experiment.dir) is str else args.ckpt_dir,
        filename="epoch{epoch:04d}",
        every_n_epochs=500,
        save_top_k=-1,
        save_last=False,
        save_weights_only=False,
        verbose=True
    )

    if args.backend is not None:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(process_group_backend=args.backend, find_unused_parameters=True)
    else:
        strategy = args.strategy

    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, periodic_checkpoint_callback],
        enable_checkpointing=True
    )
    trainer.fit(pl_model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('ckpt_autoencoder', type=str, help='Path to pretrained autoencoder checkpoint')
    parser.add_argument('ckpt_predictor', type=str, help='Path to pretrained predictor checkpoint')
    parser.add_argument('--num_bases', type=int, default=66, help='Number of bases')
    parser.add_argument('--latent_dim', type=int, default=12, help='Latent dimension of autoencoder')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--autoencoder_layers', type=int, default=3, help='Number of post layers in autoencoder')
    parser.add_argument('--flow_layers', type=int, default=24, help='Number of layers in normalizing flow')
    parser.add_argument('--flow_hidden_dim', type=int, default=192, help='Hidden dimension of flow networks')
    parser.add_argument('--eps_p', type=float, default=1e-3, help='Epsilon p for conditional AE')
    parser.add_argument('--eps_q', type=float, default=1e-1, help='Epsilon q for conditional AE')
    parser.add_argument('--scale_map', type=str, default='exp_clamp', help='Scale map for flow (exp, exp_clamp)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=5000, help='Max training steps')
    parser.add_argument('--val_interval', type=int, default=50, help='Validation interval in steps')
    parser.add_argument('--sample_num', type=int, default=64, help='Number of generated images per validation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--project', type=str, default='3dshapes-clnf', help='wandb project name')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/clnf', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--strategy', type=str, default='auto', help='Distributed training strategy (ddp, ddp_spawn, etc)')
    parser.add_argument('--backend', type=str, default=None, help='Distributed backend (nccl, gloo, etc)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training')
    parser.add_argument('--devices', type=str, default='auto', help='Number of devices (GPUs/CPUs) per node')
    args = parser.parse_args()
    main(args)
