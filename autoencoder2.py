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
    

class Autoencoder(torch.nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3*2, kernel_size=4, stride=2, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*2, 32, 32]),
            torch.nn.Conv2d(3*2, 3*4, kernel_size=4, stride=2, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*4, 16, 16]),
            torch.nn.Conv2d(3*4, 3*8, kernel_size=4, stride=2, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*8, 8, 8]),
            torch.nn.Conv2d(3*8, 3*16, kernel_size=4, stride=2, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*16, 4, 4]),
            torch.nn.Conv2d(3*16, 3*32, kernel_size=4, stride=2, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*32, 2, 2]),
            torch.nn.Conv2d(3*32, 3*64, kernel_size=2),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3*64, 3*32, kernel_size=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*32, 2, 2]),
            torch.nn.ConvTranspose2d(3*32, 3*16, kernel_size=4, stride=2, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*16, 4, 4]),
            torch.nn.ConvTranspose2d(3*16, 3*8, kernel_size=4, stride=2, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*8, 8, 8]),
            torch.nn.ConvTranspose2d(3*8, 3*4, kernel_size=4, stride=2, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*4, 16, 16]),
            torch.nn.ConvTranspose2d(3*4, 3*2, kernel_size=4, stride=2, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*2, 32, 32]),
            torch.nn.ConvTranspose2d(3*2, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid(),
        )

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon


class AutoencoderModule(pl.LightningModule):
    def __init__(self, model, sample_num=64):
        super().__init__()
        self.model = model
        self.sample_num = sample_num
        self.recon_imgs = None
            
    def on_validation_epoch_end(self):
        # 検証終了時に画像生成しwandbに記録
        if self.recon_imgs is None:
            return
        with torch.no_grad():
            img = self.recon_imgs[:self.sample_num]
            grid = tv.utils.make_grid(img.cpu(), nrow=8)
            wandb_logger = self.logger
            if hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log({f"val_generated/epoch_{self.current_epoch}": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})
            self.recon_imgs = None

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        recon = self.model(batch)[1]
        loss = torch.nn.functional.mse_loss(recon, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        recon = self.model(batch)[1]
        if self.recon_imgs is None:
            self.recon_imgs = recon.clamp(0, 1).cpu()
        loss = torch.nn.functional.mse_loss(recon, batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=5e-3, weight_decay=1e-4)
        return optimizer


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

    model = Autoencoder()

    # LightningModule化
    pl_model = AutoencoderModule(model, sample_num=args.sample_num)

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
        dirpath=args.ckpt_dir,
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
    parser.add_argument('--batch_size', type=int, default=16000, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=120000, help='Max training steps')
    parser.add_argument('--val_interval', type=int, default=50, help='Validation interval in steps')
    parser.add_argument('--sample_num', type=int, default=64, help='Number of generated images per validation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--project', type=str, default='3dshapes-autoencoder', help='wandb project name')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/autoencoder', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--strategy', type=str, default='auto', help='Distributed training strategy (ddp, ddp_spawn, etc)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training')
    parser.add_argument('--devices', type=str, default='auto', help='Number of devices (GPUs/CPUs) per node')
    args = parser.parse_args()
    main(args)
