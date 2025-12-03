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


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3*2, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*2, 64, 64]),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(3*2, 3*4, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*4, 32, 32]),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(3*4, 3*8, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*8, 16, 16]),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(3*8, 3*16, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*16, 8, 8]),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(3*16, 3*32, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*32, 4, 4]),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(3*32, 3*64, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*64, 2, 2]),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten()
        )

        self.color_predictor = torch.nn.Sequential(
            torch.nn.Linear(3*64, 128),
            torch.nn.Softplus(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 6)
        )

        self.scale_predictor = torch.nn.Sequential(
            torch.nn.Linear(3*64, 128),
            torch.nn.Softplus(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 1)
        )

        self.shape_predictor = torch.nn.Sequential(
            torch.nn.Linear(3*64, 128),
            torch.nn.Softplus(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 4)
        )

        self.orientation_predictor = torch.nn.Sequential(
            torch.nn.Linear(3*64, 128),
            torch.nn.Softplus(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.feature_extractor.forward(x)
        color = self.color_predictor.forward(features).reshape(-1, 3, 2)
        color = color / torch.norm(color, dim=-1, keepdim=True)
        scale = self.scale_predictor.forward(features)
        shape = self.shape_predictor.forward(features).softmax(dim=-1)
        orientation = self.orientation_predictor.forward(features).squeeze(-1)
        return color, scale, shape, orientation


class Dataset3DShapes(torch.utils.data.Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = self.images[self.indices[idx]]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img / 255.0

        label = self.labels[self.indices[idx]]
        label = torch.from_numpy(label).float()
        color = label[0:3].mul(2 * torch.pi)
        color = torch.stack([color.sin(), color.cos()], dim=-1)  # (3, 2)
        scale = label[3:4]
        shape = label[4].long()
        orientation = label[5].div(30)
        return img, color, scale, shape, orientation


class PredictorModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
            
    def forward(self, x):
        return self.model(x)
    
    def criterion(self, preds, targets):
        color_pred, scale_pred, shape_pred, orientation_pred = preds
        color_target, scale_target, shape_target, orientation_target = targets

        color_loss = torch.nn.functional.mse_loss(color_pred, color_target)
        scale_loss = torch.nn.functional.mse_loss(scale_pred, scale_target)
        shape_loss = torch.nn.functional.cross_entropy(shape_pred.add(1).log(), shape_target)
        orientation_loss = torch.nn.functional.mse_loss(orientation_pred, orientation_target)

        shape_acc = (shape_pred.argmax(dim=-1) == shape_target).float().mean()

        total_loss = color_loss + scale_loss + shape_loss + orientation_loss
        return total_loss, color_loss, scale_loss, shape_loss, orientation_loss, shape_acc

    def training_step(self, batch, batch_idx):
        imgs, color_target, scale_target, shape_target, orientation_target = batch
        color, scale, shape, orientation = self.model(imgs)
        loss, color_loss, scale_loss, shape_loss, orientation_loss, shape_acc = self.criterion(
            (color, scale, shape, orientation),
            (color_target, scale_target, shape_target, orientation_target)
        )
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_color_loss', color_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_scale_loss', scale_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_shape_loss', shape_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_shape_acc', shape_acc, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_orientation_loss', orientation_loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, color_target, scale_target, shape_target, orientation_target = batch
        color, scale, shape, orientation = self.model(imgs)
        loss, color_loss, scale_loss, shape_loss, orientation_loss, shape_acc = self.criterion(
            (color, scale, shape, orientation),
            (color_target, scale_target, shape_target, orientation_target)
        )
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_color_loss', color_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_scale_loss', scale_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_shape_loss', shape_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_shape_acc', shape_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_orientation_loss', orientation_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss 

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=1e-3, weight_decay=1e-1)
        return optimizer

def main(args):
    # load dataset
    dataset = h5py.File('3dshapes.h5', 'r')
    print(dataset.keys())
    images = dataset['images'][:]
    labels = dataset['labels'][:]
    n_samples = images.shape[0]

    # データ分割（8:2）
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(n_samples * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]

    # Set up model
    torch.manual_seed(0)

    model = Predictor()

    # LightningModule化
    pl_model = PredictorModule(model)

    # DataLoader
    train_data = Dataset3DShapes(images, labels, train_idx)
    val_data = Dataset3DShapes(images, labels, val_idx)
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
        num_nodes=args.num_nodes,
        devices=args.devices,
        strategy=args.strategy,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True
    )
    trainer.fit(pl_model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4000, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=120000, help='Max training steps')
    parser.add_argument('--val_interval', type=int, default=1000, help='Validation interval in steps')
    parser.add_argument('--sample_num', type=int, default=64, help='Number of generated images per validation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--project', type=str, default='3dshapes-predictor', help='wandb project name')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--strategy', type=str, default='auto', help='Distributed training strategy (ddp, ddp_spawn, etc)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training')
    parser.add_argument('--devices', type=str, default='auto', help='Number of devices (GPUs/CPUs) per node')
    args = parser.parse_args()
    main(args)