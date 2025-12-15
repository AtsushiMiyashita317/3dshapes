import torch
import pytorch_lightning as pl
import torchvision as tv
import wandb
import normflows as nf
import math
import matplotlib.pyplot as plt

class Autoencoder(torch.nn.Module):
    def __init__(self, num_post_layers=3, latent_dim=12):
        super().__init__()
        layers = [
            torch.nn.Conv2d(3, 3*2, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*2, 32, 32]),
            torch.nn.Conv2d(3*2, 3*4, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*4, 16, 16]),
            torch.nn.Conv2d(3*4, 3*8, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*8, 8, 8]),
            torch.nn.Conv2d(3*8, 3*16, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*16, 4, 4]),
            torch.nn.Conv2d(3*16, 3*32, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*32, 2, 2]),
            torch.nn.Conv2d(3*32, 3*64, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*64, 1, 1]),
        ]

        current_channels = 3*64
        for _ in range(num_post_layers):
            layers.extend([
                torch.nn.Conv2d(current_channels, current_channels // 2, kernel_size=1),
                torch.nn.Softplus(),
                torch.nn.LayerNorm([current_channels // 2, 1, 1]),
            ])
            current_channels //= 2

        layers.append(torch.nn.Conv2d(current_channels, latent_dim, kernel_size=1))

        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        layers.append(torch.nn.ConvTranspose2d(latent_dim, current_channels, kernel_size=1))

        for _ in range(num_post_layers):
            layers.extend([
                torch.nn.Softplus(),
                torch.nn.LayerNorm([current_channels, 1, 1]),
                torch.nn.ConvTranspose2d(current_channels, current_channels * 2, kernel_size=1),
            ])
            current_channels *= 2

        layers.extend([
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*64, 1, 1]),
            torch.nn.ConvTranspose2d(3*64, 3*32, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*32, 2, 2]),
            torch.nn.ConvTranspose2d(3*32, 3*16, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*16, 4, 4]),
            torch.nn.ConvTranspose2d(3*16, 3*8, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*8, 8, 8]),
            torch.nn.ConvTranspose2d(3*8, 3*4, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*4, 16, 16]),
            torch.nn.ConvTranspose2d(3*4, 3*2, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*2, 32, 32]),
            torch.nn.ConvTranspose2d(3*2, 3, kernel_size=2, stride=2),
            torch.nn.Sigmoid(),
        ])

        self.decoder = torch.nn.Sequential(*layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


class AutoencoderModule(pl.LightningModule):
    def __init__(self, model, sample_num=64):
        super().__init__()
        self.model = model
        self.recon_imgs = None
            
    def on_validation_epoch_end(self):
        # 検証終了時に画像生成しwandbに記録
        if self.recon_imgs is None:
            return
        with torch.no_grad():
            img = self.recon_imgs
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
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=5e-3, weight_decay=1e-5)
        return optimizer


# class Autoencoder(torch.nn.Module):
#     def __init__(self, latent_dim=128):
#         super().__init__()
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Conv2d(3, 3*2, kernel_size=4, stride=2, padding=1),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*2, 32, 32]),
#             torch.nn.Conv2d(3*2, 3*4, kernel_size=4, stride=2, padding=1),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*4, 16, 16]),
#             torch.nn.Conv2d(3*4, 3*8, kernel_size=4, stride=2, padding=1),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*8, 8, 8]),
#             torch.nn.Conv2d(3*8, 3*16, kernel_size=4, stride=2, padding=1),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*16, 4, 4]),
#             torch.nn.Conv2d(3*16, 3*32, kernel_size=4, stride=2, padding=1),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*32, 2, 2]),
#             torch.nn.Conv2d(3*32, 3*64, kernel_size=2),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*64, 1, 1]),
#             torch.nn.Conv2d(3*64, latent_dim, kernel_size=1),
#         )

#         self.decoder = torch.nn.Sequential(
#             torch.nn.ConvTranspose2d(latent_dim, 3*64, kernel_size=1),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*64, 1, 1]),
#             torch.nn.ConvTranspose2d(3*64, 3*32, kernel_size=2),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*32, 2, 2]),
#             torch.nn.ConvTranspose2d(3*32, 3*16, kernel_size=4, stride=2, padding=1),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*16, 4, 4]),
#             torch.nn.ConvTranspose2d(3*16, 3*8, kernel_size=4, stride=2, padding=1),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*8, 8, 8]),
#             torch.nn.ConvTranspose2d(3*8, 3*4, kernel_size=4, stride=2, padding=1),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*4, 16, 16]),
#             torch.nn.ConvTranspose2d(3*4, 3*2, kernel_size=4, stride=2, padding=1),
#             torch.nn.Softplus(),
#             torch.nn.LayerNorm([3*2, 32, 32]),
#             torch.nn.ConvTranspose2d(3*2, 3, kernel_size=4, stride=2, padding=1),
#             torch.nn.Sigmoid(),
#         )

#     def encode(self, x):
#         z = self.encoder(x)
#         return z

#     def decode(self, z):
#         z = self.decoder(z)
#         return z

#     def forward(self, x):
#         z = self.encode(x)
#         x_recon = self.decode(z)
#         return z, x_recon


# class AutoencoderModule(pl.LightningModule):
#     def __init__(self, sample_num=64):
#         super().__init__()
#         self.model = Autoencoder()
#         self.sample_num = sample_num
#         self.recon_imgs = None
        
#     def on_validation_epoch_end(self):
#         # 検証終了時に画像生成しwandbに記録
#         if self.recon_imgs is None:
#             return
#         with torch.no_grad():
#             img = self.recon_imgs[:self.sample_num]
#             grid = tv.utils.make_grid(img.cpu(), nrow=8)
#             wandb_logger = self.logger
#             if hasattr(wandb_logger, "experiment"):
#                 wandb_logger.experiment.log({f"val_generated/epoch_{self.current_epoch}": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})
#             self.recon_imgs = None

#     def encode(self, x):
#         return self.model.encode(x)

#     def decode(self, z):
#         return self.model.decode(z)

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         recon = self.model(batch)[1]
#         loss = torch.nn.functional.mse_loss(recon, batch)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         recon = self.model(batch)[1]
#         if self.recon_imgs is None:
#             self.recon_imgs = recon.clamp(0, 1).cpu()
#         loss = torch.nn.functional.mse_loss(recon, batch)
#         self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adamax(self.model.parameters(), lr=5e-3, weight_decay=1e-5)
#         return optimizer


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


class PredictorModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Predictor()
            
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


class NFModule(pl.LightningModule):
    def __init__(self, sample_num=64, beta=10.0):
        super().__init__()
        self.sample_num = sample_num
        self.beta = beta

        num_layers = 12
        input_dim = 128
        hidden_dim = 128
        half_dim = input_dim // 2

        self.autoencoder = Autoencoder(input_dim)

        base = nf.distributions.base.DiagGaussian(input_dim, trainable=False)

        flows = []
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([half_dim, hidden_dim, hidden_dim, input_dim], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(input_dim, mode='swap'))

        self.flow = nf.NormalizingFlow(base, flows)

    def on_validation_epoch_end(self):
        # 検証終了時に画像生成しwandbに記録
        with torch.no_grad():
            z = self.flow.sample(self.sample_num)[0]
            z = z.view(self.sample_num, -1, 1, 1)
            img = self.autoencoder.decode(z)
            img = img.clamp(0, 1)
            grid = tv.utils.make_grid(img.cpu(), nrow=8)
            wandb_logger = self.logger
            if hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log({f"val_generated/sample": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})

    def encode(self, x):
        z = self.autoencoder.encode(x)
        z = z.reshape(z.size(0), -1)
        return self.flow.inverse(z)
    
    def decode(self, z):
        z = self.flow.forward(z)
        z = z.reshape(z.size(0), -1, 1, 1)
        return self.autoencoder.decode(z)
    
    def sample(self, num_samples):
        z = self.flow.sample(num_samples)[0]
        z = z.reshape(num_samples, -1, 1, 1)
        return self.autoencoder.decode(z)

    def forward(self, x):
        z, recon = self.autoencoder(x)
        z = z.reshape(z.size(0), -1)
        return self.flow(z)

    def training_step(self, batch, batch_idx):
        z, recon = self.autoencoder(batch)
        z = z.reshape(z.size(0), -1)
        nll = self.flow.forward_kld(z.detach())
        recons = torch.nn.functional.mse_loss(recon, batch)
        loss = self.beta * recons + nll
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_nll', nll, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recons', recons, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z, recon = self.autoencoder(batch)
        z = z.reshape(z.size(0), -1)
        nll = self.flow.forward_kld(z)
        recons = torch.nn.functional.mse_loss(recon, batch)
        loss = self.beta * recons + nll
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_nll', nll, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recons', recons, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(list(self.autoencoder.parameters()) + list(self.flow.parameters()), lr=1e-3, weight_decay=1e-5)
        return optimizer


class CLNF(torch.nn.Module):
    def __init__(
        self, 
        ckpt_predictor,
        num_bases=66,
        latent_dim=12,
        autoencoder_layers=3,
        flow_layers=24,
        flow_hidden_dim=192,
        log_var_init=0.0,
        eps_p=1e-3,
        eps_q=1e-1,
    ):
        super().__init__()

        self.predictor = PredictorModule.load_from_checkpoint(ckpt_predictor).model.eval()

        self.latent_dim = latent_dim
        self.num_bases = num_bases

        self.autoencoder = Autoencoder(num_post_layers=autoencoder_layers, latent_dim=latent_dim)

        input_dim = latent_dim
        hidden_dim = flow_hidden_dim
        half_dim = input_dim // 2


        base = nf.distributions.base.DiagGaussian(input_dim, trainable=False)

        flows = []
        for i in range(flow_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([half_dim, hidden_dim, hidden_dim, input_dim], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(input_dim, mode='swap'))

        self.flow = nf.NormalizingFlow(base, flows)

        self.W = torch.nn.Parameter(torch.randn(num_bases, input_dim, input_dim) * (1.0 / math.sqrt(input_dim)))

        def _forward(y: torch.Tensor):
            y = y.reshape(1, -1, 1, 1)
            x = self.autoencoder.decode(y)
            _, scale, shape, _ = self.predictor.forward(x)
            y = torch.cat([scale, shape], dim=-1).squeeze(0)
            return y

        self.jacobian_fn = torch.func.vmap(torch.func.jacrev(_forward))

        self.gate = torch.nn.Parameter(torch.randn(latent_dim))
        self.log_var_null = torch.nn.Parameter(torch.tensor(log_var_init))
        self.log_var_image = torch.nn.Parameter(torch.tensor(log_var_init))

        self.register_buffer('eps_p', torch.tensor(eps_p))
        self.register_buffer('eps_q', torch.tensor(eps_q))

    def parameters(self, recurse = True):
        yield from self.flow.parameters(recurse)
        yield from self.autoencoder.parameters(recurse)
        yield self.W
        yield self.gate
        yield self.log_var_null
        yield self.log_var_image

    def encode(self, x):
        z = self.autoencoder.encode(x)
        z = z.reshape(z.size(0), -1)
        return self.flow.inverse(z)

    def decode(self, z):
        z = self.flow.forward(z)
        z = z.reshape(z.size(0), -1, 1, 1)
        x = self.autoencoder.decode(z)
        return x

    def _predict(self, x: torch.Tensor):
        # x: (3, 64, 64)
        # Returns: (output_dim,)
        x = x.unsqueeze(0)
        _, scale, shape, _ = self.predictor.forward(x)
        out = torch.cat([scale, shape], dim=-1).squeeze(0)
        return out

    def _decode(self, y: torch.Tensor):
        # y: (input_dim,)
        # Returns: (3, 64, 64)
        y = y.reshape(1, -1, 1, 1)
        x = self.autoencoder.decode(y)
        return x.squeeze(0)
    
    def _flow(self, z: torch.Tensor):
        # z: (input_dim,)
        # Returns: (input_dim,)
        z = z.unsqueeze(0)
        y = self.flow.forward(z)
        return y.squeeze(0)

    def sample_cotangent(self, x: torch.Tensor, cv: torch.Tensor):
        def _sample_cotangent_single(x: torch.Tensor, cv: torch.Tensor):
            # x: (3, 64, 64)
            # cv: (output_dim,)
            # Returns: (3, 64, 64)

            _, vjp_fn = torch.func.vjp(self._predict, x)

            cv = vjp_fn(cv)[0]

            return cv

        cv = torch.func.vmap(_sample_cotangent_single)(x, cv)

        return cv

    def encode_cotangent(self, y: torch.Tensor, cv1: torch.Tensor, cv2: torch.Tensor):
        def _encode_cotangent_single(y: torch.Tensor, cv1: torch.Tensor, cv2: torch.Tensor):

            _, vjp_fn = torch.func.vjp(self._decode, y)

            cv1 = vjp_fn(cv1)[0]
            cv2 = vjp_fn(cv2)[0]

            return cv1, cv2

        cv1, cv2 = torch.func.vmap(_encode_cotangent_single)(y, cv1, cv2)

        return cv1, cv2 

    def pullback_cotangent(self, z: torch.Tensor, cv1: torch.Tensor, cv2: torch.Tensor):
        def _pullback_cotangent_single(z: torch.Tensor, cv1: torch.Tensor, cv2: torch.Tensor):

            _, vjp_fn = torch.func.vjp(self._flow, z)

            cv1 = vjp_fn(cv1)[0]
            cv2 = vjp_fn(cv2)[0]

            return cv2, cv1

        cv1, cv2 = torch.func.vmap(_pullback_cotangent_single)(z, cv1, cv2)

        return cv1, cv2

    def log_prob(self, cv_predictor: torch.Tensor, cv_decoder: torch.Tensor, z: torch.Tensor):
        """
        Args:
            cv_predictor: (batch_size, input_dim)
            cv_decoder: (batch_size, input_dim)
            z: (batch_size, input_dim)
        Returns:
            log_prob: (batch_size,)
        """

        input_dim = cv_predictor.size(-1)

        cv_predictor = cv_predictor + torch.randn_like(cv_predictor) * self.eps_p ** 0.5
        cv_decoder = cv_decoder + torch.randn_like(cv_decoder) * self.eps_p ** 0.5

        gate = torch.sigmoid(self.gate)                                     # (input_dim,)
        null_gate = (1.0 - gate.unsqueeze(1)) * (1.0 - gate.unsqueeze(0))   # (input_dim, input_dim)
        sym_gate = gate.unsqueeze(1) * gate.unsqueeze(0)                    # (input_dim, input_dim)

        I = torch.eye(input_dim, device=z.device)                           # (input_dim, input_dim)
        W = (self.W - self.W.mT) * sym_gate                                 # (num_bases, input_dim, input_dim)

        J = torch.einsum('bm,inm->bin', z, W)                               # (B, num_bases, input_dim)
        S = torch.einsum('bin,bim->bnm', J, J)                              # (batch_size, input_dim, input_dim)
        M_sym = S + self.eps_q * I * sym_gate                               # (batch_size, input_dim, input_dim)

        M_null = self.log_var_null.neg().exp() * I * null_gate              # (input_dim, input_dim)
        M_image = self.log_var_image.neg().exp() * I * sym_gate             # (input_dim, input_dim)

        L_predictor = M_null + M_sym                                        # (batch_size, input_dim, input_dim)
        L_decoder = M_null + M_image                                        # (input_dim, input_dim)

        trace_predictor = torch.einsum('bi,bij,bj->b', cv_predictor, L_predictor, cv_predictor)  # (batch_size,)
        trace_decoder = torch.einsum('bi,ij,bj->b', cv_decoder, L_decoder, cv_decoder)          # (batch_size,)

        logdet_predictor = torch.logdet(L_predictor)  # (batch_size,)
        logdet_decoder = torch.logdet(L_decoder)      # (batch_size,)

        log_prob_predictor = 0.5 * (logdet_predictor - trace_predictor - input_dim * math.log(2 * math.pi))  # (batch_size,)
        log_prob_decoder = 0.5 * (logdet_decoder - trace_decoder - input_dim * math.log(2 * math.pi))          # (batch_size,)

        return log_prob_predictor, log_prob_decoder

    @torch.no_grad()
    def sample(self, num_samples):
        z = self.flow.sample(num_samples)[0]
        z = z.reshape(num_samples, -1, 1, 1)
        x = self.autoencoder.decode(z)
        return x

    def forward(self, img: torch.Tensor):
        y = self.autoencoder.encode(img)
        x = self.autoencoder.decode(y)

        y = y.detach().reshape(y.size(0), -1)

        z, log_det = self.flow.inverse_and_log_det(y)
        log_prob_data = self.flow.q0.log_prob(z) + log_det

        cv_predictor = torch.randn(z.size(0), 5, device=z.device)
        cv_predictor = self.sample_cotangent(x, cv_predictor)  # (B, 3, 64, 64)

        cv_decoder = torch.randn(z.size(0), 3, 64, 64, device=z.device)
        cv_predictor, cv_decoder = self.encode_cotangent(y, cv_predictor, cv_decoder)  # (B, input_dim), (B, input_dim)
        cv_predictor = cv_predictor.detach()
        cv_decoder = cv_decoder.detach()
        cv_decoder, cv_predictor = self.pullback_cotangent(z, cv_decoder, cv_predictor)  # (B, input_dim), (B, input_dim)

        norm_predictor = cv_predictor.square().mean().sqrt()
        norm_decoder = cv_decoder.square().mean().sqrt()
        cv_predictor = cv_predictor / norm_predictor
        cv_decoder = cv_decoder / norm_decoder

        log_prob_predictor, log_prob_decoder = self.log_prob(cv_predictor, cv_decoder, z)
        log_prob_predictor = log_prob_predictor - log_det - torch.log(norm_predictor).mul(z.size(-1))
        log_prob_decoder = log_prob_decoder - log_det - torch.log(norm_decoder).mul(z.size(-1))

        return x, log_prob_data, log_prob_predictor, log_prob_decoder


class CLNFModule(pl.LightningModule):
    def __init__(
        self, 
        ckpt_predictor,
        lr=1e-3,
        beta=10.0,
        sample_num=64,
        num_bases=66,
        latent_dim=12,
        autoencoder_layers=3,
        flow_layers=24,
        flow_hidden_dim=192,
        log_var_init=0.0,
        eps_p=1e-3,
        eps_q=1e-1,
    ):
        super().__init__()
        self.model = CLNF(
            ckpt_predictor,
            num_bases=num_bases,
            latent_dim=latent_dim,
            autoencoder_layers=autoencoder_layers,
            flow_layers=flow_layers,
            flow_hidden_dim=flow_hidden_dim,
            log_var_init=log_var_init,
            eps_p=eps_p,
            eps_q=eps_q,
        )
        self.sample_num = sample_num

        self.lr = lr
        self.beta = beta

    def forward(self, x):
        return self.model(x)
    
    def on_validation_epoch_end(self):
        # 検証終了時に画像生成しwandbに記録
        if hasattr(self.logger, "experiment"):
            with torch.no_grad():
                img = self.model.sample(self.sample_num)
                img = img.clamp(0, 1)
                grid = tv.utils.make_grid(img.cpu(), nrow=8)
                self.logger.experiment.log({f"val_generated/sample": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                gate = torch.sigmoid(self.model.gate).sort().values.detach().cpu().numpy()
                ax.plot(gate)
                ax.set_title("Gate Values")
                self.logger.experiment.log({f"val_generated/gate": wandb.Image(fig, caption=f"epoch {self.current_epoch}")})
                plt.close(fig)

    def training_step(self, batch, batch_idx):
        recon, log_prob_data, log_prob_predictor, log_prob_decoder = self.model.forward(batch)
        log_prob_data = log_prob_data.mean()
        log_prob_predictor = log_prob_predictor.mean()
        log_prob_decoder = log_prob_decoder.mean()
        log_prob = log_prob_data + log_prob_predictor + log_prob_decoder
        nll_loss = -log_prob
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        loss = nll_loss + self.beta * recon_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_nll', nll_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_log_prob_data', log_prob_data, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_log_prob_predictor', log_prob_predictor, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_log_prob_decoder', log_prob_decoder, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        recon, log_prob_data, log_prob_predictor, log_prob_decoder = self.model.forward(batch)
        log_prob_data = log_prob_data.mean()
        log_prob_predictor = log_prob_predictor.mean()
        log_prob_decoder = log_prob_decoder.mean()
        log_prob = log_prob_data + log_prob_predictor + log_prob_decoder
        nll_loss = -log_prob
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        loss = nll_loss + self.beta * recon_loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_nll', nll_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_log_prob_data', log_prob_data, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_log_prob_predictor', log_prob_predictor, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_log_prob_decoder', log_prob_decoder, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=3e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.3)

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'interval': 'epoch',
            #     'frequency': 1,
            # }
        }


# class CLNF(torch.nn.Module):
#     def __init__(
#         self, 
#         ckpt_autoencoder,
#         ckpt_predictor,
#         num_bases,
#         seed_scale=1.0,
#     ):
#         super().__init__()
#         self.seed_scale = seed_scale
#         self.register_buffer('eps', torch.tensor(1.0))

#         self.autoencoder = AutoencoderModule.load_from_checkpoint(ckpt_autoencoder).model.eval()
#         self.predictor = PredictorModule.load_from_checkpoint(ckpt_predictor).model.eval()

#         num_layers = 6
#         input_dim = 3*64
#         hidden_dim = 3*64
#         half_dim = input_dim // 2

#         self.base = nf.distributions.base.DiagGaussian(input_dim, trainable=False)

#         flows = []
#         for i in range(num_layers):
#             # Neural network with two hidden layers having 64 units each
#             # Last layer is initialized by zeros making training more stable
#             param_map = nf.nets.MLP([half_dim, hidden_dim, hidden_dim, input_dim], init_zeros=True)
#             # Add flow layer
#             flows.append(nf.flows.AffineCouplingBlock(param_map))
#             # Swap dimensions
#             flows.append(nf.flows.Permute(input_dim, mode='swap'))

#         self.flow = nf.NormalizingFlow(self.base, flows)

#         # self.W = torch.nn.Parameter(torch.randn(num_bases, input_dim, input_dim) / (input_dim ** 0.5))
#         self.log_sigma = torch.nn.Parameter(torch.tensor(0.0))

#         self.coef_net = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, input_dim),
#             torch.nn.ReLU(),
#             torch.nn.LayerNorm(input_dim),
#             torch.nn.Linear(input_dim, input_dim),
#             torch.nn.ReLU(),
#             torch.nn.LayerNorm(input_dim),
#             torch.nn.Linear(input_dim, input_dim),
#             torch.nn.ReLU(),
#             torch.nn.LayerNorm(input_dim),
#             torch.nn.Linear(input_dim, input_dim),
#             torch.nn.ReLU(),
#             torch.nn.LayerNorm(input_dim),
#             torch.nn.Linear(input_dim, num_bases),
#         )

#         self.vec_net = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, input_dim),
#             torch.nn.ReLU(),
#             torch.nn.LayerNorm(input_dim),
#             torch.nn.Linear(input_dim, input_dim),
#             torch.nn.ReLU(),
#             torch.nn.LayerNorm(input_dim),
#             torch.nn.Linear(input_dim, input_dim),
#             torch.nn.ReLU(),
#             torch.nn.LayerNorm(input_dim),
#             torch.nn.Linear(input_dim, input_dim),
#             torch.nn.ReLU(),
#             torch.nn.LayerNorm(input_dim),
#             torch.nn.Linear(input_dim, num_bases * input_dim),
#         )

#     def parameters(self, recurse = True):
#         yield from self.flow.parameters(recurse)
#         yield from self.coef_net.parameters(recurse)
#         yield from self.vec_net.parameters(recurse)
#         # yield self.W
#         #yield self.log_sigma

#     def encode(self, x):
#         z = self.autoencoder.encode(x)
#         z = z.reshape(z.size(0), -1)
#         return self.flow.inverse(z)

#     def decode(self, z):
#         z = self.flow.forward(z)
#         z = z.reshape(z.size(0), -1, 1, 1)
#         with torch.no_grad():
#             x = self.autoencoder.decode(z)
#         return x
    
#     def sample(self, num_samples):
#         z = self.flow.sample(num_samples)[0]
#         z = z.reshape(num_samples, -1, 1, 1)
#         with torch.no_grad():
#             x = self.autoencoder.decode(z)
#         return x

#     @torch.no_grad()
#     def sample_cotangent(self, x: torch.Tensor):
#         def _sample_cotangent_single(x: torch.Tensor, cv: torch.Tensor):
#             x = x.unsqueeze(0)
#             cv = cv.unsqueeze(0)

#             cv_color = torch.zeros_like(cv[:, :6].view(-1, 3, 2))
#             cv_scale = cv[:, 6:7]
#             cv_shape = cv[:, 7:11]
#             cv_orientation = torch.zeros_like(cv[:, 11])

#             cv = (cv_color, cv_scale, cv_shape, cv_orientation)

#             _, vjp_fn = torch.func.vjp(self.predictor.forward, x)

#             cv = vjp_fn(cv)[0]

#             return cv[0].squeeze(0)
        
#         batch_size = x.size(0)
#         cv = torch.randn(batch_size, 12, device=x.device) * self.seed_scale
#         cv = torch.func.vmap(_sample_cotangent_single)(x, cv)

#         return cv
    
#     @torch.no_grad()
#     def encode_cotangent(self, z: torch.Tensor, cv: torch.Tensor):
#         def _encode_cotangent_single(z: torch.Tensor, cv: torch.Tensor):
#             z = z.unsqueeze(0)
#             cv = cv.unsqueeze(0)

#             _, vjp_fn = torch.func.vjp(self.autoencoder.decode, z)

#             cv = vjp_fn(cv)[0]

#             return cv.squeeze(0)

#         cv = torch.func.vmap(_encode_cotangent_single)(z, cv)
#         return cv

#     def pullback_cotangent(self, z: torch.Tensor, cv: torch.Tensor):
#         def _pullback_cotangent_single(z: torch.Tensor, cv: torch.Tensor):
#             z = z.unsqueeze(0)
#             cv = cv.unsqueeze(0)

#             _, vjp_fn = torch.func.vjp(self.flow.forward, z)

#             cv = vjp_fn(cv)[0]

#             return cv.squeeze(0)

#         cv = torch.func.vmap(_pullback_cotangent_single)(z, cv)

#         return cv
    
#     def pushforward_tangent(self, z: torch.Tensor, tv: torch.Tensor):
#         # z: (B, D)
#         # tv: (B, K, D)
#         def _pushforward_tangent_single(z: torch.Tensor, tv: torch.Tensor):
#             # z: (D,)
#             # tv: (K, D)
#             z = z.unsqueeze(0)

#             def _pushforward_tangent_inner(tv: torch.Tensor):
#                 # tv: (D,)
#                 tv = tv.unsqueeze(0)
#                 tv = torch.func.jvp(self.flow.forward, (z,), (tv,))[1]
#                 return tv.squeeze(0)
            
#             jvp_fn_multi = torch.func.vmap(_pushforward_tangent_inner)
#             tv = jvp_fn_multi(tv)

#             return tv

#         tv = torch.func.vmap(_pushforward_tangent_single)(z, tv)
#         return tv

#     def log_prob(self, z: torch.Tensor, tv: torch.Tensor, cv: torch.Tensor):
#         # zW = torch.einsum('bi,mij->bmj', z, self.W)
#         # zW = zW / zW.square().sum(dim=-1, keepdim=True).mean(dim=-2, keepdim=True).sqrt()
#         # zWz = torch.einsum('bmj,bj->bm', zW, z)

#         zW = tv
#         cv = cv + self.eps.sqrt() * torch.randn_like(cv)

#         d = self.coef_net.forward(z)
#         zWWz = torch.einsum('bmj,bnj->bmn', zW, zW)
#         S = self.eps * d.exp().diag_embed() + zWWz

#         zWv = torch.einsum('bmj,bj->bm', zW, cv)
#         bilinear = torch.einsum('bm,bmn,bn->b', zWv, torch.inverse(S), zWv)
#         squared = torch.einsum('bj,bj->b', cv, cv)
#         total = (squared - bilinear) / self.eps

#         logdet = torch.logdet(S) - d.sum(dim=-1) + self.eps.log().mul(tv.size(-1) - tv.size(-2))

#         log_prob_v = -0.5 * (total + logdet + cv.size(-1) * math.log(2 * math.pi))
#         log_prob_z = self.flow.log_prob(z)

#         return log_prob_z, log_prob_v, d
    
#     def forward(self, x: torch.Tensor):
#         z = self.autoencoder.encode(x)
#         x_recon = self.autoencoder.decode(z)
#         cv = self.sample_cotangent(x_recon)
#         cv = self.encode_cotangent(z, cv)
#         z = z.reshape(z.size(0), -1)
#         cv = cv.reshape(cv.size(0), -1)
#         z = self.flow.inverse(z)
#         tv = self.vec_net.forward(z).view(z.size(0), -1, cv.size(-1))
#         tv = self.pushforward_tangent(z, tv)
#         log_prob_z, log_prob_v, d = self.log_prob(z, tv, cv)
#         return log_prob_z, log_prob_v, d, cv, self.log_sigma


# class CLNFModule(pl.LightningModule):
#     def __init__(
#         self, 
#         ckpt_autoencoder,
#         ckpt_predictor,
#         num_bases,
#         seed_scale=1.0,
#         eps_init=1e-0,
#         eps_final=1e-10,
#         eps_steps=1000,
#         sample_num=64
#     ):
#         super().__init__()
#         self.model = CLNF(
#             ckpt_autoencoder,
#             ckpt_predictor,
#             num_bases,
#             seed_scale,
#         )
#         self.sample_num = sample_num

#         # スケジューリングするスカラーパラメータ
#         self.eps_init = eps_init
#         self.eps_final = eps_final
#         self.eps_steps = eps_steps
        
#     def on_train_epoch_start(self):
#         # 線形スケジューリング例（他のスケジューラも可）
#         progress = min(self.current_epoch / max(1, self.eps_steps), 1.0)
#         new_value = self.eps_init + (self.eps_final - self.eps_init) * progress
#         self.model.eps.data.fill_(new_value)
#         self.log('scheduled_scalar', self.model.eps, on_epoch=True, prog_bar=True)

#     def forward(self, x):
#         return self.model(x)
    
#     def on_validation_epoch_end(self):
#         # 検証終了時に画像生成しwandbに記録
#         with torch.no_grad():
#             img = self.model.sample(self.sample_num)
#             img = img.clamp(0, 1)
#             grid = tv.utils.make_grid(img.cpu(), nrow=8)
#             wandb_logger = self.logger
#             if hasattr(wandb_logger, "experiment"):
#                 wandb_logger.experiment.log({f"val_generated/sample": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})

#     def training_step(self, batch, batch_idx):
#         log_prob_z, log_prob_v, d, v, log_sigma = self.model.forward(batch)
#         log_prob_z = log_prob_z.mean()
#         log_prob_v = log_prob_v.mean()
#         log_prob = log_prob_z + 0.0 * log_prob_v
#         loss = -log_prob
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         self.log('train_log_prob_z', log_prob_z, on_step=True, on_epoch=True, prog_bar=True)
#         self.log('train_log_prob_v', log_prob_v, on_step=True, on_epoch=True, prog_bar=True)
#         self.log('train_log_sigma', log_sigma, on_step=True, on_epoch=True, prog_bar=False)
#         # self.log('train_W_std', W.std(), on_step=True, on_epoch=True, prog_bar=False)
#         self.log('train_v_std', v.std(), on_step=True, on_epoch=True, prog_bar=False)
#         self.log('train_d_mean', d.mean(), on_step=True, on_epoch=True, prog_bar=False)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         log_prob_z, log_prob_v, d, v, log_sigma = self.model.forward(batch)
#         log_prob_z = log_prob_z.mean()
#         log_prob_v = log_prob_v.mean()
#         log_prob = log_prob_z + 0.0 * log_prob_v
#         loss = -log_prob
#         self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('val_log_prob_z', log_prob_z, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('val_log_prob_v', log_prob_v, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('val_log_sigma', log_sigma, on_step=False, on_epoch=True, prog_bar=False)
#         # self.log('val_W_std', W.std(), on_step=False, on_epoch=True, prog_bar=False)
#         self.log('val_v_std', v.std(), on_step=False, on_epoch=True, prog_bar=False)
#         self.log('val_d_mean', d.mean(), on_step=False, on_epoch=True, prog_bar=False)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adamax(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
#         return optimizer
        