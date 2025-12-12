import torch
import pytorch_lightning as pl
import torchvision as tv
import wandb
import normflows as nf
import math

class Autoencoder(torch.nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
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
            torch.nn.Conv2d(3*64, 3*32, kernel_size=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*32, 1, 1]),
            torch.nn.Conv2d(3*32, 3*16, kernel_size=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*16, 1, 1]),
            torch.nn.Conv2d(3*16, latent_dim, kernel_size=1),
            torch.nn.LayerNorm([latent_dim, 1, 1]),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_dim, 3*16, kernel_size=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*16, 1, 1]),
            torch.nn.ConvTranspose2d(3*16, 3*32, kernel_size=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*32, 1, 1]),
            torch.nn.ConvTranspose2d(3*32, 3*64, kernel_size=1),
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
        )

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
        num_bases=64,
        latent_dim=128,
        log_var_init=-5.0
    ):
        super().__init__()

        self.predictor = PredictorModule.load_from_checkpoint(ckpt_predictor).model.eval()

        num_layers = 12
        input_dim = latent_dim
        hidden_dim = latent_dim
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

        self.W = torch.nn.Parameter(torch.randn(num_bases, input_dim, input_dim) * (1.0 / math.sqrt(input_dim)))

        def _forward(y: torch.Tensor):
            y = y.reshape(1, -1, 1, 1)
            x = self.autoencoder.decode(y)
            _, scale, shape, _ = self.predictor.forward(x)
            y = torch.cat([scale, shape], dim=-1).squeeze(0)
            return y

        self.jacobian_fn = torch.func.vmap(torch.func.jacrev(_forward))

        self.register_buffer('eps_p', torch.tensor(1e-3))
        self.register_buffer('eps_q', torch.tensor(1e-1))

    def parameters(self, recurse = True):
        yield from self.flow.parameters(recurse)
        yield from self.autoencoder.parameters(recurse)
        yield self.W

    def encode(self, x):
        z = self.autoencoder.encode(x)
        z = z.reshape(z.size(0), -1)
        return self.flow.inverse(z)

    def decode(self, z):
        z = self.flow.forward(z)
        z = z.reshape(z.size(0), -1, 1, 1)
        x = self.autoencoder.decode(z)
        return x

    def pullback_tangent(self, z: torch.Tensor, v: torch.Tensor):
        """
        Args:
            z: (batch_size, input_dim)
            v: (batch_size, output_dim, input_dim)
        Returns:
            J: (batch_size, output_dim, input_dim)
        """
        def _forward_flow(z: torch.Tensor):
            z = z.unsqueeze(0)
            y = self.flow.forward(z)
            return y.squeeze(0)
        
        def _forward_single(z: torch.Tensor, v: torch.Tensor):
            # z: (input_dim,)
            # v: (num_bases, input_dim)
            J = torch.func.vmap(torch.func.vjp(_forward_flow, z)[1])(v)  # (num_bases, input_dim)
            return J[0]
        J = torch.func.vmap(_forward_single)(z, v)  # (batch_size, num_bases, input_dim)
        return J

    def kl_divergence(self, J_p: torch.Tensor, J_q: torch.Tensor):
        """
        Args:
            J_p: (batch_size, output_dim, input_dim)
            J_q: (batch_size, num_bases, input_dim)
        Returns:
            kl: (batch_size,)
        """

        input_dim = J_p.size(-1)
        output_dim = J_p.size(-2)
        num_bases = J_q.size(-2)

        J_p = J_p * 10

        S_p = torch.einsum('bni,bmi->bnm', J_p, J_p)                # (batch_size, output_dim, output_dim)
        S_q = torch.einsum('bni,bmi->bnm', J_q, J_q)                # (batch_size, num_bases, num_bases)
        S_pq = torch.einsum('bni,bmi->bnm', J_p, J_q)               # (batch_size, output_dim, num_bases)

        I_p = torch.eye(output_dim, device=J_p.device)              # (output_dim, output_dim)
        I_q = torch.eye(num_bases, device=J_q.device)               # (num_bases, num_bases)
        M = S_p + self.eps_p * I_p                                  # (batch_size, output_dim, output_dim)
        H = S_q + self.eps_q * I_q                                  # (batch_size, num_bases, num_bases)

        trace_pp = torch.einsum('bii->b', S_p)                      # (batch_size,)
        trace_qq = torch.einsum('bii->b', S_q)                      # (batch_size,)
        trace_pq = torch.einsum('bij,bij->b', S_pq, S_pq)           # (batch_size,)
        trace = self.eps_p * self.eps_q * input_dim \
              + self.eps_q * trace_pp \
              + self.eps_p * trace_qq \
              + trace_pq                                            # (batch_size,)

        logdet_p = torch.logdet(M) + (input_dim - output_dim) * self.eps_p.log()  # (batch_size,)
        logdet_q = torch.logdet(H) + (input_dim - num_bases) * self.eps_q.log()   # (batch_size,)
        logdet = -(logdet_p + logdet_q)

        kl = 0.5 * (trace + logdet - input_dim)

        return kl

    @torch.no_grad()
    def sample(self, num_samples):
        z = self.flow.sample(num_samples)[0]
        z = z.reshape(num_samples, -1, 1, 1)
        x = self.autoencoder.decode(z)
        return x

    def forward(self, x: torch.Tensor):
        y = self.autoencoder.encode(x)
        x_recon = self.autoencoder.decode(y)

        y = y.detach().reshape(y.size(0), -1)
        log_prob = self.flow.log_prob(y)

        z = self.flow.inverse(y)

        J_p = self.jacobian_fn(y)  # (B, output_dim, input_dim)
        J_p = self.pullback_tangent(z, J_p.detach())  # (B, output_dim, input_dim)

        J_q = torch.einsum('bi,mji->bmj', z, self.W - self.W.mT)  # (B, num_bases, input_dim)

        kl = self.kl_divergence(J_p, J_q)  # (B,)

        return x_recon, log_prob, kl


class CLNFModule(pl.LightningModule):
    def __init__(
        self, 
        ckpt_predictor,
        lr=1e-3,
        beta=10.0,
        sample_num=64,
        latent_dim=128,
        num_bases=64,
        log_eps_init=-2.0,
        log_eps_final=-6.0,
        eps_steps=5000,
    ):
        super().__init__()
        self.model = CLNF(
            ckpt_predictor,
            num_bases=num_bases,
            latent_dim=latent_dim,
        )
        self.sample_num = sample_num

        self.lr = lr
        self.beta = beta

        # self.log_eps_init = log_eps_init
        # self.log_eps_final = log_eps_final
        # self.eps_steps = eps_steps
        # self.model.eps.data.fill_(10 ** self.log_eps_init)
        
    def forward(self, x):
        return self.model(x)
    
    def on_validation_epoch_end(self):
        # 検証終了時に画像生成しwandbに記録
        with torch.no_grad():
            img = self.model.sample(self.sample_num)
            img = img.clamp(0, 1)
            grid = tv.utils.make_grid(img.cpu(), nrow=8)
            wandb_logger = self.logger
            if hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log({f"val_generated/sample": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})

    def training_step(self, batch, batch_idx):
        recon, log_prob, kl = self.model.forward(batch)
        log_prob = log_prob.mean()
        kl = kl.mean()
        nll_loss = kl - log_prob
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        loss = nll_loss + self.beta * recon_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_log_prob', log_prob, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_nll', nll_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_kl', kl, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        recon, log_prob, kl = self.model.forward(batch)
        log_prob = log_prob.mean()
        kl = kl.mean()
        nll_loss = kl - log_prob
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        loss = nll_loss + self.beta * recon_loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_log_prob', log_prob, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_nll', nll_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_kl', kl, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.3)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
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
        