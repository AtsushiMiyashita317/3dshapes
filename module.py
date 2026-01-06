import torch
import pytorch_lightning as pl
import torchvision as tv
import wandb
import normflows as nf
import math
from matplotlib import pyplot as plt

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
        num_bases_sym=66,
        num_bases_null=66,
        latent_dim=12,
        autoencoder_layers=3,
        flow_layers=24,
        flow_hidden_dim=192,
        eps_p=1e-3,
        eps_q=1e-1,
        eps_r=1e-1,
        scale_map="exp_clamp",
    ):
        super().__init__()

        self.predictor = PredictorModule.load_from_checkpoint(ckpt_predictor).model.eval()

        self.latent_dim = latent_dim
        self.num_bases_sym = num_bases_sym
        self.num_bases_null = num_bases_null

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
            flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map=scale_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(input_dim, mode='swap'))

        self.flow = nf.NormalizingFlow(base, flows)

        self.W_sym = torch.nn.Parameter(torch.randn(num_bases_sym, input_dim, input_dim) * (1.0 / math.sqrt(input_dim)))
        self.W_null = torch.nn.Parameter(torch.randn(num_bases_null, input_dim, input_dim) * (1.0 / math.sqrt(input_dim)))

        self.log_var_sym = torch.nn.Parameter(torch.zeros(input_dim))
        self.log_var_null = torch.nn.Parameter(torch.zeros(input_dim))

        self.register_buffer('eps_p', torch.tensor(eps_p))
        self.register_buffer('eps_q', torch.tensor(eps_q))
        self.register_buffer('eps_r', torch.tensor(eps_r))
        self.register_buffer('var_sym', torch.tensor(-1.0))
        self.register_buffer('var_null', torch.tensor(-1.0))

    def parameters(self, recurse = True):
        yield from self.flow.parameters(recurse)
        yield from self.autoencoder.parameters(recurse)
        yield self.W_sym
        yield self.W_null
        yield self.log_var_sym
        yield self.log_var_null

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
        # Returns: (output_dim,)
        y = y.reshape(1, -1, 1, 1)
        x = self.autoencoder.decode(y)
        return x.squeeze(0)
    
    def sample_cotangent(self, x: torch.Tensor, cv: torch.Tensor):
        def _sample_cotangent_single(x: torch.Tensor, cv: torch.Tensor):
            # x: (3, 64, 64)
            # cv: (output_dim,)
            # Returns: (input_dim,)

            _, vjp_fn = torch.func.vjp(self._predict, x)

            cv = vjp_fn(cv)[0]

            return cv

        cv = torch.func.vmap(_sample_cotangent_single)(x, cv)

        return cv
    
    def encode_cotangent(self, y: torch.Tensor, cv: torch.Tensor):
        def _encode_cotangent_single(y: torch.Tensor, cv: torch.Tensor):
            # y: (input_dim,)
            # cv: (output_dim,)
            # Returns: (input_dim,)

            _, vjp_fn = torch.func.vjp(self._decode, y)

            cv = vjp_fn(cv)[0]

            return cv

        cv = torch.func.vmap(_encode_cotangent_single)(y, cv)

        return cv

    def pullback_cotangent(self, z: torch.Tensor, cv: torch.Tensor):
        def _pullback_cotangent_single(z: torch.Tensor, cv: torch.Tensor):
            z = z.unsqueeze(0)
            cv = cv.unsqueeze(0)

            _, vjp_fn = torch.func.vjp(self.flow.forward, z)

            cv = vjp_fn(cv)[0]

            return cv.squeeze(0)

        cv = torch.func.vmap(_pullback_cotangent_single)(z, cv)

        return cv

    def log_prob(self, cv: torch.Tensor, J: torch.Tensor, log_var: torch.Tensor, eps: torch.Tensor):
        """
        Args:
            cv: (batch_size, input_dim)
            J: (batch_size, num_bases, input_dim)
        Returns:
            log_prob: (batch_size,)
        """

        input_dim = J.size(-1)
        num_bases = J.size(-2)

        S_q = torch.einsum('bin,bim->bnm', J, J)                        # (batch_size, input_dim, input_dim)
        # S_q = S_q / num_bases
        # D = log_var.neg().exp()                                   # (input_dim,)

        I_q = torch.eye(input_dim, device=J.device)                       # (input_dim, input_dim)
        norm = S_q.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
        H = S_q + eps * norm.unsqueeze(-1).unsqueeze(-1) * I_q
        M = self.eps_p + torch.einsum('bi,bi->b', cv, cv)                   # (batch_size,)

        cv = cv + self.eps_p.sqrt() * torch.randn_like(cv)
        trace = torch.einsum('bi,bij,bj->b', cv, H, cv)                # (batch_size,)

        L_H = torch.linalg.cholesky(H)
        logdet = 2 * torch.log(torch.diagonal(L_H, dim1=-2, dim2=-1)).sum(-1)
        logdet = logdet + torch.log(M) + (input_dim - 1) * torch.log(self.eps_p)  # (batch_size,)

        log_prob = 0.5 * (logdet - trace - input_dim * math.log(2 * math.pi))  # (batch_size,)

        return log_prob

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
        z, logdet = self.flow.inverse_and_log_det(y)
        log_prob_data = self.flow.q0.log_prob(z) + logdet  # (B,)

        cv = torch.randn(x.size(0), 5, device=x.device)  # (B, output_dim)
        cv = self.sample_cotangent(x.detach(), cv)  # (B, input_dim)
        cv_sym = self.encode_cotangent(y, cv.detach())  # (B, input_dim)
        cv = torch.randn_like(x)
        cv_null = self.encode_cotangent(y, cv)  # (B, input_dim)
        cv_sym = cv_sym.detach()
        cv_null = cv_null.detach()

        if self.training:
            with torch.no_grad():
                var_sym = cv_sym.square().mean()
                if self.var_sym.item() < 0:
                    self.var_sym.copy_(var_sym)
                else:
                    self.var_sym.mul_(0.9).add_(0.1 * var_sym)
                var_null = cv_null.square().mean().sqrt().clamp_min(1e-3)
                if self.var_null.item() < 0:
                    self.var_null.copy_(var_null)
                else:
                    self.var_null.mul_(0.9).add_(0.1 * var_null)

        cv_sym = cv_sym / self.var_sym.clamp_min(1e-6).sqrt()
        cv_null = cv_null / self.var_null.clamp_min(1e-6).sqrt()
        cv_sym = self.pullback_cotangent(z, cv_sym)  # (B, input_dim)
        cv_null = self.pullback_cotangent(z, cv_null)  # (B, input_dim)

        L = self.W_sym - self.W_sym.mT  # (num_bases, input_dim, input_dim)
        L = L / L.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6).sqrt()
        L = L / L.size(-1) ** 0.5
        J_sym = torch.einsum('bi,mji->bmj', z, L)  # (B, num_bases, input_dim)
        L = self.W_null - self.W_null.mT  # (num_bases, input_dim, input_dim)
        L = L / L.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6).sqrt()
        L = L / L.size(-1) ** 0.5
        J_null = torch.einsum('bi,mji->bmj', z, L)  # (B, num_null, input_dim)

        log_prob_sym = self.log_prob(cv_sym, J_sym, self.log_var_sym, self.eps_q)  # (B,)
        log_prob_null = self.log_prob(cv_null, J_null, self.log_var_null, self.eps_r)  # (B,)

        return x, log_prob_data, log_prob_sym, log_prob_null


class CLNFModule(pl.LightningModule):
    def __init__(
        self, 
        ckpt_predictor,
        lr=1e-3,
        beta=10.0,
        sample_num=64,
        num_bases_sym=66,
        num_bases_null=66,
        latent_dim=12,
        autoencoder_layers=3,
        flow_layers=24,
        flow_hidden_dim=192,
        eps_p=1e-3,
        eps_q=1e-1,
        eps_r=1e-1,
        scale_map="exp_clamp",
    ):
        super().__init__()
        self.model = CLNF(
            ckpt_predictor,
            num_bases_sym=num_bases_sym,
            num_bases_null=num_bases_null,
            latent_dim=latent_dim,
            autoencoder_layers=autoencoder_layers,
            flow_layers=flow_layers,
            flow_hidden_dim=flow_hidden_dim,
            eps_p=eps_p,
            eps_q=eps_q,
            eps_r=eps_r,
            scale_map=scale_map,
        )
        self.sample_num = sample_num

        self.lr = lr
        self.beta = beta
    
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

            L = self.model.W - self.model.W.mT
            L = L / L.square().sum((-2, -1), keepdim=True).sqrt()

            u, s, vh = torch.linalg.svd(L.flatten(1, 2), full_matrices=False)

            fig, ax = plt.subplots()
            ax.plot(s.detach().cpu().numpy())
            ax.set_title('Singular values of learned symmetry generators')
            ax.set_xlabel('Index')
            ax.set_ylabel('Singular value')
            
            if hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log({f"val_generated/singular_values": wandb.Image(fig, caption=f"epoch {self.current_epoch}")})
            plt.close(fig)

            L = vh[:self.sample_num].reshape(-1, L.size(1), L.size(2))
            n = L.size(0)
            ncol = min(8, n)
            nrow = math.ceil(n / ncol)

            fig, ax = plt.subplots(nrow, ncol, figsize=(4*ncol, 4*nrow))
            ax = ax.flatten()
            vmax = L.abs().max().item()
            vmin = -vmax

            for i in range(L.size(0)):
                im = ax[i].imshow(L[i].detach().cpu().numpy(), cmap='bwr', vmin=vmin, vmax=vmax)
                ax[i].set_title(f'Generator {i+1}')
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                fig.colorbar(im, ax=ax[i])

            if hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log({f"val_generated/symmetry_generators": wandb.Image(fig, caption=f"epoch {self.current_epoch}")})
            plt.close(fig)

    def training_step(self, batch, batch_idx):
        recon, log_prob_data, log_prob_sym, log_prob_null = self.model.forward(batch)
        log_prob_data = log_prob_data.mean()
        log_prob_sym = log_prob_sym.mean()
        log_prob_null = log_prob_null.mean()
        nll_loss = - (log_prob_data + log_prob_sym + log_prob_null)
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        loss = nll_loss + self.beta * recon_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_log_prob_data', log_prob_data, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_log_prob_sym', log_prob_sym, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_log_prob_null', log_prob_null, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_nll', nll_loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        recon, log_prob_data, log_prob_sym, log_prob_null = self.model.forward(batch)
        log_prob_data = log_prob_data.mean()
        log_prob_sym = log_prob_sym.mean()
        log_prob_null = log_prob_null.mean()
        nll_loss = - (log_prob_data + log_prob_sym + log_prob_null)
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        loss = nll_loss + self.beta * recon_loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_log_prob_data', log_prob_data, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_log_prob_sym', log_prob_sym, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_log_prob_null', log_prob_null, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_nll', nll_loss, on_step=False, on_epoch=True, prog_bar=False)
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
