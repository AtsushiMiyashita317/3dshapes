import torch
import pytorch_lightning as pl
import torchvision as tv
import wandb
import normflows as nf
import math

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
        num_bases=66,
        latent_dim=12,
        autoencoder_layers=3,
        flow_layers=24,
        flow_hidden_dim=192,
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

        self.log_var_diag = torch.nn.Parameter(torch.zeros(input_dim))

        self.register_buffer('eps_p', torch.tensor(eps_p))
        self.register_buffer('eps_q', torch.tensor(eps_q))

    def parameters(self, recurse = True):
        yield from self.flow.parameters(recurse)
        yield from self.autoencoder.parameters(recurse)
        yield self.W
        yield self.log_var_diag

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

        J_p = J_p / J_p.square().sum().sqrt()

        S_p = torch.einsum('bni,bmi->bnm', J_p, J_p)                        # (batch_size, output_dim, output_dim)
        S_q = torch.einsum('bin,bim->bnm', J_q, J_q)                        # (batch_size, input_dim, input_dim)
        S_pq = torch.einsum('bni,bmi->bnm', J_p, J_q)                       # (batch_size, output_dim, num_bases)
        D = self.log_var_diag.neg().exp()                                   # (input_dim,)

        I_p = torch.eye(output_dim, device=J_p.device)                      # (output_dim, output_dim)
        I_q = torch.eye(input_dim, device=J_q.device)                       # (input_dim, input_dim)
        M = S_p + self.eps_p * I_p                                          # (batch_size, output_dim, output_dim)
        H = S_q + D * I_q                                                   # (batch_size, input_dim, input_dim)

        norm_M = M.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
        norm_H = H.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
        M = M + 1e-3 * norm_M.unsqueeze(-1).unsqueeze(-1) * I_p
        H = H + 1e-3 * norm_H.unsqueeze(-1).unsqueeze(-1) * I_q

        trace_p = torch.einsum('bij,bij,j->b', J_p, J_p, D)                 # (batch_size,)
        trace_q = torch.einsum('bij,bij->b', J_q, J_q)                      # (batch_size,)
        trace_pq = torch.einsum('bij,bij->b', S_pq, S_pq)                   # (batch_size,)
        trace = self.eps_p * (D.sum() + trace_q) + trace_p + trace_pq       # (batch_size,)

        L_M = torch.linalg.cholesky(M)
        L_H = torch.linalg.cholesky(H)
        logdet_M = 2 * torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)).sum(-1)
        logdet_H = 2 * torch.log(torch.diagonal(L_H, dim1=-2, dim2=-1)).sum(-1)
        logdet_p = logdet_M + (input_dim - output_dim) * self.eps_p.log()   # (batch_size,)
        logdet_q = logdet_H                                                 # (batch_size,)
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
        num_bases=66,
        latent_dim=12,
        autoencoder_layers=3,
        flow_layers=24,
        flow_hidden_dim=192,
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
