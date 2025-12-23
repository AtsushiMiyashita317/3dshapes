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
        layers.append(torch.nn.LayerNorm([latent_dim, 1, 1]))

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
        scale_map="exp_clamp",
        optimize_decoder_for_pushforward=False,
    ):
        super().__init__()

        self.predictor = PredictorModule.load_from_checkpoint(ckpt_predictor).model.eval()

        self.latent_dim = latent_dim
        self.num_bases = num_bases
        self.optimize_decoder_for_pushforward = optimize_decoder_for_pushforward

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

        self.W = torch.nn.Parameter(torch.randn(num_bases, input_dim, input_dim) * (1.0 / math.sqrt(input_dim)))

        self.register_buffer('eps_p', torch.tensor(eps_p))
        self.register_buffer('eps_q', torch.tensor(eps_q))
        self.register_buffer('var_v', torch.tensor(-1.0))
        self.register_buffer('var_cv', torch.tensor(-1.0))

        def _pushforward_tangent_single(z: torch.Tensor, v: torch.Tensor):
            # z: (input_dim,)
            # v: (input_dim,)
            z = z.unsqueeze(0)
            v = v.unsqueeze(0)
            _, v = torch.func.jvp(self.flow.forward, (z,), (v,))
            return v.squeeze(0)
        
        func = torch.func.vmap(_pushforward_tangent_single, in_dims=(None, 0))
        self.pushforward_tangent_func = torch.func.vmap(func, in_dims=(0, 0))

        self.jacobian_func = torch.func.vmap(torch.func.jacfwd(self._decode))

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
    
    def pushforward_tangent(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # z: (batch_size, input_dim)
        # v: (batch_size, num_bases, input_dim)
        v = self.pushforward_tangent_func(z, v)  # (batch_size, num_bases, 3, 64, 64)
        return v

    def log_prob(self, cv: torch.Tensor, v: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cv: (batch_size, *)
            v: (batch_size, num_bases, input_dim)
            J: (batch_size, *, input_dim)
        Returns:
            log_prob: (batch_size,)
        """
        cv = cv.reshape(cv.size(0), -1)  # (batch_size, output_dim)
        J = J.reshape(J.size(0), -1, J.size(-1))  # (batch_size, output_dim, input_dim)

        # cv = cv + self.eps_p.sqrt() * torch.randn_like(cv)

        num_bases = v.size(-2)
        output_dim = cv.size(-1)

        S_pp = torch.einsum('bi,bi->b', cv, cv)
        S_qq = torch.einsum('bni,bji,bjk,bmk->bnm', v, J, J, v)
        S_pq = torch.einsum('bj,bji,bni->bn', cv, J, v)

        I = torch.eye(num_bases, device=v.device)
        # norm = S_qq.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-9)     # (batch_size,)
        # eps_q = self.eps_q * norm                      # (batch_size,)
        eps_q = self.eps_q
        M = S_qq + eps_q.unsqueeze(-1).unsqueeze(-1) * I                        # (batch_size, num_bases, num_bases)

        # w^t(eI + v^tv)w= e w^tw + w^tv^tvw
        trace = eps_q * S_pp + torch.einsum('bn,bn->b', S_pq, S_pq)  # (batch_size,)

        L_H = torch.linalg.cholesky(M)
        logdet = 2 * torch.log(torch.diagonal(L_H, dim1=-2, dim2=-1)).sum(-1)

        log_prob = 0.5 * (logdet - trace)

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

        L = self.W - self.W.mT  # (num_bases, input_dim, input_dim)
        # L = L / L.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6).sqrt()
        # L = L / L.size(-1) ** 0.5
        v = torch.einsum('bi,mji->bmj', z, L)  # (B, num_bases, input_dim)
        
        v = self.pushforward_tangent(z, v)  # (B, num_bases, input_dim)

        if self.optimize_decoder_for_pushforward:
            J = self.jacobian_func(y)  # (B, input_dim, 3, 64, 64)
        else:
            with torch.no_grad():
                J = self.jacobian_func(y)  # (B, input_dim, 3, 64, 64)

        if self.training:
            with torch.no_grad():
                var = v.square().mean((0,1)).sum()
                if self.var_v.item() < 0:
                    self.var_v.copy_(var)
                else:
                    self.var_v.mul_(0.9).add_(0.1 * var)
                var = cv.square().mean(0).sum()
                if self.var_cv.item() < 0:
                    self.var_cv.copy_(var)
                else:
                    self.var_cv.mul_(0.9).add_(0.1 * var)

        v = v / v.square().mean((0,1), keepdim=True).clamp_min(1e-6).sqrt()
        v = v / v.size(1)
        cv = cv / self.var_cv.clamp_min(1e-6).sqrt()
        cv = cv.detach()

        log_prob_cotangent = self.log_prob(cv, v, J)  # (B,)

        return x, log_prob_data, log_prob_cotangent


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
        scale_map="exp_clamp",
        optimize_decoder_for_pushforward=False,
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
            scale_map=scale_map,
            optimize_decoder_for_pushforward=optimize_decoder_for_pushforward,
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
        recon, log_prob_data, log_prob_cotangent = self.model.forward(batch)
        log_prob_data = log_prob_data.mean()
        log_prob_cotangent = log_prob_cotangent.mean()
        nll_loss = - (log_prob_data + log_prob_cotangent)
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        loss = nll_loss + self.beta * recon_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_log_prob_data', log_prob_data, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_log_prob_cotangent', log_prob_cotangent, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_nll', nll_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_var_v', self.model.var_v, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_var_cv', self.model.var_cv, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        recon, log_prob_data, log_prob_cotangent = self.model.forward(batch)
        log_prob_data = log_prob_data.mean()
        log_prob_cotangent = log_prob_cotangent.mean()
        nll_loss = - (log_prob_data + log_prob_cotangent)
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        loss = nll_loss + self.beta * recon_loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_log_prob_data', log_prob_data, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_log_prob_cotangent', log_prob_cotangent, on_step=False, on_epoch=True, prog_bar=False)
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
