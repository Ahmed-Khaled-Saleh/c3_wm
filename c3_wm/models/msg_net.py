__all__ = ['Tensor', 'BaseVAE', 'VectorQuantizer', 'ResidualLayer', 'VQVAE']

import torch
from torch import nn
from torch.nn import functional as F


from typing import List, Callable, Union, Any, TypeVar, Tuple

Tensor = TypeVar('torch.tensor')

from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, decay=0.99, eps=1e-5, dead_threshold=None):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.dead_threshold = dead_threshold

        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed', embedding.clone())
        self.register_buffer('usage', torch.ones(num_embeddings), persistent=False)
        self.register_buffer('initialized', torch.tensor(False), persistent=False)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.initialized.fill_(True)

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)

        if self.training and self.dead_threshold is not None and not self.initialized:
            pick = torch.randint(0, flat_latents.size(0), (self.K,), device=latents.device)
            self.embedding.copy_(flat_latents.detach()[pick])
            self.ema_embed.copy_(self.embedding)
            self.cluster_size.fill_(1.0)
            self.initialized.fill_(True)

        dist = (torch.sum(flat_latents ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding ** 2, dim=1)
                - 2 * torch.matmul(flat_latents, self.embedding.t()))

        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)

        avg_probs = encoding_one_hot.mean(0).detach()
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized_latents = torch.matmul(encoding_one_hot, self.embedding)
        quantized_latents = quantized_latents.view(latents_shape)

        if self.training:
            self.cluster_size.mul_(self.decay).add_(
                encoding_one_hot.sum(0).detach(), alpha=1 - self.decay
            )
            dw = torch.matmul(encoding_one_hot.t(), flat_latents.detach())
            self.ema_embed.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            smoothed = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
            self.embedding.copy_(self.ema_embed / smoothed.unsqueeze(1))

            if self.dead_threshold is not None:
                self.usage.mul_(self.decay).add_(encoding_one_hot.sum(0), alpha=1 - self.decay)
                dead = self.usage < self.dead_threshold
                if dead.any():
                    pick = torch.randint(0, flat_latents.size(0), (int(dead.sum()),), device=latents.device)
                    fresh = flat_latents.detach()[pick]
                    self.embedding[dead] = fresh
                    self.ema_embed[dead] = fresh
                    self.cluster_size[dead] = 1.0
                    self.usage[dead] = 1.0

        vq_loss = self.beta * F.mse_loss(quantized_latents.detach(), latents)

        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss, perplexity


class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = [64, 128, 256, 512, 512],
                 beta: float = 0.25,
                 img_size: int = 224,
                 num_residual_layers: int = 3,
                 pre_quant_activation: bool = True,
                 dead_threshold: float | None = None,
                 recon_loss: str = "l1",
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.num_residual_layers = num_residual_layers
        assert recon_loss in ("l1", "mse", "l1+mse"), recon_loss
        self.recon_loss = recon_loss

        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(self.num_residual_layers):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU() if pre_quant_activation else nn.Identity())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta, dead_threshold=dead_threshold)

        modules = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[-1],
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(self.num_residual_layers):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.LeakyReLU())

        hidden_dims_reversed = hidden_dims[::-1]
        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims_reversed[i], hidden_dims_reversed[i + 1],
                                       kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims_reversed[-1], out_channels=3,
                                   kernel_size=4, stride=2, padding=1),
                nn.Tanh())
        )

        self.decoder = nn.Sequential(*modules)


    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss, perplexity = self.vq_layer(encoding)

        return [self.decode(quantized_inputs), input, vq_loss, perplexity]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        vq_loss = args[2]
        perplexity = args[3]
        weights = kwargs.get('weights')

        if weights is not None:
            pixel_weights = F.interpolate(weights[:, None].float(), size=input.shape[-2:], mode='nearest')
        else:
            pixel_weights = (input > -0.9).float() * 9.0 + 1.0
        err = (recons - input).abs()
        per_pixel = {"l1": err, "mse": err ** 2, "l1+mse": err + err ** 2}[self.recon_loss]
        recons_loss = (per_pixel * pixel_weights).mean()


        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss': vq_loss,
                'Perplexity': perplexity}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:

        return self.forward(x)[0]


    def get_message_indices(self, obs):
        encoding = self.encode(obs)[0]

        latents = encoding.permute(0, 2, 3, 1).contiguous()
        latents = latents.view(-1, self.embedding_dim)

        dist = (torch.sum(latents ** 2, dim=1, keepdim=True)
                + torch.sum(self.vq_layer.embedding ** 2, dim=1)
                - 2 * torch.matmul(latents, self.vq_layer.embedding.t()))

        indices = torch.argmin(dist, dim=1)
        return indices.view(-1, 7, 7)
