import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

class LearnablePrototypes(nn.Module):
    def __init__(self, num_domains, prototype_dim):
        super(LearnablePrototypes, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_domains, prototype_dim))

    def forward(self, domain_label):
        return self.prototypes[domain_label]

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, use_sn=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.use_sn = use_sn
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

        if self.use_sn:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
            if self.learned_sc:
                self.conv1x1 = nn.utils.spectral_norm(self.conv1x1)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2), upsample=False, use_sn=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.use_sn = use_sn
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        if self.use_sn:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
            if self.learned_sc:
                self.conv1x1 = nn.utils.spectral_norm(self.conv1x1)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class Generator(BaseModel):
    def __init__(self, img_size=256, style_dim=64,
                 max_conv_dim=512, use_sn=False):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0),
            nn.Tanh()
        )

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        self.encoder_channels = [dim_in]
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True, use_sn=use_sn))
            self.encoder_channels.append(dim_out)
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True, use_sn=use_sn))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True, use_sn=use_sn))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, use_sn=use_sn))

    def _encode(self, x):
        x = self.from_rgb(x)
        f_enc = [x]
        for block in self.encode:
            x = block(x)
            f_enc.append(x)
        return x, f_enc

    def _decode(self, x, s):
        for block in self.decode:
            x = block(x, s)
        x_gen = self.to_rgb(x)
        return x_gen

    def forward(self, x, s):
        x_enc, f_enc = self._encode(x)
        x_gen = self._decode(x_enc, s)
        return x_gen, f_enc


class MappingNetwork(BaseModel):
    def __init__(
            self,
            style_dim=64,
            num_domains=2):

        super().__init__()
        
        self.style_dim = style_dim
        self.num_domains = num_domains
        self.prototypes_learnable = LearnablePrototypes(num_domains, style_dim)

    def forward(self, d):
        batch_size = d.size(0)
        s = torch.zeros(batch_size, self.style_dim, device=d.device)
        for i in range(self.num_domains):
            mask = (d == i)
            if mask.any():
                s[mask] = self.prototypes_learnable(domain_label=i).to(d.device)
        return s

    def get_prototypes(self):
        return self.prototypes_learnable

class StyleEncoder(BaseModel):
    def __init__(
            self,
            img_size=256,
            style_dim=64,
            num_domains=2,
            max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(
            dim_out, int(style_dim * num_domains))
        self.num_domains = num_domains
        self.style_dim = style_dim

        self.prototypes = torch.zeros((num_domains, style_dim))
        self.prototype_counts = torch.zeros(num_domains)

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)

        out = self.unshared(h)  # (batch, num_domains * style_dim)
        out = out.view(out.size(0), self.num_domains, self.style_dim)
        idx = torch.arange(y.size(0), device=y.device)
        s = out[idx, y]  # (batch, style_dim)

        with torch.no_grad():
            for i in range(self.num_domains):
                mask = (y == i)
                if mask.any():
                    self.prototypes[i] += s[mask].sum(dim=0).to(self.prototypes.device)
                    self.prototype_counts[i] += mask.sum().to(self.prototypes.device)

        return s

    def get_prototypes(self):
        return self.prototypes / self.prototype_counts[:, None]



class Discriminator(BaseModel):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512, use_sn=False):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        if use_sn:
            blocks += [nn.utils.spectral_norm(nn.Conv2d(3, dim_in, 3, 1, 1))]
        else:
            blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True, use_sn=use_sn)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        if use_sn:
            blocks += [nn.utils.spectral_norm(
                nn.Conv2d(dim_out, dim_out, 4, 1, 0))]
        else:
            blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]

        blocks += [nn.LeakyReLU(0.2)]

        if use_sn:
            blocks += [nn.utils.spectral_norm(
                nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        else:
            blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.arange(y.size(0), device=y.device)
        out = out[idx, y]  # (batch)
        return out


class StarGANv2(nn.Module):

    def __init__(
            self,
            generator,
            mapping_network,
            style_encoder):
        super(StarGANv2, self).__init__()
        self.generator = generator
        self.mapping_network = mapping_network
        self.style_encoder = style_encoder

    def forward(self, x_src, d_ref, x_ref=None):
        if x_ref is not None:
            s_ref = self.style_encoder(x_ref, d_ref)
            ref_source = 'style_encoder'
        else:
            s_ref = self.mapping_network(d_ref)
            ref_source = 'mapping_network'
        x_gen, f_enc = self.generator(x_src, s_ref)
        return x_gen, s_ref, ref_source

    def interpolate(
            self,
            x_src,
            d_ref,
            style_indexes,
            style_weights,
            x_ref=None):
        if x_ref is not None:
            s_ref = self.style_encoder(x_ref, d_ref)
        else:
            s_ref = self.mapping_network(d_ref)

        styles = s_ref[style_indexes] * style_weights[:, :, None]
        s_final = styles.sum(dim=1)
        x_gen, _ = self.generator(x_src, s_final)
        return x_gen

    def get_prototypes(self):
        # 从 mapping_network 和 style_encoder 中获取 prototypes
        mapping_prototypes = self.mapping_network.get_prototypes()
        style_encoder_prototypes = self.style_encoder.get_prototypes()
        return {
            'mapping_prototypes': mapping_prototypes,
            'style_encoder_prototypes': style_encoder_prototypes
        }


class StarGanv2Aug(StarGANv2):

    def __init__(
            self,
            generator,
            mapping_network,
            style_encoder,
            num_styles,
            device):
        super(StarGanv2Aug, self).__init__(
            generator, mapping_network, style_encoder)
        self.num_styles = num_styles
        self.num_domains = style_encoder.num_domains
        self.dist_style_weights = torch.distributions.dirichlet.Dirichlet(
            0.4 * torch.ones(num_styles, device=device))

    def set_requires_grad(self, model, flag):
        for p in model.parameters():
            p.requires_grad = flag


class StyleAugMix(StarGanv2Aug):

    def __init__(
            self,
            generator,
            mapping_network,
            style_encoder,
            num_styles,
            device):
        super(StyleAugMix, self).__init__(
            generator, mapping_network, style_encoder, num_styles,
            device)

    def forward(self, x, y, d):
        x_aug, y_aug = self.style_aug_mix(x, y, d)

        return x_aug, y_aug

    def style_aug_mix(self, x, y, d):
        N = x.size(0)
        style_indexes = torch.randn(N, N, device=x.device).argsort(
            dim=-1)[:, :self.num_styles]
        style_weights = self.dist_style_weights.sample(
            [style_indexes.shape[0]]).to(x.device)
        if d is not None:
            d_ref = d
        else:
            d_ref =torch.randint(
                0, self.num_domains, (N, ), device=x.device).long()
        x_gen = self.interpolate(
            x, d_ref, style_indexes,
            style_weights)
        return x_gen, y
    
    def get_all_prototypes(self):
        prototypes = self.get_prototypes()
        return prototypes
