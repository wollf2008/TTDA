import tqdm
from collections import OrderedDict

import torch
from torch import autograd
from torch import nn
from torch.nn import functional as F
from torch.distributions.beta import Beta
from torchvision.transforms import functional as TF
from torchvision.models import vgg16
from torchvision.models._utils import IntermediateLayerGetter

from .utils import NonGeometricTransform, Normalize, requires_grad


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, logits, target):
        probs = F.softmax(logits, 1)
        nll_loss = (- target * torch.log(probs)).sum(1).mean()

        return nll_loss


def compute_grad_gp(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


class VGG16Extractor(nn.Module):

    def __init__(
            self,
            vgg_layers=["relu2_1", "relu3_1"],
            normalize_01=True,
            normalize_imagenet=True):
        super(VGG16Extractor, self).__init__()
        self.normalize_01 = normalize_01
        self.normalize_imagenet = normalize_imagenet
        self.normalization_01 = Normalize(
            mean=(-1, -1, -1), std=(2, 2, 2))
        self.normalizion_imagenet = Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        vgg = vgg16(pretrained=True).features
        layer2name = {
            "3": "relu1_2",
            "6": "relu2_1",
            "8": "relu2_2",
            "11": "relu3_1",
            "15": "relu3_3",
            "18": "relu4_1",
            "22": "relu4_3",
            "25": "relu5_1",
        }
        name2layer = {name: layer for layer, name in layer2name.items()}

        return_layers = {
            name2layer[layer_name]: layer_name for layer_name in vgg_layers
            if layer_name in name2layer
        }
        self.features_extractor = IntermediateLayerGetter(
            vgg, return_layers=return_layers)
        self.layer2channel = {
            "relu1_2": 64,
            "relu2_1": 128,
            "relu2_2": 128,
            "relu3_1": 256,
            "relu3_3": 256,
            "relu4_1": 512,
            "relu4_3": 512,
            "relu5_1": 512,
        }

    def forward(self, x):
        x = self.normalization_01(x) if self.normalize_01 else x
        x = self.normalizion_imagenet(x) if self.normalize_imagenet else x
        features = self.features_extractor(x)
        return features


class DomainInvariantPerceptualLoss(nn.Module):

    def __init__(self, vgg_content_layers, vgg_style_layers):
        super().__init__()
        self.vgg_content_layers = vgg_content_layers
        self.vgg_style_layers = vgg_style_layers
        self.vgg = VGG16Extractor(
            list(set(vgg_content_layers+vgg_style_layers)))
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x_gen, x_src):
        x_gen = F.interpolate(x_gen, size=(224, 224), mode="bilinear")
        x_src = F.interpolate(x_src, size=(224, 224), mode="bilinear")
        feat_gen = self.vgg(x_gen)
        feat_src = self.vgg(x_src)
        domain_invariant_perceptual_loss = 0
        for layer_name in self.vgg_content_layers:
            domain_invariant_perceptual_loss = domain_invariant_perceptual_loss + F.mse_loss(
                F.instance_norm(feat_gen[layer_name]), F.instance_norm(feat_src[layer_name]))

        for layer_name in self.vgg_style_layers:
            mse_mean = F.mse_loss(
                feat_gen[layer_name].mean(dim=[-2, -1]), feat_src[layer_name].mean(dim=[-2, -1]))
            mse_std = F.mse_loss(
                feat_gen[layer_name].std(dim=[-2, -1]), feat_src[layer_name].std(dim=[-2, -1]))
            domain_invariant_perceptual_loss = domain_invariant_perceptual_loss + mse_mean + mse_std
        return domain_invariant_perceptual_loss


class SemanticRelationConstrativeLoss(nn.Module):

    def __init__(
            self,
            temperature,
            sampled_patches,
            encoded_features_channels,
            mlp_depth):
        super().__init__()
        self.temperature = temperature
        self.sampled_patches = sampled_patches
        self.mlps = nn.ModuleList([])
        self.num_layers = len(encoded_features_channels)
        for input_channels in encoded_features_channels:
            mlp = []
            for _ in range(mlp_depth-1):
                mlp.extend([
                    nn.Linear(input_channels, input_channels),
                    nn.LeakyReLU(0.2)])
            mlp.append(nn.Linear(input_channels, input_channels))
            mlp = nn.Sequential(*mlp)
            self.mlps.append(mlp)

    def forward(
            self,
            features_src: list,
            features_gen: list):

        loss = 0
        for feat_src, feat_gen, mlp in zip(
                features_src, features_gen, self.mlps):
            N, C, H, W = feat_src.shape

            # Flatten representation and permute axis to have channel
            # axis at the end.
            feat_src = feat_src.reshape([N, C, -1])
            feat_gen = feat_gen.reshape([N, C, -1])
            n_patches = feat_gen.size(-1)

            # Sampled K sampled representations depending on the
            # sampled patches ratio
            K = min(self.sampled_patches, n_patches)
            indexes = torch.multinomial(
                torch.ones(N, n_patches), num_samples=K).reshape(
                    N, 1, K).repeat([1, C, 1]).to(feat_src.device)

            samp_feat_src = torch.gather(feat_src, 2, indexes)
            samp_feat_gen = torch.gather(feat_gen, 2, indexes)

            # Reshape tensor (N, C, num_sampled) --> (N, num_sampled, C)
            samp_feat_src = samp_feat_src.permute(0, 2, 1)
            samp_feat_gen = samp_feat_gen.permute(0, 2, 1)

            # Project each patch representation with mlp
            samp_feat_src = mlp(samp_feat_src)
            samp_feat_gen = mlp(samp_feat_gen)

            # Normalize representations
            z = F.normalize(samp_feat_src, p=2, dim=-1)
            w = F.normalize(samp_feat_gen, p=2, dim=-1)

            # Compute the losses
            src_loss = self.semantic_consistency_loss(
                z, w)
            patch_dcne_loss = (self.dnce_loss(z, w) + self.dnce_loss(w, z))/2
            loss += src_loss + patch_dcne_loss
        return loss / self.num_layers

    def semantic_consistency_loss(
            self,
            z: torch.Tensor,
            w: torch.Tensor):
        N, K, _ = z.shape
        sim_z = torch.bmm(z, z.permute(0, 2, 1))
        sim_w = torch.bmm(w, w.permute(0, 2, 1))
        non_diag_mask = ~torch.eye(K, device=z.device)[None, ...].repeat([N, 1, 1]).bool()
        P = torch.masked_select(sim_z, non_diag_mask).reshape(
            N, K, K-1)
        Q = torch.masked_select(sim_w, non_diag_mask).reshape(
            N, K, K-1)
        return self.jsd(P, Q)

    def dnce_loss(
            self,
            z: torch.Tensor,
            w: torch.Tensor):
        N, K, _ = z.shape
        sim = torch.bmm(z, w.permute(0, 2, 1)) / self.temperature

        diag_mask = torch.eye(K, device=z.device)[None, ...].repeat([N, 1, 1]).bool()
        non_diag_mask = ~diag_mask

        pos_sim = torch.masked_select(sim, diag_mask).reshape(N, -1)
        neg_sim = torch.masked_select(sim, non_diag_mask).reshape(
            N, K, K-1)
        losses = -pos_sim + torch.logsumexp(neg_sim, dim=-1)
        loss = losses.mean()
        return loss

    def jsd(self, P_logits, Q_logits):
        P_probs = F.softmax(P_logits, dim=1)
        Q_probs = F.softmax(Q_logits, dim=1)

        M_probs = (P_probs + Q_probs)/2
        loss = 0.0
        loss += F.kl_div(F.log_softmax(P_logits, dim=1),
                         M_probs, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(Q_logits, dim=1),
                         M_probs, reduction="batchmean")

        return loss/2


class JSD(nn.Module):

    def __init__(self, temperature=1):
        super(JSD, self).__init__()
        self. temperature = temperature

    def forward(self, P_logits, Q_logits):
        P_probs = F.softmax(P_logits, dim=1)
        Q_probs = F.softmax(Q_logits, dim=1)

        M_probs = (P_probs + Q_probs)/2
        loss = 0.0
        loss += F.kl_div(F.log_softmax(P_logits, dim=1),
                         M_probs, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(Q_logits, dim=1),
                         M_probs, reduction="batchmean")
        return loss/2
