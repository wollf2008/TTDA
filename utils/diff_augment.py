import torch
import torch.nn as nn
import numpy as np
import random
import math
from kornia.geometry.transform import rotate, scale
from kornia import augmentation as K


class DiffAugment(nn.Module):
    """
    This class implement a variant of DiffAugment for training GANs effciently
    composed of:
        - flips
        - integer translations
        - continuous rotations
        - isotropic and anisotropic scaling
        - cutout
    """

    def __init__(self, p):
        super(DiffAugment, self).__init__()
        self.p = p

    def forward(self, images: torch.Tensor, p=None) -> torch.Tensor:

        prob = p if p is not None else self.p
        transformed_imgs = images.clone()

        # Perform horizontal, vertical flips
        images_flipped = torch.rand(
            images.size(0), device=images.device) <= prob
        if images_flipped.any().item() > 0:
            transformed_imgs[images_flipped] = transformed_imgs[images_flipped].flip(
                dims=(-2,))

        images_flipped = torch.rand(
            images.size(0), device=images.device) <= prob
        if images_flipped.any().item() > 0:
            transformed_imgs[images_flipped] = transformed_imgs[images_flipped].flip(
                dims=(-1,))

        # Perform integer translation
        images_translated = torch.rand(
            images.size(0), device=images.device) <= prob
        if images_translated.any().item() > 0:
            transformed_imgs[images_translated] = integer_translation(
                transformed_imgs[images_translated])

        # Perform continuous rotations
        images_rotated = torch.rand(
            images.size(0), device=images.device) <= prob
        if images_rotated.any().item() > 0:
            angles = torch.randint(0, 180, (transformed_imgs[images_rotated].size(0),),
                                   device=images.device).type(images.type())
            transformed_imgs[images_rotated] = rotate(
                transformed_imgs[images_rotated], angles, padding_mode='reflection')

        # Perform isotropic scaling
        images_scaling = torch.rand(
            images.size(0), device=images.device) <= prob
        if images_scaling.any().item() > 0:
            transformed_imgs[images_scaling] = scale(
                transformed_imgs[images_scaling],
                scale_factor=torch.ones(images_scaling.float().sum().int().item(), 2, device=images.device) * torch.from_numpy(
                    np.random.lognormal(mean=0, sigma=(0.2 * math.log(2)) ** 2,
                                        size=(images_scaling.float().sum().int().item(), 1))).float().to(images.device),
                center=torch.ones(images_scaling.float().sum(
                ).int().item(), 2, device=images.device)
                * 0.5 * torch.tensor(images.shape[2:], device=images.device),
                padding_mode="reflection")

        # Perform anisotropic scaling
        images_scaling = torch.rand(
            images.size(0), device=images.device) <= prob
        if images_scaling.any().item() > 0:
            transformed_imgs[images_scaling] = scale(
                transformed_imgs[images_scaling],
                scale_factor=torch.ones(images_scaling.float().sum().int().item(), 2, device=images.device) * torch.from_numpy(
                    np.random.lognormal(mean=0, sigma=(0.2 * math.log(2)) ** 2,
                                        size=(images_scaling.float().sum().int().item(), 2))).float().to(images.device),
                center=torch.ones(images_scaling.float().sum(
                ).int().item(), 2, device=images.device)
                * 0.5 * torch.tensor(images.shape[2:], device=images.device),
                padding_mode="reflection")

        # Perform cutout
        transformed_imgs = K.RandomErasing(
            scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0,
            same_on_batch=False, p=prob)(transformed_imgs)
        return transformed_imgs


def integer_translation(images: torch.Tensor) -> torch.Tensor:
    """
    Function implements integer translation augmentation
    """
    # Get translation index
    translation_index = (int(images.shape[-2] * random.uniform(-0.125, 0.125)),
                         int(images.shape[-1] * random.uniform(-0.125, 0.125)))
    # Apply translation
    return torch.roll(images, shifts=translation_index, dims=(-2, -1))
