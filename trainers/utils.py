import random
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import kornia.augmentation as K
from kornia.augmentation.container import ImageSequential


def random_uniform(a, b):
    return (a - b) * random.random() + b


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.Tensor(mean))
        self.register_buffer("std", torch.Tensor(std))

    def forward(self, x):
        x = (x - self.mean[None, :, None, None])/self.std[None, :, None, None]
        return x


class RandomAdjustSharpness():

    def __init__(self, sharpness_factor):
        self.sharpness_factor = sharpness_factor

    def __call__(self, x):
        sharpness_factor = random_uniform(
            1-self.sharpness_factor, 1 + self.sharpness_factor)
        x = TF.adjust_sharpness(x, sharpness_factor)
        return x


class RandomSolarize():

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x):
        threshold = random_uniform(
            self.threshold[0], self.threshold[1])
        x = TF.solarize(x, threshold)
        return x


class RandomPosterize():

    def __init__(self, bits=[4, 8]):
        self.bits = bits

    def __call__(self, x):
        bits = random.choice(list(range(self.bits[0], self.bits[1]+1)))
        x = TF.posterize(x, bits)
        return x


class RandomTranslation(nn.Module):

    def __init__(
            self,
            ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        shift_x, shift_y = int(x.size(2) * self.ratio +
                               0.5), int(x.size(3) * self.ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1,
                                      size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1,
                                      size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[
            grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
        return x


class MirrorRandomTranslation(nn.Module):

    def __init__(
            self,
            ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        shift_x, shift_y = int(x.size(2) * self.ratio), int(x.size(3) * self.ratio)

        translation_x = torch.randint(-shift_x, shift_x + 1,
                                      size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1,
                                      size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = grid_x + translation_x
        grid_y = grid_y + translation_y

        neg_offsets_grid_x, pos_offsets_grid_x = grid_x < 0, grid_x >= x.size(2)
        neg_offsets_grid_y, pos_offsets_grid_y = grid_y < 0, grid_y >= x.size(3)

        grid_x[neg_offsets_grid_x] = torch.abs(grid_x[neg_offsets_grid_x])
        grid_x[pos_offsets_grid_x] = 2*x.size(2)-1 - grid_x[pos_offsets_grid_x]

        grid_y[neg_offsets_grid_y] = torch.abs(grid_y[neg_offsets_grid_y])
        grid_y[pos_offsets_grid_y] = 2*x.size(3)-1 - grid_y[pos_offsets_grid_y]
        x_pad = x
        x = x_pad.permute(0, 2, 3, 1).contiguous()[
            grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
        return x


class RandAugment():

    def __init__(self, p):
        self.p = p
        self.transforms_list = [
            transforms.RandomAffine(0, shear=[0, 0, 0.3, 0.3]),
            transforms.RandomAffine(0, shear=[0.3, 0.3, 0, 0]),
            transforms.RandomAffine(0, translate=[0, 0.3]),
            transforms.RandomAffine(0, translate=[0.3, 0]),
            transforms.RandomAffine(degrees=30),
            transforms.ColorJitter(hue=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(brightness=0.5),
            RandomAdjustSharpness(0.3),
            RandomPosterize(bits=[4, 8]),
            RandomSolarize(threshold=[0, 1]),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1)
        ]

    def __call__(self, x):
        num_transforms = len(self.transforms_list)
        transforms_indexes = random.sample(
            list(range(num_transforms)), num_transforms)
        for idx in transforms_indexes:
            if random.random() <= self.p:
                transform = self.transforms_list[idx]
                x = transform(x)
        return x


class RandAugmentGPU(nn.Module):

    def __init__(
            self,
            p):
        super().__init__()
        self.unnormalize = K.Normalize(
            mean=(-1, -1, -1),
            std=(2, 2, 2))
        self.rand_augmentations = [
            K.ColorJitter(p=p, brightness=0.9, same_on_batch=False),
            K.ColorJitter(p=p, hue=0.9, same_on_batch=False),
            K.ColorJitter(p=p, contrast=0.9, same_on_batch=False),
            K.RandomErasing(p=p, scale=(0, 0.2), same_on_batch=False),
            # K.RandomEqualize(p=p, same_on_batch=False),
            K.RandomInvert(p=p, same_on_batch=False),
            # K.RandomPosterize(p=p, bits=[4, 8], same_on_batch=False),
            K.RandomAffine(p=p, degrees=[-30, 30], same_on_batch=False),
            K.RandomSharpness(p=p, sharpness=[0.1, 1.9], same_on_batch=False),
            K.RandomAffine(p=p, degrees=0, shear=(-0.1,
                                                  0.1, 0, 0), same_on_batch=False),
            K.RandomAffine(p=p, degrees=0, shear=(
                0, 0, -0.1, 0.1), same_on_batch=False),
            K.RandomSolarize(p=p, thresholds=[0, 1], same_on_batch=False),
            K.RandomAffine(p=p, degrees=0, translate=[
                           0.3, 0], same_on_batch=False),
            K.RandomAffine(p=p, degrees=0, translate=[
                           0, 0.3], same_on_batch=False),
        ]
        self.rand_augment_transform = ImageSequential(
            *self.rand_augmentations,
            same_on_batch=False,
            random_apply=len(self.rand_augmentations)
        )
        self.normalize = K.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5))

    def forward(self, x):
        x = self.unnormalize(x)
        x = self.rand_augment_transform(x)
        x = self.normalize(x)
        return x


class DiffAugment(nn.Module):

    def __init__(
            self,
            img_size,
            p):
        super().__init__()
        self.normalize = K.Normalize(
            mean=(-1, -1, -1),
            std=(2, 2, 2))
        self.augmentations = [
            # K.ColorJitter(p=p, brightness=0.9, same_on_batch=False),
            # K.ColorJitter(p=p, hue=0.9, same_on_batch=False),
            # K.ColorJitter(p=p, contrast=0.9, same_on_batch=False),
            K.RandomResizedCrop(
                size=img_size, scale=(0.8, 1), ratio=(0.9, 1.1), p=p),
            MirrorRandomTranslation(ratio=0.3),
            K.RandomErasing(p=p, scale=(0, 0.3), same_on_batch=False),
            K.RandomGaussianBlur(
                kernel_size=(5, 5), sigma=(2, 2), border_type='reflect', same_on_batch=False, p=p),
            K.RandomHorizontalFlip(same_on_batch=False, p=p),
            K.RandomVerticalFlip(same_on_batch=False, p=p),
        ]
        self.augment = ImageSequential(
            *self.augmentations,
            same_on_batch=False,
            random_apply=len(self.augmentations)
        )
        self.unnormalize = K.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5))

    def forward(self, x, normalize=False):
        if normalize:
            x = self.normalize(x)
        x = self.augment(x)
        if normalize:
            x = self.unnormalize(x)
        return x


class NonGeometricTransform(nn.Module):

    def __init__(
            self,
            p):
        super().__init__()
        self.unnormalize = K.Normalize(
            mean=(-1, -1, -1),
            std=(2, 2, 2))
        self.transforms = [
            K.ColorJitter(p=p, brightness=0.5, same_on_batch=False),
            K.ColorJitter(p=p, hue=0.5, same_on_batch=False),
            K.ColorJitter(p=p, contrast=0.5, same_on_batch=False),
            # K.RandomSolarize(p=p, thresholds=[0, 1], same_on_batch=False),
            # K.RandomInvert(p=p, same_on_batch=False),
        ]
        self.augment = ImageSequential(
            *self.transforms,
            same_on_batch=False,
            random_apply=len(self.transforms)
        )
        self.normalize = K.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5))

    def forward(self, x, normalize=False):
        if normalize:
            x = self.unnormalize(x)
        x = self.augment(x)
        if normalize:
            x = self.normalize(x)
        return x


def requires_grad(model, flag):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model.module.set_requires_grad(flag)
    else:
        model.requires_grad(flag)
