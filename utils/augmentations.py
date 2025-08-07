import random
import torch
from PIL import Image


class RandomRotation90():

    def __init__(self):
        pass

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x_rot = self.pytorch_rotate(x)
        else:
            x_rot = self.pil_rotate(x)
        return x_rot

    def pil_rotate(self, x):
        rotations = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        rotation = rotations[random.randint(0, 3)]
        if rotation is not None:
            x = x.transpose(rotation)
        return x

    def pytorch_rotate(self, x):
        return torch.rot90(x, k=torch.randint(0, 4), dims=[-2, -1])


class TwoCropsTransform():

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return self.transform1(x), self.transform2(x)
