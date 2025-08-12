from copy import deepcopy
from collections import OrderedDict

import torch
from torch import nn


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay
        self.ema_model = deepcopy(model)

        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_model = self.ema_model.eval()

    @torch.no_grad()
    def update_ema_weights(self, current_model):
        model_params = OrderedDict(current_model.named_parameters())
        ema_model_params = OrderedDict(self.ema_model.named_parameters())
        model_buffers = OrderedDict(current_model.named_buffers())
        ema_model_buffers = OrderedDict(self.ema_model.named_buffers())
        # check if both model contains the same set of keys
        assert model_params.keys() == ema_model_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            ema_model_params[name].sub_(
                (1. - self.decay) * (ema_model_params[name] - param))

        # # check if both model contains the same set of keys
        assert model_buffers.keys() == ema_model_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            ema_model_buffers[name].copy_(buffer)

    def forward(self, inputs, *args, **kwargs):
        if self.training:
            print("EMA model is used in training mode!")
            return self.ema_model(inputs, *args, **kwargs)
        else:
            return self.ema_model(inputs, *args, **kwargs)
