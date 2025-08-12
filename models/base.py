from torch import nn


class BaseModel(nn.Module):

    def _init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def set_requires_grad(self, flag):
        assert flag in [True, False], "flag should be a boolean"
        for p in self.parameters():
            p.requires_grad = flag
