import torch
from framework import Framework


class kfoodframework(Framework):
    def __init__(self, model, criterion, pre_processing=None):
        super(kfoodframework, self).__init__()
        self.pre_processing = pre_processing
        self.add_module('model', model, flag_train=True)
        self.add_module('criterion', criterion, flag_train=True)

    def __call__(self, x, label=None):
        if self.pre_processing is not None:
            with torch.set_grad_enabled(False):
                x = self.pre_processing(x)

        x = self.modules['model'](x)

        if label is None:
            x = self.modules['criterion'](x)
            return x
        else:
            loss = self.modules['criterion'](x, label)
            return loss