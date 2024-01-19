import torch
from .framework import Framework


class DANN(Framework):
    def __init__(self, model, criterion):
        super(DANN, self).__init__()
        self.add_module('model', model, flag_train=True)
        self.add_module('criterion', criterion, flag_train=True)

    def __call__(self, x,alpha=None, label=None,domain_label=None):
        x = self.modules['model'](x)

        if label is None:
            x = self.modules['criterion'](x)
            return x
        else:
            loss = self.modules['criterion'](x,alpha, label,domain_label)
            return loss