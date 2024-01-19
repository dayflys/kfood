from abc import ABCMeta, abstractmethod
from typing import Optional
import torch
import copy

class Framework(metaclass=ABCMeta):
    def __init__(self):
        self.modules = {}
        self.freeze_modules = {}
        self.trainable_modules = {}
        self.device = 'cpu'
        self.ddp = False

    @abstractmethod
    def __call__(self, x: torch.Tensor, *labels: Optional[torch.Tensor]):
        pass

    def add_module(self, key, model, flag_train=True):
        self.modules[key] = model
        self.set_module_trainability(key, flag_train)

    def set_module_trainability(self, key, flag_train):
        for param in self.modules[key].parameters():
            param.requires_grad=flag_train
        if not flag_train:
            self.modules[key].eval()

    def cuda(self):
        for key in self.modules.keys():
            self.modules[key].cuda()
        self.device = 'cuda'

    def get_parameters(self):
        params = []
        for key, model in self.modules.items():
            if self.is_trainable(key):
                params += list(model.parameters())
        return params

    def get_num_trainable_parameters(self):
        num = 0
        for key, model in self.modules.items():
            if self.is_trainable(key):
                for param in model.parameters():
                    num += param.numel()
        return num

    def copy_state_dict(self):
        output = {}
        for key, model in self.modules.items():
            if 0 < len(model.state_dict().keys()):
                output[key] = copy.deepcopy(model.state_dict())

        return output

    def load_state_dict(self, state_dict):
        for key, params in state_dict.items():
            self.modules[key].load_state_dict(params)

    def eval(self):
        for key in self.modules.keys():
            if self.is_trainable(key):
                self.modules[key].eval()

    def train(self):
        for key in self.modules.keys():
            if self.is_trainable(key):
                self.modules[key].train()

    def is_trainable(self, key):
        flag_param_exist = 0 < len(self.modules[key].state_dict().keys())
        flag_require_grad = False
        if flag_param_exist:
            for param in self.modules[key].parameters():
                flag_require_grad = param.requires_grad
                if flag_require_grad:
                    break
        return flag_param_exist and flag_require_grad

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
