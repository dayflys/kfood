import torch 
import torch.nn as nn 

class CCE(torch.nn.Module):
    def __init__(self, embedding_size, num_class, class_weight=None):
        super(CCE, self).__init__()
        if class_weight is not None and type(class_weight) is list:
            class_weight = torch.FloatTensor(class_weight)
        self.softmax_loss = nn.CrossEntropyLoss(weight=class_weight)
        self.fc1 = nn.Linear(embedding_size, num_class)
        self.bn1 = nn.BatchNorm1d(num_class)

    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        if label is not None:
            loss = self.softmax_loss(x, label)
            return loss
        else:
            return x