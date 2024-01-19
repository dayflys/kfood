import torch 
import torch.nn as nn 
from torch.autograd import Function

class CCE(torch.nn.Module):
    def __init__(self, embedding_size, num_class, class_weight=None):
        super(CCE, self).__init__()
        if class_weight is not None and type(class_weight) is list:
            class_weight = torch.FloatTensor(class_weight)
        self.softmax_loss = nn.CrossEntropyLoss(weight=class_weight)
        self.classifier = nn.Sequential(
            nn.Linear(1024,embedding_size),
            nn.ReLU(),
            nn.Linear(in_features=embedding_size, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=num_class)
        )
        self.fc1 = nn.Linear(embedding_size, num_class)
        self.bn1 = nn.BatchNorm1d(num_class)

    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        x = self.classifier(x)
        if label is not None:
            loss = self.softmax_loss(x, label)
            return loss
        else:
            return x
        
        

class Discriminator(nn.Module):
    def __init__(self,embedding_size,num_class,class_weight=None):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(1024,embedding_size),
            # nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(in_features=embedding_size, out_features=num_class),
        )
        if class_weight is not None and type(class_weight) is list:
            class_weight = torch.FloatTensor(class_weight)
        self.softmax_loss = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, x,alpha,label):
        reversed_input = ReverseLayerF.apply(x, alpha)
        x = self.discriminator(reversed_input)
        
        loss = self.softmax_loss(x,label)
        return loss
        
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
class DANNLoss(nn.Module):
    def __init__(self,embedding_size,cls_num_class,disc_num_class,cls_class_weight=None,disc_class_weight=None):
        super(DANNLoss, self).__init__()
        self.classifier = CCE(embedding_size,cls_num_class,cls_class_weight)
        self.discriminator = Discriminator(embedding_size,disc_num_class,disc_class_weight)
        
    def forward(self,x,alpha=None,label=None,domain_label=None):
        if label is not None:
            cls_loss = self.classifier(x,label)
            disc_loss = self.discriminator(x,alpha,domain_label)
            return cls_loss,disc_loss
        else:
            cls_pred = self.classifier(x)
            return cls_pred
    
    