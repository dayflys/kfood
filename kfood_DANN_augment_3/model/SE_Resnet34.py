import torch.nn as nn
import torch

class SEResent34(nn.Module):
    def __init__(self,embedding):
        super(SEResent34,self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        conv_list = [64,128,256,512]
        self.stage1 = nn.ModuleList([residueblcok(conv_list[0],conv_list[0],True) if i == 0 else residueblcok(conv_list[0],conv_list[0]) for i in range(3)])
        self.stage2 = nn.ModuleList([residueblcok(conv_list[0],conv_list[1],True) if i == 0 else residueblcok(conv_list[1],conv_list[1]) for i in range(4)])
        self.stage3 = nn.ModuleList([residueblcok(conv_list[1],conv_list[2],True) if i == 0 else residueblcok(conv_list[2],conv_list[2]) for i in range(6)])
        self.stage4 = nn.ModuleList([residueblcok(conv_list[2],conv_list[3],True) if i == 0 else residueblcok(conv_list[3],conv_list[3]) for i in range(3)])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.embedding = nn.Sequential(
            nn.Linear(512,embedding),
            nn.BatchNorm1d(embedding)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self,x):
        batch,_,_,_ = x.size()
        
        x = self.first(x)
        
        for module in self.stage1:
            x = module(x)
        
        for module in self.stage2:
            x = module(x)
            
        for module in self.stage3:
            x = module(x)
        
        for module in self.stage4:
            x = module(x)
        x = self.gap(x).view(batch,-1)
        x = self.embedding(x)
        return x 
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1) 
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x
    
class residueblcok(nn.Module):
    def __init__(self,in_channel,out_channel,first=False):
        super(residueblcok,self).__init__()
        self.first = first
        self.se = SEBlock(out_channel)
        self.start = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        if first:
            self.pointwise = nn.Conv2d(in_channel,out_channel,1,stride=2)
            self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,padding=1,stride=2),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel,out_channel,3,padding=1)
        )
        else:
            self.pointwise = nn.Conv2d(in_channel,out_channel,1)
            self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel,out_channel,3,padding=1)
        )


    
    def forward(self,x):
        x = self.start(x)
        x1 = self.pointwise(x)
        x2 = self.conv2(x)
        x3 = self.se(x2)
        x = x2*x3 + x1
        
        return x
    
class ASP(nn.Module):
    
    def __init__(self, in_channels):
        super(ASP, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=in_channels, kernel_size=1),
            nn.Softmax(dim=2)
        )
    
    def forward(self, x):
        
        w = self.attention(x)
        w_mean = torch.sum(x * w, dim=2)
        w_std = torch.sqrt(( torch.sum((x**2) * w, dim=2) - w_mean**2 ).clamp(min=1e-5))
        
        x = torch.cat((w_mean, w_std), dim = 1)
        
        return x