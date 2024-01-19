import torch.nn as nn 

class ResNet18(nn.Module):
    def __init__(self,embedding_size):
        super(ResNet18,self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,stride=2,padding=1)
        )
        for i in range(4):
            if i == 0:
                setattr(self, '{}{:d}'.format("layer", 1), ResidualBlock(64,128))
            elif i == 3:
                setattr(self, '{}{:d}'.format("layer", i+1), ResidualBlock(512, 512,last=True))
            else:
                setattr(self, '{}{:d}'.format("layer", i+1), ResidualBlock(64*(2**i),64*(2**(i+1))))

        self.linear = nn.Linear(512,embedding_size)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        batch,_,_,_ = x.size()
        x = self.first(x)
        for i in range(4):
            x = getattr(self, '{}{:d}'.format("layer", i+1))(x)

        x = self.avgpool(x).view(batch,-1)
        x = self.linear(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,last=False):
        super(ResidualBlock,self).__init__()

        self.first_conv = BasicBlock(in_channel,in_channel)
        if last:
            self.second_conv = BasicBlock(in_channel,in_channel)
        else:
            self.second_conv = BasicBlock(in_channel,out_channel,downsample=True)

    def forward(self,x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,downsample=False):
        super(BasicBlock,self).__init__()
        self.downsample = downsample
        stride = 1
        if downsample:
            self.down=nn.Sequential(
                nn.Conv2d(in_channel,out_channel,1,stride=2),
                nn.BatchNorm2d(out_channel)
            )
            stride = 2

        self.residue = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,1,stride=stride),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel,out_channel,1),
                nn.BatchNorm2d(out_channel)
            )
        self.last = nn.ReLU()

    def forward(self,x):
        x1 = self.residue(x)
        if self.downsample:
            identity = self.down(x)
        else: identity = x
        x = self.last(x1+identity)
        return x
