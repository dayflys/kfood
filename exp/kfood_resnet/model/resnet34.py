import torch.nn as nn 

class ResNet34(nn.Module):
    def __init__(self,embedding_size):
        super(ResNet34,self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,stride=2,padding=1)
        )
        layer_list = [3,4,6,3]
        for i in range(4):
            setattr(self, '{}{:d}'.format("layer", i+1), self.make_layer(64*(2**i),layer_list[i]))


        self.linear = nn.Linear(1024,embedding_size)

        self.avgpool = nn.AdaptiveAvgPool2d(1)


    def make_layer(self,conv,num):
        stage = nn.ModuleList()
        for i in range(num):
            if i == num-1 :
                stage.append(BasicBlock(conv,conv*2, downsample=True))
            else:
                stage.append(BasicBlock(conv,conv))
        return nn.Sequential(*stage)



    def forward(self,x):
        batch,_,_,_ = x.size()
        x = self.first(x)
        for i in range(4):
            x = getattr(self, '{}{:d}'.format("layer", i+1))(x)

        x = self.avgpool(x).view(batch,-1)
        x = self.linear(x)
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