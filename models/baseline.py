"""From-scratch ResNet-18 + spatial MLP trajectory baseline (~20M parameters)."""
import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,channels,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(channels)
        self.conv2=nn.Conv2d(channels,channels,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(channels)
        self.skip=nn.Identity() if stride==1 and in_channels==channels else nn.Sequential(
            nn.Conv2d(in_channels,channels,1,stride,bias=False),nn.BatchNorm2d(channels))
        self.act=nn.ReLU(inplace=True)

    def forward(self,x):
        residual=self.skip(x)
        x=self.act(self.bn1(self.conv1(x)))
        return self.act(self.bn2(self.conv2(x))+residual)


class ResNet18Trajectory(nn.Module):
    def __init__(self,steps=10,pooling_size=4,mlp_hidden=1024,latent_dim=512):
        super().__init__()
        self.steps=steps
        self.stem=nn.Sequential(nn.Conv2d(4,64,7,2,3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                nn.MaxPool2d(3,2,1))
        blocks=[];previous=64
        for stage,channels in enumerate([64,128,256,512]):
            blocks.extend([BasicBlock(previous,channels,1 if stage==0 else 2),BasicBlock(channels,channels)])
            previous=channels
        self.backbone=nn.Sequential(*blocks)
        self.pool=nn.AdaptiveAvgPool2d((pooling_size,pooling_size))
        self.head=nn.Sequential(nn.Linear(512*pooling_size**2+2,mlp_hidden),nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden,latent_dim),nn.ReLU(inplace=True),nn.Linear(latent_dim,steps*4))
        for module in self.modules():
            if isinstance(module,nn.Conv2d):nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,bev,ego):
        if bev.dtype==torch.uint8:bev=bev.float()/255
        features=self.pool(self.backbone(self.stem(bev))).flatten(1)
        return self.head(torch.cat([features,ego],dim=1)).reshape(-1,self.steps,4)


def masked_trajectory_loss(prediction,target,valid,action_mask):
    import torch.nn.functional as F
    weights=valid.to(prediction.dtype)*action_mask[:,None].to(prediction.dtype)
    elements=F.smooth_l1_loss(prediction.float(),target.float(),reduction='none').mean(-1)
    return (elements*weights).sum()/weights.sum().clamp_min(1)
