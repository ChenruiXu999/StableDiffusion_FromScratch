import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_attention(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(1,channels)

    def forward(self,x):
        # x: (Batch_Size, Features, Height, Width)
        residue = x 
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features). Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
        x = x.transpose(-1, -2)
        # Perform self-attention WITHOUT mask
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width) 
        x += residue
        # (Batch_Size, Features, Height, Width)
        return x 

class VAE_ResitualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super.__init__()
        self.groupnorm_1 = nn.GroupNorm(32,in_channels)#组归一化是对批量归一化（Batch Normalization）的一种改进，它将通道分成若干组，然后对每一组内的数据进行归一化,32是组组数
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.groupnorm_2 = nn.GroupNorm(32,out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
        
    def forward(self,x):
        residual = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x+self.residual_layer(residual)
    


