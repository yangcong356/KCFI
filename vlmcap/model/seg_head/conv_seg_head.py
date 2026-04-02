import torch.nn as nn
    
class ConvSegHead(nn.Module):
    def __init__(self, input_channels, output_channels, image_size):
        super(ConvSegHead, self).__init__()

        self.linear = nn.Linear(in_features=input_channels,out_features=256)
        self.upsample = nn.Upsample(size=(image_size, image_size),mode='bilinear', align_corners=False)
        self.conv2d = nn.Conv2d(in_channels=256, out_channels=output_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.linear(x)
        
        B, N, C = x.size()
        H = int(N ** 0.5)
        W = H

        x = self.upsample(x.permute(0, 2, 1).reshape(B, C, H, W))
        x = self.conv2d(x)

        return x