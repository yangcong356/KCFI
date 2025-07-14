import torch
import torch.nn as nn
from torch.nn import functional as F


class Flow_Block(nn.Module):

    def __init__(self, in_channels, kernel_size) -> None:
        super(Flow_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False, groups=in_channels*2)
        self.insnorm1 =nn.InstanceNorm2d(in_channels*2)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels, in_channels*2, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False, groups=in_channels)
        self.insnorm3 =nn.InstanceNorm2d(in_channels*2)
        self.gelu3 = nn.GELU()
        self.predict_flow = nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.insnorm1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.insnorm3(x)
        x = self.gelu3(x)
        x = self.predict_flow(x)

        return x

class KCPM(nn.Module):
    """Key Change Perception Module.

    Args:
        in_channels (int): Input channels of features.
    """

    def __init__(self,
                 in_channels):
        super(KCPM, self).__init__()
        self.in_channels = in_channels // 4
        
        kernel_size = 5
        self.flow_make_g = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False, groups=in_channels*2),
            nn.InstanceNorm2d(in_channels*2),
            nn.GELU(),
            nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=True),
        )
        # 使用ModuleList存储Flow_Block实例
        self.flows = nn.ModuleList([
            Flow_Block(in_channels=self.in_channels, kernel_size=kernel_size)
            for _ in range(4)
        ])
    
    def forward(self, x1, x2, fusion_policy=None):
        """Forward function."""
        B, N, C = x1.size()
        H = int(N ** 0.5)
        W = H

        x1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
        x2 = x2.permute(0, 2, 1).reshape(B, C, H, W)

        x1_chunks = torch.chunk(x1, chunks=4, dim=1)
        x2_chunks = torch.chunk(x2, chunks=4, dim=1)

        outputs = [
            torch.cat((a_chunk, b_chunk), dim=1)
            for a_chunk, b_chunk in zip(x1_chunks, x2_chunks)
        ]

        x1_feats = []
        x2_feats = []

        # 使用循环来处理每个Flow_Block和对应的输出
        for flow_block, out, x1_c, x2_c in zip(
                self.flows, outputs, x1_chunks, x2_chunks):
            flow = flow_block(out)
            f1, f2 = torch.chunk(flow, 2, dim=1)
            x1_feat = self.warp(x1_c, f1) - x2_c
            x2_feat = self.warp(x2_c, f2) - x1_c
            x1_feats.append(x1_feat)
            x2_feats.append(x2_feat)

        x1_feat = torch.cat(x1_feats, dim=1)
        x2_feat = torch.cat(x2_feats, dim=1)

        output_l = torch.cat([x1_feat, x2_feat], dim=1)
        flow_g = self.flow_make_g(output_l)
        f1_g, f2_g = torch.chunk(flow_g, 2, dim=1)
        x1_feat_g = self.warp(x1_feat, f1_g) - x2_feat
        x2_feat_g = self.warp(x2_feat, f2_g) - x1_feat
        
        if fusion_policy is None:
            return x1_feat_g, x2_feat_g
        
        output = self.fusion(x1_feat_g, x2_feat_g, fusion_policy)
        output = output.view(B, C, N).permute(0, 2, 1)
        return output


    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output

    @staticmethod
    def fusion(x1, x2, policy):
        """Specify the form of feature fusion"""
        
        _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
        assert policy in _fusion_policies, f'The fusion policies {_fusion_policies} are supported'
        
        if policy == 'concat':
            x = torch.cat([x1, x2], dim=2)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'abs_diff':
            x = torch.abs(x1 - x2)

        return x