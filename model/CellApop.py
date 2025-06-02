import torch
import torch.nn as nn
from .common import LayerNorm2d, UpSampleLayer, OpSequential, MLPBlock
from .rep_vit import rep_vit_m1, RepViTBlock, _make_divisible, Conv2d_BN
from thop import profile

class ResidualBlock(nn.Module):
    """不影响输入输出分辨率的处理层"""
    def __init__(self, channels, use_se=True, use_hs=True):
        super(ResidualBlock, self).__init__()
        
        self.process = RepViTBlock(
            inp=channels, 
            hidden_dim=2 * channels,  # 这个值必须是inp的2倍
            oup=channels,             # 输出通道数等于输入通道数
            kernel_size=3, 
            stride=1,                 # stride=1确保尺寸不变
            use_se=use_se, 
            use_hs=use_hs
        )
        
        self.ffn = nn.Sequential(
            Conv2d_BN(channels, channels * 2, 1, 1, 0),
            nn.GELU(),
            Conv2d_BN(channels * 2, channels, 1, 1, 0)
        )
        
        self.norm = LayerNorm2d(channels)

    def forward(self, x):
        identity = x
        x = self.process(x)
        x = self.ffn(x)
        return self.norm(x + identity)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_se=True, use_hs=True):
        super(DecoderBlock, self).__init__()
        
        self.process = RepViTBlock(
            inp=in_channels, hidden_dim=2 * in_channels, oup=in_channels,
            kernel_size=3, stride=1, use_se=use_se, use_hs=use_hs
        )
        
        self.pre_cat_conv = Conv2d_BN(in_channels, out_channels, 1, 1, 0)
        
        self.upsample = nn.ConvTranspose2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        
        self.upsample_norm = LayerNorm2d(out_channels)
        self.upsample_act = nn.GELU()
        
        self.post_cat_conv = Conv2d_BN(out_channels + skip_channels, out_channels, 1, 1, 0)
        
        self.ffn = nn.Sequential(
            Conv2d_BN(out_channels, out_channels * 2, 1, 1, 0),
            nn.GELU(),
            Conv2d_BN(out_channels * 2, out_channels, 1, 1, 0)
        )

    def forward(self, x, skip):
        x = self.process(x)
        x = self.pre_cat_conv(x)
        x = self.upsample_act(self.upsample_norm(self.upsample(x)))
        
        if x.shape[2:] != skip.shape[2:]:
            skip = torch.nn.functional.interpolate(
                skip, size=x.shape[2:], mode='bilinear', align_corners=False
            )
        
        x = torch.cat([x, skip], dim=1)
        x = self.post_cat_conv(x)
        return self.ffn(x)

class CellApop(nn.Module):
    def __init__(self, img_size=1024, num_classes=1, encoder_fuse=True, freeze=False):
        super(CellApop, self).__init__()
        
        # 保存freeze参数
        self.freeze = freeze
        
        self.encoder = rep_vit_m1(img_size=img_size, fuse=encoder_fuse, freeze=freeze)
        
        # 通道配置 - 确保与编码器输出匹配
        self.channels = [48, 96, 192, 384]  # stage0, stage1, stage2, stage3
        encoder_out_channels = 256 if encoder_fuse else 384
        
        # re_conv层 - 将编码器输出调整到stage3通道数
        self.re_conv = nn.Sequential(
            Conv2d_BN(encoder_out_channels, self.channels[3], 3, 2, 1),
            nn.GELU()
        )
        
        # 残差块配置：[re_conv后, stage3后, stage2后, stage1后]
        residual_counts = [2, 6, 3, 3]
        se_hs_configs = [(True, True), (True, True), (True, False), (True, False)]
        
        # 创建残差块 - 修复通道数分配
        self.residual_blocks = nn.ModuleDict()
        stage_names = ['re_conv', 'stage3', 'stage2', 'stage1']
        stage_channels = [self.channels[3], self.channels[2], self.channels[1], self.channels[0]]  # [384, 192, 96, 48]
        
        for stage_name, count, channels, (use_se, use_hs) in zip(stage_names, residual_counts, stage_channels, se_hs_configs):
            self.residual_blocks[stage_name] = nn.ModuleList([
                ResidualBlock(channels, use_se=use_se, use_hs=use_hs) for _ in range(count)
            ])
        
        # 解码器配置
        decoder_configs = [
            (self.channels[3], self.channels[2], self.channels[2], True, True),   # 384->192 (stage3->2)
            (self.channels[2], self.channels[1], self.channels[1], True, False),  # 192->96  (stage2->1)
            (self.channels[1], self.channels[0], self.channels[0], True, False),  # 96->48   (stage1->0)
        ]
        
        # 创建解码器
        self.decoders = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, use_se, use_hs) 
            for in_ch, skip_ch, out_ch, use_se, use_hs in decoder_configs
        ])
        
        # 最终输出层
        self.final_upsample = nn.Sequential(
            Conv2d_BN(self.channels[0], 32, 3, 1, 1), nn.GELU(),
            Conv2d_BN(32, 16, 3, 1, 1), nn.GELU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            Conv2d_BN(16, num_classes, 1, 1, 0)
        )
    
    def train(self, mode=True):
        """重写train方法以支持freeze参数"""
        super(CellApop, self).train(mode)
        
        if self.freeze:
            # 冻结编码器（已经在RepViT的train方法中处理）
            self.encoder.eval()
            
            # 冻结解码器部分
            self.re_conv.eval()
            
            # 冻结所有残差块
            for stage_blocks in self.residual_blocks.values():
                for block in stage_blocks:
                    block.eval()
            
            # 冻结解码器模块
            for decoder in self.decoders:
                decoder.eval()
            
            # 冻结最终上采样层
            self.final_upsample.eval()
            
            # 冻结所有参数
            for param in self.parameters():
                param.requires_grad = False
    
    def _apply_residual_blocks(self, x, stage_name):
        """应用指定阶段的残差块"""
        for block in self.residual_blocks[stage_name]:
            x = block(x)
        return x

    def forward(self, x):
        features = self.encoder(x)
        
        # 编码器输出处理
        x = self.re_conv(features['final'])
        x = self._apply_residual_blocks(x, 're_conv')
        
        # 解码器处理
        skip_features = [features['stage2'], features['stage1'], features['stage0']]
        stage_names = ['stage3', 'stage2', 'stage1']
        
        for decoder, skip, stage_name in zip(self.decoders, skip_features, stage_names):
            x = decoder(x, skip)
            x = self._apply_residual_blocks(x, stage_name)
        
        return self.final_upsample(x), features['final']

if __name__ == '__main__':
    model = CellApop(img_size=1024, num_classes=1, encoder_fuse=True, freeze=True)
    model.eval()
    input_tensor, _ = torch.randn(1, 3, 1024, 1024)
    output = model(input_tensor)
    print(f"Final output shape: {output.shape}")
    # 计算模型的FLOPs
    flops, _ = profile(model, inputs=(input_tensor,))
    print(f"FLOPs: {flops/1e9:.2f}G")
    # 计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
