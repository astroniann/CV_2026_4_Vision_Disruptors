"""
3D U-Net for medical image synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    3D Convolutional Block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class DownBlock3D(nn.Module):
    """Encoder block: ConvBlock -> MaxPool"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        
        self.conv_block = ConvBlock3D(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class UpBlock3D(nn.Module):
    """Decoder block: Upsample -> Concat -> ConvBlock"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels // 2, 
            kernel_size=2, stride=2
        )
        self.conv_block = ConvBlock3D(in_channels, out_channels, dropout)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for MRI synthesis
    
    Architecture:
    - 4 encoder levels (down blocks)
    - Bottleneck
    - 4 decoder levels (up blocks)
    - Skip connections between encoder and decoder
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=32,
        num_levels=4,
        dropout=0.1
    ):
        """
        Args:
            in_channels: Number of input modalities (e.g., 3 for T1, T2, FLAIR)
            out_channels: Number of output modalities (e.g., 1 for T1ce)
            base_channels: Number of channels in first layer
            num_levels: Number of encoder/decoder levels
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        channels = base_channels
        
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else channels // 2
            self.encoders.append(DownBlock3D(in_ch, channels, dropout))
            channels *= 2
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(channels // 2, channels, dropout)
        
        # Decoder (upsampling path)
        self.decoders = nn.ModuleList()
        
        for i in range(num_levels):
            self.decoders.append(UpBlock3D(channels, channels // 2, dropout))
            channels //= 2
        
        # Final convolution
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, in_channels, D, H, W)
        
        Returns:
            Output tensor (B, out_channels, D, H, W)
        """
        # Encoder path with skip connections
        skips = []
        
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_unet():
    """Test U-Net forward pass"""
    
    print("Testing 3D U-Net...")
    
    # Create model
    model = UNet3D(
        in_channels=3,
        out_channels=1,
        base_channels=32,
        num_levels=4,
        dropout=0.1
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 1
    x = torch.randn(batch_size, 3, 240, 240, 155)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 1, 240, 240, 155), "Output shape mismatch!"
    
    print("✅ U-Net test passed!")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\nTesting on GPU...")
        model = model.cuda()
        x = x.cuda()
        
        with torch.no_grad():
            output = model(x)
        
        print(f"GPU output shape: {output.shape}")
        print("✅ GPU test passed!")
    else:
        print("\n⚠️ CUDA not available, skipping GPU test")


if __name__ == '__main__':
    test_unet()