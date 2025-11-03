import torch
import torch.nn as nn

# ── Efficient Channel Attention ──────────────────────────────────────────────
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # x: [N, C, H, W]
        y = self.avg_pool(x)                                # [N, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)                 # [N, 1, C]
        y = self.conv(y)                                    # [N, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [N, C, 1, 1]
        return x * y.expand_as(x)                           # [N, C, H, W]

# ── MCP-X Module ─────────────────────────────────────────────────────────────
class MCPX(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_paths):
        super().__init__()
        # Multiple branches: num_paths 3×3 Conv→BN→ReLU
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(inplace=True)
            ) for _ in range(num_paths)
        ])
        # Expert Routing: global average pooling, then 1×1 Conv to get num_paths weights, followed by Softmax
        self.attn_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, num_paths, kernel_size=1),
            nn.Softmax(dim=1)
        )
        # Concatenate all branches, then 1×1 Conv→BN→ReLU to restore to out_ch
        self.merge_conv = nn.Sequential(
            nn.Conv2d(hid_ch * num_paths, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # Channel attention
        self.eca      = ECABlock(out_ch)
        # Residual shortcut: in_ch→out_ch
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: [N, in_ch, H, W]
        w = self.attn_weights(x)  # [N, num_paths, 1, 1]
        outs = []
        for i, p in enumerate(self.paths):
            o  = p(x)                    # [N, hid_ch, H, W]
            wi = w[:, i:i+1, :, :].expand_as(o)
            outs.append(o * wi)          # Weighted branch output
        cat    = torch.cat(outs, dim=1)  # [N, hid_ch * num_paths, H, W]
        merged = self.merge_conv(cat)    # [N, out_ch, H, W]
        attn   = self.eca(merged)        # [N, out_ch, H, W]
        return attn + self.residual(x)   # Residual addition

# ── MCP-X without ECA (only multi-branch + routing + residual) ─────────────────────────────
class MCPX_NoECA(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_paths):
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(inplace=True)
            ) for _ in range(num_paths)
        ])
        self.attn_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, num_paths, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d(hid_ch * num_paths, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        w = self.attn_weights(x)
        outs = []
        for i, p in enumerate(self.paths):
            o = p(x)
            wi = w[:, i:i+1, :, :].expand_as(o)
            outs.append(o * wi)
        cat    = torch.cat(outs, dim=1)
        merged = self.merge_conv(cat)
        # Direct residual addition, no ECA
        return merged + self.residual(x)

# ── MetaCortexNet (Full Version) ─────────────────────────────────────────────────
class MetaCortexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Stage 1: Shallow encoding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # Stage 2: MCP-X module
        self.mcpx  = MCPX(in_ch=64, hid_ch=16, out_ch=64, num_paths=4)

        # Stage 3: BRSE decoding
        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.enc2  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv= nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.outc  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

        # Stage 4: Meta-Causal Attention
        self.attn_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Stage 5: Classification head
        self.gap       = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final  = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        # Stage 1
        x = self.relu1(self.conv1(x))  # [N, 32, 224, 224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112, 112]
        # Stage 2
        x = self.mcpx(x)               # [N, 64, 112, 112]
        # Stage 3 (BRSE)
        x = self.relu3(self.enc1(x))   # [N, 64, 112, 112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N, 128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64, 112, 112]
        x = self.relu6(self.outc(x))   # [N, 64, 112, 112]
        # Stage 4 (Attention)
        x = self.attn_pool(x)          # [N, 64, 32, 32]
        C, H, W = x.shape[1:]
        x = x.view(N, C, H*W).permute(0, 2, 1)  # [N, H*W, C]
        x, _ = self.attn(x, x, x)      # [N, H*W, C]
        x = x.permute(0, 2, 1).view(N, C, H, W) # [N, 64, 32, 32]
        # Stage 5 (Classification)
        x = self.gap(x).view(N, -1)    # [N, 64]
        return self.fc_final(x)        # [N, num_classes]

# ── Baseline model: Remove MCP-X, keep only shallow + BRSE + Attention ─────────────────────
class BaselineNet(nn.Module):
    """
    Baseline: Skip MCP-X module
    """
    def __init__(self, num_classes):
        super().__init__()
        # Stage 1: Shallow encoding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        # Skip Stage 2 (MCP-X)

        # Stage 3: BRSE
        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.enc2  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5  = nn.ReLU(inplace=True)
        self.outc   = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6  = nn.ReLU(inplace=True)

        # Stage 4: Attention
        self.attn_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Stage 5: Classification head
        self.gap      = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        x = self.relu1(self.conv1(x))  # [N, 32, 224, 224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112, 112]
        # Stage 3 (BRSE)
        x = self.relu3(self.enc1(x))   # [N, 64, 112, 112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N, 128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64, 112, 112]
        x = self.relu6(self.outc(x))   # [N, 64, 112, 112]
        # Stage 4 (Attention)
        x = self.attn_pool(x)          # [N, 64, 32, 32]
        C, H, W = x.shape[1:]
        x = x.view(N, C, H*W).permute(0, 2, 1)  # [N, H*W, C]
        x, _ = self.attn(x, x, x)      # [N, H*W, C]
        x = x.permute(0, 2, 1).view(N, C, H, W) # [N, 64, 32, 32]
        # Stage 5 (Classification)
        x = self.gap(x).view(N, -1)    # [N, 64]
        return self.fc_final(x)        # [N, num_classes]

# ── MetaCortexNet without Attention ────────────────────────────────────────
class MetaCortexNet_NoAttn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # MCP-X (can be replaced with MCPX_NoECA or MCPX as needed)
        self.mcpx  = MCPX(in_ch=64, hid_ch=16, out_ch=64, num_paths=4)

        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2,2)
        self.enc2  = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv= nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.outc  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

        # Removed MultiheadAttention
        # self.attn_pool = nn.AdaptiveAvgPool2d((32,32))
        # self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        self.gap      = nn.AdaptiveAvgPool2d((1,1))
        self.fc_final = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        x = self.relu1(self.conv1(x))  # [N, 32, 224,224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112,112]
        x = self.mcpx(x)               # [N, 64, 112,112]
        x = self.relu3(self.enc1(x))   # [N, 64, 112,112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N,128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64,112,112]
        x = self.relu6(self.outc(x))   # [N, 64,112,112]
        # Stage4: No Attention
        x = self.gap(x).view(N, -1)    # [N,64]
        return self.fc_final(x)        # [N, num_classes] 
import torch.nn as nn

# ── Efficient Channel Attention ──────────────────────────────────────────────
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # x: [N, C, H, W]
        y = self.avg_pool(x)                                # [N, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)                 # [N, 1, C]
        y = self.conv(y)                                    # [N, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [N, C, 1, 1]
        return x * y.expand_as(x)                           # [N, C, H, W]

# ── MCP-X Module ─────────────────────────────────────────────────────────────
class MCPX(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_paths):
        super().__init__()
        # Multiple branches: num_paths 3×3 Conv→BN→ReLU
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(inplace=True)
            ) for _ in range(num_paths)
        ])
        # Expert Routing: global average pooling, then 1×1 Conv to get num_paths weights, followed by Softmax
        self.attn_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, num_paths, kernel_size=1),
            nn.Softmax(dim=1)
        )
        # Concatenate all branches, then 1×1 Conv→BN→ReLU to restore to out_ch
        self.merge_conv = nn.Sequential(
            nn.Conv2d(hid_ch * num_paths, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # Channel attention
        self.eca      = ECABlock(out_ch)
        # Residual shortcut: in_ch→out_ch
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: [N, in_ch, H, W]
        w = self.attn_weights(x)  # [N, num_paths, 1, 1]
        outs = []
        for i, p in enumerate(self.paths):
            o  = p(x)                    # [N, hid_ch, H, W]
            wi = w[:, i:i+1, :, :].expand_as(o)
            outs.append(o * wi)          # Weighted branch output
        cat    = torch.cat(outs, dim=1)  # [N, hid_ch * num_paths, H, W]
        merged = self.merge_conv(cat)    # [N, out_ch, H, W]
        attn   = self.eca(merged)        # [N, out_ch, H, W]
        return attn + self.residual(x)   # Residual addition

# ── MCP-X without ECA (only multi-branch + routing + residual) ─────────────────────────────
class MCPX_NoECA(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_paths):
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(inplace=True)
            ) for _ in range(num_paths)
        ])
        self.attn_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, num_paths, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d(hid_ch * num_paths, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        w = self.attn_weights(x)
        outs = []
        for i, p in enumerate(self.paths):
            o = p(x)
            wi = w[:, i:i+1, :, :].expand_as(o)
            outs.append(o * wi)
        cat    = torch.cat(outs, dim=1)
        merged = self.merge_conv(cat)
        # Direct residual addition, no ECA
        return merged + self.residual(x)

# ── MetaCortexNet (Full Version) ─────────────────────────────────────────────────
class MetaCortexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Stage 1: Shallow encoding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # Stage 2: MCP-X module
        self.mcpx  = MCPX(in_ch=64, hid_ch=16, out_ch=64, num_paths=4)

        # Stage 3: BRSE decoding
        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.enc2  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv= nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.outc  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

        # Stage 4: Meta-Causal Attention
        self.attn_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Stage 5: Classification head
        self.gap       = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final  = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        # Stage 1
        x = self.relu1(self.conv1(x))  # [N, 32, 224, 224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112, 112]
        # Stage 2
        x = self.mcpx(x)               # [N, 64, 112, 112]
        # Stage 3 (BRSE)
        x = self.relu3(self.enc1(x))   # [N, 64, 112, 112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N, 128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64, 112, 112]
        x = self.relu6(self.outc(x))   # [N, 64, 112, 112]
        # Stage 4 (Attention)
        x = self.attn_pool(x)          # [N, 64, 32, 32]
        C, H, W = x.shape[1:]
        x = x.view(N, C, H*W).permute(0, 2, 1)  # [N, H*W, C]
        x, _ = self.attn(x, x, x)      # [N, H*W, C]
        x = x.permute(0, 2, 1).view(N, C, H, W) # [N, 64, 32, 32]
        # Stage 5 (Classification)
        x = self.gap(x).view(N, -1)    # [N, 64]
        return self.fc_final(x)        # [N, num_classes]

# ── Baseline model: Remove MCP-X, keep only shallow + BRSE + Attention ─────────────────────
class BaselineNet(nn.Module):
    """
    Baseline: Skip MCP-X module
    """
    def __init__(self, num_classes):
        super().__init__()
        # Stage 1: Shallow encoding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        # Skip Stage 2 (MCP-X)

        # Stage 3: BRSE
        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.enc2  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5  = nn.ReLU(inplace=True)
        self.outc   = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6  = nn.ReLU(inplace=True)

        # Stage 4: Attention
        self.attn_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Stage 5: Classification head
        self.gap      = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        x = self.relu1(self.conv1(x))  # [N, 32, 224, 224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112, 112]
        # Stage 3 (BRSE)
        x = self.relu3(self.enc1(x))   # [N, 64, 112, 112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N, 128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64, 112, 112]
        x = self.relu6(self.outc(x))   # [N, 64, 112, 112]
        # Stage 4 (Attention)
        x = self.attn_pool(x)          # [N, 64, 32, 32]
        C, H, W = x.shape[1:]
        x = x.view(N, C, H*W).permute(0, 2, 1)  # [N, H*W, C]
        x, _ = self.attn(x, x, x)      # [N, H*W, C]
        x = x.permute(0, 2, 1).view(N, C, H, W) # [N, 64, 32, 32]
        # Stage 5 (Classification)
        x = self.gap(x).view(N, -1)    # [N, 64]
        return self.fc_final(x)        # [N, num_classes]

# ── MetaCortexNet without Attention ────────────────────────────────────────
class MetaCortexNet_NoAttn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # MCP-X (can be replaced with MCPX_NoECA or MCPX as needed)
        self.mcpx  = MCPX(in_ch=64, hid_ch=16, out_ch=64, num_paths=4)

        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2,2)
        self.enc2  = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv= nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.outc  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

        # Removed MultiheadAttention
        # self.attn_pool = nn.AdaptiveAvgPool2d((32,32))
        # self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        self.gap      = nn.AdaptiveAvgPool2d((1,1))
        self.fc_final = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        x = self.relu1(self.conv1(x))  # [N, 32, 224,224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112,112]
        x = self.mcpx(x)               # [N, 64, 112,112]
        x = self.relu3(self.enc1(x))   # [N, 64, 112,112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N,128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64,112,112]
        x = self.relu6(self.outc(x))   # [N, 64,112,112]
        # Stage4: No Attention
        x = self.gap(x).view(N, -1)    # [N,64]
        return self.fc_final(x)        # [N, num_classes] 
import torch.nn as nn

# ── Efficient Channel Attention ──────────────────────────────────────────────
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # x: [N, C, H, W]
        y = self.avg_pool(x)                                # [N, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)                 # [N, 1, C]
        y = self.conv(y)                                    # [N, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [N, C, 1, 1]
        return x * y.expand_as(x)                           # [N, C, H, W]

# ── MCP-X Module ─────────────────────────────────────────────────────────────
class MCPX(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_paths):
        super().__init__()
        # Multiple branches: num_paths 3×3 Conv→BN→ReLU
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(inplace=True)
            ) for _ in range(num_paths)
        ])
        # Expert Routing: global average pooling, then 1×1 Conv to get num_paths weights, followed by Softmax
        self.attn_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, num_paths, kernel_size=1),
            nn.Softmax(dim=1)
        )
        # Concatenate all branches, then 1×1 Conv→BN→ReLU to restore to out_ch
        self.merge_conv = nn.Sequential(
            nn.Conv2d(hid_ch * num_paths, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # Channel attention
        self.eca      = ECABlock(out_ch)
        # Residual shortcut: in_ch→out_ch
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: [N, in_ch, H, W]
        w = self.attn_weights(x)  # [N, num_paths, 1, 1]
        outs = []
        for i, p in enumerate(self.paths):
            o  = p(x)                    # [N, hid_ch, H, W]
            wi = w[:, i:i+1, :, :].expand_as(o)
            outs.append(o * wi)          # Weighted branch output
        cat    = torch.cat(outs, dim=1)  # [N, hid_ch * num_paths, H, W]
        merged = self.merge_conv(cat)    # [N, out_ch, H, W]
        attn   = self.eca(merged)        # [N, out_ch, H, W]
        return attn + self.residual(x)   # Residual addition

# ── MCP-X without ECA (only multi-branch + routing + residual) ─────────────────────────────
class MCPX_NoECA(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_paths):
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(inplace=True)
            ) for _ in range(num_paths)
        ])
        self.attn_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, num_paths, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d(hid_ch * num_paths, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        w = self.attn_weights(x)
        outs = []
        for i, p in enumerate(self.paths):
            o = p(x)
            wi = w[:, i:i+1, :, :].expand_as(o)
            outs.append(o * wi)
        cat    = torch.cat(outs, dim=1)
        merged = self.merge_conv(cat)
        # Direct residual addition, no ECA
        return merged + self.residual(x)

# ── MetaCortexNet (Full Version) ─────────────────────────────────────────────────
class MetaCortexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Stage 1: Shallow encoding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # Stage 2: MCP-X module
        self.mcpx  = MCPX(in_ch=64, hid_ch=16, out_ch=64, num_paths=4)

        # Stage 3: BRSE decoding
        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.enc2  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv= nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.outc  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

        # Stage 4: Meta-Causal Attention
        self.attn_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Stage 5: Classification head
        self.gap       = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final  = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        # Stage 1
        x = self.relu1(self.conv1(x))  # [N, 32, 224, 224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112, 112]
        # Stage 2
        x = self.mcpx(x)               # [N, 64, 112, 112]
        # Stage 3 (BRSE)
        x = self.relu3(self.enc1(x))   # [N, 64, 112, 112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N, 128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64, 112, 112]
        x = self.relu6(self.outc(x))   # [N, 64, 112, 112]
        # Stage 4 (Attention)
        x = self.attn_pool(x)          # [N, 64, 32, 32]
        C, H, W = x.shape[1:]
        x = x.view(N, C, H*W).permute(0, 2, 1)  # [N, H*W, C]
        x, _ = self.attn(x, x, x)      # [N, H*W, C]
        x = x.permute(0, 2, 1).view(N, C, H, W) # [N, 64, 32, 32]
        # Stage 5 (Classification)
        x = self.gap(x).view(N, -1)    # [N, 64]
        return self.fc_final(x)        # [N, num_classes]

# ── Baseline model: Remove MCP-X, keep only shallow + BRSE + Attention ─────────────────────
class BaselineNet(nn.Module):
    """
    Baseline: Skip MCP-X module
    """
    def __init__(self, num_classes):
        super().__init__()
        # Stage 1: Shallow encoding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        # Skip Stage 2 (MCP-X)

        # Stage 3: BRSE
        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.enc2  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5  = nn.ReLU(inplace=True)
        self.outc   = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6  = nn.ReLU(inplace=True)

        # Stage 4: Attention
        self.attn_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Stage 5: Classification head
        self.gap      = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        x = self.relu1(self.conv1(x))  # [N, 32, 224, 224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112, 112]
        # Stage 3 (BRSE)
        x = self.relu3(self.enc1(x))   # [N, 64, 112, 112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N, 128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64, 112, 112]
        x = self.relu6(self.outc(x))   # [N, 64, 112, 112]
        # Stage 4 (Attention)
        x = self.attn_pool(x)          # [N, 64, 32, 32]
        C, H, W = x.shape[1:]
        x = x.view(N, C, H*W).permute(0, 2, 1)  # [N, H*W, C]
        x, _ = self.attn(x, x, x)      # [N, H*W, C]
        x = x.permute(0, 2, 1).view(N, C, H, W) # [N, 64, 32, 32]
        # Stage 5 (Classification)
        x = self.gap(x).view(N, -1)    # [N, 64]
        return self.fc_final(x)        # [N, num_classes]

# ── MetaCortexNet without Attention ────────────────────────────────────────
class MetaCortexNet_NoAttn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # MCP-X (can be replaced with MCPX_NoECA or MCPX as needed)
        self.mcpx  = MCPX(in_ch=64, hid_ch=16, out_ch=64, num_paths=4)

        self.enc1  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2,2)
        self.enc2  = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv= nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.outc  = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

        # Removed MultiheadAttention
        # self.attn_pool = nn.AdaptiveAvgPool2d((32,32))
        # self.attn      = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        self.gap      = nn.AdaptiveAvgPool2d((1,1))
        self.fc_final = nn.Linear(64, num_classes)

    def forward(self, x):
        N = x.size(0)
        x = self.relu1(self.conv1(x))  # [N, 32, 224,224]
        x = self.relu2(self.conv2(x))  # [N, 64, 112,112]
        x = self.mcpx(x)               # [N, 64, 112,112]
        x = self.relu3(self.enc1(x))   # [N, 64, 112,112]
        x = self.pool(x)               # [N, 64, 56, 56]
        x = self.relu4(self.enc2(x))   # [N,128, 56, 56]
        x = self.relu5(self.deconv(x)) # [N, 64,112,112]
        x = self.relu6(self.outc(x))   # [N, 64,112,112]
        # Stage4: No Attention
        x = self.gap(x).view(N, -1)    # [N,64]
        return self.fc_final(x)        # [N, num_classes] 