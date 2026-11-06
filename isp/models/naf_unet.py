"""NAFNet-style U-Net (Chen et al., ECCV 2022: "Simple Baselines for Image Restoration").

Used for the hybrid teacher (packed RAW + noise map -> linear RGB) and the end-to-end baseline
(packed RAW -> sRGB). Output is produced at packed (half) resolution with 12 channels and
PixelShuffle'd to full resolution, so the network also learns demosaicing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, 2 * c, 1)
        self.conv2 = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)  # depthwise
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))  # simplified channel attention
        self.conv3 = nn.Conv2d(c, c, 1)
        self.conv4 = nn.Conv2d(c, 2 * c, 1)
        self.conv5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFUNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=12, width=32, enc=(1, 1, 2), middle=4, dec=(1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.encoders, self.downs, self.ups, self.decoders = (nn.ModuleList() for _ in range(4))
        c = width
        for n in enc:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.ending = nn.Conv2d(width, out_ch, 3, padding=1)
        self.factor = 2 ** len(enc)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.factor, (-w) % self.factor
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        f = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            f = enc(f)
            skips.append(f)
            f = down(f)
        f = self.middle(f)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            f = dec(up(f) + skip)
        return self.ending(f)[..., :h, :w]
