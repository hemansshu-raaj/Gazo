"""Small plain-conv U-Net student: Conv+ReLU only (no LayerNorm / attention), so it exports cleanly
to ONNX and quantises well to INT8 for CPU / mobile inference."""
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class StudentUNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=12, width=16, levels=3):
        super().__init__()
        chans = [width * 2 ** i for i in range(levels + 1)]
        self.inc = _block(in_ch, chans[0])
        self.downs = nn.ModuleList(nn.Sequential(nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1),
                                                 nn.ReLU(inplace=True), _block(chans[i + 1], chans[i + 1]))
                                   for i in range(levels))
        self.ups = nn.ModuleList(nn.Sequential(nn.Conv2d(chans[i + 1], chans[i] * 4, 1), nn.PixelShuffle(2))
                                 for i in reversed(range(levels)))
        self.decs = nn.ModuleList(_block(chans[i], chans[i]) for i in reversed(range(levels)))
        self.out = nn.Conv2d(chans[0], out_ch, 3, padding=1)
        self.factor = 2 ** levels

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.factor, (-w) % self.factor
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        f = self.inc(x)
        skips = [f]
        for d in self.downs:
            f = d(f)
            skips.append(f)
        skips.pop()
        for up, dec in zip(self.ups, self.decs):
            f = dec(up(f) + skips.pop())
        return self.out(f)[..., :h, :w]
