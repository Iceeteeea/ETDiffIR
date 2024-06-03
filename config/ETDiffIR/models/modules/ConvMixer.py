import torch
from torch import nn
# from module_util import convNetBlock as ConvNextB

class PatchEmbeddings(nn.Module):
    def __init__(self, d_model, patch_size, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x

class ConvMixerLayer(nn.Module):
    def __init__(self, d_model, kernel_size):
        super().__init__()
        self.depth_wise_conv = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, groups=d_model, padding=(kernel_size - 1) // 2)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm2d(d_model)
        self.point_wise_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.act2 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(d_model)

    def forward(self, x):
        residual = x
        x = self.depth_wise_conv(x)
        x = self.act1(x)
        x = self.norm1(x)
        x += residual
        x = self.point_wise_conv(x)
        x = self.act2(x)
        x = self.norm2(x)

        return x

class ConvMixer(nn.Module):
    def __init__(self, d_model, kernel_size, n_layers: int,patch_emb):
        super().__init__()
        self.patch_emb = patch_emb
        self.conv_mixer_layers = nn.ModuleList([])
        self.conv_mixer_layer = ConvMixerLayer(d_model=d_model, kernel_size=kernel_size)
        for i in range(n_layers):
            self.conv_mixer_layers.append(self.conv_mixer_layer)

    def forward(self, x):
        x = self.patch_emb(x)
        print(x.shape)
        for layer in self.conv_mixer_layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda")
    model = ConvMixer(d_model=3, kernel_size=7, n_layers=4, patch_emb=ConvNextB(3))
    model.to(device)
    img = torch.ones([1, 3, 256, 256])
    img = img.to(device)
    output = model(img)
    print(output.shape)