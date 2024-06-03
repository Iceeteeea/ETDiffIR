import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools
#以下三个都加上.
from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,newResBlock,convNetBlock,
    LinearAttention, Attention,
    PreNorm, Residual)

from .attention import SpatialTransformer
from .AccUNet_arch import HANCBlock


class ResPath(nn.Module):

    def __init__(self, in_chnls, n_lvl):
        super().__init__()
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        self.bn = nn.BatchNorm2d(in_chnls)

        for i in range(n_lvl):
            self.convs.append(convNetBlock(in_chnls))
            self.bns.append(nn.BatchNorm2d(in_chnls))

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
        return self.bn(x)

class myConditionalUNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, ch_mult=[1, 2, 4, 8],n_filts=64,
                 context_dim=512, use_degra_context=True, use_image_context=False, upscale=1, k1s=[3, 3, 2, 1], k2s=[3, 3, 3, 2]):
        super().__init__()
        self.depth = len(ch_mult)
        self.upscale = upscale  # not used
        self.context_dim = -1 if context_dim is None else context_dim
        self.use_image_context = use_image_context
        self.use_degra_context = use_degra_context
        self.pool = torch.nn.MaxPool2d(2)
        print("using myUnet")


        num_head_channels = 32
        dim_head = num_head_channels

        block_class = functools.partial(newResBlock, conv=default_conv, act=NonLinearity())
        block_class_o = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc * 2, nf, 7)

        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        if self.context_dim > 0 and use_degra_context:
            self.prompt = nn.Parameter(torch.rand(1, time_dim))
            self.text_mlp = nn.Sequential(
                nn.Linear(context_dim, time_dim), NonLinearity(),
                nn.Linear(time_dim, time_dim))
            self.prompt_mlp = nn.Linear(time_dim, time_dim)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        ch_mult = [1] + ch_mult
        self.init_HANC = nn.Sequential(HANCBlock(nf, out_channels=n_filts, k=3, inv_fctr=3), HANCBlock(n_filts, n_filts, k=3, inv_fctr=3))
        for i in range(self.depth):
            dim_in = n_filts * ch_mult[i]
            dim_out = n_filts * ch_mult[i + 1]
            # print(dim_in, dim_out)

            num_heads_in = dim_in // num_head_channels
            num_heads_out = dim_out // num_head_channels
            # print(num_heads_in, num_heads_out)
            dim_head_in = dim_in // num_heads_in

            if use_image_context and context_dim > 0:
                att_down = LinearAttention(dim_in) if i < 3 else SpatialTransformer(dim_in, num_heads_in, dim_head,
                                                                                    depth=1, context_dim=context_dim)
                att_up = LinearAttention(dim_out) if i < 3 else SpatialTransformer(dim_out, num_heads_out, dim_head,
                                                                                   depth=1, context_dim=context_dim)
            else:
                att_down = LinearAttention(dim_in)  # if i < 2 else Attention(dim_in)
                att_up = LinearAttention(dim_out)  # if i < 2 else Attention(dim_out)

            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim, k=k1s[i]),  # b1
                # block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),  # b2
                Residual(PreNorm(dim_in, att_down)),  # atttention
                ResPath(dim_in, self.depth - i),
                Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out),
                # self.pool
                # downsample
            ]))


            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim, k=k2s[i]),
                Residual(PreNorm(dim_out, att_up)),
                Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = nf * ch_mult[-1]
        num_heads_mid = mid_dim // num_head_channels
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim, k=3)
        if use_image_context and context_dim > 0:
            self.mid_attn = Residual(PreNorm(mid_dim, SpatialTransformer(mid_dim, num_heads_mid, dim_head, depth=1,
                                                                         context_dim=context_dim)))
        else:
            self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim, k=3)

        self.final_res_block = block_class_o(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)
        # self.rspth1 = ResPath(nf, 4)
        # self.rspth2 = ResPath(nf * 2, 3)
        # self.rspth3 = ResPath(nf * 4, 2)
        # self.rspth4 = ResPath(nf * 8, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, xt, cond, time, text_context=None, image_context=None):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)

        x = xt - cond
        x = torch.cat([x, cond], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(time)
        if self.context_dim > 0:
            if self.use_degra_context and text_context is not None:
                prompt_embedding = torch.softmax(self.text_mlp(text_context), dim=1) * self.prompt
                prompt_embedding = self.prompt_mlp(prompt_embedding)
                t = t + prompt_embedding

            if self.use_image_context and image_context is not None:
                image_context = image_context.unsqueeze(1)

        h = []
        x = self.init_HANC(x) # 1, 64, 256, 256

        for b1, attn, rspth, downsample in self.downs:  # 这一块对应UNet左半边编码器区域
            x = b1(x, t) #第二次：输入1 64 128 128 输出1 128 128 128 loop1 1 64 256 256 loop2 1 64 128 128
            # h.append(x)
            # print(x.shape)

            # x = b2(x, t)
            x = attn(x, context=image_context) # loop1 1 64 256 256
            x1 = rspth(x)

            h.append(x1)

            x = downsample(x) # loop1 1 64 128 128



        x = self.mid_block1(x, t)
        x = self.mid_attn(x, context=image_context)
        x = self.mid_block2(x, t)

        for b1, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            # h.pop()


            # x = torch.cat([x, h.pop()], dim=1)
            # x = b2(x, t)

            x = attn(x, context=image_context)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W].contiguous()

        return x

if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    import numpy as np

    device = torch.device("cuda")
    model = myConditionalUNet(in_nc=3, out_nc=3, nf=64, ch_mult=[1, 2, 4, 8], context_dim=512, use_degra_context=True, use_image_context=True, n_filts=64)
    model.to(device)
    img = torch.ones([4, 3, 256, 256])
    img = img.to(device)
    cond = torch.randn([4, 3, 256, 256])
    cond = cond.to(device)
    time = 5
    text_context = torch.randn([4, 512])
    text_context = text_context.to(device)
    image_context = torch.randn([4, 512])
    image_context = image_context.to(device)
    output = model(img, cond, time, text_context=text_context, image_context=image_context)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    # print(flop_count_table(FlopCountAnalysis(model, (img, cond, time, text_context, image_context))))
    print(output.shape)

