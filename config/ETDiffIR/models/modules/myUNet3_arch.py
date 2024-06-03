import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools

# 以下两个前面有.
from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler, newResBlock3,newResBlock2,
    LinearAttention, Attention,
    PreNorm, Residual)

from .attention import SpatialTransformer


class myConditionalUNet3(nn.Module):
    def __init__(self, in_nc, out_nc, nf, ch_mult=[1, 2, 4, 4], 
                    context_dim=512, use_degra_context=True, use_image_context=False, upscale=1):
        super().__init__()
        self.depth = len(ch_mult)
        self.upscale = upscale # not used
        self.context_dim = -1 if context_dim is None else context_dim
        self.use_image_context = use_image_context
        self.use_degra_context = use_degra_context

        num_head_channels = 32
        dim_head = num_head_channels

        block_class = functools.partial(newResBlock3, conv=default_conv, act=NonLinearity())
        block_class2 = functools.partial(newResBlock3, conv=default_conv, act=NonLinearity(), is_right=True)


        self.init_conv = default_conv(in_nc*2, nf, 7)
        
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

        for i in range(self.depth):
            dim_in = nf * ch_mult[i]
            dim_out = nf * ch_mult[i+1]
            num_heads_in = dim_in // num_head_channels
            num_heads_out = dim_out // num_head_channels
            # print(num_heads_in, num_heads_out)

            dim_head_in = dim_in // num_heads_in

            if use_image_context and context_dim > 0:
                att_down = LinearAttention(dim_in) if i < 3 else SpatialTransformer(dim_in, num_heads_in, dim_head, depth=1, context_dim=context_dim)
                att_up = LinearAttention(dim_out) if i < 3 else SpatialTransformer(dim_out, num_heads_out, dim_head, depth=1, context_dim=context_dim)
            else:
                att_down = LinearAttention(dim_in) # if i < 2 else Attention(dim_in)
                att_up = LinearAttention(dim_out) # if i < 2 else Attention(dim_out)

            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim), # b1
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim), # b2
                Residual(PreNorm(dim_in, att_down)), #atttention
                Downsample(dim_in, dim_out) if i != (self.depth-1) else default_conv(dim_in, dim_out) # downsample
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class2(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class2(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, att_up)),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = nf * ch_mult[-1]
        num_heads_mid = mid_dim // num_head_channels
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        if use_image_context and context_dim > 0:
            self.mid_attn = Residual(PreNorm(mid_dim, SpatialTransformer(mid_dim, num_heads_mid, dim_head, depth=1, context_dim=context_dim)))
        else:
            self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class2(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

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

        x = self.init_conv(x) # 1, 64, 256, 256
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
        # print(x.shape)
        for b1, b2, attn, downsample in self.downs: # 这一块对应UNet左半边编码器区域
            x = b1(x, t) #输入 1 64 128 128 输出1 64 128 128  loop1 1 64 256 256 loop2 1 64 128 128
            h.append(x)

            x = b2(x, t)# loop1 1 64 256 256 # loop2 1 64 128 128


            x = attn(x, context=image_context) # loop1 1 64 256 256 loop2 1 64 128 128

            h.append(x)

            x = downsample(x) # loop1 1 64 128 128 loop2 1 128 64 64



        x = self.mid_block1(x, t)
        x = self.mid_attn(x, context=image_context)
        x = self.mid_block2(x, t)
        # 1 512 32 32
        # print("start")
        # for i in h:
        #     print(i.shape)
        # print("end")

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1) # 1 768 32 32
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            # print(x.shape)


            x = attn(x, context=image_context)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W].contiguous()
        # print("std:{}".format(x.std()))
        # if torch.isnan(x.std()):
        #     print("output is nan in")
        
        return x

if __name__ == '__main__':
    import numpy as np
    device = torch.device("cuda")
    model = myConditionalUNet3(in_nc=3, out_nc=3, nf=64, ch_mult=[1, 2, 4, 8], context_dim=512, use_degra_context=True, use_image_context=True)
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
    print(output.shape)

