import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F

class selfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super(selfAttention, self).__init__()
        if in_channel % n_head != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (in_channel, n_head)
            )

        self.num_attention_heads = n_head
        self.norm_layer = nn.GroupNorm(norm_groups, in_channel)
        self.key_layer = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.query_layer = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.value_layer = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.out_layer = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, h, w = input.shape
        n_head = self.num_attention_heads
        head_dim = channel // n_head

        norm = self.norm_layer(input)
        key = self.key_layer(norm).view(batch, n_head, head_dim, h, w)
        query = self.query_layer(norm).view(batch, n_head, head_dim, h, w)
        value = self.value_layer(norm).view(batch, n_head, head_dim, h, w)

        attention_scores = torch.einsum('bnchw,bncxy->bnhwxy', (query, key)).contiguous()
        attention_scores = attention_scores / math.sqrt(channel)
        attention_scores = attention_scores.view(batch, n_head, h, w, -1)
        attention_probs = F.softmax(attention_scores, dim = -1)
        attention_probs = attention_probs.view(batch, n_head, h, w, h, w)

        out = torch.einsum('bnhwxy, bncxy->bnchw', (attention_probs, value)).contiguous()
        out = self.out_layer(out.view(batch, channel, h, w))
        return out + input

features = torch.rand((32, 20, 16, 16))
attention = selfAttention(20, 2, 10)
result = attention.forward(features)
print(result.shape)

'''
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))
        return out + input

features2 = torch.rand((32, 20, 64, 64))
attention2 = selfAttention(20, 2, 10)
result2 = attention2.forward(features2)
print(result2.shape)
'''

