import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class PatchEmbedding(nn.Module): # pretty overkill, but need this structure in order to load image-models weights
    def __init__(self, patch_size, in_channels, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        out = self.proj(x)
        return out

    
class MLP(nn.Sequential):
    def __init__(self, dim, ff_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.drop2 = nn.Dropout(dropout)
        
        
class EncoderBlock(nn.Module): # layerscale, droppath?
    def __init__(self, dim, ff_dim, num_heads, dropout, attention_dropout):
        super().__init__()
        # layer norm -> mhsa -> layer norm -> mlp
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiheadAttention(dim, num_heads, dropout, attention_dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, ff_dim, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        out = x + self.mlp(self.norm2(x))
        return out

    
class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, attention_dropout, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        #assert isinstance(self.dim/self.num_heads, int)
        self.h = self.dim // self.num_heads
    
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.h).permute(2, 0, 3, 1, 4) # [3, batch_size, num_heads, sequence_length, attention_height]
        q, k, v = qkv.unbind(0) # q, k, v of shape [b, n, s, h]
        
        k = k.transpose(-2, -1)
        attention = torch.matmul(q, k)
        attention = attention * (self.h ** -0.5)
        attention = self.attn_drop(F.softmax(attention, dim=-1))
        
        out = torch.matmul(attention, v) # b, h, s, w
        out = out.permute(0, 2, 1, 3) # b, s, h, w 
        out = out.flatten(2) # b, s, d
        
        out = self.proj(out)
        out = self.proj_drop(out)
        return out