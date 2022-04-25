import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class MLP(nn.Sequential):
    def __init__(self, dim, ff_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout_2 = nn.Dropout(dropout)
        
class EncoderBlock(nn.Module):
    def __init__(self, dim, ff_dim, num_heads, dropout, attention_dropout):
        super().__init__()
        # layer norm -> mhsa -> dropout -> layer norm -> mlp
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiheadAttention(dim, num_heads, dropout, attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, ff_dim, dropout)
        
    def forward(self, x):
        h = self.norm1(x)
        h = self.attn(h)
        h = self.dropout(h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        out = x + h
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
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        k = k.transpose(-2, -1)
        score = torch.matmul(q, k)
        score = score * (self.h ** -0.5)
        score = self.attn_drop(F.softmax(score, dim=-1))
        
        out = torch.matmul(score, v) # b, h, s, w
        out = out.permute(0, 2, 1, 3) # b, s, h, w 
        out = out.flatten(2) # b, s, d
        
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out