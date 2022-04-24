import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from vit.model import *

class MLP(nn.Sequential):
    def __init__(self, dim, ff_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(dim, ff_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, dim)
        self.dropout_2 = nn.Dropout(dropout)
        
class EncoderBlock(nn.Module):
    def __init__(self, dim, ff_dim, num_heads, dropout, attention_dropout):
        super().__init__()
        # layer norm -> mhsa -> dropout -> layer norm -> mlp
        self.ln_1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, ff_dim, dropout)
        
    def forward(self, x):
        h = self.ln_1(x)
        h, _ = self.self_attention(query=h, key=h, value=h, need_weights=False)
        h = self.dropout(h)
        x = x + h
        h = self.ln_2(x)
        h = self.mlp(h)
        out = x + h
        return out
    
class MultiheadAttention2(nn.Module):
    def __init__(self, dim, num_heads, attention_dropout):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        assert self.dim / self.num_heads == 0
        self.h = self.dim / self.num_heads
    
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, x):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_k(v)
    
        q = q.view(q.shape[0], self.num_heads, q.shape[1], -1)
        k = k.view(k.shape[0], self.num_heads, k.shape[1], -1)
        v = v.view(v.shape[0], self.num_heads, v.shape[1], -1)
        
        k = k.transpose(-2, -1)
        score = torch.matmul(q, k)
        score = score * (self.h ** -0.5)
        score = self.dropout(F.softmax(score))
        
        out = torch.matmul(score, v) # b, h, s, w
        out = out.permute(0, 2, 1, 3) # b, s, h, w 
        out = out.flatten(2) # b, s, d
        
        return out
    
class Encoder(nn.Module):
    def __init__(self, seq_len, num_layers, dim, ff_dim, num_heads, dropout, attention_dropout):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.dropout = nn.Dropout(dropout)
        
        blocks = OrderedDict()
        for i in range(num_layers):
            key = f'encoder_layer_{i}'
            blocks[key] = EncoderBlock(dim, ff_dim, num_heads, dropout, attention_dropout)
            
        self.layers = nn.Sequential(blocks)
        self.ln = nn.LayerNorm(dim, eps=1e-6)
    
    def forward(self, x):
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.layers(x)
        out = self.ln(x)
        
        return out