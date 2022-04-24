import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import OrderedDict

class VisionTransformer(nn.Module):
    def __init__(self, **params):
        super().__init__()
        # static
        self.input_channels = params['input_channels']
        self.dim = params['dim']
        self.hidden_dim = params['hidden_dim']
        self.patch_size = params['patch_size']
        self.img_size = params['img_size']
        self.dropout = params['dropout']
        self.attention_dropout = params['attention_dropout']
        self.num_layers = params['num_layers']
        self.num_heads = params['num_heads']
        self.num_classes = params['num_classes']
        
        self.fine_tune = params['fine_tune']
        
        # dynamic
        self.embed_height = self.img_size // self.patch_size
        self.embed_width = self.img_size // self.patch_size
        self.seq_len = self.embed_height * self.embed_width + 1 # add one here for token
        
        # network structure
        self.patch_embed = nn.Conv2d(in_channels=self.input_channels, out_channels=self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.dim))
        self.blocks = nn.Sequential(*[
            EncoderBlock(self.dim, self.hidden_dim, self.num_heads, self.dropout, self.attention_dropout) for i in range(self.num_layers)
        ])
        self.head = nn.Linear(self.dim, self.num_classes)
        
        if self.fine_tune is not None:
            self.ft = nn.Linear(self.num_classes, self.fine_tune) # fine_tune is the modified num classes
        
    def process_input(self, x):
        # inputs are of shape (b, c, h, w)
        # outputs should be of shape (b, s, d)
        x = self.patch_embed(x)
        x = x.flatten(2)
        out = x.permute(0, 2, 1)
        return out
        
    def forward(self, x):
        b = x.shape[0] # batch size
        x = self.process_input(x)
        
        # concatenate token
        token = self.cls_token.expand(b, -1, -1)
        x = torch.cat([token, x], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        
        # extract token
        x = x[:, 0]
        out = self.head(x)
        
        # readjust to fine tuning dataset
        if self.fine_tune is not None:
            out = self.ft(out)
            
        return out
    
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
        self.attn = MultiheadAttention2(dim, num_heads, dropout, attention_dropout)
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
    
class MultiheadAttention2(nn.Module):
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