import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from vit.torch_network import *


class PTVisionTransformer(nn.Module):
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
        self.representation_layer = params['representation_layer']
        
        self.fine_tune = params['fine_tune']
        
        # dynamic
        self.embed_height = self.img_size // self.patch_size
        self.embed_width = self.img_size // self.patch_size
        self.seq_len = self.embed_height * self.embed_width + 1 # add one here for token
        
        # network structure
        self.conv_proj = nn.Conv2d(in_channels=self.input_channels, out_channels=self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.encoder = Encoder(self.seq_len, self.num_layers, self.dim, self.hidden_dim, self.num_heads, self.dropout, self.attention_dropout)
        
        self.head_layers = OrderedDict()
        if self.representation_layer is None:
            self.head_layers['head'] = nn.Linear(self.dim, self.num_classes)
        # else...
            # pre logits, etc
        
        self.heads = nn.Sequential(self.head_layers) 
        
        # fine_tune is the modified num classes
        self.fine_tune = nn.Linear(self.num_classes, self.fine_tune) if self.fine_tune else nn.Identity()
        
    def process_input(self, x):
        # inputs are of shape (b, c, h, w)
        # outputs should be of shape (b, s, d)
        x = self.conv_proj(x)
        x = x.flatten(2)
        out = x.permute(0, 2, 1)
        return out
        
    def forward(self, x):
        b = x.shape[0] # batch size
        x = self.process_input(x)
        
        # concatenate token
        token = self.class_token.expand(b, -1, -1)
        x = torch.cat([token, x], dim=1)
        x = self.encoder(x)
        
        # extract token
        x = x[:, 0]
        out = self.heads(x)
        out = self.fine_tune(out)
            
        return out