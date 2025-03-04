import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.nn import ModuleList

class llamaAttention(nn.Module):
    def __init__(self, config, masking=True, attention_type="causal"):
        super(llamaAttention, self).__init__()
        self.config = config
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_type = attention_type
        self.masking = masking

        def forward(self, x):
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            if masking:
                # Causal attention
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.config.hidden_size ** 0.5)
                if self.masking:
                    mask = torch.tril(torch.ones(attn_weights.size(-2), attn_weights.size(-1))).to(attn_weights.device)
                    attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, v)
            else:
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.config.hidden_size ** 0.5)
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, v)
            return attn_output

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.layernorm(attn_output + x)
    
class llamaLayer(nn.Module):
    def __init__(self, config):
        super(llamaLayer, self).__init__()
        self.config = config
        self.attention = llamaAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        attn_output = self.attention(x)
        ffn_output = self.ffn(attn_output)
        return self.layernorm2(ffn_output + x)

class llamaSS(nn.Module):
    def __init__(self, config):
        super(llamaSS, self).__init__()
        self.config = config
        self.llamaembedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.llamaLayers = ModuleList([llamaLayer(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.llamaembedding(x)
        for layer in self.llamaLayers:
            x = layer(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.output(x)
        return x
    
