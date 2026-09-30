"""Frozen-feature predictors sharing the same projection and output head."""
import math
import torch
from torch import nn


class ReactionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=21, aggregation='mean', layers=2, positional_encoding=True):
        super().__init__()
        if aggregation not in {'mean', 'temporal', 'query'}:
            raise ValueError(aggregation)
        self.aggregation = aggregation
        self.positional_encoding = positional_encoding
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes))
        if aggregation == 'temporal':
            block = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*4, dropout=0., batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        if aggregation == 'query':
            self.queries = nn.Parameter(torch.randn(num_classes, hidden_dim) / math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, frames, mask):
        if frames.ndim != 3 or mask.shape != frames.shape[:2] or not mask.any(1).all():
            raise ValueError('Each sequence needs at least one valid frame')
        h = self.projection(frames)
        attention = None
        if self.aggregation == 'mean':
            pooled = (h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
            logits = self.head(pooled)
        elif self.aggregation == 'temporal':
            position = torch.arange(h.shape[1], device=h.device, dtype=h.dtype)[:,None]
            freq = torch.exp(torch.arange(0,self.hidden_dim,2,device=h.device,dtype=h.dtype)*(-math.log(10000)/self.hidden_dim))
            encoding = torch.zeros_like(h[0])
            encoding[:,0::2] = torch.sin(position*freq)
            encoding[:,1::2] = torch.cos(position*freq)
            h = torch.cat([self.cls.expand(len(h),-1,-1),h+encoding if self.positional_encoding else h],dim=1)
            valid = torch.cat([torch.ones(len(mask),1,device=mask.device,dtype=torch.bool),mask],dim=1)
            pooled = self.temporal(h,src_key_padding_mask=~valid)[:,0]
            logits = self.head(pooled)
        else:
            scores = torch.einsum('cd,btd->bct',self.queries,h)/math.sqrt(self.hidden_dim)
            attention = scores.masked_fill(~mask[:,None,:],-torch.inf).softmax(-1)
            pooled = torch.einsum('bct,btd->bcd',attention,h)
            logits = self.head(pooled).diagonal(dim1=1,dim2=2)
        return logits, attention
