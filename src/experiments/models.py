"""Frozen-feature predictors sharing the same projection and output head."""
import math
import torch
from torch import nn


class ReactionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=21, aggregation='mean', layers=2,
                 positional_encoding=True, evidence_dim=0, fusion=None, auxiliary_vad=False, description_dim=0):
        super().__init__()
        if aggregation not in {'mean', 'temporal', 'query', 'shared_query'}:
            raise ValueError(aggregation)
        self.aggregation = aggregation
        self.positional_encoding = positional_encoding
        if evidence_dim not in {0, 11} or fusion not in {None, 'global_peak', 'global_global', 'peak_peak'}:
            raise ValueError('Invalid evidence or fusion configuration')
        if fusion is not None and aggregation != 'mean':
            raise ValueError('Global/peak fusion uses the shared mean-pooling projection')
        self.input_dim, self.evidence_dim, self.fusion = input_dim, evidence_dim, fusion
        if auxiliary_vad and (aggregation != 'mean' or fusion is not None):
            raise ValueError('The initial VAD auxiliary comparison uses the B1 pooled representation')
        if description_dim and (description_dim != input_dim or aggregation != 'mean'
                                or evidence_dim or fusion or auxiliary_vad):
            raise ValueError('Description fusion requires two equally sized inputs and plain mean pooling')
        self.description_dim = description_dim
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.head = nn.Sequential(nn.Linear(hidden_dim * (2 if fusion or description_dim else 1), hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes))
        if aggregation == 'temporal':
            block = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*4, dropout=0., batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        if aggregation in {'query', 'shared_query'}:
            self.queries = nn.Parameter(torch.randn(num_classes, hidden_dim) / math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim
        if evidence_dim:
            # Add this after all common parameters to preserve their initialization.
            self.evidence_projection = nn.Linear(evidence_dim, hidden_dim, bias=False)
        self.auxiliary_vad = auxiliary_vad
        if auxiliary_vad:
            self.vad_head = nn.Linear(hidden_dim, 3)
        if description_dim:
            self.description_projection = nn.Sequential(nn.LayerNorm(description_dim), nn.Linear(description_dim, hidden_dim))

    def forward(self, frames, mask, return_vad=False):
        if frames.ndim != 3 or mask.shape != frames.shape[:2] or not mask.any(1).all():
            raise ValueError('Each sequence needs at least one valid frame')
        expected_width = self.input_dim + self.evidence_dim + int(self.fusion is not None) + self.description_dim
        if frames.shape[-1] != expected_width:
            raise ValueError('Input packet width differs from visual/evidence/fusion configuration')
        h = self.projection(frames[..., :self.input_dim])
        if self.evidence_dim:
            h = h + self.evidence_projection(frames[..., self.input_dim:self.input_dim + self.evidence_dim])
        attention = None
        if self.description_dim:
            other = self.description_projection(frames[..., self.input_dim:])
            pooled = (torch.cat([h, other], dim=-1) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            logits = self.head(pooled)
        elif self.fusion:
            indicator = frames[..., -1]
            if not ((indicator == 0) | (indicator == 1)).all():
                raise ValueError('Peak mask must contain only zero or one')
            peak_mask = (indicator == 1) & mask
            if not peak_mask.any(1).all():
                raise ValueError('Every clip needs a selected peak')
            global_pool = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            peak_pool = (h * peak_mask.unsqueeze(-1)).sum(1) / peak_mask.sum(1, keepdim=True)
            pools = {'global': global_pool, 'peak': peak_pool}
            pooled = torch.cat([pools[key] for key in self.fusion.split('_')], dim=-1)
            logits = self.head(pooled)
        elif self.aggregation == 'mean':
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
            if self.aggregation == 'shared_query':
                # Keep every learned query active and match query-model capacity.
                # Only remove the association between a reaction and its own map.
                attention = attention.mean(1, keepdim=True).expand_as(attention)
            pooled = torch.einsum('bct,btd->bcd',attention,h)
            logits = self.head(pooled).diagonal(dim1=1,dim2=2)
        if return_vad:
            return logits, attention, self.vad_head(pooled) if self.auxiliary_vad else None
        return logits, attention
