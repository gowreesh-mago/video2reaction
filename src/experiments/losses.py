import torch
from torch.nn import functional as F


def distribution_loss(logits, target, kind='kl', cosine_weight=0.2, ranking_weight=0.1, delta=0.03, margin=0.1):
    logq = F.log_softmax(logits, dim=-1)
    q = logq.exp()
    kl = F.kl_div(logq, target, reduction='batchmean')
    if kind == 'ce':
        return -(target*logq).sum(-1).mean()
    if kind == 'kl':
        return kl
    cosine = (1-F.cosine_similarity(q,target,dim=-1)).mean()
    if kind == 'cosine':
        return cosine
    if kind == 'js':
        logm = ((target+q)*0.5).clamp_min(1e-12).log()
        return 0.5*(F.kl_div(logm,target,reduction='batchmean') + (q*(logq-logm)).sum(-1).mean())
    if kind == 'composite':
        pairs = target[:,:,None] > target[:,None,:] + delta
        penalties = F.relu(margin-logits[:,:,None]+logits[:,None,:])
        ranking = ((penalties*pairs).sum((1,2))/pairs.sum((1,2)).clamp_min(1)).mean()
        return kl+cosine_weight*cosine+ranking_weight*ranking
    raise ValueError(f'Unknown loss: {kind}')
