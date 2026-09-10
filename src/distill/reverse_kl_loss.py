from __future__ import annotations

import torch
import torch.nn.functional as F


def reverse_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Token-level reverse KL: KL(student || teacher)."""
    log_p_s = F.log_softmax(student_logits, dim=-1)
    p_s = log_p_s.exp()
    log_p_t = F.log_softmax(teacher_logits, dim=-1)
    kl = (p_s * (log_p_s - log_p_t)).sum(dim=-1)
    if mask is not None:
        kl = kl * mask
        denom = mask.sum().clamp_min(1.0)
        return kl.sum() / denom
    return kl.mean()
