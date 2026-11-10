from __future__ import annotations

import torch
import torch.nn.functional as F


def reverse_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor | None = None, *, reduction: str = "token_mean") -> torch.Tensor:
    """KL(student || teacher); sequence_sum implements Eq. (5)'s token sum."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("Expected matching [batch, response tokens, vocabulary] logits")
    if reduction not in {"token_mean", "sequence_sum"}:
        raise ValueError("Unknown reverse-KL reduction")
    log_p_s = F.log_softmax(student_logits.float(), dim=-1)
    p_s = log_p_s.exp()
    log_p_t = F.log_softmax(teacher_logits.float(), dim=-1)
    kl = (p_s * (log_p_s - log_p_t)).sum(dim=-1)
    if mask is not None:
        if mask.shape != kl.shape:
            raise ValueError("Response mask must match [batch, tokens]")
        kl = kl * mask
    if reduction == "sequence_sum":
        return kl.sum(dim=-1).mean()
    if mask is not None:
        denom = mask.sum().clamp_min(1.0)
        return kl.sum() / denom
    return kl.mean()
