import torch

from src.distill.reverse_kl_loss import reverse_kl_loss


def test_reverse_kl_loss_non_negative() -> None:
    student = torch.tensor([[[2.0, 0.5], [1.0, 1.0]]])
    teacher = torch.tensor([[[1.5, 1.0], [2.0, 0.1]]])
    loss = reverse_kl_loss(student, teacher)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_reverse_kl_loss_with_mask() -> None:
    student = torch.tensor([[[2.0, 0.5], [1.0, 1.0]]])
    teacher = torch.tensor([[[1.5, 1.0], [2.0, 0.1]]])
    mask = torch.tensor([[1.0, 0.0]])
    loss = reverse_kl_loss(student, teacher, mask=mask)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0
