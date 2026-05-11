import torch

from adsorbgen.flow import smooth_lddt_loss


def test_smooth_lddt_loss_lower_for_matching_pair_geometry():
    true = torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    pred_good = true.clone()
    pred_bad = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 4.0, 0.0]]])
    mask = torch.tensor([[True, True, True]])

    assert smooth_lddt_loss(pred_good, true, mask) < smooth_lddt_loss(pred_bad, true, mask)


def test_smooth_lddt_loss_ignores_unselected_atoms():
    true = torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    pred = true.clone()
    pred[:, 2] = 100.0
    mask = torch.tensor([[True, True, False]])

    base = smooth_lddt_loss(true, true, mask)
    changed = smooth_lddt_loss(pred, true, mask)

    assert torch.allclose(base, changed)
