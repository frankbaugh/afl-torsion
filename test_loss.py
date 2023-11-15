import torch
import torch.nn.functional as F

def torsion_loss(
    pred_angles,
    true_chi,
    period_mask,
    angle_mask,
    chi_weight,
    angle_norm_weight,
    eps=1e-6,
    eps_mask=1e-4
):
    """
    in: chi_i, nb comparison of two angles (α and β) represented as points on the unit
    circle by an L2 norm is equivalent to the cosine of the angle difference
    also calculate with alt_trutch (180deg rotation for ) and pick least loss so
    that it can converge on either
    sin(a ± π) = -sin(a)
    cos(a ± π) = -cos(a) 
    """
    print(f"Input: pred_angles - {pred_angles.shape}, true_chi - {true_chi.shape}, period_mask - {period_mask.shape}, angle_mask - {angle_mask.shape}")

    normalized_pred_angles = F.normalize(pred_angles, p=2, dim=-1)
    print(f"normalized_pred_angles: {normalized_pred_angles.shape}")

    shifted_mask = (1 - 2 * period_mask).unsqueeze(-1)  # (non) symmetric angles = (-)1. Broadcasts for masking
    true_chi_shifted = shifted_mask * true_chi
    print(f"shifted_mask: {shifted_mask.shape}, true_chi_shifted: {true_chi_shifted.shape}")

    sq_chi_error = torch.sum((true_chi - normalized_pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum((true_chi_shifted - normalized_pred_angles) ** 2, dim=-1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)  # [*, N, 4]
    print(f"sq_chi_error: {sq_chi_error.shape}")

    loss = torch.sum(angle_mask * sq_chi_error, dim=(-1, -2))  # [*,]
    loss /= (eps_mask + torch.sum(angle_mask, dim=(-1, -2)))
    print(f"loss: {loss.shape}")

    angle_norm = torch.sqrt(torch.sum(pred_angles ** 2, dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.0)  # [*, N, 4]
    print(f"angle_norm: {angle_norm.shape}, norm_error: {norm_error.shape}")

    angle_norm_loss = torch.sum(angle_mask * norm_error, dim=(-1, -2))  # [*,]
    angle_norm_loss /= (eps_mask + torch.sum(angle_mask, dim=(-1, -2)))
    print(f"angle_norm_loss: {angle_norm_loss.shape}")

    loss = chi_weight * loss + angle_norm_weight * angle_norm_loss
    loss = torch.mean(loss)  # Average over batch dimension at the end

    print(f"Final Loss: {loss.item()}")

    return loss

# Example usage with random test data
batch_size = 2
N = 3
pred_angles = torch.randn(batch_size, N, 4, 2)
true_chi = torch.randn(batch_size, N, 4, 2)
period_mask = torch.randint(0, 2, size=(batch_size, N, 4)).float()
angle_mask = torch.randint(0, 2, size=(batch_size, N, 4)).float()
chi_weight = 1.0
angle_norm_weight = 1.0

torsion_loss(pred_angles, true_chi, period_mask, angle_mask, chi_weight, angle_norm_weight)
