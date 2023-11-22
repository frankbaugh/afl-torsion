def torsion_loss(
    pred_angles, #   [*, N, 4, 2]
    true_chi, #      [*, N, 4, 2]
    period_mask, #   [*, N, 4]
    angle_mask, #    [*, N, 4]
    chi_weight,
    angle_norm_weight,
    eps=1e-6, 
    eps_mask=1e-4  
    
):
    # TODO: Make this classification into bins instead. 
    """
    in: chi_i, nb comparison of two angles (α and β) represented as points on the unit
    circle by an L2 norm is equivalent to the cosine of the angle difference
    also calculate with alt_trutch (180deg rotation for ) and pick least loss so
    that it can converge on either
    sin(a ± π) = -sin(a)
    cos(a ± π) = -cos(a) 
    """
    normalized_pred_angles = F.normalize(pred_angles, p=2, dim=-1)
    
    shifted_mask = (1 - 2 * period_mask).unsqueeze(-1) # (non) symmetric angles = (-)1. Broadcasts for masking
    true_chi_shifted = shifted_mask * true_chi
    
    
    sq_chi_error = torch.sum((true_chi - normalized_pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum((true_chi_shifted - normalized_pred_angles) ** 2, dim=-1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted) # [*, N, 4]

    
    """
     This confuses me because there are many dimensions, and maybe each element
     oF the batch should be handled differentl
     
     Another concern is why the maksing takes okec after the minmum lss is talen; this alloa
     the error in angles which do not even exist to creep in to the selection of the corrext truth
     
     Open fold has absurd code to mask the loss, see line 346 of loss.py
    """
    loss = torch.sum(angle_mask * sq_chi_error, dim=(-1, -2)) # [*,]
    loss /= (eps_mask + torch.sum(angle_mask, dim=(-1, -2)))
    
    angle_norm = torch.sqrt(
        torch.sum(pred_angles ** 2, dim=-1) + eps
    )
    norm_error = torch.abs(angle_norm - 1.0) # [*, N, 4]
    
    angle_norm_loss = torch.sum(angle_mask * norm_error, dim=(-1, -2)) # [*,]
    angle_norm_loss /= (eps_mask + torch.sum(angle_mask, dim=(-1, -2)))

    loss = chi_weight * loss + angle_norm_weight * angle_norm_loss
    loss = torch.mean(loss)  # Average over batch dimension at the end


    return loss