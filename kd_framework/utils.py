import torch
import torch.nn.functional as F

# utility functions 
# 1) spatial attention

def spatial_attention(feature_map):
    """
    Computes 2D spatial attention from feature map.
    Input:  feature_map of shape [B, C, H, W]
    Output: attention map of shape [B, 1, H, W]
    """
    return feature_map.pow(2).mean(1, keepdim=True)  # mean across channels


# 2) Distillation loss

def distillation_loss(student_logits, teacher_logits, 
                      student_feats, teacher_feats,
                      targets, temperature=4.0, alpha=0.7, beta=250.0):
    """
    Combined KD loss with attention transfer
    Args:
        student_logits: Output logits from student
        teacher_logits: Output logits from teacher
        student_att: List of student attention maps
        teacher_att: List of teacher attention maps
        targets: Ground truth labels
    Returns:
        Total loss
    """
    # Cross-entropy loss with ground truth
    ce_loss = F.cross_entropy(student_logits, targets)

    # KL divergence on logits (soft labels)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # Attention Transfer loss (MSE between attention maps)
    att_loss = 0
    
    for sa, ta in zip(student_feats, teacher_feats):
        sa_att = spatial_attention(sa)
        ta_att = spatial_attention(ta)
        ta_att_resized = F.interpolate(ta_att, size=sa_att.shape[2:], mode='bilinear', align_corners=False)
        
        #print(f"Student attention shape: {sa_att.shape}, Teacher attention shape: {ta_att.shape}, Resized: {ta_att_resized.shape}")

        att_loss += F.mse_loss(sa_att, ta_att_resized)

    # for sa, ta in zip(student_feats, teacher_feats):
    #     att_loss += F.mse_loss(spatial_attention(sa), spatial_attention(ta))

    # Combine losses
    total_loss = (1. - alpha) * ce_loss + alpha * kd_loss + beta * att_loss
    return total_loss