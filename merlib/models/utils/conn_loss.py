import torch.nn.functional as F
import torch
def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    # (B,C) @ (C,B) = (B,B)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    # neg_cos_matrix = cos_matrix - 0.7
    neg_cos_matrix = cos_matrix - 0.4
    # neg_cos_matrix = cos_matrix - 0.1
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss