import torch


def BPRLoss(pos_pred_scores, neg_pred_scores, mean=True):
    loss = -torch.log(torch.sigmoid(pos_pred_scores - neg_pred_scores) + 1e-10)
    if mean:
        loss = loss.mean()
    return loss