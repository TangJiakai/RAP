import numpy as np
import torch


def Recall(pos_len, pos_idx, topk_list):
    recall = np.cumsum(pos_idx, axis=1) / pos_len
    result = [recall[:, k-1] for k in topk_list]
    return result


def Precision(pos_len, pos_idx, topk_list):
    precision = np.cumsum(pos_idx, axis=1) / (np.arange(pos_idx.shape[1]) + 1)
    result = [precision[:, k-1] for k in topk_list]
    return result


def NDCG(pos_len, pos_idx, topk_list):
    len_rank = np.full_like(pos_len, pos_idx.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)
    idcg_len = np.ravel(idcg_len).astype(np.int64)

    iranks = np.zeros_like(pos_idx, dtype=np.float32)
    iranks[:, :] = np.arange(pos_idx.shape[1]) + 1
    idcg = np.cumsum(1 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]
    
    ranks = np.zeros_like(pos_idx, dtype=np.float32)
    ranks[:, :] = np.arange(pos_idx.shape[1]) + 1
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.where(pos_idx, dcg, 0)
    dcg = np.cumsum(dcg, axis=1)
    dcg = dcg / idcg

    result = [dcg[:, k-1] for k in topk_list]
    return result


def HitRatio(pos_len, pos_idx, topk_list):
    hit = np.cumsum(pos_idx, axis=1)
    hit = (hit > 0).astype(int)
    result = [hit[:, k-1] for k in topk_list]
    return np.array(result)


def MRR(pos_len, pos_idx, topk_list):
    idxs = np.argmax(pos_idx, axis=1)
    mrr = np.zeros_like(pos_idx, dtype=np.float32)
    for row, idx in enumerate(idxs):
        if pos_idx[row, idx] > 0:
            mrr[row, idx:] = 1 / (idx + 1)
        else:
            mrr[row, idx:] = 0
    result = [mrr[:, k-1] for k in topk_list]
    return result


def evaluate_funs(history_item_ids, pos_item_ids, pred_scores, score_metric="NDCG@10", topk_list=[10,20], metric_list=[Recall, NDCG, HitRatio, MRR]):
    scores_tensor = torch.concat(pred_scores, dim=0)
    for i, hist_ids in enumerate(history_item_ids):
        scores_tensor[i, hist_ids] = -np.inf
    _, topk_idx = torch.topk(scores_tensor, max(topk_list), dim=-1)
    
    scores_tensor = scores_tensor.numpy()
    topk_idx = topk_idx.numpy()

    pos_matrix = np.zeros_like(scores_tensor)
    for i, pos_ids in enumerate(pos_item_ids):
        pos_matrix[i, pos_ids] = 1
    pos_len_arr = np.sum(pos_matrix, axis=1, keepdims=True)
    pos_idx = pos_matrix[np.arange(pos_matrix.shape[0])[:,np.newaxis], topk_idx]

    results = {}

    for metric in metric_list:
        metric_results = metric(pos_len_arr, pos_idx, topk_list)
        for topk, res in zip(topk_list, metric_results):
            mean_res = np.mean(res)
            results[f'{metric.__name__}@{topk}'] = round(mean_res, 7)

    return results[score_metric], results
    
    