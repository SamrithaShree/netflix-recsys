import numpy as np
import math

def precision_at_k(recommended, ground_truth, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(ground_truth))
    return hits / k

def recall_at_k(recommended, ground_truth, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0

def dcg_at_k(recommended, ground_truth, k):
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, r in enumerate(recommended_k):
        if r in ground_truth:
            dcg += 1.0 / math.log2(i + 2)
    return dcg

def idcg_at_k(ground_truth, k):
    idcg = 0.0
    n = min(len(ground_truth), k)
    for i in range(n):
        idcg += 1.0 / math.log2(i + 2)
    return idcg

def ndcg_at_k(recommended, ground_truth, k):
    dcg = dcg_at_k(recommended, ground_truth, k)
    idcg = idcg_at_k(ground_truth, k)
    return dcg / idcg if idcg > 0 else 0.0

def batch_eval(users, test_dict, rec_dict, k=10):
    precisions, recalls, ndcgs = [], [], []
    for user in users:
        gt_items = test_dict.get(user, [])
        rec_items = rec_dict.get(user, [])
        precisions.append(precision_at_k(rec_items, gt_items, k))
        recalls.append(recall_at_k(rec_items, gt_items, k))
        ndcgs.append(ndcg_at_k(rec_items, gt_items, k))
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)
