import torch
from typing import List
from utils import compute_cosine_dis
import numpy as np

# 加权聚合本地模型更新
def aggregate_local_model_updates_by_weight(local_model_updates, weights_of_select_clients):
    # 要求所有updates都在cpu
    assert all([all([v.device == torch.device('cpu') for v in local_model_updates[i].values()]) for i in range(len(local_model_updates))])

    print('本轮被聚合的客户端数量: ', len(local_model_updates))
    assert len(local_model_updates) == len(weights_of_select_clients)
    global_model_update = {k: torch.zeros_like(local_model_updates[0][k]) for k in local_model_updates[0] if "num_batches_tracked" not in k}
    for k in global_model_update:
        for i in range(len(local_model_updates)):
            global_model_update[k] += local_model_updates[i][k] * weights_of_select_clients[i]
    return global_model_update

# 平均聚合本地模型更新
def aggregate_local_updates_by_avg(local_model_updates):
    # 要求所有updates都在cpu
    assert all([all([v.device == torch.device('cpu') for v in local_model_updates[i].values()]) for i in range(len(local_model_updates))])
    
    global_model_update = {k: torch.zeros_like(local_model_updates[0][k]) for k in local_model_updates[0] if "num_batches_tracked" not in k}
    for k in global_model_update:
        for i in range(len(local_model_updates)):
            global_model_update[k] += local_model_updates[i][k]
        global_model_update[k] /= len(local_model_updates)
    return global_model_update

def TrimmedMean(inputs: List[torch.Tensor], num_excluded):
    inputs_tensor = torch.stack(inputs, dim=0)
    largest, _ = torch.topk(inputs_tensor, num_excluded, 0)
    neg_smallest, _ = torch.topk(-inputs_tensor, num_excluded, 0)
    new_stacked = torch.cat([inputs_tensor, -largest, neg_smallest]).sum(0)
    new_stacked /= len(inputs_tensor) - 2 * num_excluded
    return new_stacked

def MultiKrum(inputs: List[torch.Tensor], num_byzantine: int):
    def compute_euclidean_distance(v1, v2):
        return (v1 - v2).norm() ** 2

    def pairwise_euclidean_distances(vectors):
        n = len(vectors)
        vectors = [v.flatten() for v in vectors]
        distances = {i: {} for i in range(n - 1)}
        for i in range(n - 1):
            for j in range(i + 1, n):
                distances[i][j] = compute_euclidean_distance(vectors[i], vectors[j])
        return distances

    def compute_scores(distances, i, n, f):
        s = [distances[j][i] for j in range(i)] + [distances[i][j] for j in range(i + 1, n)]
        return sum(sorted(s)[:n - f - 2])

    def multi_krum(distances, n, f, m):
        if n < 1:
            raise ValueError(f"Number of workers should be positive integer. Got {n}.")
        if m < 1 or m > n:
            raise ValueError(f"Number of workers for aggregation should be >=1 and <= {n}. Got {m}.")
        if 2 * f + 2 > n:
            raise ValueError(f"Too many Byzantine workers: 2 * {f} + 2 >= {n}.")
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                if distances[i][j] < 0:
                    raise ValueError(f"The distance between node {i} and {j} should be non-negative: Got {distances[i][j]}.")
        
        scores = [(i, compute_scores(distances, i, n, f)) for i in range(n)]
        return [x[0] for x in sorted(scores, key=lambda x: x[1])][:m]

    n = len(inputs)  # 总 worker 数量
    k = n // 2  # 选择一半的 worker

    if num_byzantine >= n // 2:
        raise ValueError(f"Number of Byzantine workers should be less than half of total workers. Got {num_byzantine} out of {n}.")

    updates = torch.stack(inputs, dim=0)
    distances = pairwise_euclidean_distances(updates)
    top_k_indices = multi_krum(distances, n, num_byzantine, k)
    values = torch.stack([updates[i] for i in top_k_indices], dim=0).mean(dim=0)
    return values

def PPBR(inputs: List[torch.Tensor]):
    # 将输入的tensor列表堆叠成一个tensor
    updates = torch.stack(inputs, dim=0)
    
    # 计算每个模型的得分
    def get_scores_by_dismatrix(matrix):
        k = matrix.shape[1] // 2
        sorted_matrix = np.sort(matrix, axis=1)
        sorted_matrix[:, :k] = 0
        return np.sum(sorted_matrix, axis=1)
    
    # 计算余弦相似度矩阵
    dis_matrix = compute_cosine_dis(updates)
    
    # 计算得分并排序
    all_scores = get_scores_by_dismatrix(dis_matrix)
    sorted_indices = np.argsort(all_scores)
    
    # 选择得分最高的一半模型
    k = len(inputs) // 2
    top_k_indices = sorted_indices[-k:]

    print('PPBR selected_indices: ', top_k_indices)
    
    # 计算选中模型的平均值
    selected_updates = updates[top_k_indices]
    aggregated_update = torch.mean(selected_updates, dim=0)
    
    return aggregated_update