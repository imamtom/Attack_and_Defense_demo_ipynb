import torch
from typing import List

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