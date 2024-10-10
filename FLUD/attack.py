import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import l2_distance_of_2_updates, max_abs_of_sliding_window, update_update_convert_to_vector, compute_euclid_dis, layer_wise_align, test_acc, test_asr
import gc
import numpy as np

# 定义单个客户端的训练函数
def train_and_get_local_update_of_single_client(client_index, local_model, local_optimizer, train_loader, local_epochs, test_acc_loader, device):
    local_model.to(device)
    # 记录本轮初始的模型参数
    intial_model = copy.deepcopy(local_model)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 记录训练loss
    train_loss = 0.0

    # 每个客户端的训练数据
    for epoch in range(local_epochs):
        local_model.train()
        for images, labels in train_loader:
            # 检查images是否被归一化
            # print(images.max(), images.min(), images.mean(), images.std())
            local_optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)
            # 前向传播
            # print("images", images)
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 裁剪梯度
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1)
            local_optimizer.step()
            train_loss += loss.item()

    # loss
    train_loss /= len(train_loader)
    print('Client{} local model, Epoch: {}, Loss: {:.4f}'.format(client_index, epoch, train_loss))
    
    # if client_index == 0: 那么测试acc
    if client_index == 0:
        local_model.eval()
        acc = test_acc(local_model, test_acc_loader, device)
        print('Client{} local model, Test Acc: {:.4f}'.format(client_index, acc))

    # 计算本轮训练后的模型参数与初始模型参数的差值
    local_model_update = {k: (local_model.state_dict()[k] - intial_model.state_dict()[k]) for k in intial_model.state_dict() if "num_batches_tracked" not in k}
    
    # 将模型参数转移到cpu
    local_model.to('cpu')
    # 清除 GPU 缓存
    torch.cuda.empty_cache()
    # 强制垃圾回收
    gc.collect()
    return local_model_update, train_loss

# 定义compriomised_clients的攻击手段 attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise', 'ALIE', 'MinMax', 'IPM', 'Backdoor']
def train_and_get_local_update_of_attack_LabelFlipping(client_index, local_model, local_optimizer, train_loader, local_epochs, test_acc_loader, device, num_class):
    # 记录本轮初始的模型参数
    intial_model = copy.deepcopy(local_model).to('cpu')

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 记录训练loss
    train_loss = 0.0
    local_model.to(device)
    for epoch in range(local_epochs):
        local_model.train()
        for images, labels in train_loader:
            # 修改标签为 9-label, 然后训练
            labels = num_class - 1 - labels
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            local_optimizer.zero_grad()
            loss.backward()
            # 裁剪梯度
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1)
            local_optimizer.step()
            train_loss += loss.item()

    # loss
    train_loss /= len(train_loader)
    print('Client{} local model, Epoch: {}, Loss: {:.4f}'.format(client_index, epoch, train_loss))
    
    # if client_index == 0: 那么测试acc
    if client_index == 0:
        local_model.eval()
        acc = test_acc(local_model, test_acc_loader, device)
        print('Client{} local model, Test Acc: {:.4f}'.format(client_index, acc))

    # 将模型参数转移到cpu
    local_model = local_model.to('cpu')
    # 清除 GPU 缓存
    torch.cuda.empty_cache()
    # 强制垃圾回收
    gc.collect()
    
    # 计算本轮训练后的模型参数与初始模型参数的差值
    local_model_update = {k: (local_model.state_dict()[k] - intial_model.state_dict()[k]) for k in intial_model.state_dict() if "num_batches_tracked" not in k}
    del local_model
    return local_model_update, train_loss
        
# 定义SignFlipping的攻击函数
def train_and_get_local_update_of_attack_SignFlipping(client_index, local_model, local_optimizer, train_loader, local_epochs, test_acc_loader, device):
    # 记录本轮初始的模型参数
    intial_model = copy.deepcopy(local_model.state_dict())
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 记录训练loss
    train_loss = 0.0
    local_model.to(device)
    for epoch in range(local_epochs):
        local_model.train()
        # 计算完梯度都会取反
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            local_optimizer.zero_grad()
            loss.backward()
            # 裁剪梯度
            # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1)

            train_loss += loss.item()
            for param in local_model.parameters():
                param.grad.data = -param.grad.data
            local_optimizer.step()
            
    # loss
    train_loss /= len(train_loader)
    print('Client{} local model, Epoch: {}, Loss: {:.4f}'.format(client_index, epoch, train_loss))
    
    # if client_index == 0: 那么测试acc
    if client_index == 0:
        local_model.eval()
        acc = test_acc(local_model, test_acc_loader, device)
        print('Client{} local model, Test Acc: {:.4f}'.format(client_index, acc))

    # 将模型参数转移到cpu
    local_model = local_model.to('cpu')
    # 清除 GPU 缓存
    torch.cuda.empty_cache()
    # 强制垃圾回收
    gc.collect()


    # 计算本轮训练后的模型参数与初始模型参数的差值
    local_model_update = {k: (local_model.state_dict()[k] - intial_model[k]) for k in intial_model if "num_batches_tracked" not in k}
    return local_model_update, train_loss

# 定义Noise的攻击函数
def train_and_get_local_update_of_attack_Noise(client_index, local_model):
    noise_mean=0
    noise_std=1
    # 模型更新为和模型参数同形状的正态分布
    local_model_update = {k: torch.normal(mean=noise_mean, std=noise_std, size=local_model.state_dict()[k].shape) for k in local_model.state_dict() if "num_batches_tracked" not in k}
    train_loss = 0
    return local_model_update, train_loss

# 定义IPM的攻击函数
def train_and_get_local_update_of_attack_IPM(benign_update_mean, scale):
    # 相反方向的scale倍缩放
    local_model_update = {k: -scale * benign_update_mean[k] for k in benign_update_mean if "num_batches_tracked" not in k}
    train_loss = 0
    return local_model_update, train_loss

# 定义ALIE的攻击函数
def train_and_get_local_update_of_attack_ALIE(num_clients, num_byzantine, benign_update_mean, benign_update_std):
    #  计算z_max
    s = torch.floor_divide(num_clients, 2) + 1 - num_byzantine
    cdf_value = (num_clients - num_byzantine - s) / (num_clients - num_byzantine)
    dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
    z_max = dist.icdf(cdf_value)
    # update = mean + std * z_max
    # 对每个entry进行缩放
    local_model_update = {k: benign_update_mean[k] + benign_update_std[k] * z_max for k in benign_update_mean if "num_batches_tracked" not in k}
    train_loss = 0
    return local_model_update, train_loss

# 定义MinMax的攻击函数
def minmax_attack_by_binary_search(benign_update_mean, benign_update_std, benign_updates, benign_clients_index):
    # 所有良性梯度的mean, 标准差, 良性梯度的之间距离的最大值
    all_distances_of_benign_updates = torch.stack([l2_distance_of_2_updates(benign_updates[i], benign_updates[j]) for i in benign_clients_index for j in benign_clients_index])
    threshold = all_distances_of_benign_updates.max()

    low = 0
    high = 5
    while abs(high - low) > 0.01:
        mid = (low + high) / 2
        # mal_update = torch.stack([mean_grads - mid * deviation])
        malicious_update = {k: benign_update_mean[k] - mid * benign_update_std[k] for k in benign_update_mean if "num_batches_tracked" not in k}
        # loss = torch.cdist(mal_update, benign_updates, p=2).max()
        loss = torch.stack([l2_distance_of_2_updates(malicious_update, benign_updates[i]) for i in benign_clients_index]).max()
        if loss < threshold:
            low = mid
        else:
            high = mid
    train_loss = 0
    return malicious_update, train_loss


def flud_attack_by_binary_search(benign_update_mean, benign_update_std, benign_updates, benign_clients_index, num_clients, num_byzantine, given_size):
    # for key in benign_update_mean:
    #     benign_update_mean[key] = -1 * benign_update_mean[key]
    #     benign_update_std[key] = -1 * benign_update_std[key]
    
    window_size = given_size
    # benign_updates 是一个list, 里面是每个benign_client的model_update
    benign_digests_list = []
    for i in benign_updates:
        benign_digests_list.append(max_abs_of_sliding_window(update_update_convert_to_vector(benign_updates[i]), window_size))

    # 计算欧氏距离矩阵
    distance_matrix = compute_euclid_dis(benign_digests_list)
    
    # 给定一个距离矩阵 和 一组的距离, 返回该距离在距离矩阵中的排名, 从小到大的排名
    def get_ranks_of_distance(distances, distance_matrix):
        # 查看distances 是否为 tesnor
        if isinstance(distances, torch.Tensor):
            distances = distances.cpu().numpy()
        # distance[i] 在 distance_matrix[i]中排第几
        ranks = []
        for i in range(distance_matrix.shape[0]):
            ranks.append(np.sum(distance_matrix[i] < distances[i])+1)
        return ranks
    # 给定ranks向量, 和一个阈值, 返回向量中<=阈值的个数
    def count_less_than_or_equal_to_threshold(ranks, threshold):
        return sum([1 for rank in ranks if rank <= threshold])
    
    rank_threshold = num_clients // 2 - num_byzantine # at least the number of benign clients that mark the byzantine update as the neighbors
    low = 0
    high = 5
    while abs(high - low) > 0.0001:
        mid = (low + high) / 2
        # mal_update = torch.stack([mean_grads - mid * deviation])
        malicious_update = {k: benign_update_mean[k] - mid * benign_update_std[k] for k in benign_update_mean if "num_batches_tracked" not in k}
        # 恶意update的digest
        malicious_digest = max_abs_of_sliding_window(update_update_convert_to_vector(malicious_update), window_size)
        # 计算malicious_digest到所有benign_digests的距离, 放到一个list中
        assert len(malicious_digest) == len(benign_digests_list[0])
        distances = [torch.norm(malicious_digest - benign_digests, p=2).cpu().numpy() for benign_digests in benign_digests_list]
        # 计算被认为是邻居的数量(distances[i]在 distance_matrix[i]中排名在threshold以内)
        ranks = get_ranks_of_distance(distances, distance_matrix) # distances[i] 在 distance_matrix[i]的排名, 从小到大
        # 计算count
        count = count_less_than_or_equal_to_threshold(ranks, rank_threshold)
        
        if count >= rank_threshold:
            low = mid
        else:
            high = mid
    print('count', count)
    print('low: ', low)
    print('high: ', high)

    train_loss = 0
    return malicious_update, train_loss