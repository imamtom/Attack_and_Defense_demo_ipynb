import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import l2_distance_of_2_updates, max_abs_of_sliding_window, update_update_convert_to_vector, compute_euclid_dis, layer_wise_align_on_vector, test_acc, test_asr
from utils import get_model_updates
import gc
import numpy as np
from typing import List

# 定义单个客户端的训练函数
def train_and_get_local_vector_of_single_client(client_index, local_model, local_optimizer, train_loader, local_epochs, test_acc_loader, layers_to_aggregate, device):
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
    local_model_update = get_model_updates(intial_model, local_model, layers_to_aggregate)
    flatten_local_model_update = update_update_convert_to_vector(local_model_update)
    
    # 清除 GPU 缓存
    torch.cuda.empty_cache()
    # 强制垃圾回收
    gc.collect()

    return flatten_local_model_update, train_loss

# 定义compriomised_clients的攻击手段 attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise', 'ALIE', 'MinMax', 'IPM', 'Backdoor']
def train_and_get_local_vector_of_attack_LabelFlipping(client_index, local_model, local_optimizer, train_loader, local_epochs, test_acc_loader, layers_to_aggregate, num_class, device):
    local_model.to(device)
    # 记录本轮初始的模型参数
    intial_model = copy.deepcopy(local_model)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 记录训练loss
    train_loss = 0.0

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
            # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1)
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
    local_model_update = get_model_updates(intial_model, local_model, layers_to_aggregate)
    flatten_local_model_update = update_update_convert_to_vector(local_model_update)
    
    # 清除 GPU 缓存
    torch.cuda.empty_cache()
    # 强制垃圾回收
    gc.collect()

    return flatten_local_model_update, train_loss
        
# 定义SignFlipping的攻击函数
def train_and_get_local_update_of_attack_SignFlipping(client_index, local_model, local_optimizer, train_loader, local_epochs, test_acc_loader, layers_to_aggregate, device):
    local_model.to(device)
    # 记录本轮初始的模型参数
    intial_model = copy.deepcopy(local_model)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 记录训练loss
    train_loss = 0.0

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
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1)

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

    # 计算本轮训练后的模型参数与初始模型参数的差值
    local_model_update = get_model_updates(intial_model, local_model, layers_to_aggregate)
    flatten_local_model_update = update_update_convert_to_vector(local_model_update)
    
    # 清除 GPU 缓存
    torch.cuda.empty_cache()
    # 强制垃圾回收
    gc.collect()

    return flatten_local_model_update, train_loss

# 定义Noise的攻击函数
def train_and_get_local_vector_of_attack_Noise(benign_vector_mean: torch.Tensor):
    noise_mean=0
    noise_std=1
    # vector为和benign_vector_mean同形状的正态分布
    noise = torch.normal(mean=noise_mean, std=noise_std, size=benign_vector_mean.size())
    train_loss = 0
    return noise, train_loss

# 定义IPM的攻击函数
def train_and_get_local_vector_of_attack_IPM(benign_vector_mean, scale):
    # 相反方向的scale倍缩放
    noise = -benign_vector_mean * scale
    train_loss = 0
    return noise, train_loss

# 定义ALIE的攻击函数
def train_and_get_local_vector_of_attack_ALIE(num_clients, num_byzantine, benign_vector_mean, benign_vector_std):
    #  计算z_max
    s = torch.floor_divide(num_clients, 2) + 1 - num_byzantine
    cdf_value = (num_clients - num_byzantine - s) / (num_clients - num_byzantine)
    dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
    z_max = dist.icdf(cdf_value)
    # noise = mean + std * z_max
    # 对每个entry进行缩放
    noise = benign_vector_mean + benign_vector_std * z_max
    train_loss = 0
    return noise, train_loss

# 定义MinMax的攻击函数
def minmax_attack_by_binary_search(benign_vector_mean, benign_vector_std, benign_vectors: List[torch.Tensor]):
    benign_vectors = torch.stack(benign_vectors)
    # benign_vectors之间的L2距离
    all_distances_of_benign_vector = torch.cdist(benign_vectors, benign_vectors, p=2)
    threshold = all_distances_of_benign_vector.max()

    low = 0
    high = 5
    while abs(high - low) > 0.01:
        mid = (low + high) / 2
        malicious_vector = torch.stack([benign_vector_mean - mid * benign_vector_std])
        loss = torch.cdist(malicious_vector, benign_vectors, p=2).max()
        if loss < threshold:
            low = mid
        else:
            high = mid
    print('low: ', low)
    print('high: ', high)
    
    train_loss = 0
    return benign_vector_mean - mid * benign_vector_std, train_loss


def flud_attack_by_binary_search(benign_vector_mean, benign_vector_std, benign_vectors:List[torch.Tensor], num_clients, num_byzantine, given_size):
    window_size = given_size
    # benign_updates 是一个list, 里面是每个benign_client的model_update
    benign_l_infnity_list = []
    for vector in benign_vectors:
        benign_l_infnity_list.append(max_abs_of_sliding_window(vector, window_size))

    l_infnity_vectors_benign = torch.stack(benign_l_infnity_list)
    all_distances_of_benign_l_infnity_vectors = torch.cdist(l_infnity_vectors_benign, l_infnity_vectors_benign, p=2)
    threshold = all_distances_of_benign_l_infnity_vectors.max()
    low = 0
    high = 5
    while abs(high - low) > 0.01:
        mid = (low + high) / 2
        l_infinity_malicious = torch.stack([max_abs_of_sliding_window(benign_vector_mean - mid * benign_vector_std, given_size)])
        loss = torch.cdist(l_infinity_malicious, l_infnity_vectors_benign, p=2).max()
        if loss < threshold:
            low = mid
        else:
            high = mid
    print('low: ', low)
    print('high: ', high)

    train_loss = 0
    return benign_vector_mean - mid * benign_vector_std, train_loss