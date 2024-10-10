import torch
import random
import numpy as np
import os
import torch.nn as nn


# 给定参数, 返回保存结果的目录
def get_save_results_dir(dataset_name, iid, alpha, num_clients, attack_method, defense_method):
    if iid == True:
        results_dir = '~/results/{}_iid-{}_numclients-{}_attack-{}_defense-{}_'.format(dataset_name, iid, num_clients, attack_method, defense_method)
    else:
        results_dir = '~/results/{}_iid-{}_alpha-{}_numclients-{}_attack-{}_defense-{}_'.format(dataset_name, iid, alpha, num_clients, attack_method, defense_method)
    # 以及一个4bytes随机的字符串, 
    # 加上年月日时分秒
    import datetime
    results_dir += datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    import uuid
    results_dir += str(uuid.uuid4())[:4] + '/'
    return results_dir


def set_GPU(gpu_id):
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        if gpu_id < torch.cuda.device_count():
            print(f"Using GPU: {gpu_id}")
            device = torch.device("cuda:{}".format(gpu_id))
        else:
            print(f"GPU id {gpu_id} is not available, using GPU: 0")
            device = torch.device("cuda:0")
    else:
        print("No GPU available")
        device = torch.device("cpu")
    return device

def set_random_seeds(seed_value=42, use_cuda=True):
        """
        设置所有相关随机数生成器的种子
        
        Args:
        seed_value (int): 要使用的种子值
        use_cuda (bool): 是否设置CUDA的随机种子（如果使用GPU）
        """
        # 设置 Python 内置随机模块的种子
        random.seed(seed_value)
        
        # 设置 NumPy 的随机种子
        np.random.seed(seed_value)
        
        # 设置 PyTorch 的随机种子
        torch.manual_seed(seed_value)
        
        if use_cuda and torch.cuda.is_available():
            # 设置 CUDA 的随机种子（如果使用 GPU）
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)  # 如果使用多 GPU
            
            # 设置 cudnn 的随机种子
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 设置 Python 哈希种子（影响某些 Python 对象的哈希值）
        os.environ['PYTHONHASHSEED'] = str(seed_value)


# 定义测试acc函数
def test_acc(global_model, test_acc_loader, device):
    global_model.to(device)
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_acc_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 定义测试asr函数
def test_asr(model, test_asr_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_asr_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print('predicted: ', predicted)
            # print('labels: ', labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# 定义一个函数, 计算两个local_model_update的L2距离
def l2_distance_of_2_updates(model_update1, model_update2):
    distance = 0
    for k in model_update1:
        if "num_batches_tracked" not in k:
            distance += torch.norm(model_update1[k] - model_update2[k], p=2) ** 2
    return torch.sqrt(distance)


# 使用滑动窗口对一维向量进行采样, 返回max(abs())
def max_abs_of_sliding_window(vector, window_size):
    # 先转为绝对值
    vector = torch.abs(vector)
    # 然后用max_pool1d, ceil = True, 保证最后一个窗口不会被舍弃
    max_pool = nn.MaxPool1d(window_size, stride=window_size, ceil_mode=True)
    vector = vector.unsqueeze(0)
    vector = max_pool(vector)
    return vector.squeeze()

# 使用滑动窗口对维向量进行采样, 求出max()
def maxpool_of_sliding_window(vector, kernel_size):
    # 计算vector的长度
    length = vector.size(0)
    w = kernel_size * 2
    h = length // w

    vector = vector[:w*h]
    # reshape, 使得vector的长度是kernel_size的整数倍
    vector = vector.view(1, 1, h, w)

    y = torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=True, return_indices=False)
    # 然后将y转为1维张量
    y = y.view(-1)
    return y

# layer_wise_align: 这里将每个卷积层算出来一个max,每个矩阵算出来一个max. 返回一个vector
def layer_wise_align(vector):
    # vector是一个model_update, 里面有很多层的参数
    layers_tensor = [torch.sign(layer_tensor)*torch.max(torch.abs(layer_tensor)) for layer_tensor in [vector[k] for k in vector]]
    # 将他们拼接为一个tensor
    for i in range(len(layers_tensor)):
        # print('layers_tensor[i]: ', layers_tensor[i].shape)
        layers_tensor[i] = layers_tensor[i].flatten()
    layers_tensor = torch.cat(layers_tensor, dim=0)
    # print('layers_tensor: ', layers_tensor.shape)
    return layers_tensor    



# 传入一个模型更新, 将它转为一个向量, 并返回这个向量
def update_update_convert_to_vector(model_update):
    # print([k for k in model_update])
    return torch.cat([model_update[k].flatten() for k in model_update])

# 计算向量之间的两两欧氏距离
def compute_euclid_dis(vectors_list):
    num = len(vectors_list)
    dis_max = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            dis_max[i, j] =torch.sqrt(torch.sum((vectors_list[i] - vectors_list[j]) ** 2))
            dis_max[j, i] = dis_max[i, j]
    return dis_max

# 距离矩阵转为投票向量
def votes_by_dismatrix(dis_matrix):
    # 获取每行前half_clients个最小距离值的索引
    half_ = dis_matrix.shape[0] // 2
    top_indices = np.argsort(dis_matrix, axis=1)[:, :half_]
    # 创建一个全零的矩阵，然后将每行的[half_clients个最小距离值的索引]位置标为1
    result_matrix = np.zeros_like(dis_matrix)
    rows, cols = np.indices(top_indices.shape)
    result_matrix[rows, top_indices] = 1
    column_sums = np.sum(result_matrix, axis=0)
    return column_sums

def extract_state_dict(model):
    """
    1. 输入模型, 提取模型的state_dict，如果存在则排除num_batches_tracked
    
    Args:
        model (torch.nn.Module): 输入的模型
    
    Returns:
        dict: 不包含num_batches_tracked的state_dict
    """
    return {name: param.clone().detach() for name, param in model.state_dict().items()
            if 'num_batches_tracked' not in name}

def synchronization_local_model(local_model, global_state_dict):
    """
    2. 使用全局模型的state_dict更新本地模型
    
    Args:
        local_model (torch.nn.Module): 本地模型
        global_state_dict (dict): 全局模型的state_dict
    """
    local_state_dict = local_model.state_dict()
    
    for name, param in global_state_dict.items():
        if 'num_batches_tracked' not in name:
            local_state_dict[name].copy_(param)
    local_model.load_state_dict(local_state_dict)
    return local_model  

# 给定一个state_dict, 提取其中的shape
def extract_shapes_from_state_dict(state_dict):
    shape_dict = {}
    for name, param in state_dict.items():
        shape_dict[name] = param.shape
    return shape_dict

# 给定一个一维向量, 以及一个shape_dict, 将这个向量转为state_dict
def vector_to_state_dict(vector, shape_dict, benign_update_length_):
    assert benign_update_length_ == len(vector)
    state_dict = {}
    start = 0
    for name, shape in shape_dict.items():
        length = np.prod(shape)
        state_dict[name] = vector[start:start+length].view(shape)
        start += length
    return state_dict

# 给定所有digest, 被投票选出的客户端的index
def defense_vote_for_clients(all_digests_list):
    half_clients = len(all_digests_list) // 2
    # 计算欧氏距离
    distance_matrix = compute_euclid_dis(all_digests_list)
    # 投票
    votes = votes_by_dismatrix(distance_matrix)
    # 对votes中小于10的元素, 用0代替
    votes[votes < half_clients] = 0
    # 获取那些非0 的元素的index, 并转为list
    non_zero_clients_index = list(np.nonzero(votes)[0])
    # 被聚合的客户端的index
    aggregated_clients_index = non_zero_clients_index
    return aggregated_clients_index

# 传入模型, 返回模型的参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)