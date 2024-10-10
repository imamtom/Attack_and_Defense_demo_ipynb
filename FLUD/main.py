from utils import set_GPU, set_random_seeds, test_acc, test_asr, get_save_results_dir, extract_shapes_from_state_dict, extract_state_dict, synchronization_local_model, count_parameters
from utils import update_update_convert_to_vector, max_abs_of_sliding_window, defense_vote_for_clients, maxpool_of_sliding_window, vector_to_state_dict, layer_wise_align, compute_euclid_dis
import os
import random
from models import FashionCNN, MLP, ResNet10, MobileNetV2, LSTM, ResNet34
from datasets import ImagenetteDataset, ImagenetteDataset_server, ImagenetteDataset_per_client, create_rgb_trigger, get_transform, get_transform_add_trigger
import copy
import torch
import numpy as np
import torch.optim as optim
from defense import aggregate_local_model_updates_by_weight, TrimmedMean
from attack import train_and_get_local_update_of_single_client, train_and_get_local_update_of_attack_LabelFlipping, train_and_get_local_update_of_attack_SignFlipping, train_and_get_local_update_of_attack_Noise, train_and_get_local_update_of_attack_IPM, minmax_attack_by_binary_search, train_and_get_local_update_of_attack_ALIE, flud_attack_by_binary_search
from typing import List
import gc
import argparse


def run(attack_method, defense_method, num_clients, poisoned_client_portion, poison_data_portion, dataset_name, gpu_id, iid, alpha, global_rounds, local_epochs, local_learning_rate, local_momentum, batch_size, given_size, target_label):

    # 存储config.json
    config = {
        "attack_method": attack_method,
        "defense_method": defense_method,
        "num_clients": num_clients,
        "poisoned_client_portion": poisoned_client_portion,
        "poison_data_portion": poison_data_portion,
        "dataset_name": dataset_name,
        "gpu_id": gpu_id,
        "iid": iid,
        "alpha": alpha,
        "global_rounds": global_rounds,
        "local_epochs": local_epochs,
        "local_learning_rate": local_learning_rate,
        "local_momentum": local_momentum,
        "batch_size": batch_size,
        "given_size": given_size,
        "target_label": target_label,
    }

    # 输出config
    print("config: ", config)

    # 攻击类型限制范围
    attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise-(0,1)', 'ALIE', 'MinMax', 'IPM-0.1', 'IPM-100', 'Backdoor', 'NoAttack', 'Adaptive']

    assert attack_method in attack_methods, "attack_method should be one of {}".format(attack_methods)
    if attack_method == 'Backdoor':
        backdoor = True
    else:
        backdoor = False

    defense_methods = ['FedAvg', 'FLUD', 'RowSample', 'AlignSample', 'Median', 'Trimmed-Mean', 'ELSA', 'Multi-krum', 'PPBR', 'RFBDS']
    if defense_method not in defense_methods:
        raise ValueError("defense_method should be one of {}".format(defense_methods))


    num_classes_dict = {"MNIST": 10, "FashionMNIST": 10, "CIFAR10": 10, "ImageNet12": 12, "AgNews": 4}
    half_clients = num_clients // 2

    # 建立dataset_name: MNIST, FashionMNIST, CIFAR10和模型的映射
    dataset_name_to_model_map = {
        'MNIST': MLP(28*28, 128, 256, 10),
        'FashionMNIST': FashionCNN(),
        'CIFAR10': ResNet10(),
        'ImageNet12': MobileNetV2(12),
        'AgNews': LSTM()
    }

    if dataset_name not in num_classes_dict:
        raise ValueError("dataset_name should be one of {}".format(num_classes_dict.keys()))

    if dataset_name == "ImageNet12":
        all_dataset_train = ImagenetteDataset(root_dir='/scratch/wenjie/imagenet12/train', transform=None)
        all_dataset_val = ImagenetteDataset(root_dir='/scratch/wenjie/imagenet12/eval', transform=None)

        # 创建一个特定模式的trigger
        specific_pattern = [0, 1, 0, 1, 0, 1, 0, 1, 0]  # 棋盘模式
        trigger_img = create_rgb_trigger(specific_pattern, size=9)

    num_classes = num_classes_dict[dataset_name]


    # 每round选择的客户端比例
    selected_clients_portion = 1
    device = set_GPU(gpu_id)
    selected_clients_index = random.sample(range(num_clients), int(num_clients * selected_clients_portion))
    # 排序index 
    selected_clients_index.sort()


    # 用数据集, iid, 攻击类型, 年月日时分秒, 作为存储config.json和results.json的目录
    save_reslult_dir = get_save_results_dir(dataset_name, iid, alpha, num_clients, attack_method, defense_method)
    save_reslult_dir = os.path.expanduser(save_reslult_dir)
    if not os.path.exists(save_reslult_dir):
        os.makedirs(save_reslult_dir)
    print("save_reslult_dir: ", save_reslult_dir)


    # 存储config.json
    import json
    with open(save_reslult_dir + 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # 存储结果的result.json的路径
    results_json_path = save_reslult_dir + 'result.json'

    # ==================================== 数据集加载 ==========================================================
    malicious_clinet_indices = range(int(num_clients * poisoned_client_portion))

    print("所有数据集的类别数量: ")
    print(all_dataset_train.get_class_num())
    clear_transform_client = get_transform(is_train=True)
    trigger_transform_client = get_transform_add_trigger(trigger_img, is_train=True)
    # split dataset
    client_dataset_instances = all_dataset_train.split(num_clients, iid=iid, alpha=alpha)

    # plot_data_distribution(client_dataset_instances)
    if backdoor:
        # 对数据进行投毒
        for i in malicious_clinet_indices:
            client_dataset_instances[i].set_trigger_img_indices(poison_data_portion, target_label)

    for i in range(len(client_dataset_instances)):
        # 打印每个客户端的数据分布
        if attack_method == 'Backdoor':
            print(f"Client {i}'s data distribution: ", client_dataset_instances[i].get_class_num_with_trigger())
        else:
            print(f"Client {i}'s data distribution: ", client_dataset_instances[i].get_class_num())
        client_dataset_instances[i].set_transform(clear_transform_client, trigger_transform_client)

    # 初始化每个客户端的train_loader
    train_loaders = [torch.utils.data.DataLoader(client_dataset_instances[i], batch_size=batch_size, shuffle=True) for i in range(num_clients)]

    clear_transform_server = get_transform(is_train=False)
    trigger_transform_server = get_transform_add_trigger(trigger_img, is_train=False)

    eval_acc_dataset_instance = all_dataset_val.to_server_dataset(used_for='test_acc', target_label=target_label)
    eval_asr_dataset_instance = all_dataset_val.to_server_dataset(used_for='test_asr', target_label=target_label)

    eval_acc_dataset_instance.set_transform(clear_transform_server, trigger_transform_server)
    eval_asr_dataset_instance.set_transform(clear_transform_server, trigger_transform_server)

    test_acc_loader = torch.utils.data.DataLoader(eval_acc_dataset_instance, batch_size=batch_size, shuffle=False)
    test_asr_loader = torch.utils.data.DataLoader(eval_asr_dataset_instance, batch_size=batch_size, shuffle=False)

    # ======================================== 模型加载 ======================================================
    # 根据dataset_name 对模型进行初始化
    global_model = dataset_name_to_model_map[dataset_name]
    local_models = [copy.deepcopy(global_model) for _ in range(num_clients)]

    global_model.to('cpu')
    local_models = [local_model.to('cpu') for local_model in local_models]

    # 打印可训练的参数数量
    print("Number of trainable parameters: ", count_parameters(global_model))

    # 定义优化器
    # global_optimizer = optim.SGD(global_model.parameters(), lr = global_learning_rate)
    local_optimizers = [optim.SGD(local_models[i].parameters(), lr= local_learning_rate, momentum=local_momentum) for i in range(num_clients)]

    # 获取每个客户端的训练集数量, 并转为权重
    number_of_samples_each_client_list = [len(client_dataset_instances[i]) for i in range(num_clients)]
    weights_of_select_clients = [number_of_samples_each_client_list[i] / sum(number_of_samples_each_client_list) for i in selected_clients_index]

    # ==============================================================================================

    # 恶意客户端的数量 和 index
    num_byzantine = int(num_clients * poisoned_client_portion)
    malicious_client_index = list(range(num_byzantine))

    # 良性客户端的index
    benign_clients_index = list(range(num_byzantine, num_clients))

    print('malicious_clients_index: ', malicious_client_index)  
    print('benign_clients_index: ', benign_clients_index)

    # 训练
    for round in range(global_rounds):
        print('Global round: {}=========================='.format(round))
        global_model_dict_withoutnumbatches = extract_state_dict(global_model)
        for i in range(num_clients):
            local_models[i] = synchronization_local_model(local_models[i], global_model_dict_withoutnumbatches)

        # 记录每个客户端的模型更新 和 损失
        malicious_updates, benign_updates, clients_losses = {}, {}, {}

        # 获取所有良性客户端的更新
        for i in benign_clients_index:
            benign_update, loss = train_and_get_local_update_of_single_client(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs, test_acc_loader, device)
            benign_updates[i] = benign_update
            clients_losses[i] = loss

        # 计算每个entry的均值
        benign_update_mean = {k: torch.mean(torch.stack([benign_updates[i][k] for i in benign_clients_index]), dim=0) for k in global_model_dict_withoutnumbatches}
        # 计算每个entry的标准差
        benign_update_std = {k: torch.std(torch.stack([benign_updates[i][k] for i in benign_clients_index]), dim=0) for k in global_model_dict_withoutnumbatches}

        print('attack method: ', attack_method) 
        # 获取所有恶意客户端的更新
        if attack_method == 'LabelFlipping':
            for i in malicious_client_index:
                malicious_update, loss = train_and_get_local_update_of_attack_LabelFlipping(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs, test_acc_loader, device, num_classes)
                malicious_updates[i] = malicious_update
                clients_losses[i] = loss
        elif attack_method == 'Backdoor' or attack_method == 'NoAttack':
            for i in malicious_client_index:
                malicious_update, loss = train_and_get_local_update_of_single_client(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs, test_acc_loader, device)
                malicious_updates[i] = malicious_update
                clients_losses[i] = loss
        elif attack_method == 'SignFlipping':
            for i in malicious_client_index:
                malicious_update, loss = train_and_get_local_update_of_attack_SignFlipping(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs, test_acc_loader, device)
                malicious_updates[i] = malicious_update
                clients_losses[i] = loss
        elif attack_method == 'Noise-(0,1)':
            for i in malicious_client_index:
                malicious_update, loss = train_and_get_local_update_of_attack_Noise(i, local_models[i])
                malicious_updates[i] = malicious_update
                clients_losses[i] = loss
        elif attack_method == 'IPM-0.1':
            ipm_update, loss = train_and_get_local_update_of_attack_IPM(benign_update_mean, 0.1)
            # malicious updates都是一样的
            malicious_updates = {i: ipm_update for i in malicious_client_index}
            for i in malicious_client_index:
                clients_losses[i] = loss
        elif attack_method == 'IPM-100':
            ipm_update, loss = train_and_get_local_update_of_attack_IPM(benign_update_mean, 100)
            # malicious updates都是一样的
            malicious_updates = {i: ipm_update for i in malicious_client_index}
            for i in malicious_client_index:
                clients_losses[i] = loss
        elif attack_method == 'MinMax':
            minmax_update, loss = minmax_attack_by_binary_search(benign_update_mean, benign_update_std, benign_updates, benign_clients_index)
            # malicious updates都是一样的
            malicious_updates = {i: minmax_update for i in malicious_client_index}
            for i in malicious_client_index:
                clients_losses[i] = loss
        elif attack_method == 'ALIE':
            alie_update, loss = train_and_get_local_update_of_attack_ALIE(num_clients, num_byzantine, benign_update_mean, benign_update_std)
            # malicious updates都是一样的
            malicious_updates = {i: alie_update for i in malicious_client_index}
            for i in malicious_client_index:
                clients_losses[i] = loss
        elif attack_method == 'Adaptive':
            adaptive_update, loss = flud_attack_by_binary_search(benign_update_mean, benign_update_std, benign_updates, benign_clients_index, num_clients, num_byzantine, given_size)
            # malicious updates都是一样的
            malicious_updates = {i: adaptive_update for i in malicious_client_index}
            for i in malicious_client_index:
                clients_losses[i] = loss
        else:
            raise ValueError('Invalid attack method')
        
        # 假设 benign_update_mean 是你的 state_dict
        benign_update_shape = extract_shapes_from_state_dict(benign_update_mean)
        benign_update_length = sum([np.prod(shape) for shape in benign_update_shape.values()])
        # print('benign_update_shape: ', benign_update_shape)
        print('benign_update_length: ', benign_update_length)
        print('defense_method: ', defense_method)

        # 对恶意更新进行防御
        if defense_method == 'FedAvg':
            # 用于聚合的所有客户端的index
            aggregated_clients_index = malicious_client_index + benign_clients_index
            aggregated_clients_weights = [weights_of_select_clients[i] for i in aggregated_clients_index]

            # 对所有客户端的更新进行加权平均
            global_model_update = aggregate_local_model_updates_by_weight([malicious_updates[i] for i in malicious_client_index] + [benign_updates[i] for i in benign_clients_index], weights_of_select_clients)
        elif defense_method == 'FLUD':
            # 防御方法: 计算所有更新的拍平的向量, 对这个向量使用滑动窗口采样中的max(abs(model update)), 记作 model digest
            window_size = given_size
            model_digest = {}
            for i in range(num_clients):
                if i in malicious_client_index:
                    model_digest[i] = max_abs_of_sliding_window(update_update_convert_to_vector(malicious_updates[i]), window_size)
                else:
                    model_digest[i] = max_abs_of_sliding_window(update_update_convert_to_vector(benign_updates[i]), window_size)
            # 取出所有digest, 放到一个list中
            all_digests_list = [model_digest[i] for i in range(num_clients)]
            # 被聚合的客户端的index
            aggregated_clients_index = defense_vote_for_clients(all_digests_list)
            # 他们的权重
            aggregated_clients_weights = [weights_of_select_clients[i] for i in aggregated_clients_index]

            print('aggregated_clients_index: ', aggregated_clients_index)
            
            # 对non zero客户端的更新进行加权平均, 其中包括了恶意客户端和良性客户端
            global_model_update = aggregate_local_model_updates_by_weight([malicious_updates[i] for i in aggregated_clients_index if i in malicious_client_index] + [benign_updates[i] for i in aggregated_clients_index if i in benign_clients_index], aggregated_clients_weights)

        elif defense_method == 'RowSample':
            # 防御方法: 计算所有更新的拍平的向量, 对这个向量使用滑动窗口采样中的max(abs(model update)), 记作 model digest
            window_size = given_size
            model_digest = {}
            for i in range(num_clients):
                if i in malicious_client_index:
                    model_digest[i] = update_update_convert_to_vector(malicious_updates[i])
                else:
                    model_digest[i] = update_update_convert_to_vector(benign_updates[i])
            # 取出所有digest, 放到一个list中
            all_digests_list = [model_digest[i] for i in range(num_clients)]
            # 被聚合的客户端的index
            aggregated_clients_index = defense_vote_for_clients(all_digests_list, distance_matrix)
            # 他们的权重
            aggregated_clients_weights = [weights_of_select_clients[i] for i in aggregated_clients_index]

            print('aggregated_clients_index: ', aggregated_clients_index)
            
            # 对non zero客户端的更新进行加权平均, 其中包括了恶意客户端和良性客户端
            global_model_update = aggregate_local_model_updates_by_weight([malicious_updates[i] for i in aggregated_clients_index if i in malicious_client_index] + [benign_updates[i] for i in aggregated_clients_index if i in benign_clients_index], aggregated_clients_weights)

        elif defense_method == 'AlignSample':
            # 防御方法: 计算所有更新的拍平的向量, 对这个向量使用滑动窗口采样中的max(abs(model update)), 记作 model digest
            window_size = given_size
            model_digest = {}
            for i in range(num_clients):
                if i in malicious_client_index:
                    model_digest[i] = layer_wise_align(malicious_updates[i])
                else:
                    model_digest[i] = layer_wise_align(benign_updates[i])
            # 取出所有digest, 放到一个list中
            all_digests_list = [model_digest[i] for i in range(num_clients)]
            # 被聚合的客户端的index
            aggregated_clients_index = defense_vote_for_clients(all_digests_list, distance_matrix)
            # 他们的权重
            aggregated_clients_weights = [weights_of_select_clients[i] for i in aggregated_clients_index]

            print('aggregated_clients_index: ', aggregated_clients_index)
            
            # 对non zero客户端的更新进行加权平均, 其中包括了恶意客户端和良性客户端
            global_model_update = aggregate_local_model_updates_by_weight([malicious_updates[i] for i in aggregated_clients_index if i in malicious_client_index] + [benign_updates[i] for i in aggregated_clients_index if i in benign_clients_index], aggregated_clients_weights)

        elif defense_method == 'MaxPoolSample':
            # 防御方法: 计算所有更新的拍平的向量, 对这个向量使用滑动窗口采样中的max(abs(model update)), 记作 model digest
            window_size = 5
            model_digest = {}
            for i in range(num_clients):
                if i in malicious_client_index:
                    model_digest[i] = maxpool_of_sliding_window(update_update_convert_to_vector(malicious_updates[i]), window_size)
                else:
                    model_digest[i] = maxpool_of_sliding_window(update_update_convert_to_vector(benign_updates[i]), window_size)
            # 取出所有digest, 放到一个list中
            all_digests_list = [model_digest[i] for i in range(num_clients)]
            # 被聚合的客户端的index
            aggregated_clients_index = defense_vote_for_clients(all_digests_list, distance_matrix)
            # 他们的权重
            aggregated_clients_weights = [weights_of_select_clients[i] for i in aggregated_clients_index]

            print('aggregated_clients_index: ', aggregated_clients_index)
            
            # 对non zero客户端的更新进行加权平均, 其中包括了恶意客户端和良性客户端
            global_model_update = aggregate_local_model_updates_by_weight([malicious_updates[i] for i in aggregated_clients_index if i in malicious_client_index] + [benign_updates[i] for i in aggregated_clients_index if i in benign_clients_index], aggregated_clients_weights)

        elif defense_method == 'Median':
            def median(inputs: List[torch.Tensor]):
                inputs_tensor = torch.stack(inputs, dim=0)
                values_upper, _ = inputs_tensor.median(dim=0)
                values_lower, _ = (-inputs_tensor).median(dim=0)
                return (values_upper - values_lower) / 2
            model_digest = {}
            for i in range(num_clients):
                if i in malicious_client_index:
                    model_digest[i] = update_update_convert_to_vector(malicious_updates[i])
                else:
                    model_digest[i] = update_update_convert_to_vector(benign_updates[i])
            # 取出所有digest, 放到一个list中
            all_digests_list = [model_digest[i] for i in range(num_clients)]
            # 计算中位数
            median_value = median(all_digests_list)
            global_model_update = vector_to_state_dict(median_value, benign_update_shape, benign_update_length)


        elif defense_method == 'Trimmed-Mean':
            # 除去每个维度上的最大的poisoned_client_portion百分比和最小的poisoned_client_portion的百分比, 剩下的值求平均
            model_digest = {}
            for i in range(num_clients):
                if i in malicious_client_index:
                    model_digest[i] = update_update_convert_to_vector(malicious_updates[i])
                else:
                    model_digest[i] = update_update_convert_to_vector(benign_updates[i])
            # 取出所有digest, 放到一个list中
            all_digests_list = [model_digest[i] for i in range(num_clients)]
            num_excluded = num_byzantine    
            trimmed_mean_value = TrimmedMean(all_digests_list, num_excluded)
            global_model_update = vector_to_state_dict(trimmed_mean_value, benign_update_shape, benign_update_length)
            
        elif defense_method == 'ELSA':
            model_digest = {}
            for i in range(num_clients):
                if i in malicious_client_index:
                    model_digest[i] = update_update_convert_to_vector(malicious_updates[i])
                else:
                    model_digest[i] = update_update_convert_to_vector(benign_updates[i])
            # 取出所有digest, 放到一个list中
            all_digests_list = [model_digest[i] for i in range(num_clients)]

            l2norms = [torch.norm(update).item() for update in all_digests_list]
            # 计算中位数
            median_value = np.mean(l2norms)

            # 每个update进行判断, 如果l2norms大于median_value, 则进行缩放
            for i in range(num_clients):
                if l2norms[i] > median_value:
                    all_digests_list[i] = all_digests_list[i] * (median_value / l2norms[i])
            
            global_model_vector = torch.mean(torch.stack(all_digests_list), dim=0)
            global_model_update = vector_to_state_dict(global_model_vector, benign_update_shape, benign_update_length)
    
        elif defense_method == 'Multi-krum':
            def Multikrum(inputs: List[torch.Tensor], num_byzantine: int):
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
            model_digest = {}
            for i in range(num_clients):
                if i in malicious_client_index:
                    model_digest[i] = update_update_convert_to_vector(malicious_updates[i])
                else:
                    model_digest[i] = update_update_convert_to_vector(benign_updates[i])
            # 取出所有digest, 放到一个list中
            all_digests_list = [model_digest[i] for i in range(num_clients)]
            global_model_vector = Multikrum(all_digests_list, num_byzantine)
            global_model_update = vector_to_state_dict(global_model_vector, benign_update_shape, benign_update_length)

        elif defense_method == 'PPBR':
            def cosine_similarity_based_aggregation(inputs: List[torch.Tensor]):
                # 将输入的tensor列表堆叠成一个tensor
                updates = torch.stack(inputs, dim=0)
                
                # 计算余弦相似度矩阵
                def compute_cosine_similarity(updates):
                    num = len(updates)
                    dis_max = np.zeros((num, num))
                    for i in range(num):
                        for j in range(i + 1, num):
                            dis_max[i, j] = torch.nn.functional.cosine_similarity(
                                updates[i, :], updates[j, :], dim=0
                            ).item()  # 转换为Python标量
                            dis_max[j, i] = dis_max[i, j]
                    dis_max[dis_max == -np.inf] = -1
                    dis_max[dis_max == np.inf] = 1
                    dis_max[np.isnan(dis_max)] = 1
                    return dis_max
                
                # 计算每个模型的得分
                def get_scores_by_dismatrix(matrix):
                    k = matrix.shape[1] // 2
                    sorted_matrix = np.sort(matrix, axis=1)
                    sorted_matrix[:, :k] = 0
                    return np.sum(sorted_matrix, axis=1)
                
                # 计算余弦相似度矩阵
                dis_matrix = compute_cosine_similarity(updates)
                
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
            model_digest = {}
            
            for i in range(num_clients):
                if i in malicious_client_index:
                    model_digest[i] = update_update_convert_to_vector(malicious_updates[i])
                else:
                    model_digest[i] = update_update_convert_to_vector(benign_updates[i])
            # 取出所有digest, 放到一个list中
            all_digests_list = [model_digest[i] for i in range(num_clients)]
            global_model_vector = cosine_similarity_based_aggregation(all_digests_list)
            global_model_update = vector_to_state_dict(global_model_vector, benign_update_shape, benign_update_length)

        elif defense_method == 'RFBDS': 
            """ Compresses updates using AlignSample,
                clusters them with OPTICS, and clips the updates using
                l2 norm. """
            # 防御方法: 计算所有更新的拍平的向量, 对这个向量使用滑动窗口采样中的max(abs(model update)), 记作 model digest
            model_digest = {}
            for i in range(num_clients):
                if i in malicious_client_index:
                    model_digest[i] = layer_wise_align(malicious_updates[i])
                else:
                    model_digest[i] = layer_wise_align(benign_updates[i])
            # 取出所有digest, 放到一个list中
            all_digests_list = [model_digest[i] for i in range(num_clients)]
            # 计算欧氏距离
            distance_matrix = compute_euclid_dis(all_digests_list)

            # 聚类
            from sklearn.cluster import AgglomerativeClustering
            def _cluster_updates(dis_max):
                clustering = AgglomerativeClustering(
                    metric="precomputed", linkage="single", n_clusters=2
                )
                clustering.fit(dis_max)

                flag = 1 if np.sum(clustering.labels_) > len(dis_max) // 2 else 0
                selected_idxs = [
                    idx for idx, label in enumerate(clustering.labels_) if label == flag
                ]

                return selected_idxs
            
            selected_idxs = _cluster_updates(distance_matrix)
            print('RFBDS selected_idxs: ', selected_idxs)

            # 计算l2 norm
            l2norms = [torch.norm(digest).item() for digest in all_digests_list]

            # 将列表转换为 NumPy 数组
            my_array = np.array(l2norms)

            # 求聚类的簇的数量, 选这个数量的l2norms作为阈值rho
            k = len(selected_idxs)

            # 使用 numpy.partition 对数组进行分区
            kth_largest_value = np.partition(my_array, -k)[-k]
            rho = kth_largest_value
            

            # 对于每个选中的客户端, 如果l2norm大于rho, 则进行缩放, 顺便从vector转回state_dict
            for i in selected_idxs:
                if l2norms[i] > rho:
                    all_digests_list[i] = all_digests_list[i] * (rho / l2norms[i])
                all_digests_list[i] = vector_to_state_dict(all_digests_list[i], benign_update_shape, benign_update_length)

            selected_updates = [all_digests_list[i] for i in selected_idxs]
            selected_updates_weights = [weights_of_select_clients[i] for i in selected_idxs]
            # 加权平均
            global_model_update = aggregate_local_model_updates_by_weight(selected_updates, selected_updates_weights)
            
        else:
            raise ValueError('Invalid defense method')

        # 用 global optimizer 更新全局模型
        # global_model.to('cpu')
        global_model = global_model.to('cpu')
        # 清除 GPU 缓存
        torch.cuda.empty_cache()
        # 强制垃圾回收
        gc.collect()
        
        global_model_update = {k: global_model_update[k].to('cpu') for k in global_model_update}

        # 只聚合weight和bias
        # for name, param in global_model.named_parameters():
        #     param.data += global_model_update[name]

        # # 用 global_model_update 更新 global_model的state_dict
        global_dict = copy.deepcopy(global_model.state_dict())

        for name in global_model_update:
            global_dict[name] += global_model_update[name]
        global_model.load_state_dict(global_dict)


        
        # 测试
        accuracy = test_acc(global_model, test_acc_loader, device)
        asr = test_asr(global_model, test_asr_loader, device)
        print('Global round: {}, Accuracy: {:.4f}, ASR: {:.4f}'.format(round, accuracy, asr))

        # 对字典client_losses进行排序
        clients_losses = dict(sorted(clients_losses.items(), key=lambda x: x[0]))



        # 存储结果
        result = {
            "training_iteration": round,
            "acc_top_1": accuracy,
            "test_asr": asr,
            "train_loss": clients_losses
        }
        with open(results_json_path, 'a') as f:
            json.dump(result, f)
            f.write('\n')

if __name__ == '__main__':
    
    seed = 42
    set_random_seeds(seed)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    # def run(attack_method, defense_method, num_clients, poisoned_client_portion, poison_data_portion, dataset_name, gpu_id, 
    # iid, alpha, global_rounds, 
    # local_epochs, local_learning_rate, local_momentum, batch_size, given_size, target_label):
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--attack_method', type=str, default='NoAttack', help='Attack method to be applied')
    parser.add_argument('--defense_method', type=str, default='FedAvg', help='Defense method to be applied')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--poisoned_client_portion', type=float, default=0.4, help='Poisoned client portion')
    parser.add_argument('--poison_data_portion', type=float, default=0.5, help='Poison data portion')
    parser.add_argument('--dataset_name', type=str, default='ImageNet12', help='Name of the dataset')
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU id to use')
    parser.add_argument('--iid', type=str2bool, default=True, help='Boolean indicating if the data distribution is IID')
    parser.add_argument('--alpha', type=float, default=10, help='Alpha value for non-IID data distribution')
    parser.add_argument('--global_rounds', type=int, default=50, help='Number of global rounds')
    parser.add_argument('--local_epochs', type=int, default=3, help='Number of local epochs')
    parser.add_argument('--local_learning_rate', type=float, default=0.1, help='Local learning rate')
    parser.add_argument('--local_momentum', type=float, default=0.9, help='Local momentum')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--given_size', type=int, default=4096, help='window size for FLUD defense method')
    parser.add_argument('--target_label', type=int, default=0, help='Target label for backdoor attack')
    args = parser.parse_args()
    
    run(args.attack_method, args.defense_method, args.num_clients, args.poisoned_client_portion, args.poison_data_portion, args.dataset_name, args.gpu_id, 
        args.iid, args.alpha, args.global_rounds, 
        args.local_epochs, args.local_learning_rate, args.local_momentum, args.batch_size, args.given_size, args.target_label)
    
