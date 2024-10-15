from utils import set_GPU, set_random_seeds, test_acc, test_asr, get_save_results_dir, extract_shapes_from_state_dict, extract_state_dict, synchronization_local_model, count_parameters
from utils import update_update_convert_to_vector, max_abs_of_sliding_window, defense_vote_for_clients, maxpool_of_sliding_window, vector_to_state_dict, layer_wise_align_on_vector, compute_euclid_dis
from utils import _mean, _median, _weighted_mean, _std, _OPTICS_for_vectors
import os
import random
from models import FashionCNN, MLP, ResNet10, MobileNetV2, LSTM, ResNet34
from datasets import ImagenetteDataset, ImagenetteDataset_server, ImagenetteDataset_per_client, create_rgb_trigger, get_transform, get_transform_add_trigger
import copy
import torch
import numpy as np
import torch.optim as optim
from defense import aggregate_local_model_updates_by_weight, TrimmedMean, MultiKrum, PPBR
from attack import train_and_get_local_vector_of_single_client, train_and_get_local_vector_of_attack_LabelFlipping, train_and_get_local_update_of_attack_SignFlipping, train_and_get_local_vector_of_attack_Noise, train_and_get_local_vector_of_attack_IPM, minmax_attack_by_binary_search, train_and_get_local_vector_of_attack_ALIE, flud_attack_by_binary_search
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

    if attack_method == 'Noise':
        attack_method = 'Noise-(0,1)'
    if attack_method == 'IPM01':
        attack_method = 'IPM-0.1'
    # 攻击类型限制范围
    attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise-(0,1)', 'ALIE', 'MinMax', 'IPM-0.1', 'IPM-100', 'Backdoor', 'NoAttack', 'Adaptive']

    assert attack_method in attack_methods, "attack_method should be one of {}".format(attack_methods)
    if attack_method == 'Backdoor':
        backdoor = True
    else:
        backdoor = False

    defense_methods = ['FedAvg', 'FLUD', 'RowSample', 'AlignSample', 'MaxPoolSample', 'Median', 'Trimmed-Mean', 'ELSA', 'Multi-krum', 'PPBR', 'RFBDS']
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
    # selected_clients_portion = 1
    selected_clients_index = range(num_clients)


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

    # ==================================== GPU设置 ==========================================================
    
    device = set_GPU(gpu_id)
    num_workers_for_dataloader = 8
    
    # ==================================== 数据集加载 ==========================================================
    malicious_client_indices = list(range(int(num_clients * poisoned_client_portion)))
    benign_client_indices = list(range(int(num_clients * poisoned_client_portion), num_clients))
    num_byzantine = len(malicious_client_indices)
    print("malicious_client_indices: ", malicious_client_indices)
    print("benign_client_indices: ", benign_client_indices)


    print("所有数据集的类别数量: ")
    print(all_dataset_train.get_class_num())
    clear_transform_client = get_transform(is_train=True)
    trigger_transform_client = get_transform_add_trigger(trigger_img, is_train=True)
    # split dataset
    client_dataset_instances = all_dataset_train.split(num_clients, iid=iid, alpha=alpha)

    # plot_data_distribution(client_dataset_instances)
    if backdoor:
        # 对数据进行投毒
        for i in malicious_client_indices:
            client_dataset_instances[i].set_trigger_img_indices(poison_data_portion, target_label)

    for i in range(len(client_dataset_instances)):
        # 打印每个客户端的数据分布
        if attack_method == 'Backdoor':
            print(f"Client {i}'s data distribution: ", client_dataset_instances[i].get_class_num_with_trigger())
        else:
            print(f"Client {i}'s data distribution: ", client_dataset_instances[i].get_class_num())
        client_dataset_instances[i].set_transform(clear_transform_client, trigger_transform_client)

    # 初始化每个客户端的train_loader
    train_loaders = [torch.utils.data.DataLoader(client_dataset_instances[i], batch_size=batch_size, shuffle=True, num_workers = num_workers_for_dataloader) for i in range(num_clients)]

    clear_transform_server = get_transform(is_train=False)
    trigger_transform_server = get_transform_add_trigger(trigger_img, is_train=False)

    eval_acc_dataset_instance = all_dataset_val.to_server_dataset(used_for='test_acc', target_label=target_label)
    eval_asr_dataset_instance = all_dataset_val.to_server_dataset(used_for='test_asr', target_label=target_label)

    eval_acc_dataset_instance.set_transform(clear_transform_server, trigger_transform_server)
    eval_asr_dataset_instance.set_transform(clear_transform_server, trigger_transform_server)

    test_acc_loader = torch.utils.data.DataLoader(eval_acc_dataset_instance, batch_size=batch_size, shuffle=False, num_workers = num_workers_for_dataloader)
    test_asr_loader = torch.utils.data.DataLoader(eval_asr_dataset_instance, batch_size=batch_size, shuffle=False, num_workers = num_workers_for_dataloader)

    # ======================================== 模型加载 ======================================================
    # 根据dataset_name 对模型进行初始化
    global_model = dataset_name_to_model_map[dataset_name]
    local_models = [copy.deepcopy(global_model) for _ in range(num_clients)]

    global_model.to(device)
    local_models = [local_model.to(device) for local_model in local_models]

    # 打印可训练的参数数量
    print("Number of trainable parameters: ", count_parameters(global_model))

    # 定义优化器
    # global_optimizer = optim.SGD(global_model.parameters(), lr = global_learning_rate)
    local_optimizers = [optim.SGD(local_models[i].parameters(), lr= local_learning_rate, momentum=local_momentum) for i in range(num_clients)]

    # 获取每个客户端的训练集数量, 并转为权重
    number_of_samples_each_client_list = [len(client_dataset_instances[i]) for i in range(num_clients)]
    weights_of_select_clients = [number_of_samples_each_client_list[i] / sum(number_of_samples_each_client_list) for i in selected_clients_index]
    # 转为tensor
    # weights_of_select_clients = torch.tensor(weights_of_select_clients, device=device)
    # 模型的哪些层是需要被聚合的
    layers_to_aggregate = []
    for name, param in global_model.state_dict().items():
        if "num_batches_tracked" not in name:
            layers_to_aggregate.append(name)
    
    # 需要被聚合的层的shape
    name_shapes_to_aggregate = extract_shapes_from_state_dict(global_model.state_dict(), layers_to_aggregate)
    benign_update_length = sum([np.prod(shape) for shape in name_shapes_to_aggregate.values()])
    # print('name_shapes_to_aggregate: ', name_shapes_to_aggregate)
    print('benign_update_length: ', benign_update_length)



    # ==============================================================================================
    # 训练
    for round in range(global_rounds):
        print('Global round: {}=========================='.format(round))
        global_model_dict = extract_state_dict(global_model, layers_to_aggregate)
        for i in range(num_clients):
            local_models[i] = synchronization_local_model(local_models[i], global_model_dict, layers_to_aggregate)

        # 记录每个客户端的模型更新 和 损失
        malicious_vectors, benign_vectors, clients_losses = {}, {}, {}

        # 获取所有良性客户端的更新
        for i in benign_client_indices:
            benign_vectors[i], clients_losses[i] = train_and_get_local_vector_of_single_client(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs, test_acc_loader, layers_to_aggregate, device)

        # 计算每个entry的均值
        benign_vector_mean = _mean([benign_vectors[i] for i in benign_client_indices])
        # 计算每个entry的标准差
        benign_vector_std = _std([benign_vectors[i] for i in benign_client_indices])

        print('attack method: ', attack_method) 
        # 获取所有恶意客户端的更新
        if attack_method == 'LabelFlipping':
            for i in malicious_client_indices:
                malicious_update, loss = train_and_get_local_vector_of_attack_LabelFlipping(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs, test_acc_loader, layers_to_aggregate, num_classes, device)
                malicious_vectors[i] = malicious_update
                clients_losses[i] = loss
        elif attack_method == 'Backdoor' or attack_method == 'NoAttack':
            for i in malicious_client_indices:
                malicious_update, loss = train_and_get_local_vector_of_single_client(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs, test_acc_loader, layers_to_aggregate, device)
                malicious_vectors[i] = malicious_update
                clients_losses[i] = loss
        elif attack_method == 'SignFlipping':
            for i in malicious_client_indices:
                malicious_update, loss = train_and_get_local_update_of_attack_SignFlipping(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs, test_acc_loader, layers_to_aggregate, device)
                malicious_vectors[i] = malicious_update
                clients_losses[i] = loss
        elif attack_method == 'Noise-(0,1)':
            for i in malicious_client_indices:
                malicious_update, loss = train_and_get_local_vector_of_attack_Noise(benign_vector_mean)
                malicious_vectors[i] = malicious_update.to(device)
                clients_losses[i] = loss
        elif attack_method == 'IPM-0.1':
            ipm_vector, loss = train_and_get_local_vector_of_attack_IPM(benign_vector_mean, 0.1)
            # malicious updates都是一样的
            malicious_vectors = {i: ipm_vector.to(device) for i in malicious_client_indices}
            for i in malicious_client_indices:
                clients_losses[i] = loss
        elif attack_method == 'IPM-100':
            ipm_vector, loss = train_and_get_local_vector_of_attack_IPM(benign_vector_mean, 100)
            # malicious updates都是一样的
            malicious_vectors = {i: ipm_vector.to(device) for i in malicious_client_indices}
            for i in malicious_client_indices:
                clients_losses[i] = loss
        elif attack_method == 'MinMax':
            minmax_update, loss = minmax_attack_by_binary_search(benign_vector_mean, benign_vector_std, [benign_vectors[i] for i in benign_client_indices])
            # malicious updates都是一样的
            malicious_vectors = {i: minmax_update.to(device) for i in malicious_client_indices}
            for i in malicious_client_indices:
                clients_losses[i] = loss
        elif attack_method == 'ALIE':
            alie_vector, loss = train_and_get_local_vector_of_attack_ALIE(num_clients, num_byzantine, benign_vector_mean, benign_vector_std)
            # malicious updates都是一样的
            malicious_vectors = {i: alie_vector.to(device) for i in malicious_client_indices}
            for i in malicious_client_indices:
                clients_losses[i] = loss
        elif attack_method == 'Adaptive':
            adaptive_update, loss = flud_attack_by_binary_search(benign_vector_mean, benign_vector_std, [benign_vectors[i] for i in benign_client_indices], num_clients, num_byzantine, given_size)
            # malicious updates都是一样的
            malicious_vectors = {i: adaptive_update.to(device) for i in malicious_client_indices}
            for i in malicious_client_indices:
                clients_losses[i] = loss
        else:
            raise ValueError('Invalid attack method')
        
        # ============================== 防御方法 ========================================
        # 对恶意更新进行防御
        if defense_method == 'FedAvg':
            # 用于聚合的所有客户端的index
            aggregated_clients_index = malicious_client_indices + benign_client_indices
            aggregated_clients_weights = torch.tensor([weights_of_select_clients[i] for i in aggregated_clients_index], device=device)
            # 对所有客户端的更新进行加权平均
            global_model_vector = _weighted_mean([malicious_vectors[i] for i in aggregated_clients_index if i in malicious_client_indices] + [benign_vectors[i] for i in aggregated_clients_index if i in benign_client_indices], aggregated_clients_weights)
        elif defense_method == 'FLUD':
            # 防御方法: 计算所有更新的拍平的向量, 对这个向量使用滑动窗口采样中的max(abs(model update)), 记作 model digest
            window_size = given_size
            model_digest = {}
            for i in range(num_clients):
                if i in malicious_client_indices:
                    model_digest[i] = max_abs_of_sliding_window(malicious_vectors[i], window_size)
                else:
                    model_digest[i] = max_abs_of_sliding_window(benign_vectors[i], window_size)
            # 取出所有digest, 放到一个list中
            aggregated_clients_index = defense_vote_for_clients([model_digest[i] for i in range(num_clients)])
            # 他们的权重
            aggregated_clients_weights = torch.tensor([weights_of_select_clients[i] for i in aggregated_clients_index], device=device)
            print('aggregated_clients_index: ', aggregated_clients_index)
            
            # 对non zero客户端的更新进行加权平均, 其中包括了恶意客户端和良性客户端
            global_model_vector = _weighted_mean([malicious_vectors[i] for i in aggregated_clients_index if i in malicious_client_indices] + [benign_vectors[i] for i in aggregated_clients_index if i in benign_client_indices], aggregated_clients_weights)

        elif defense_method == 'RowSample':
            # 被聚合的客户端的index
            aggregated_clients_index = defense_vote_for_clients([malicious_vectors[i] for i in malicious_client_indices] + [benign_vectors[i] for i in benign_client_indices])
            # 他们的权重
            aggregated_clients_weights = torch.tensor([weights_of_select_clients[i] for i in aggregated_clients_index], device=device)

            print('aggregated_clients_index: ', aggregated_clients_index)
            
            # 对non zero客户端的更新进行加权平均, 其中包括了恶意客户端和良性客户端
            global_model_vector = _weighted_mean([malicious_vectors[i] for i in aggregated_clients_index if i in malicious_client_indices] + [benign_vectors[i] for i in aggregated_clients_index if i in benign_client_indices], aggregated_clients_weights)

        elif defense_method == 'AlignSample':
            # 被聚合的客户端的index
            aggregated_clients_index = defense_vote_for_clients([layer_wise_align_on_vector(malicious_vectors[i], name_shapes_to_aggregate) for i in malicious_client_indices] + [layer_wise_align_on_vector(benign_vectors[i], name_shapes_to_aggregate) for i in benign_client_indices])
            # 他们的权重
            aggregated_clients_weights = torch.tensor([weights_of_select_clients[i] for i in aggregated_clients_index], device=device)

            print('aggregated_clients_index: ', aggregated_clients_index)
            
            # 对non zero客户端的更新进行加权平均, 其中包括了恶意客户端和良性客户端
            global_model_vector = _weighted_mean([malicious_vectors[i] for i in aggregated_clients_index if i in malicious_client_indices] + [benign_vectors[i] for i in aggregated_clients_index if i in benign_client_indices], aggregated_clients_weights)

        elif defense_method == 'MaxPoolSample':
            kernel_size = 5
            # 被聚合的客户端的index
            aggregated_clients_index = defense_vote_for_clients([maxpool_of_sliding_window(malicious_vectors[i], kernel_size) for i in malicious_client_indices] + [maxpool_of_sliding_window(benign_vectors[i], kernel_size) for i in benign_client_indices])
            # 他们的权重
            aggregated_clients_weights = torch.tensor([weights_of_select_clients[i] for i in aggregated_clients_index], device=device)

            print('aggregated_clients_index: ', aggregated_clients_index)
            
            # 对non zero客户端的更新进行加权平均, 其中包括了恶意客户端和良性客户端
            global_model_vector = _weighted_mean([malicious_vectors[i] for i in aggregated_clients_index if i in malicious_client_indices] + [benign_vectors[i] for i in aggregated_clients_index if i in benign_client_indices], aggregated_clients_weights)

        elif defense_method == 'Median':
            # 计算中位数
            median_value = _median([malicious_vectors[i] for i in malicious_client_indices] + [benign_vectors[i] for i in benign_client_indices])
            global_model_vector = median_value


        elif defense_method == 'Trimmed-Mean':
            num_excluded = num_byzantine    
            trimmed_mean_value = TrimmedMean([malicious_vectors[i] for i in malicious_client_indices] + [benign_vectors[i] for i in benign_client_indices], num_excluded)
            global_model_vector = trimmed_mean_value
            
        elif defense_method == 'ELSA':
            # 取出所有digest, 放到一个list中
            all_digests_list = [malicious_vectors[i] for i in malicious_client_indices] + [benign_vectors[i] for i in benign_client_indices]
            # 计算平均l2范数
            l2norms = [torch.norm(update).item() for update in all_digests_list]
            median_value = np.mean(l2norms)

            # 每个update进行判断, 如果l2norms大于median_value, 则进行缩放
            for i in range(num_clients):
                if l2norms[i] > median_value:
                    all_digests_list[i] = all_digests_list[i] * (median_value / l2norms[i])
            
            global_model_vector = _mean(all_digests_list)
    
        elif defense_method == 'Multi-krum':
            # 取出所有digest, 放到一个list中
            all_digests_list = [malicious_vectors[i] for i in malicious_client_indices] + [benign_vectors[i] for i in benign_client_indices]

            global_model_vector = MultiKrum(all_digests_list, num_byzantine)

        elif defense_method == 'PPBR':
            # 取出所有digest, 放到一个list中
            all_digests_list = [malicious_vectors[i] for i in malicious_client_indices] + [benign_vectors[i] for i in benign_client_indices]
            global_model_vector = PPBR(all_digests_list)

        elif defense_method == 'RFBDS': 
            """ Compresses updates using AlignSample,
                clusters them with OPTICS, and clips the updates using
                l2 norm. """
            # 防御方法: 计算所有更新的拍平的向量, 对这个向量使用滑动窗口采样中的max(abs(model update)), 记作 model digest
            # 取出所有digest, 放到一个list中
            all_digests_list = [layer_wise_align_on_vector(malicious_vectors[i], name_shapes_to_aggregate) for i in malicious_client_indices] + [layer_wise_align_on_vector(benign_vectors[i], name_shapes_to_aggregate) for i in benign_client_indices]

            # 计算欧氏距离
            distance_matrix = compute_euclid_dis(all_digests_list)
            # 对所有客户端的digest进行聚类
            selected_idxs = _OPTICS_for_vectors(distance_matrix)
            print('RFBDS selected_idxs: ', selected_idxs)

            # 计算l2 norm
            l2norms = [torch.norm(digest).item() for digest in all_digests_list]

            # 求聚类的簇的数量, 选这个数量的l2norms作为阈值rho
            k = len(selected_idxs)
            rho =  np.partition(np.array(l2norms), -k)[-k]
            

            # 对于每个选中的客户端, 如果l2norm大于rho, 则进行缩放, 顺便从vector转回state_dict
            for i in selected_idxs:
                if l2norms[i] > rho:
                    all_digests_list[i] = all_digests_list[i] * (rho / l2norms[i])

            selected_updates_weights = torch.tensor([weights_of_select_clients[i] for i in selected_idxs], device=device)
            # 加权平均
            global_model_vector = _weighted_mean([all_digests_list[i] for i in selected_idxs], selected_updates_weights)
            
        else:
            raise ValueError('Invalid defense method')

        # 清除 GPU 缓存
        torch.cuda.empty_cache()
        # 强制垃圾回收
        gc.collect()
        
        global_model_vector = vector_to_state_dict(global_model_vector, name_shapes_to_aggregate, benign_update_length)

        # 只聚合weight和bias
        # for name, param in global_model.named_parameters():
        #     param.data += global_model_update[name]

        # # 用 global_model_update 更新 global_model的state_dict
        global_dict = global_model.state_dict()

        for name in global_model_vector:
            if name in layers_to_aggregate:
                global_dict[name] += global_model_vector[name]
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
    
