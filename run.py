
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse


# 定义训练模型, 为逻辑回归模型
# 可训练的参数数量: 7840 = 28*28*10 + 10
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

# 定义MLP, 28*28-128-256-10
# 可训练的参数数量: 137074 = 28*28*128 + 128 + 128*256 + 256 + 256*10 + 10
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out


# 用来训练FashionMNIST数据集的CNN模型
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop(out)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)

        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet10():
    return ResNet(BasicBlock, [1, 1, 1, 1])

# 用ResNet10模型进行测试
# model = ResNet10()
# print(model)

# 定义单个客户端的训练函数
def train_and_get_local_update_of_single_client(client_index, local_model, local_optimizer, train_loader, local_epochs):
    # 记录本轮初始的模型参数
    intial_model = copy.deepcopy(local_model.state_dict())
    
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
            
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            local_optimizer.zero_grad()
            loss.backward()
            # 裁剪梯度
            # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1)
            # 更新参数
            local_optimizer.step()
            train_loss += loss.item()
            
    # loss
    train_loss /= len(train_loader)
    print('Client{} local model, Epoch: {}, Loss: {:.4f}'.format(client_index, epoch, train_loss))
    # 计算本轮训练后的模型参数与初始模型参数的差值
    local_model_update = {k: local_model.state_dict()[k] - intial_model[k] for k in intial_model}
    # print("local_model_update: ", local_model_update)
    return local_model_update, train_loss

# 定义compriomised_clients的攻击手段 attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise', 'ALIE', 'MinMax', 'IPM', 'Backdoor']
def train_and_get_local_update_of_attack_LabelFlipping(client_index, local_model, local_optimizer, train_loader, local_epochs):
    # 记录本轮初始的模型参数
    intial_model = copy.deepcopy(local_model.state_dict())
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 记录训练loss
    train_loss = 0.0
    for epoch in range(local_epochs):
        local_model.train()
        for images, labels in train_loader:
            # 修改标签为 9-label, 然后训练
            labels = 9 - labels
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            local_optimizer.zero_grad()
            loss.backward()
            # 裁剪梯度
            # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1)
            # 更新参数
            local_optimizer.step()
            train_loss += loss.item()
    # loss
    train_loss /= len(train_loader)
    print('Client{} local model, Epoch: {}, Loss: {:.4f}'.format(client_index, epoch, train_loss))
    # 计算本轮训练后的模型参数与初始模型参数的差值
    local_model_update = {k: local_model.state_dict()[k] - intial_model[k] for k in intial_model}
    return local_model_update, train_loss
        
# 定义SignFlipping的攻击函数
def train_and_get_local_update_of_attack_SignFlipping(client_index, local_model, local_optimizer, train_loader, local_epochs):
    # 记录本轮初始的模型参数
    intial_model = copy.deepcopy(local_model.state_dict())
    
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
            
    # loss, 为什么是nan? 因为计算的梯度是负数, 但是计算的loss是正数, 所以loss会是nan
    train_loss /= len(train_loader)
    print('Client{} local model, Epoch: {}, Loss: {:.4f}'.format(client_index, epoch, train_loss))
    # 计算本轮训练后的模型参数与初始模型参数的差值
    local_model_update = {k: local_model.state_dict()[k] - intial_model[k] for k in intial_model}
    return local_model_update, train_loss

# 定义Noise的攻击函数
def train_and_get_local_update_of_attack_Noise(client_index, local_model):
    noise_mean=0
    noise_std=1
    # 模型更新为和模型参数同形状的正态分布
    local_model_update = {k: torch.normal(mean=noise_mean, std=noise_std, size=local_model.state_dict()[k].shape).to(device) for k in local_model.state_dict()}
    train_loss = 0
    return local_model_update, train_loss

# 定义IPM的攻击函数
def train_and_get_local_update_of_attack_IPM(benign_update_mean, scale):
    # 相反方向的scale倍缩放
    local_model_update = {k: -scale * benign_update_mean[k] for k in benign_update_mean}
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
    local_model_update = {k: benign_update_mean[k] + benign_update_std[k] * z_max for k in benign_update_mean}
    train_loss = 0
    return local_model_update, train_loss


# 定义一个函数, 计算两个local_model_update的L2距离
def l2_distance_of_2_updates(model_update1, model_update2):
    distance = 0
    for k in model_update1:
        distance += torch.norm(model_update1[k] - model_update2[k], p=2) ** 2
    return torch.sqrt(distance)


def minmax_attack_by_binary_search(benign_update_mean, benign_update_std, benign_updates, benign_clients_index):
    # 所有良性梯度的mean, 标准差, 良性梯度的之间距离的最大值
    all_distances_of_benign_updates = torch.stack([l2_distance_of_2_updates(benign_updates[i], benign_updates[j]) for i in benign_clients_index for j in benign_clients_index])
    threshold = all_distances_of_benign_updates.max()

    low = 0
    high = 5
    while abs(high - low) > 0.01:
        mid = (low + high) / 2
        # mal_update = torch.stack([mean_grads - mid * deviation])
        malicious_update = {k: benign_update_mean[k] - mid * benign_update_std[k] for k in benign_update_mean}
        # loss = torch.cdist(mal_update, benign_updates, p=2).max()
        loss = torch.stack([l2_distance_of_2_updates(malicious_update, benign_updates[i]) for i in benign_clients_index]).max()
        if loss < threshold:
            low = mid
        else:
            high = mid
    train_loss = 0
    return malicious_update, train_loss


# 读取每个客户端的数据
def load_multiple_client_training_data(num_clients, read_dir):
    clients_images_list = [[] for _ in range(num_clients)]
    clients_labels_list = [[] for _ in range(num_clients)]
    clients_modified_labels_list = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        data = np.load(read_dir + 'client{}_data.npz'.format(i))
        clients_images_list[i] = data['images']
        clients_labels_list[i] = data['labels']
        clients_modified_labels_list[i] = data['modified_labels']

    test_acc_data = np.load(read_dir + 'test_acc_data.npz')
    test_acc_images = test_acc_data['images']
    test_acc_labels = test_acc_data['labels']

    test_asr_data = np.load(read_dir + 'test_asr_data.npz')
    test_asr_images = test_asr_data['images']
    test_asr_labels = test_asr_data['labels']

    return clients_images_list, clients_labels_list, clients_modified_labels_list, test_acc_images, test_acc_labels, test_asr_images, test_asr_labels

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

# 生成保存目录的函数
def get_processed_data_dir(dataset_name, iid, alpha, num_clients, backdoor, poisoned_client_portion, poison_data_portion):
    if iid == True:
        data_dir = '~/processed_data/{}_iid-{}_numclients-{}_backdoor-{}_poisonedclientsportion-{}_poisoneddataportion-{}/'.format(dataset_name, iid, num_clients, backdoor, poisoned_client_portion, poison_data_portion)
    else:
        data_dir = '~/processed_data/{}_iid-{}_alpha-{}_numclients-{}_backdoor-{}_poisonedclientsportion-{}_poisoneddataportion-{}/'.format(dataset_name, iid, alpha, num_clients, backdoor, poisoned_client_portion, poison_data_portion)
    return data_dir


# 定义函数, 获取并返回每个客户端的训练集数量
def get_client_dataset_numbers(clients_images_numpuy_list, num_clients):
    number_of_samples_each_client_list = [0] * num_clients
    for i in range(num_clients):
        number_of_samples_each_client_list[i] = len(clients_images_numpuy_list[i])
    return number_of_samples_each_client_list


# 定义测试函数
def test_acc(global_model, test_loader):
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 定义投毒成功率测试函数
def test_asr(model, test_loader_poison_test_10percent):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader_poison_test_10percent:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def aggregate_local_model_updates_by_weight(local_model_updates, weights_of_select_clients):
    print('本轮被聚合的客户端数量: ', len(local_model_updates))
    assert len(local_model_updates) == len(weights_of_select_clients)
    global_model_update = {k: torch.zeros_like(local_model_updates[0][k]) for k in local_model_updates[0]}
    for k in global_model_update:
        for i in range(len(local_model_updates)):
            global_model_update[k] += local_model_updates[i][k] * weights_of_select_clients[i]
    return global_model_update

def aggregate_local_updates_by_avg(local_model_updates):
    global_model_update = {k: torch.zeros_like(local_model_updates[0][k]) for k in local_model_updates[0]}
    for k in global_model_update:
        for i in range(len(local_model_updates)):
            global_model_update[k] += local_model_updates[i][k]
        global_model_update[k] /= len(local_model_updates)
    return global_model_update


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
# 使用滑动窗口对一维向量进行采样, 返回max(abs())
def max_abs_of_sliding_window(vector, window_size):
    # 先转为绝对值
    vector = torch.abs(vector)
    # 然后用max_pool1d, ceil = True, 保证最后一个窗口不会被舍弃
    max_pool = nn.MaxPool1d(window_size, stride=window_size, ceil_mode=True)
    vector = vector.unsqueeze(0)
    vector = max_pool(vector)
    return vector.squeeze()

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
    # 获取每行前10个最小距离值的索引
    top_indices = np.argsort(dis_matrix, axis=1)[:, :10]
    # 创建一个全零的矩阵，然后将每行的[10个最小距离值的索引]位置标为1
    result_matrix = np.zeros_like(dis_matrix)
    rows, cols = np.indices(top_indices.shape)
    result_matrix[rows, top_indices] = 1
    column_sums = np.sum(result_matrix, axis=0)
    return column_sums


device = None

def run(dataset_name, attack_method, defense_method, iid, alpha, given_size, gpu_id):
    # 在这里添加你的代码逻辑
    print(f"Dataset Name: {dataset_name}")
    print(f"Attack Method: {attack_method}")
    print(f"Defense Method: {defense_method}")
    print(f"IID: {iid}")
    print(f"Alpha: {alpha}")
    print(f"Given Size: {given_size}")
    
    if attack_method == 'Noise':
        attack_method = 'Noise-(0,1)'

    if attack_method == 'IPM01':
        attack_method = 'IPM-0.1'
    
    if attack_method == 'IPM100':
        attack_method = 'IPM-100'

    # 攻击类型限制范围
    attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise-(0,1)', 'ALIE', 'MinMax', 'IPM-0.1', 'IPM-100', 'Backdoor', 'NoAttack']
    assert attack_method in attack_methods, "attack_method should be one of {}".format(attack_methods)

    if attack_method == 'Backdoor':
        backdoor = True
    else:
        backdoor = False
    print("backdoor attack: ", backdoor)

    # poisioned client portion
    poisoned_client_portion = 0.4

    # poisioned data portion
    poison_data_portion = 0.5
    print("poisoned data portion in each client: ", poison_data_portion)

    # target label
    target_label = 0
    print("target label: ", target_label)

    # 设置数据集的超参数
    num_clients = 20
    num_classes = 10

    # 被comprised的客户端的index
    comprised_client_idx = range(num_clients)
    print("comprised_client_idx: ", comprised_client_idx)

    # 读取数据所在的目录
    read_dir = get_processed_data_dir(dataset_name, iid, alpha, num_clients, backdoor, poisoned_client_portion, poison_data_portion)
    print("read_dir: ", read_dir)
    # 扩展波浪号到用户主目录
    read_dir = os.path.expanduser(read_dir)
    # 检查目录是否存在, 不存在则发出警告
    if not os.path.exists(read_dir):
        print("Warning: the directory does not exist!")


    # 建立dataset_name: MNIST, FashionMNIST, CIFAR10和模型的映射
    dataset_name_to_model_map = {
        'MNIST': MLP(28*28, 128, 256, 10),
        'FashionMNIST': FashionCNN(),
        # 'FashionMNIST': MLP(28*28, 128, 256, 10),
        'CIFAR10': ResNet10()
    }

    global device 
    device = set_GPU(gpu_id)

    # 根据dataset_name 对模型进行初始化
    global_model = dataset_name_to_model_map[dataset_name]
    local_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    # 模型转为device
    global_model.to(device)
    local_models = [local_model.to(device) for local_model in local_models]

    # 打印参数数量的函数
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 并打印出global_model的结构
    print('global_model: ', global_model)
    # 打印可训练的参数数量
    print("Number of trainable parameters: ", count_parameters(global_model))

    # 考虑所有攻击方式

    global_learning_rate = 1

    # 定义超参数
    batch_size = 128
    local_learning_rate = 0.1
    local_momentum = 0.9 # 动量
    local_weight_decay = 0
    # local_weight_decay = 5e-4 # L2正则化系数

    # global_rounds是全局迭代次数
    global_rounds = 50
    # local_epochs是每个客户端的本地训练次数
    local_epochs = 10

    # 每round选择的客户端比例
    selected_clients_portion = 1
    selected_clients_index = random.sample(range(num_clients), int(num_clients * selected_clients_portion))
    # 排序index 
    selected_clients_index.sort()

    # 加载数据集
    clients_images_numpuy_list, clients_labels_list, clients_modified_labels_list, test_acc_images, test_acc_labels, test_asr_images, test_asr_labels = load_multiple_client_training_data(num_clients, read_dir)
    print("clients_images_list: ", len(clients_images_numpuy_list))
    print("clients_images_list[0]: ", clients_images_numpuy_list[0].shape, type(clients_images_numpuy_list[0]))
    print("clients_images_list[0][0]", clients_images_numpuy_list[0][0].shape, type(clients_images_numpuy_list[0][0]))

    # 训练数据集转为可训练的格式
    for i in range(num_clients):
        clients_images_numpuy_list[i] = torch.from_numpy(clients_images_numpuy_list[i]).float()
        # labels是整数
        clients_labels_list[i] = torch.from_numpy(clients_labels_list[i]).long()
        clients_modified_labels_list[i] = torch.from_numpy(clients_modified_labels_list[i]).long()


    # 转为dataloader
    # 标签会影响测试集的准确率
    train_loaders = [DataLoader(torch.utils.data.TensorDataset(clients_images_numpuy_list[i], clients_modified_labels_list[i]), batch_size=batch_size, shuffle=True) for i in range(num_clients)]

    # 测试acc的数据集转为可训练的格式
    test_acc_images = torch.from_numpy(test_acc_images).float()
    test_acc_labels = torch.from_numpy(test_acc_labels).long()
    # 转为dataloader
    test_acc_dataset = torch.utils.data.TensorDataset(test_acc_images, test_acc_labels)
    test_acc_loader = torch.utils.data.DataLoader(dataset=test_acc_dataset, batch_size=len(test_acc_images), shuffle=False)

    # 测试asr的数据集转为可训练的格式
    test_asr_images = torch.from_numpy(test_asr_images).float()
    test_asr_labels = torch.from_numpy(test_asr_labels).long()
    # 转为dataloader
    test_asr_dataset = torch.utils.data.TensorDataset(test_asr_images, test_asr_labels)
    test_asr_loader = torch.utils.data.DataLoader(dataset=test_asr_dataset, batch_size=len(test_asr_images), shuffle=False)
    
    
    # 打印每个客户端的训练集数量
    for i in range(num_clients):
        print("Client{} dataset size: {}".format(i, len(clients_images_numpuy_list[i])))
    # 打印测试acc数据集的数量
    print("Test acc dataset size: ", len(test_acc_images))
    # 打印测试asr数据集的数量
    print("Test asr dataset size: ", len(test_asr_images))
    

    # 定义优化器
    global_optimizer = optim.SGD(global_model.parameters(), lr = global_learning_rate)
    # local_optimizers = [optim.SGD(local_models[i].parameters(), lr = local_learning_rate) for i in range(num_clients)]
    local_optimizers = [optim.SGD(local_models[i].parameters(), lr= local_learning_rate, momentum=local_momentum, weight_decay=local_weight_decay) for i in range(num_clients)]

    # 用数据集, iid, 攻击类型, 年月日时分秒, 作为存储config.json和results.json的目录
    save_reslult_dir = get_save_results_dir(dataset_name, iid, alpha, num_clients, attack_method, defense_method)
    save_reslult_dir = os.path.expanduser(save_reslult_dir)
    if not os.path.exists(save_reslult_dir):
        os.makedirs(save_reslult_dir)
    print("save_reslult_dir: ", save_reslult_dir)

    # 存储config.json
    config = {
        "defense_method": defense_method,
        "batch_size": batch_size,
        "attack_method": attack_method,
        "local_epochs": local_epochs,
        "local_learning_rate": local_learning_rate,
        "num_clients": num_clients,
        "num_classess": num_classes,
        "join_ratio": selected_clients_portion,
        "dataset": read_dir,
        "global_rounds": global_rounds,
        "given_size": 4096,
    }
    # 存储config.json
    import json
    with open(save_reslult_dir + 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # 存储结果的result.json的路径
    results_json_path = save_reslult_dir + 'result.json'


    # 获取每个客户端的训练集数量, 并转为权重
    number_of_samples_each_client_list = get_client_dataset_numbers(clients_images_numpuy_list, num_clients)
    weights_of_select_clients = [number_of_samples_each_client_list[i] / sum(number_of_samples_each_client_list) for i in selected_clients_index]

    # 训练
    for round in range(global_rounds):
        print('Global round: {}=========================='.format(round))

        # 下发全局模型
        for i in range(num_clients):
            local_models[i].load_state_dict(global_model.state_dict())

        
        # 恶意客户端的数量 和 index
        num_byzantine = int(num_clients * poisoned_client_portion)
        malicious_client_index = list(range(num_byzantine))

        # 良性客户端的index
        benign_clients_index = list(range(num_byzantine, num_clients))

        print('malicious_clients_index: ', malicious_client_index)  
        print('benign_clients_index: ', benign_clients_index)

        # 记录每个客户端的模型更新 和 损失
        malicious_updates, benign_updates, clients_losses = {}, {}, {}

        # 获取所有良性客户端的更新
        for i in benign_clients_index:
            benign_update, loss = train_and_get_local_update_of_single_client(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs)
            benign_updates[i] = benign_update
            clients_losses[i] = loss
            # print('update of client{}: '.format(i), benign_updates[i])  
        
        # 计算每个entry的均值
        benign_update_mean = {k: torch.mean(torch.stack([benign_updates[i][k] for i in benign_clients_index]), dim=0) for k in global_model.state_dict()}
        # 计算每个entry的标准差
        benign_update_std = {k: torch.std(torch.stack([benign_updates[i][k] for i in benign_clients_index]), dim=0) for k in global_model.state_dict()}
        
        # print('benign_update_mean: ', benign_update_mean)
        # print('benign_update_std: ', benign_update_std)
        

        print('attack method: ', attack_method) 
        # 获取所有恶意客户端的更新
        if attack_method == 'LabelFlipping':
            for i in malicious_client_index:
                malicious_update, loss = train_and_get_local_update_of_attack_LabelFlipping(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs)
                malicious_updates[i] = malicious_update
                clients_losses[i] = loss
        elif attack_method == 'Backdoor' or attack_method == 'NoAttack':
            for i in malicious_client_index:
                malicious_update, loss = train_and_get_local_update_of_single_client(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs)
                malicious_updates[i] = malicious_update
                clients_losses[i] = loss
        elif attack_method == 'SignFlipping':
            for i in malicious_client_index:
                malicious_update, loss = train_and_get_local_update_of_attack_SignFlipping(i, local_models[i], local_optimizers[i], train_loaders[i], local_epochs)
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
            print('ALIE update: ', alie_update)
            malicious_updates = {i: alie_update for i in malicious_client_index}
            for i in malicious_client_index:
                clients_losses[i] = loss
        else:
            raise ValueError('Invalid attack method')



        print('defense_method: ', defense_method)
        # 这里用于计算全局模型的更新
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
            # 计算欧氏距离
            distance_matrix = compute_euclid_dis(all_digests_list)
            # print('distance_matrix: ', distance_matrix)
            # 投票
            votes = votes_by_dismatrix(distance_matrix)
            # 对votes中小于10的元素, 用0代替
            votes[votes < 10] = 0
            # 获取那些非0 的元素的index, 并转为list
            non_zero_clients_index = list(np.nonzero(votes)[0])

            # 被聚合的客户端的index
            aggregated_clients_index = non_zero_clients_index
            # 他们的权重
            aggregated_clients_weights = [weights_of_select_clients[i] for i in aggregated_clients_index]


            # aggregated_clients_index 中恶意客户端的index
            aggregated_malicious_client_index = [i for i in aggregated_clients_index if i in malicious_client_index]
            # aggregated_clients_index 中良性客户端的index
            aggregated_benign_clients_index = [i for i in aggregated_clients_index if i in benign_clients_index]

            # 对non zero客户端的更新进行加权平均, 其中包括了恶意客户端和良性客户端
            global_model_update = aggregate_local_model_updates_by_weight([malicious_updates[i] for i in aggregated_malicious_client_index] + [benign_updates[i] for i in aggregated_benign_clients_index], aggregated_clients_weights)
            
        # 被聚合的客户端的index
        print('aggregated_clients_index: ', aggregated_clients_index)
        # 打印权重
        print('aggregated_clients_weights: ', aggregated_clients_weights)
        

        # 计算全局模型
        # global_model_update = aggregate_local_model_updates_by_weight([malicious_updates[i] for i in malicious_client_index] + [benign_updates[i] for i in benign_clients_index] , weights_of_select_clients)
        # global_model_update = aggregate_local_updates_by_avg(local_model_updates)

        # print('global_model_update: ', global_model_update) 

        # 用 global optimizer 更新全局模型
        for name, param in global_model.named_parameters():
            param.data += global_model_update[name]
        
        # 测试
        accuracy = test_acc(global_model, test_acc_loader)
        asr = test_asr(global_model, test_asr_loader)
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


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--attack_method', type=str, required=True, help='Attack method to use')
    parser.add_argument('--defense_method', type=str, required=True, help='Defense method to be applied')
    parser.add_argument('--iid', type=str2bool, default=True, help='Boolean indicating if the data distribution is IID')
    parser.add_argument('--alpha', type=str, default='10', help='Alpha value, a numeric parameter')
    parser.add_argument('--given_size', type=int, default=4096, help='window size for FLUD defense method')
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU id to use')
    args = parser.parse_args()
    
    run(args.dataset_name, args.attack_method, args.defense_method, args.iid, args.alpha, args.given_size, args.gpu_id)
