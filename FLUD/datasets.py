from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import random
from matplotlib import pyplot as plt


def create_rgb_trigger(pattern=None, size=9, colors=None):
    """
    创建一个size x size的RGB trigger，由3x3的彩色方块组成
    
    参数:
    pattern : 一个长度为(size/3)^2的列表，指定每个3x3块的颜色索引
               如果不指定，则随机生成
    size : 触发器的大小，默认为9
    colors : 一个包含两种RGB颜色的列表，默认为黑色和白色
    
    返回:
    trigger : size x size x 3的PIL Image对象，RGB图案
    """
    block_num = (size // 3) ** 2
    if pattern is None:
        pattern = np.random.randint(0, 2, block_num)
    elif len(pattern) != block_num:
        raise ValueError(f"Pattern must be a list of length {block_num}")
    
    if colors is None:
        colors = [(0, 0, 0), (255, 255, 255)]  # 默认黑白
    
    trigger = np.zeros((size, size, 3), dtype=np.uint8)
    
    for i in range(size // 3):
        for j in range(size // 3):
            color = colors[pattern[i * (size // 3) + j]]
            trigger[i*3:i*3+3, j*3:j*3+3] = color
    
    return Image.fromarray(trigger, mode='RGB')

class AddTrigger(object):
    def __init__(self, trigger_img):
        self.trigger_img = trigger_img

    def __call__(self, img):
        """
        将不透明的 trigger_img 添加到输入图像的右下角
        
        参数:
        img : PIL Image 对象
        
        返回:
        PIL Image 对象，右下角添加了不透明的 trigger_img
        """
        img_np = np.array(img)
        trigger_np = np.array(self.trigger_img)
        
        h, w = img_np.shape[:2]
        th, tw = trigger_np.shape[:2]
        
        # 计算 trigger 应该放置的位置
        y_offset = h - th
        x_offset = w - tw
        
        # 如果输入图像是灰度图，将其转换为 RGB
        if len(img_np.shape) == 2:
            img_np = np.stack((img_np,) * 3, axis=-1)
        
        # 直接将 trigger 覆盖到图像的右下角，不考虑透明度
        img_np[y_offset:, x_offset:, :3] = trigger_np[:, :, :3]
        
        return Image.fromarray(img_np)
    
# 定义预处理步骤
def get_transform(is_train=True):
    # ImageNet数据集的均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(256),  # 首先将图像调整为稍大的尺寸
            transforms.CenterCrop(224),  # 然后从中心裁剪出所需的224x224大小
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform

# get_transform_trigger
def get_transform_add_trigger(trigger_img, is_train=True):
    # ImageNet数据集的均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(256),  # 首先将图像调整为稍大的尺寸
            transforms.CenterCrop(224),  # 然后从中心裁剪出所需的224x224大小
            AddTrigger(trigger_img),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform


class ImagenetteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)} # class名字到class index的字典映射
        self.num_labels = len(self.classes)
        self.images = self._load_images() # 一个list,每个元素是(img_path, label)的tuple
        # 如果访问实例.images[i]返回的是一个元组(img_path, label)
        # 如果访问实例[i]返回的是一个元组(img, label)
        self.labels_array = np.array([label for _, label in self.images])

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    # 设置transform
    def set_transform(self, transform):
        self.transform = transform


    def random_select_delete_remain(self, proportion):
        # 从数据集中随即保留proportion比例的数据,
        import random
        random.shuffle(self.images)
        self.images = self.images[:int(len(self.images)*proportion)]
    # 打印每个类别的数量
    def get_class_num(self):
        class_num = {}
        for _, label in self.images:
            if label not in class_num:
                class_num[label] = 1
            else:
                class_num[label] += 1
        return class_num
    # split dataset into num_client parts
    def split_iid(self, num_clients):
        label_numpy = self.labels_array
        num_classes = self.num_labels
        clientid_to_each_label_indices = {i:{ j:{} for j in range(num_classes)} for i in range(num_clients)}
        for class_index in range(num_classes):
            label_index = np.where(label_numpy == class_index)[0]
            num_label = len(label_index)
            # 计算每个客户端应该分配的样本数量, 余数部分均匀分配到前面的客户端
            num_samples_per_client = num_label // num_clients
            remaining_samples = num_label % num_clients     
            # 该类别下, 每个客户端分到的样本索引
            label_index_dict = {}
            start_index = 0
            for client_index in range(num_clients):
                if client_index < remaining_samples:
                    label_index_dict[client_index] = label_index[start_index: start_index + num_samples_per_client + 1]
                    start_index += num_samples_per_client + 1
                else:
                    label_index_dict[client_index] = label_index[start_index: start_index + num_samples_per_client]
                    start_index += num_samples_per_client
            # 更新clientid_to_label_indices
            for client_index in range(num_clients):
                clientid_to_each_label_indices[client_index][class_index] = label_index_dict[client_index]
        return clientid_to_each_label_indices

    def split_image_data_dirichlet(self, num_clients, alpha):
        num_classes = self.num_labels
        clientid_to_each_label_indices = {i:{ j:{} for j in range(num_classes)} for i in range(num_clients)}
        labels_numpy = self.labels_array
        # 每个客户端至少有least_num_samples个样本
        least_num_samples = 1
        # 定义一个比例, 让每个客户端的数据数量在总体数据中的比例至少达到这个比例
        threshold_proportion = 1 / num_clients * 0.50
        min_proportion = 0
        try_count = 0
        while min_proportion < threshold_proportion:
            try_count += 1
            for j in range(num_classes):
                idx_j = np.where(labels_numpy == j)[0]
                # 确保每个客户端至少有一个样本
                initial_split = np.array_split(idx_j[:least_num_samples*num_clients], num_clients)
                remaining_indices = idx_j[least_num_samples*num_clients:]

                # 生成迪利克雷分布
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients)) # 等价于np.random.dirichlet([alpha] * num_clients)
                remaining_splits = np.split(remaining_indices, (proportions * len(remaining_indices)).astype(int).cumsum()[:-1])
                for i in range(num_clients):
                    indices = np.concatenate((initial_split[i], remaining_splits[i] if i < len(remaining_splits) else []))
                    clientid_to_each_label_indices[i][j] = indices  
            # 计算每个客户端的数据比例
            min_proportion = 1
            for i in range(num_clients):
                client_proportion = 0
                for j in range(num_classes):
                    client_proportion += len(clientid_to_each_label_indices[i][j])
                client_proportion /= len(labels_numpy)
                min_proportion = min(min_proportion, client_proportion)  
        
        # 统计每个客户端的数据比例
        proportions_each_client = []
        for i in range(num_clients):
            client_proportion = 0
            for j in range(num_classes):
                client_proportion += len(clientid_to_each_label_indices[i][j])
            client_proportion /= len(labels_numpy)
            proportions_each_client.append(client_proportion)
        proportions_each_client = np.array(proportions_each_client).round(4)
        print("each client's proportion of processing data: ", proportions_each_client)
        print("each client's min threshold of data proportion : ", threshold_proportion)
        print("try_count: ", try_count)
        return clientid_to_each_label_indices
    
    # split dataset, return a set of dataset
    def split(self, num_client, iid=True, alpha=1):
        client_dataset_instances = []
        if iid:
            clientid_to_each_label_indices = self.split_iid(num_client)
        else:
            clientid_to_each_label_indices = self.split_image_data_dirichlet(num_client, alpha)
        
        for client_id in range(num_client):
            images_of_client = []
            for lable_id in clientid_to_each_label_indices[client_id]:
                for idx in clientid_to_each_label_indices[client_id][lable_id]:
                    images_of_client.append(self.images[idx])
            client_dataset_instances.append(ImagenetteDataset_per_client(client_id, images_of_client))

        return client_dataset_instances

    # 转为server端的数据集
    def to_server_dataset(self, used_for='test_acc', target_label=None):
        if used_for not in ['test_acc', 'test_asr']:
            raise ValueError("used_for must be 'test_acc' or 'test_asr'")
        if used_for == 'test_acc':
            return ImagenetteDataset_server(self.images, used_for=used_for, target_label=target_label)
        elif used_for == 'test_asr':
            if target_label == None:
                raise ValueError("target_label must be specified when used_for is 'test_asr'")
            # 去掉label == target_label的数据, 剩余数据集的label均变为target_label
            images = [(img_path, target_label) for img_path, label in self.images if label != target_label]
            return ImagenetteDataset_server(images, used_for=used_for, target_label=target_label)

class ImagenetteDataset_server(Dataset):
    # 传入 一个list images,每个元素是(image, label)的tuple
    def __init__(self, images, used_for='test_acc', target_label=None, clear_transform=None, trigger_transform=None):
        used_for_list = ['test_acc', 'test_asr']
        if used_for not in used_for_list:
            raise ValueError("used_for must be 'test_acc' or 'test_asr'")
        self.used_for = used_for # 用来决定访问__getitem__时返回的数据是否带有trigger
        self.target_label = target_label
        self.images = images
        self.clear_transform = clear_transform
        self.trigger_transform = trigger_transform
        self.num_labels = len(set([label for _, label in self.images]))
        self.labels_array = np.array([label for _, label in self.images])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.used_for == 'test_acc': 
            image = self.clear_transform(image)
        elif self.used_for == 'test_asr':
            image = self.trigger_transform(image)
        else:
            raise ValueError("used_for must be 'test_acc' or 'test_asr'")
        return image, label
    # 统计每个类别的数量
    def get_class_num(self):
        class_num = {}
        for _, label in self.images:
            if label not in class_num:
                class_num[label] = 1
            else:
                class_num[label] += 1
        return class_num
    # 设置transform 
    def set_transform(self, clear_transform, trigger_transform):
        if self.used_for == 'test_acc':
            self.clear_transform = clear_transform
        elif self.used_for == 'test_asr':
            self.trigger_transform = trigger_transform

    
class ImagenetteDataset_per_client(Dataset):
    # 传入 一个list images,每个元素是(image, label)的tuple
    def __init__(self, client_id, images, clear_transform=None, trigger_transform=None):
        self.client_id = client_id
        self.images = images
        self.clear_transform = clear_transform
        self.trigger_transform = trigger_transform
        self.num_labels = len(set([label for _, label in self.images]))
        self.labels_array = np.array([label for _, label in self.images])
        self.trigger_label_array = self.labels_array
        self.trigger_img_indices = []
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if idx in self.trigger_img_indices:
            # 需要
            image = self.trigger_transform(image)
            label = self.trigger_label_array[idx]
        else:
            image = self.clear_transform(image)
            label = label
        return image, label
    
    # 统计每个类别的数量
    def get_class_num(self):
        class_num = {}
        for _, label in self.images:
            if label not in class_num:
                class_num[label] = 1
            else:
                class_num[label] += 1
        return class_num
    # 注入trigger
    def set_trigger_img_indices(self, poison_data_portion, target_label):
        num_classes = self.num_labels
        labels_numpy = self.labels_array
        num_poison_samples = int(len(self.images) * poison_data_portion)
        label_to_label_indices = {i: np.where(labels_numpy == i)[0] for i in range(num_classes)}
        # 非target_label数据标签看作是优先被投毒的数据
        prior_poison_idx = np.empty((0,), dtype=int)
        for class_id in range(num_classes):
            if class_id != target_label:
                prior_poison_idx = np.concatenate((prior_poison_idx, label_to_label_indices[class_id]))
        
        # 如果非target_label的数据数量大于poison_data_portion比例的数据, 则从prior_poison_idx 随机选择poison_data_portion比例的数据
        print("prior_poison_idx: ", len(prior_poison_idx))
        print("num_poison_samples: ", num_poison_samples)
        if len(prior_poison_idx) >= int(num_poison_samples):
            poison_idx = np.random.choice(prior_poison_idx, int(num_poison_samples), replace=False)
        else:
            # 说明非target_label的数据不够, 需要从target_label中选择
            # 收集所有非target_label的数据
            poison_idx = prior_poison_idx
            # 计算需要从target_label中选择的数量
            supplement_num = num_poison_samples - len(prior_poison_idx)
            poison_idx = np.concatenate((poison_idx, label_to_label_indices[target_label][:supplement_num]))
        print("被植入trigger的样本的poison_idx: ", len(poison_idx))
        # 对poison_idx中的样本注入trigger
        self.trigger_img_indices = poison_idx
        # 同时更改trigger_label_array
        self.trigger_label_array = np.array([target_label if idx in poison_idx else label for idx, label in enumerate(self.trigger_label_array)])

    # 设置两个transform
    def set_transform(self, clear_transform, trigger_transform):
        self.clear_transform = clear_transform
        self.trigger_transform = trigger_transform

    # 获取每个类别的数量, 以及每个类别中被置入trigger的数量
    def get_class_num_with_trigger(self):
        class_num = {} # class_num[label] = number of samples
        class_num_trigger = {} # class_num_trigger[label] = number of samples with trigger
        for idx in range(len(self.images)):
            _, label = self.images[idx]
            if label not in class_num:
                class_num[label] = 1
                class_num_trigger[label] = 0
            else:
                class_num[label] += 1
            if idx in self.trigger_img_indices:
                class_num_trigger[label] += 1
        return class_num, class_num_trigger
    
# 画图, 传入多个数据集实例, 画出每个client的数据分布
def plot_data_distribution(client_dataset_instances):
    for i in range(len(client_dataset_instances)):
        class_num = client_dataset_instances[i].get_class_num()
        plt.bar(class_num.keys(), class_num.values())
        plt.title(f"Client {i}'s data distribution")
        plt.show()

# 根据clientid_to_label_indices中的index数量, 画出每个客户端的数据分布
def plot_data_distribution(client_dataset_instances):
    num_clients = len(client_dataset_instances)
    num_classes = client_dataset_instances[0].num_labels
    import matplotlib.pyplot as plt
    class_num_list = []
    for i in range(len(client_dataset_instances)):
        class_num = client_dataset_instances[i].get_class_num()
        class_num_list.append(class_num)
        print(f"Client {i}'s data distribution: ", class_num)

    plt.figure(figsize=(4, 5))
    # 每个点的坐标为(client_index, label)
    # 每个点的大小为number_of_samples
    # 每个点的颜色为client_index
    # 生成num_clients种颜色
    colors = plt.cm.jet(np.linspace(0, 1, num_clients))
    for client_index in range(num_clients):
        for label in range(num_classes):
            number_of_samples = class_num_list[client_index].get(label, 0)
            plt.scatter(client_index, label, s=number_of_samples, color=colors[client_index], alpha=0.5)
    # 坐标刻度分别是是0到num_clients-1, 0到num_classes-1
    plt.xticks(range(num_clients))
    plt.yticks(range(num_classes))
    plt.xlabel('client id')
    plt.ylabel('class')
    plt.title('Data distribution of each client')
    plt.show()

# 画出每个客户端的数据分布, 包括被注入trigger的数据, 每个客户端是一个柱状图, 所有柱状图放在一个图中
def plot_data_distribution_with_trigger(client_dataset_instances):
    # 子图数量
    num_clients = len(client_dataset_instances)
    # 计算行数和列数
    num_rows = (num_clients + 1) // 2  # 向上取整
    num_cols = 2
    # 创建子图
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    # 确保 axs 是二维数组
    axs = np.array(axs).reshape(num_rows, num_cols)

    num_clients = len(client_dataset_instances)
    num_classes = client_dataset_instances[0].num_labels
    class_num_list = []
    class_num_trigger_list = []
    for i in range(len(client_dataset_instances)):
        class_num, class_num_trigger = client_dataset_instances[i].get_class_num_with_trigger()
        class_num_list.append(class_num)
        class_num_trigger_list.append(class_num_trigger)

    for i in range(num_clients):
        row = i // 2
        col = i % 2
        class_num = class_num_list[i]
        class_num_trigger = class_num_trigger_list[i]
        axs[row, col].bar(class_num.keys(), class_num.values(),
                    color='b',
                    label='clear data')
        axs[row, col].bar(class_num_trigger.keys(), class_num_trigger.values(),
                    color='r',
                    label='trigger data')
        axs[row, col].set_title(f"Client {i}")
        axs[row, col].legend()
    plt.show()
    
# 显示样本
def display_image(img, method='normalize', title=None):
    """
    Display a PyTorch tensor as an image.
    
    Args:
    img (torch.Tensor): The input image tensor (C, H, W).
    method (str): 'normalize' or 'clip' to handle out-of-range values.
    title (str, optional): Title for the plot.
    """
    with torch.no_grad():
        if method == 'normalize':
            img_processed = (img - img.min()) / (img.max() - img.min())
        elif method == 'clip':
            img_processed = torch.clamp(img, 0, 1)
        else:
            raise ValueError("Method must be either 'normalize' or 'clip'")
        
        # Ensure the tensor is on CPU and convert to numpy array
        img_processed = img_processed.cpu().permute(1, 2, 0).numpy()
        
        plt.figure(figsize=(2, 2))
        plt.imshow(img_processed)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

