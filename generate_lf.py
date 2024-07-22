# 为训练生成训练命令
# 生成的命令会保存在bash文件中
# 生成的命令会使用gpu 0

import os
import sys
import time

# 生成的命令会保存在bash文件中
bash_file = 'train_gpu_0.sh'

dataset_name = ['MNIST', 'FashionMNIST', 'CIFAR10'] 

# attack_methods = ['LabelFlipping']

defense_methods = ['FedAvg', 'FLUD']

alphas = ['0.1', '1', '10']

# 生成的命令会使用gpu 0, 固定参数 --attack_method LabelFlipping 

# 生成的命令会保存在bash文件中
with open(bash_file, 'w') as f:
    for dataset in dataset_name:
        for alpha in alphas:
            for defense_method in defense_methods:
                f.write('python run.py --dataset {} --attack_method LabelFlipping  --defense_method {} --iid False --alpha {} --given_size 4096 --gpu_id 0\n'.format(dataset, defense_method, alpha))