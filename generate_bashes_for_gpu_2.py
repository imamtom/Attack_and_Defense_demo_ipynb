# 为训练生成训练命令

import os
import sys
import time

# 生成的命令会保存在bash文件中
bash_file = 'train_gpu_2.sh'

dataset_name = ['MNIST', 'FashionMNIST', 'CIFAR10'] 

attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise', 'ALIE', 'MinMax', 'IPM01', 'IPM100', 'Backdoor', 'NoAttack']

defense_methods = ['FedAvg', 'FLUD']

# 固定参数 --iid False --alpha 1 --given_size 4096 --gpu_id 2

# 生成的命令会保存在bash文件中
with open(bash_file, 'w') as f:
    for dataset in dataset_name:
        for attack_method in attack_methods:
            for defense_method in defense_methods:
                f.write('python run.py --dataset {} --attack_method {} --defense_method {} --iid False --alpha 1 --given_size 4096 --gpu_id 2\n'.format(dataset, attack_method, defense_method))