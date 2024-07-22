# 为训练生成训练命令

import os
import sys
import time

# 生成的命令会保存在bash文件中
bash_file = 'train_gpu_1.sh'

dataset_name = ['MNIST', 'FashionMNIST', 'CIFAR10'] 

# attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise', 'ALIE', 'MinMax', 'IPM01', 'IPM100', 'Backdoor', 'NoAttack']

defense_methods = ['RowSample', 'AlignSample', 'MaxPoolSample']

# 生成的命令会使用gpu 1, 固定参数 --iid True --alpha 10 --given_size 4096 --gpu_id 1

# 生成的命令会保存在bash文件中
with open(bash_file, 'w') as f:
    for dataset in dataset_name:
        for defense_method in defense_methods:
            f.write('python run_sample.py --dataset {} --attack_method Backdoor --defense_method {} --iid True --alpha 10 --given_size 4096 --gpu_id 1\n'.format(dataset, defense_method))