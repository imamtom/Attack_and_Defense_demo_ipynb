import os

defense_methods = ['FedAvg', 'FLUD', 'RowSample', 'AlignSample', 'MaxPoolSample', 'Median', 'Trimmed-Mean', 'ELSA', 'Multi-krum', 'PPBR', 'RFBDS']
attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise', 'ALIE', 'MinMax', 'IPM-01', 'IPM-100', 'Backdoor', 'NoAttack', 'Adaptive']
# python main.py --dataset ImageNet12 --attack_method Backdoor --defense_method FLUD --iid True --local_learning_rate 0.1 --batch_size 128 --local_epochs 3 --alpha 1 --given_size 4096 --gpu_id 1
# 根据攻击手段和防御手段生成对应的 bash 文件

bash_sentences = []
bash_sentence = 'python main.py --dataset ImageNet12 --attack_method {} --defense_method {} --iid True --local_learning_rate 0.1 --batch_size 128 --local_epochs 3 --alpha 1 --given_size 4096 --gpu_id 1'
for attack_method in attack_methods:
    for defense_method in defense_methods:
        bash_sentences.append(bash_sentence.format(attack_method, defense_method))

with open('all_imagenet12_bashes.sh', 'w') as f:
    for bash_sentence in bash_sentences:
        f.write(bash_sentence + '\n')
    f.close()


