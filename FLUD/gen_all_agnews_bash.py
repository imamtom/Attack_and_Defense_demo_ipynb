import os

defense_methods = ['FedAvg', 'FLUD', 'RowSample', 'AlignSample', 'MaxPoolSample', 'Median', 'Trimmed-Mean', 'ELSA', 'Multi-krum', 'PPBR', 'RFBDS']
attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise', 'ALIE', 'MinMax', 'IPM-01', 'IPM-100', 'Backdoor', 'NoAttack', 'Adaptive']
# # python main.py --dataset AgNews --attack_method Backdoor --defense_method FedAvg --num_clients 20 --iid True --local_learning_rate 0.1 --global_rounds 30 --batch_size 128 --local_epochs 1 --alpha 0.1 --given_size 4096 --gpu_id 0
# 根据攻击手段和防御手段生成对应的 bash 文件

bash_sentences = []
bash_sentence = 'python main.py --dataset AgNews --attack_method {} --defense_method {} --num_clients 20 --iid True --local_learning_rate 0.1 --global_rounds 30 --batch_size 128 --local_epochs 1 --alpha 0.1 --given_size 4096 --gpu_id 0'
for attack_method in attack_methods:
    for defense_method in defense_methods:
        bash_sentences.append(bash_sentence.format(attack_method, defense_method))

with open('all_agnews_bashes.sh', 'w') as f:
    for bash_sentence in bash_sentences:
        f.write(bash_sentence + '\n')
    f.close()


