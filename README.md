# Attack_and_Defense_demo_ipynb
simulation of Attack_and_Defense for Federated Learning

### 数据生成

生成过程需要在`fl_dataset_load_and_train.ipynb`中指定`dataset_name`, `backdoor`, `iid`, `alpha`, `target_label`.

建议先不要动`poisoned_client_portion`, `poison_data_portion`, `num_clients`, `num_classes = 10` 因为后续训练过程中有些是写死的.

它会在你的用户目录下生成一个`test_imaget_data`目录用于存储下载的原始训练集和测试集文件, 并在你的用户目录下生成一个`processed_data`目录用于处理好的数据集.


### 数据训练

训练过程需要在`fl_dataset_load_and_train.ipynb`中指定`dataset_name`, `iid`, `alpha`, `num_clients`, `backdoor`, `poisoned_client_portion`, `poison_data_portion`

攻击方式目前集成了
- LabelFlipping
- SignFlipping
- Noise-(0,1)
- ALIE 
- MinMax
- IPM-0.1
- IPM-100
- Backdoor
- NoAttack

防御方式目前集成了
- FLUD
- FedAvg

训练步骤
- 目前预设 id 为`0-7` 的客户端为恶意, `8-19`的客户端为良性
- 根据客户端的类型, 逐个训练良性的/伪造恶意的本地模型更新
- 服务器采用样本数量的使用鲁棒聚合方法本地模型更新

### 批量运行

``` bash
nohup bash train_gpu_0.sh > output_gpu0.txt 2>&1 &
nohup bash train_gpu_1.sh > output_gpu1.txt 2>&1 &
nohup bash train_gpu_2.sh > output_gpu2.txt 2>&1 &
nohup bash train_gpu_3.sh > output_gpu3.txt 2>&1 &
```