import torch

def extract_state_dict(model):
    """
    1. 输入模型, 提取模型的state_dict，如果存在则排除num_batches_tracked
    
    Args:
        model (torch.nn.Module): 输入的模型
    
    Returns:
        dict: 不包含num_batches_tracked的state_dict
    """
    return {name: param.clone().detach() for name, param in model.state_dict().items()
            if 'num_batches_tracked' not in name}

def update_local_model(local_model, global_state_dict):
    """
    2. 使用全局模型的state_dict更新本地模型
    
    Args:
        local_model (torch.nn.Module): 本地模型
        global_state_dict (dict): 全局模型的state_dict
    """
    local_state_dict = local_model.state_dict()
    
    for name, param in global_state_dict.items():
        if name in local_state_dict:
            local_state_dict[name].copy_(param)
    
    local_model.load_state_dict(local_state_dict)

def aggregate_state_dicts(state_dicts):
    """
    3. 聚合所有本地模型的state_dict, 得到一个聚合的state_dict
    
    Args:
        state_dicts (list): 包含多个本地模型state_dict的列表
    
    Returns:
        dict: 聚合后的state_dict
    """
    aggregated = {}
    num_models = len(state_dicts)
    
    for name in state_dicts[0].keys():
        aggregated[name] = sum(sd[name] for sd in state_dicts) / num_models
    
    return aggregated

def update_global_model(global_model, aggregated_state_dict):
    """
    4. 聚合的state_dict更新全局模型
    
    Args:
        global_model (torch.nn.Module): 全局模型
        aggregated_state_dict (dict): 聚合后的state_dict
    """
    global_state_dict = global_model.state_dict()
    
    for name, param in aggregated_state_dict.items():
        if name in global_state_dict:
            global_state_dict[name].copy_(param)
    
    global_model.load_state_dict(global_state_dict)

# 示例用法
def federated_learning_round(global_model, local_models):
    """
    执行一轮联邦学习
    
    Args:
        global_model (torch.nn.Module): 全局模型
        local_models (list): 本地模型列表
    """
    # 1. 提取全局模型的state_dict
    global_state_dict = extract_state_dict(global_model)
    
    # 2. 更新所有本地模型
    for local_model in local_models:
        update_local_model(local_model, global_state_dict)
    
    # 假设这里进行了本地训练
    # local_training(local_models)
    
    # 3. 提取并聚合所有本地模型的state_dict
    local_state_dicts = [extract_state_dict(model) for model in local_models]
    aggregated_state_dict = aggregate_state_dicts(local_state_dicts)
    
    # 4. 使用聚合的state_dict更新全局模型
    update_global_model(global_model, aggregated_state_dict)

# 主函数示例
def main():
    # 初始化全局模型（可以是ResNet10或ResNet34）
    global_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
    
    # 模拟多个本地模型（实际应用中这些应该分布在不同的客户端）
    num_clients = 3
    local_models = [torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False) for _ in range(num_clients)]
    
    # 执行多轮联邦学习
    num_rounds = 5
    for _ in range(num_rounds):
        federated_learning_round(global_model, local_models)

if __name__ == "__main__":
    main()