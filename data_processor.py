import torch
from torch.utils.data import Dataset
from config import Config


config = Config()


class PromptDataset(Dataset):
    """提示数据集"""
    def __init__(self, prompts):
        self.prompts = prompts


    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]
    

def load_prompts(file_path):
    with open (file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    return prompts

def create_dataloader(prompts, batch_size):
    """创建数据加载器"""

    dataset = PromptDataset(prompts)
    #返回批处理数据
    batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    return batches

def tokenize_prompts(tokenizer, prompts):
    """将提示文本转换为模型输入"""
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,                #如果序列超过 max_length，自动截断到最大长度
        max_length=config.MAX_SEQ_LEN,
        return_tensors="pt"             #返回pytorch张量
    )
    # 转移到设备
    input_ids = inputs.input_ids.to(config.DEVICE)
    attention_mask = inputs.attention_mask.to(config.DEVICE)
    return input_ids, attention_mask
