import numpy as np
import torch
import os
from config import Config


config = Config()

def create_mask_from_lengths(seq_lengths, max_len):
    """
    根据序列长度创建掩码
    
    参数:
        seq_lengths: 每个序列的长度 [batch]
        max_len: 最大序列长度
    
    返回:
        mask: 序列掩码张量 [batch, max_len]
    """
    batch_size = len(seq_lengths)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for i, length in enumerate(seq_lengths):
        mask[i, :length] = 1.0
    return mask.to(config.DEVICE)

def normalize_advantages(advantages):
    """标准化优势函数"""
    mean = advantages.mean()
    std = advantages.std() + 1e-8
    return (advantages - mean) / std

def save_checkpoint(model, optimizer, step, path):
    """保存训练检查点"""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Saved checkpoint at step {step} to {path}")

def load_checkpoint(model, optimizer, path):
    """加载训练检查点"""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"Loaded checkpoint from step {start_step}")
        return start_step
    return 0

def compute_response_lengths(responses):
    """计算响应序列的实际长度"""
    return [(r != 0).sum().item() for r in responses]
