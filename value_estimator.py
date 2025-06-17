import torch
from config import Config

config = Config()


def compute_advatages(rewards, values, masks, last_value, gamma=config.GAMMA, gae_lambda=config.GAE_LAMBDA):
    """
    使用GAE（广义优势估计）计算优势函数
    
    参数:
        rewards: 每个时间步的奖励 [batch, seq]
        values: 价值函数的预测值 [batch, seq+1]
        masks: 标记序列长度的掩码 (1=有效, 0=填充) [batch, seq]
        last_value: 最后状态的价值 [batch, 1]
        gamma: 折扣因子
        gae_lambda: GAE参数
    
    返回:
        advantages: 优势函数 [batch, seq]
        returns: 回报值 [batch, seq]
    """

    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # 处理last_value的形状
    if last_value.dim() == 1:
        next_value = last_value  # 已经是[batch]形状
    else:
        next_value = last_value.squeeze(-1)  # 安全地从[batch,1]变为[batch]
    
    # 存储当前GAE值
    gae = 0
    
    # 从后向前计算优势
    for t in reversed(range(seq_len)):
        # 计算下一状态的值
        if t == seq_len - 1:
            next_values = next_value
        else:
            next_values = values[:, t+1]
        
        # 计算TD误差：δₜ = rₜ + γ * V(sₜ₊₁) * mask - V(sₜ)
        td_error = rewards[:, t] + gamma * next_values * masks[:, t] - values[:, t]
        
        # 更新GAE：Aₜ = δₜ + γ * λ * Aₜ₊₁ * mask
        gae = td_error + gamma * gae_lambda * masks[:, t] * gae
        
        # 存储优势
        advantages[:, t] = gae
        
        # 计算回报：Rₜ = Aₜ + V(sₜ)
        returns[:, t] = advantages[:, t] + values[:, t]
    
    return advantages, returns