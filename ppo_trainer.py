import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import Config

config = Config()

class PPOTrainer:
    """PPO训练器"""
    def __init__(self, policy_model, ref_model, value_model, optimizer):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.value_model = value_model
        self.optimizer = optimizer
        self.device = config.DEVICE
        self.clip_eps = config.CLIP_EPSILON
        self.value_coef = config.VALUE_COEF
        self.kl_coef = config.KL_COEF
        self.entropy_coef = config.ENTROPY_COEF

    def train_step(self, batch):
        """
        执行一个PPO训练步骤
        
        参数:
            batch: 包含输入、响应、概率和值的字典
        
        返回:
            loss_dict: 包含各项损失的字典
        """
        # 解包数据集
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        response_ids = batch['response_ids'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        returns = batch['returns'].to(self.device)
        prompt_lens = batch['prompt_lens']

        # 计算当前策略的概率
        outputs = self.policy_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        #正确提取响应部分的logits
        resp_start_idx = prompt_lens - 1
        resp_end_idx = resp_start_idx + response_ids.size(1)
        resp_logits = logits[:, resp_start_idx:resp_end_idx, :]

        # 维度检查确保兼容性
        assert resp_logits.size(1) == response_ids.size(1), (
            f"Logits序列长度({resp_logits.size(1)})与响应ID长度({response_ids.size(1)})不匹配！"
            f"提示长度:{prompt_lens}, 总序列长度:{logits.size(1)}"
        )

        # 计算新的对数概率（添加epsilon防止log(0)）
        probs = F.softmax(resp_logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        
        # 使用正确维度的gather操作
        new_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)

        # 计算重要性采样比
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 裁剪PPO目标函数
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 计算KL散度惩罚（参考模型不计算梯度）
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits[:, resp_start_idx:resp_end_idx, :]
            ref_probs = F.softmax(ref_logits, dim=-1)
        
        kl_div = F.kl_div(
            input=log_probs, 
            target=ref_probs,
            log_target=False,
            reduction='none'
        ).sum(dim=-1)
        kl_penalty = self.kl_coef * kl_div.mean()

        # 计算熵奖励（添加epsilon防止log(0)）
        entropy = -torch.sum(probs * log_probs, dim=-1)
        entropy_bonus = self.entropy_coef * entropy.mean()

        # 获取完整序列的值函数估计（确保输出是[batch_size, seq_len]形状）
        full_values = self.value_model(input_ids, attention_mask=attention_mask)
        resp_values = full_values[:, resp_start_idx:resp_end_idx]
        
        # 计算值函数损失
        value_loss = F.mse_loss(resp_values, returns) * self.value_coef

        # 总损失
        total_loss = policy_loss + kl_penalty - entropy_bonus + value_loss

        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), config.MAX_GRAD_NORM)
        self.optimizer.step()

        # 返回损失指标
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_div': kl_div.mean().item(),
            'entropy': entropy.mean().item(),
            'mean_reward': rewards.mean().item()
        }

        return loss_dict