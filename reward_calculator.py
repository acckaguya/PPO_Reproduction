import torch
from config import Config

config = Config()


def compute_rewards(reward_model, reward_tokenizer, policy_tokenizer, prompts, responses):
    """
    使用奖励模型计算完整响应的奖励
    
    参数:
        reward_model: 奖励模型
        reward_tokenizer: 奖励模型的tokenizer
        policy_tokenizer: 策略模型的tokenizer
        prompts: 原始提示列表
        responses: 生成的响应token列表
    
    返回:
        rewards: 每个响应的奖励值 [batch]
    """

    # 将响应token解码为文本
    decoded_responses = []
    for resp in responses:
        valid_tokens = resp[resp != 0]  # 跳过填充token
        text = policy_tokenizer.decode(valid_tokens, skip_special_tokens=True)
        decoded_responses.append(text)
    
    # 拼接提示和响应
    full_texts = [f"{p} {r}" for p, r in zip(prompts, decoded_responses)]
    
    # 使用奖励模型的tokenizer编码
    inputs = reward_tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LEN,
        return_tensors="pt",
        return_token_type_ids=False
    ).to(config.DEVICE)
    
    # 计算奖励
    with torch.no_grad():
        rewards = reward_model(**inputs)
    
    return rewards

def add_kl_penalty(log_probs_policy, log_probs_ref, kl_coef=config.KL_COEF):
    """
    计算KL散度惩罚项
    
    参数:
        log_probs_policy: 策略模型的对数概率 [batch, seq]
        log_probs_ref: 参考模型的对数概率 [batch, seq]
        kl_coef: KL惩罚系数
    
    返回:
        kl_penalty: KL散度惩罚值 [batch, seq]
    """
    kl_div = log_probs_policy - log_probs_ref
    kl_penalty = kl_coef * kl_div

    return kl_penalty


def add_entropy_bonus(probs, entropy_coef=config.ENTROPY_COEF):
    """
    计算熵奖励以鼓励探索
    
    参数:
        probs: token 概率 [batch, vocab]
        entropy_coef: 熵奖励系数
    
    返回:
        entropy: 熵奖励值 [batch, 1]
    """
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy_coef * entropy.unsqueeze(-1)