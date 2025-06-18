import torch
import torch.nn.functional as F
from config import Config

config = Config()

def generate_rollout(policy_model, input_ids, attention_mask):
    """
    使用策略模型生成响应轨迹

    参数：
        policy_model: 策略模型
        input_ids:输入token IDs [batch_size, seq]
        attention_mask:注意力掩码 [batch_size, seq]

    返回:
        response_tokens: 响应token IDs [batch, resp_len]
        log_probs: 每个token的对数概率 [batch, resp_len]
        values: 每个token的值函数预测 [batch, resp_len+1]
    """

    batch_size = input_ids.size(0)
    response_length = config.NUM_RESPONSE_TOKENS


    response_tokens = torch.zeros(
        (batch_size, response_length),
        dtype=torch.long,
        device=config.DEVICE
    )
    log_probs = torch.zeros(
        (batch_size, response_length),
        device=config.DEVICE
    )

    current_input = input_ids


    for step in range(response_length):
        #获取下一个token的概率
        with torch.no_grad():
            outputs = policy_model(current_input, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :] #获取最后一个token的logits[batch_size, vocab_size],相当于获取策略的分布

        #应用采样参数
        logits = logits / config.TEMPERATURE
        if config.TOP_K > 0:
            #应用top-k过滤,找到前k个token，将第k个token的logits作为阈值，小于阈值设为负无穷
            indices_to_remove = logits < torch.topk(logits, config.TOP_K)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        
        if config.TOP_P < 1.0:
            #应用top-p采样
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)


            #移除累积概率高于p的token
            sorted_indices_to_remove = cumulative_probs > config.TOP_P
            #确保至少保留一个token
            sorted_indices_to_remove[..., 0] = 0

            #创建用于删除的掩码
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        #概率归一化，将logits转为概率分布
        probs = F.softmax(logits, dim=-1)               #probs[batch_size, vocab_size]
        tokens = torch.multinomial(probs, 1).squeeze(1) #multinomial根据probs随机采样一个token，并让形状从 [batch_size, 1] 压缩为 [batch_size]，即每个样本一个token ID

        #结果
        response_tokens[:, step] = tokens
        log_probs[:, step] = torch.log(probs.gather(1, tokens.unsqueeze(1))[..., 0])


        #更新用于下一步
        current_input = torch.cat([current_input, tokens.unsqueeze(1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size,1), device=config.DEVICE)], dim=1)

    return response_tokens, log_probs, attention_mask