from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from config import Config


config = Config()

class PolicyModel(nn.Module):
    """策略模型封装，用于生成文本响应"""

    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(config.POLICY_MODEL_PATH).to(config.DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(config.POLICY_MODEL_PATH)

        #添加特殊标记eos_token（结束标记）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask):
        """
        参数：
            input_ids: token ID 张量 形状[batch_size, sequence_length]
            attention_mask: 注意力掩码张量(与 input_ids 形状相同)，用于区分真实 token 和填充 token
        """
        return self.model(input_ids, attention_mask=attention_mask)

    def get_tokenizer(self):
        return self.tokenizer    
        

class ReferenceModel(nn.Module):
    """参考模型封装，用于计算KL散度"""

    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            config.REF_MODEL_PATH
        ).to(config.DEVICE).eval() 

        #不更新参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        #禁用梯度计算
        with torch.no_grad():
            return self.model(input_ids, attention_mask=attention_mask)
        

class ValueModel(nn.Module):
    """价值函数封装，用于估计状态价值"""

    def __init__(self):
        super().__init__()
        #价值模型与策略模型共享参数
        self.transformer = AutoModelForCausalLM.from_pretrained(
            config.VALUE_MODEL_PATH
        ).to(config.DEVICE)


        #添加一个头用于预测值
        self.value_head = nn.Linear(
            self.transformer.config.hidden_size, 1
        ).to(config.DEVICE)


    def forward(self, input_ids, attention_mask=None):
        #返回所有位置的价值
        outputs = self.transformer(
            input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        #价值头应用到每个位置 [batch_size, seq_len]
        values = self.value_head(hidden_states).squeeze(-1)
        return values  #形状:[batch_size, seq_len]
    

class RewardModel(nn.Module):
    """奖励模型封装，用于计算文本奖励"""

    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.REWARD_MODEL_PATH
        ).to(config.DEVICE).eval()


        #冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.REWARD_MODEL_PATH
        )

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            return outputs.logits.squeeze(-1)
        
    def get_tokenizer(self):
        return self.tokenizer