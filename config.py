import torch

class Config:
    """训练配置参数"""

    #模型设置
    POLICY_MODEL_PATH = r"/root/autodl-tmp/hugging-face/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    REF_MODEL_PATH = r"/root/autodl-tmp/hugging-face/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    VALUE_MODEL_PATH = r"/root/autodl-tmp/hugging-face/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    REWARD_MODEL_PATH = r"/root/autodl-tmp/hugging-face/hub/models--OpenAssistant--reward-model-deberta-v3-large-v2/snapshots/c355404efa9ad2ad069f3a197cae0523c14244fc"


    #训练超参数
    BATCH_SIZE = 8              #PPO每次更新的批次大小
    ROLLOUT_BATCH_SIZE = 8      #每次生成的轨迹的批量
    LEARNING_RATE = 1e-5        #学习率
    GAMMA = 0.99                #折扣因子
    GAE_LAMBDA = 0.85           #GAE参数
    CLIP_EPSILON = 0.1          #PPO裁剪范围
    KL_COEF = 0.1               #KL惩罚系数
    VALUE_COEF = 0.5            #值函数损失系数
    ENTROPY_COEF = 0.05         #熵奖励系数
    MAX_GRAD_NORM = 1.0         #梯度裁剪值
    PPO_EPOCHS = 4              #PPO更新轮次


    #文本生成参数
    MAX_SEQ_LEN = 128           #最大序列长度
    NUM_RESPONSE_TOKENS = 128    #生成响应token数
    TOP_K = 50                  #Top-k采样
    TOP_P = 0.9                 #Top-p采样
    TEMPERATURE = 1.0           #温度


    #训练设置
    NUM_UPDATES = 20          #总训练次数
    SAVE_INTERVAL = 100         #保存间隔
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    #输入数据
    PROMPT_FILE = "./prompts.txt"