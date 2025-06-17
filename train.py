import time
import numpy as np
import torch
from tqdm import tqdm
from config import Config
from models import PolicyModel, ReferenceModel, ValueModel, RewardModel
from data_processor import *
from rollout_generator import generate_rollout
from reward_calculator import compute_rewards
from ppo_trainer import PPOTrainer
from value_estimator import compute_advatages
from utils import *
import matplotlib.pyplot as plt 

def main():
    #初始化配置
    config = Config()
    
    #创建损失记录结构
    policy_losses = []
    value_losses = []
    mean_rewards = []
    steps = 0
    
    #加载模型
    print("Loading models...")
    policy_model = PolicyModel().to(config.DEVICE)
    ref_model = ReferenceModel().to(config.DEVICE)
    value_model = ValueModel().to(config.DEVICE)
    reward_model = RewardModel().to(config.DEVICE)

    #获取tokenizer
    policy_tokenizer = policy_model.get_tokenizer()
    reward_tokenizer = reward_model.get_tokenizer()

    #创建优化器 (只优化策略模型和值模型)
    optimizer = torch.optim.AdamW(
        list(policy_model.parameters()) + list(value_model.parameters()),
        lr=config.LEARNING_RATE
    )

    #初始化PPO训练器
    trainer = PPOTrainer(policy_model, ref_model, value_model, optimizer)

    #加载提示数据
    print("Loading prompts...")
    prompts = load_prompts(config.PROMPT_FILE)
    dataloader = create_dataloader(prompts, config.ROLLOUT_BATCH_SIZE)

    #训练循环
    print("Starting training...")
    start_step = 0  # 如果需要恢复训练，可以加载检查点
    step = start_step

    for epoch in range(config.NUM_UPDATES):
        epoch_losses = []
        epoch_rewards = []
        
        for batch_prompts in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_UPDATES}"):
            # 跳过最后的不完整批次
            if len(batch_prompts) < config.ROLLOUT_BATCH_SIZE:
                continue
                
            # 1. Tokenize prompts
            input_ids, attention_mask = tokenize_prompts(policy_tokenizer, batch_prompts)
            
            # 2. 生成响应和轨迹
            response_ids, log_probs, _ = generate_rollout(
                policy_model, input_ids, attention_mask
            )
            
            # 3. 计算响应序列的实际长度
            resp_lengths = compute_response_lengths(response_ids)
            resp_masks = create_mask_from_lengths(resp_lengths, config.NUM_RESPONSE_TOKENS)
            
            # 4. 计算每个完整响应的奖励
            rewards = compute_rewards(
                reward_model, 
                reward_tokenizer,
                batch_prompts,
                response_ids
            )
            
            # 5. 扩展奖励到每个时间步 (稀疏奖励)
            step_rewards = torch.zeros_like(log_probs)
            for i in range(step_rewards.size(0)):
                # 只在序列最后位置分配奖励
                step_rewards[i, resp_lengths[i]-1] = rewards[i]
            
            # 6. 估计值函数
            with torch.no_grad():
                # 估计序列最后位置的值
                last_values = value_model(
                    input_ids[:, -1].unsqueeze(1),  # 使用最后一个输入token
                    attention_mask[:, -1].unsqueeze(1)
                )
            
            # 7. 计算优势函数和回报
            advantages, returns = compute_advatages(
                step_rewards, 
                log_probs,  # 临时作为占位符
                resp_masks, 
                last_values
            )
            
            # 8. 标准化优势函数
            advantages = normalize_advantages(advantages)
            
            # 9. 准备训练批数据
            full_ids = torch.cat([input_ids, response_ids], dim=1)
            full_attention_mask = torch.cat([attention_mask, torch.ones_like(response_ids)], dim=1)
            prompt_lens = input_ids.size(1)  # 获取提示部分长度
            batch_data = {
                'input_ids': full_ids,
                'attention_mask': full_attention_mask,
                'response_ids': response_ids,
                'old_log_probs': log_probs,
                'rewards': rewards,
                'advantages': advantages,
                'returns': returns,
                'prompt_lens': prompt_lens
            }
            
            # 10. 执行PPO更新 (多个小epoch)
            for _ in range(config.PPO_EPOCHS):
                loss_dict = trainer.train_step(batch_data)
                epoch_losses.append(loss_dict['total_loss'])
                epoch_rewards.append(loss_dict['mean_reward'])
                
                #记录每次更新的损失
                policy_losses.append(loss_dict['policy_loss'])
                value_losses.append(loss_dict['value_loss'])
                mean_rewards.append(loss_dict['mean_reward'])
                steps += 1
                
                #每50步打印一次损失
                if steps % 50 == 0:
                    print(f"Step {steps}: "
                          f"Policy Loss={loss_dict['policy_loss']:.4f}, "
                          f"Value Loss={loss_dict['value_loss']:.4f}, "
                          f"Reward={loss_dict['mean_reward']:.4f}")
            
            #更新步数计数器
            step += 1
            
            # #定期保存模型
            # if step % config.SAVE_INTERVAL == 0:
            #     save_path = f"checkpoints/ppo_model_step_{step}.pt"
            #     save_checkpoint(policy_model, optimizer, step, save_path)
                
        
        # 打印epoch统计信息
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")
    
    # 训练结束，保存最终模型
    #save_checkpoint(policy_model, optimizer, step, "./results/final_ppo_model.pt")
    print("Training completed!")


    # 绘制最终的策略损失和价值损失曲线
    plt.figure(figsize=(12, 6))
    
    # 策略损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(policy_losses, 'b-', label='Policy Loss')
    plt.title('Final Policy Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 价值损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(value_losses, 'r-', label='Value Loss')
    plt.title('Final Value Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/Loss.png')
    plt.show()
if __name__ == "__main__":
    main()