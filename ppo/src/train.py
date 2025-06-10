# 导入标准库 json，用于读取 json 格式的数据
import json

# 导入 Hugging Face 的 datasets 工具，用于构建 Dataset 对象
from datasets import Dataset

# 从 TRL（Transformer Reinforcement Learning）库导入：
# - PPOTrainer：用于 Proximal Policy Optimization（PPO）训练的 Trainer
# - PPOConfig：PPO 的训练参数配置
from trl import PPOTrainer, PPOConfig

# 导入 PyTorch，用于张量操作
import torch

# 导入自定义数据整理函数（collator），用于将数据整理成 batch
from ppo.src.collator import collate_queries

# 导入 PPO 阶段使用的模型加载函数
from ppo.src.model import load_ppo_model

def load_queries(path: str) -> Dataset:
    """
    读取 queries.json 文件，将每一行 JSON 对象解析成 Dataset 格式。
    
    参数：
    - path: JSON 文件路径。每行应为 {"query": "..."}

    返回：
    - Hugging Face Dataset 对象
    """
    items = []
    # 打开查询文件（每行一个 json）
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            items.append(json.loads(line))  # 解析每一行的 json 数据
    return Dataset.from_list(items)  # 转换为 Dataset 格式，方便训练

def main():
    # 加载 PPO 模型和 tokenizer
    model_path = r'D:\work\models\Meta-Llama-3.1-8B-Instruct'
    # 这里 reward_adapter 是之前训练好的奖励模型 LoRA 权重
    tokenizer, model = load_ppo_model(model_path, reward_adapter="./reward_model")

    # 加载查询数据（人类输入的 prompts）
    dataset = load_queries("./data/queries.json")

    # PPO 配置参数
    ppo_conf = PPOConfig(
        kl_penalty="full",  # KL 整体惩罚模式（用于限制生成结果偏离初始模型）
        ppo_epochs=3,       # 每个 batch 执行 3 次 PPO 更新
        batch_size=2,       # 每次训练的大 batch size
        mini_batch_size=1   # 每个 mini-batch 的大小（更小有助于显存优化）
    )

    # 初始化 PPO Trainer
    trainer = PPOTrainer(
        config=ppo_conf,       # PPO 配置参数
        model=model,           # 当前训练的策略模型
        ref_model=None,        # 参考模型，设置为 None 表示不使用 KL 比较模型
        tokenizer=tokenizer,   # 分词器
        dataset=dataset,       # 训练数据
        data_collator=collate_queries  # 自定义 collator，用于组织 batch 数据
    )

    # 采样与训练主循环
    for batch in trainer.dataloader:
        # 生成模型回复
        responses = trainer.generate(
            batch,               # 输入 batch
            return_prompt=False, # 不返回 prompt，只返回生成的回复
            min_length=-1,       # 不限制最小长度
            top_k=0,             # 关闭 top-k 采样
            top_p=1,             # 允许所有概率（即不限制 nucleus 采样）
            do_sample=True,      # 启用随机采样
            pad_token_id=tokenizer.pad_token_id,  # 指定 padding token
            max_new_tokens=32    # 每次生成最多 32 个新 token
        )

        # 计算每个生成回复的奖励分数
        scores = []
        for q, r in zip(batch, responses):
            # 拼接 query 和 response，构造完整输入
            inp = torch.cat([q, r], dim=0).unsqueeze(0)  # 添加 batch 维度
            # 通过模型计算奖励分数（取最后一个 token 的奖励）
            score = trainer.model.compute_reward_score(input_ids=inp)[0, -1, 0]
            scores.append(score)

        # 执行一次 PPO 更新
        trainer.step(batch, responses, scores)

    # 保存训练好的模型
    trainer.save_pretrained("./ppo_model")

if __name__ == "__main__":
    main()
