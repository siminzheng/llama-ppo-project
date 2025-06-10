# llama-ppo-project
手动定义一个reward模型与ppo模型，来实现一个对llama3的简单近端策略优化

模型目录架构：
```text
gpt-ppo-project/
├── README.md
├── requirements.txt
├── reward/                       # Reward 模型相关
│   ├── data/
│   │   └── preference.json       # 偏好数据（JSONL）
│   ├── src/
│   │   ├── __init__.py
│   │   ├── dataset.py            # 读取与预处理 Dataset
│   │   ├── model.py              # 构建与加载 Reward 模型
│   │   └── train.py              # Reward 模型训练脚本
│   └── utils/
│       └── logging.py            # 日志/警告工具
└── ppo/                          # PPO 强化学习相关
    ├── data/
    │   └── queries.json          # 查询数据（JSONL）
    ├── src/
    │   ├── __init__.py
    │   ├── collator.py           # 定义批处理 Collator
    │   ├── model.py              # 构建带价值头的 LM
    │   └── train.py              # PPO 训练脚本
    └── utils/
        └── logging.py            # 日志/警告工具

```

# Llama PPO Project

本仓库包含两个子项目：

1. **reward/**: 用于训练奖励模型（Reward Model），基于序列分类。
2. **ppo/**: 基于训练好的奖励模型对因果语言模型进行 PPO 强化学习微调。

## 快速开始

```bash
pip install -r requirements.txt

reward模型训练：
python reward/src/train.py

ppo模型训练：
python ppo/src/train.py

```
