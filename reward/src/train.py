# 从 PEFT（参数高效微调）库中导入：
from peft import LoraConfig, TaskType

# 从 TRL（Transformer Reinforcement Learning）库中导入：
from trl import RewardTrainer, RewardConfig

# 从 reward 模块的 dataset 文件中导入加载偏好数据集的方法
from reward.src.dataset import load_preference_dataset

# 从 reward 模块的 model 文件中导入：
# - build_bnb_config：构建 4-bit 量化配置
# - load_reward_model：加载带 LoRA 的 Reward 模型
from reward.src.model import build_bnb_config, load_reward_model

# 从 reward 工具模块中导入关闭警告的函数
from reward.utils.logging import disable_warnings

def process_pref(example):
    """
    数据预处理函数，将输入的单条数据（包含 question、chosen、rejected 三个字段）
    拼接成两个完整的输入，并分别进行分词处理，格式化为训练所需的字典格式。

    输入示例：
    {
        "question": "请帮我写一首诗。",
        "chosen": "好的，下面是我写的诗...",
        "rejected": "我不知道。"
    }

    输出：
    {
        "input_ids_chosen": [...],
        "attention_mask_chosen": [...],
        "input_ids_rejected": [...],
        "attention_mask_rejected": [...]
    }
    """
    tokenizer = process_pref.tokenizer  # 从函数属性中读取提前绑定的 tokenizer
    chosen = example["question"] + example["chosen"]  # 拼接用户问题和模型优选回答
    rejected = example["question"] + example["rejected"]  # 拼接用户问题和模型被拒绝的回答
    tc = tokenizer(chosen)  # 分词优选回答
    tr = tokenizer(rejected)  # 分词被拒绝回答
    return {
        "input_ids_chosen":      tc["input_ids"],       # 优选回答的输入 ID
        "attention_mask_chosen": tc["attention_mask"],  # 优选回答的 Attention Mask
        "input_ids_rejected":    tr["input_ids"],       # 被拒绝回答的输入 ID
        "attention_mask_rejected":tr["attention_mask"]  # 被拒绝回答的 Attention Mask
    }

def main():
    disable_warnings()  # 关闭无关警告（美化训练日志）

    # ===================== 配置与模型加载 =====================
    model_path = r'D:\work\models\Meta-Llama-3.1-8B-Instruct'  # 指定本地模型路径

    bnb_config = build_bnb_config()  # 构建 4-bit 量化配置
    tokenizer, model = load_reward_model(model_path, bnb_config)  # 加载 tokenizer 和 reward 模型

    # ===================== 数据加载与预处理 =====================
    dataset = load_preference_dataset("./data/preference.json")  # 加载偏好训练数据集

    process_pref.tokenizer = tokenizer  # 将 tokenizer 挂载到 process_pref 函数，便于后续调用
    dataset = dataset.map(process_pref, remove_columns=["question", "chosen", "rejected"])
    # 批量应用数据预处理，将 question/chosen/rejected 转换为训练输入，删除原始字段

    # ===================== 配置训练器 =====================
    config = RewardConfig(output_dir="./reward_model")  # RewardTrainer 配置，输出路径指定为 ./reward_model
    config.num_train_epochs = 1  # 设置训练 epoch 数为 1
    config.per_device_train_batch_size = 1  # 每个设备（显卡）训练的 batch size 设置为 1

    # 初始化 RewardTrainer（用于奖励模型训练）
    trainer = RewardTrainer(
        model=model,             # 训练的 reward 模型
        tokenizer=tokenizer,     # tokenizer
        args=config,             # RewardTrainer 配置参数
        train_dataset=dataset    # 训练数据集
    )

    trainer.train()  # 启动训练
    trainer.save_model("./reward_model")  # 保存训练好的 reward 模型到指定路径

if __name__ == "__main__":
    main()  # 程序入口，执行主函数
