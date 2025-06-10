# 从 PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）库导入：
# - LoraConfig：LoRA 参数配置类
# - TaskType：任务类型枚举
from peft import LoraConfig, TaskType

# 从 transformers 库导入：
# - BitsAndBytesConfig：用于 4-bit 量化配置
# - AutoTokenizer：自动加载 tokenizer 工具
from transformers import BitsAndBytesConfig, AutoTokenizer

# 从 TRL（Transformer Reinforcement Learning）库导入：
# - AutoModelForCausalLMWithValueHead：支持 Value Head 的因果语言模型（用于 PPO 微调）
from trl import AutoModelForCausalLMWithValueHead

def build_tokenizer(model_path: str):
    """
    构建分词器。

    参数：
    - model_path：预训练模型路径。

    返回：
    - tokenizer：加载好的 tokenizer 对象。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)  # 从指定路径加载分词器，禁用 fast tokenizer（更稳定）
    tokenizer.padding_side = "right"  # 指定 padding 填充方向为右侧
    tokenizer.pad_token = tokenizer.eos_token  # 指定 padding token 为 eos（防止 padding 报错）
    return tokenizer

def build_bnb_config():
    """
    构建 4-bit 量化配置，节省显存。
    
    与 Reward 模型使用的量化配置相同，保证训练流程兼容。
    
    返回：
    - BitsAndBytesConfig 对象。
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,                # 启用 4-bit 量化加载
        bnb_4bit_use_double_quant=True,   # 使用双层量化（进一步降低显存占用）
        bnb_4bit_quant_type="nf4",        # 采用 nf4 量化类型（噪声友好）
        bnb_4bit_compute_dtype=torch.bfloat16  # 训练时计算精度为 bfloat16
    )

def load_ppo_model(model_path: str, reward_adapter: str, peft_r: int = 8):
    """
    加载支持 PPO（Proximal Policy Optimization，近端策略优化）训练的模型：
    - 基于因果语言模型（Causal LM）
    - 带 Value Head（用于计算奖励）
    - 注入 LoRA 微调结构
    - 支持 4-bit 量化低显存训练

    参数：
    - model_path：预训练语言模型路径
    - reward_adapter：已经训练好的 Reward Model 的 LoRA adapter 路径（用于奖励评估）
    - peft_r：LoRA 的秩（秩越高，训练参数越多）

    返回：
    - tokenizer：用于文本处理的分词器
    - model：带 Value Head 的语言模型
    """
    tokenizer = build_tokenizer(model_path)  # 构建 tokenizer
    bnb_cfg = build_bnb_config()  # 构建 4-bit 量化配置

    # 配置 LoRA 参数
    peft_cfg = LoraConfig(
        r=peft_r,  # LoRA 的秩，控制注入参数规模
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ],  # 选择注入 LoRA 的模块（通常是投影层）
        task_type=TaskType.CAUSAL_LM,  # 任务类型为因果语言模型
        lora_alpha=16,  # LoRA 缩放因子
        lora_dropout=0.05  # LoRA dropout 比例，增加训练鲁棒性
    )

    # 加载支持 PPO 的语言模型，包含：
    # - LoRA 配置（peft_config）
    # - 已训练的 reward adapter
    # - 4-bit 量化配置（quantization_config）
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        reward_adapter=reward_adapter,
        peft_config=peft_cfg,
        quantization_config=bnb_cfg
    )

    model.to("cuda")  # 将模型加载到 GPU 上
    return tokenizer, model  # 返回 tokenizer 和模型
