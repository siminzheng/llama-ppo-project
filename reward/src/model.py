下面是对你提供的代码的逐行注释：

```python
import torch  # 导入 PyTorch，用于张量运算和模型部署

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
# 从 PEFT（Parameter-Efficient Fine-Tuning）库中导入：
# - LoraConfig：用于配置 LoRA（低秩适配器）参数
# - TaskType：定义任务类型的枚举（比如序列分类）
# - get_peft_model：将 LoRA 注入到基础模型中
# - prepare_model_for_kbit_training：将模型转换为 k-bit 训练模式（如 4-bit）

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
# 从 Hugging Face Transformers 库中导入：
# - AutoTokenizer：用于加载预训练 tokenizer
# - BitsAndBytesConfig：配置 4-bit / 8-bit 量化参数
# - AutoModelForSequenceClassification：用于加载用于分类的预训练模型

def build_tokenizer(model_path: str):
    """
    构建并返回 tokenizer：
      - 从指定路径加载
      - 将 padding 放在序列的右侧
      - 将 pad_token 设置为 eos_token（方便对齐）
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # 加载 tokenizer，use_fast=False 表示禁用 Rust 版本的 tokenizer（有时为兼容性需求）
    tokenizer.padding_side = "right"
    # 将填充方向设置为在序列右侧填充
    tokenizer.pad_token = tokenizer.eos_token
    # 将 pad_token id 设置为 eos_token id，保证模型在填充时不会产生新的 token
    return tokenizer

def build_bnb_config():
    """
    构建并返回 BitsAndBytesConfig，用于 4-bit 量化设置：
      - load_in_4bit=True：启用 4-bit 加载
      - bnb_4bit_use_double_quant=True：使用双重量化以提高精度
      - bnb_4bit_quant_type="nf4"：选择 NF4 量化类型
      - bnb_4bit_compute_dtype=torch.float16：指定计算时的数据类型为半精度
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

def load_reward_model(model_path: str, bnb_config: BitsAndBytesConfig, peft_r: int = 8):
    """
    加载并返回可训练的 Reward 模型，包含：
      1. Tokenizer 构建
      2. 量化模型加载（4-bit）
      3. k-bit 训练准备（冻结部分原始权重）
      4. LoRA 注入
      5. 模型迁移到 GPU
    参数：
      - model_path: 预训练模型或本地 checkpoint 路径
      - bnb_config: BitsAndBytesConfig 实例，用于量化设置
      - peft_r: LoRA 的低秩 r 值，默认 8
    返回：
      - tokenizer, model
    """
    # 1. 构建 tokenizer
    tokenizer = build_tokenizer(model_path)

    # 2. 从预训练模型加载一个用于序列分类的模型（num_labels=1 表示回归/单值打分）
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        quantization_config=bnb_config  # 传入 4-bit 量化配置
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # 确保模型的 pad_token_id 与 tokenizer 对齐

    # 3. 准备 k-bit 训练（冻结大部分原始参数，只保留少数用于微调）
    model = prepare_model_for_kbit_training(model)

    # 4. 配置并应用 LoRA
    peft_cfg = LoraConfig(
        r=peft_r,  # LoRA 的秩
        target_modules=[
            "q_proj","v_proj","k_proj","o_proj",  # 自注意力层的投影矩阵
            "gate_proj","down_proj","up_proj"      # MLP 层的投影矩阵
        ],
        task_type=TaskType.SEQ_CLS,  # 序列分类任务
        lora_alpha=16,               # LoRA α 参数，控制缩放
        lora_dropout=0.05            # LoRA Dropout，防止过拟合
    )
    model = get_peft_model(model, peft_cfg)
    # 将 LoRA 模块注入到模型中

    model.print_trainable_parameters()
    # 打印并确认哪些参数是可训练的（应该只有 LoRA 参数）

    model.to("cuda")
    # 将模型移动到 GPU

    return tokenizer, model
```
