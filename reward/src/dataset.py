

import json
from datasets import Dataset

def load_preference_dataset(path: str) -> Dataset:
    """
    从 JSONL 文件读取偏好样本，并返回 HuggingFace Dataset 对象。
    每行应包含 {"question":..., "chosen":..., "rejected":...}。
    """
    items = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            items.append(json.loads(line))
    return Dataset.from_list(items)
