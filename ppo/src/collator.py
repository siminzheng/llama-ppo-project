import torch

def collate_queries(batch, tokenizer):
    """
    将每条 query 文本分词为 tensor 并移到 GPU，
    返回一个 list of 1-D LongTensor。
    """
    queries = []
    for item in batch:
        q_ids = tokenizer(item["query"], return_tensors="pt")["input_ids"].squeeze(0).to("cuda")
        queries.append(q_ids)
    return queries
