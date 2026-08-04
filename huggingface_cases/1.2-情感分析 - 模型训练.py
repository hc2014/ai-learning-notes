#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# =============================================
# Step 1: 准备数据
# =============================================

# 把你自己的训练数据填在这里
# label: 0 = 差评/负面, 1 = 好评/正面  (根据你的实际需求调整)
data = [
    {"text": "这家店味道太差了，再也不来了", "label": 0},
     {"text": "老板人怪好的,青菜里面还有肉，我吃出来一条肉虫", "label": 0},
    {"text": "环境很好，服务也很周到，推荐！", "label": 1},
    {"text": "等了半个小时才上菜，太慢了", "label": 0},
    {"text": "菜品新鲜，分量足，性价比很高", "label": 1},
    {"text": "一般般吧，没有想象中好吃", "label": 0},
    {"text": "你看看你一天天的就知道吃", "label": 0},
    {"text": "服务员态度很差，体验极差", "label": 0},
    {"text": "朋友推荐的，果然名不虚传", "label": 1},
    {"text": "价格偏贵，味道平平无奇", "label": 0},
    {"text": "每次来都不会失望，五星好评", "label": 1},
    {"text": "你看看你一天天的就知道吃", "label": 0},
{"text": "一天到晚就知道吃，啥也不干", "label": 0},
{"text": "除了吃你还会干什么？", "label": 0},
{"text": "就知道吃，胖成这样了", "label": 0},
    # ... 继续添加你的数据，数据越多效果越好
]

# 转为 HuggingFace Dataset 并划分训练集/测试集
dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"训练集: {len(dataset['train'])} 条")
print(f"测试集: {len(dataset['test'])} 条")


# =============================================
# Step 2: 加载本地模型和分词器
# =============================================

# ⚠️ 改成你本地模型的实际路径
checkpoint = "./local_models/roberta-base-finetuned-dianping-chinese"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2  # 二分类：正面/负面
)

# 自动检测 GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(f"使用设备: {device}")


# =============================================
# Step 3: 数据预处理
# =============================================

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

print("正在预处理数据...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 动态补齐
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# =============================================
# Step 4: 配置训练参数并开始训练
# =============================================

def compute_metrics(eval_pred):
    """用 numpy 计算准确率，无需联网"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": float(accuracy)}

training_args = TrainingArguments(
    output_dir="./results/roberta-dianping-finetuned",  # 训练结果保存路径
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,           # 微调学习率要小，避免破坏预训练权重
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,           # 数据少可以多跑几轮
    weight_decay=0.01,
    logging_steps=5,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("开始训练...")
trainer.train()


# =============================================
# Step 5: 保存微调后的模型
# =============================================

save_path = "./local_models/roberta-dianping-my-finetuned"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n模型已保存到: {save_path}")


# =============================================
# Step 6: 重新加载新模型并推理测试
# =============================================

import gc  # 引入垃圾回收模块，用于清理显存/内存

print("\n========== 开始验证保存后的新模型 ==========")

# 1. 清理内存：删除旧模型和分词器，释放显存/内存
del model
del tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 2. 重新从硬盘加载微调后保存的新模型
# 这才是真正验证我们保存下来的文件是否完整可用
print(f"正在从硬盘重新加载新模型: {save_path}")
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=2)

# 自动检测 GPU/CPU 并切换为评估模式
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()  # 关闭 Dropout 等训练机制，保证推理结果稳定

# 3. 封装推理函数
def predict(text):
    """用微调后的新模型预测"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_id = logits.argmax().item()
        confidence = probs[0][predicted_id].item()

    label = "好评" if predicted_id == 1 else "差评"
    return label, confidence

# 4. 测试几条新文本
test_texts = [
    "你看看你一天天的就知道吃",
    "一天到晚就知道吃，啥也不干",
    "这家店味道太差了，再也不来了",
    "环境很好，服务也很周到，推荐！",
    "菜品新鲜，分量足，性价比很高",
]

print("\n========== 新模型推理结果 ==========")
for t in test_texts:
    label, score = predict(t)
    print(f"文本: {t}")
    print(f"  -> 预测: {label} (置信度: {score:.4f})\n")