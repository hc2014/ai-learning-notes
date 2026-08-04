import os
from transformers import pipeline

local_model_path = "./local_models/roberta-base-finetuned-dianping-chinese"

# --- 修改点 3: 从本地路径加载模型 ---
classifier = pipeline(
    task="sentiment-analysis",
    model=local_model_path  # 这里传入本地路径，而不是模型的在线名称
)


# --- 预测部分保持不变 ---
result = classifier("这个手机屏幕太烂了，反应很慢！")
print(result)

result2 = classifier("物流很快，包装很精美，五星好评。")
print(result2)

#对中文支持不太友好
result2 = classifier("你看你一天天的，就知道吃。")
print(result2)