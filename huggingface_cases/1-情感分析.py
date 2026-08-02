import os
from transformers import pipeline

# --- 修改点 1: 注释掉或删除网络相关的环境变量设置 ---
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "./autodl-tmp/models"

# --- 修改点 2: 定义本地模型的路径 ---
# 请务必将下面的路径替换成你电脑上模型文件夹的真实路径
# 例如，如果你把模型放在了 E:\Demo\agentDemo\local_models\roberta-base-finetuned-dianping-chinese
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