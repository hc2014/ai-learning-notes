import os

from ultralytics import YOLO

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "000000000139.jpg")
weights_path = os.path.join(base_dir, "yolo12n.pt")

print(f"Base directory: {base_dir}")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"找不到图片文件: {image_path}")

# 优先使用本地权重；如果不存在则自动下载 YOLOv12 权重
if os.path.exists(weights_path):
    print(f"加载本地权重: {weights_path}")
    model = YOLO(weights_path)
else:
    print("未找到本地权重，正在下载 YOLOv12 权重...")
    model = YOLO("yolo12n.pt")

# 使用 YOLOv12 对图片进行目标检测
results = list(model(image_path, stream=False, conf=0.25, imgsz=640))

if not results:
    raise RuntimeError("未检测到任何目标。")

result = results[0]

# 保存检测结果图片
output_path = os.path.join(base_dir, "predicted_000000000139.jpg")
result.save(filename=output_path)
print(f"检测结果已保存到: {output_path}")

# 在当前环境下尝试显示结果
try:
    result.show()
except Exception as e:
    print(f"显示结果时跳过: {e}")

# 如果存在验证集配置文件，则进行评估
coco_config = os.path.join(base_dir, "coco.yaml")
if os.path.exists(coco_config):
    metrics = model.val(data=coco_config, save_json=True)
    print("mAP:", metrics.box.map)
else:
    print("未找到 coco.yaml，跳过评估步骤。")

