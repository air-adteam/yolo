import cv2
import torch
import numpy as np
from PIL import Image

# 加载模型
model,_ = torch.hub.load('weights', 'yolov5s', pretrained=True)

# 读取图像
img = cv2.imread('runs/detect/exp6/val__20230518113558.bmp')

# 进行对象检测
results = model(Image.fromarray(img[..., ::-1]))

# 循环处理每个检测结果
for detection in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = detection.tolist()

    # 当置信度大于阈值时
    if conf > 0.5:
        # 计算边界框坐标
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 获取ROI并读取
        roi = img[y1:y2, x1:x2]
        cv2.imshow("ROI", roi)

        # 将ROI在原始图像上可视化
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)