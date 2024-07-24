# 导入ultralytics库中的YOLO类
from ultralytics import YOLO

imgsz = 160
model = YOLO(
    r"./yolov8/digits_1665dataset_imgsz160_v8n_Adam/weights/best.pt"
)

model.export(format="rknn", imgsz=imgsz, opset=19, simplify=True)
