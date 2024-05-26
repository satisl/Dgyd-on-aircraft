from ultralytics import YOLO

imgsz = 256
model = YOLO(
    r"D:\Double-digit-yolo-detection-on-aircraft\yolov8\digits_1665dataset_imgsz160_v8n_Adam\weights\best.pt"
)

model.export(format="rknn", imgsz=imgsz, opset=19, simplify=True)
