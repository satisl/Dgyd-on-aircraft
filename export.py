from ultralytics import YOLO

imgsz = 160
model = YOLO(fr'D:\Double-digit-yolo-detection-on-aircraft\yolov8\5_1100dataset_imgsz160_v8n_Adam/weights/best.pt')

model.export(format='engine', imgsz=imgsz, half=True)
