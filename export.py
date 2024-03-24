from ultralytics import YOLO

model = YOLO(r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\5_700dataset_imgsz160_v8n_Adam/weights/best.pt')

model.export(format='engine', imgsz=160, half=True)
