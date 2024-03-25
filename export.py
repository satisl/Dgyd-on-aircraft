from ultralytics import YOLO

model = YOLO(r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\2_400dataset_imgsz640_v8n_SGD/weights/best.pt')

model.export(format='engine', imgsz=640, half=True)
