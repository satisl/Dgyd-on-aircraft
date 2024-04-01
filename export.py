from ultralytics import YOLO

imgsz = 640
model = YOLO(fr'D:\Double-digit-yolo-detection-on-aircraft\yolov8\7_400dataset_imgsz640_v8n_SGD/weights/best.pt')

model.export(format='engine', imgsz=imgsz, half=True)
