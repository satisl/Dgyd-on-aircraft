from ultralytics import YOLO

imgsz = 640
model = YOLO(r'D:\Double-digit-yolo-detection-on-aircraft\yolov8n.pt')

model.export(format='engine', imgsz=imgsz, half=True)
