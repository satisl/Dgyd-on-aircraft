from ultralytics import YOLO



imgsz = 96
model = YOLO(f'yolov8/789_800dataset_imgsz{imgsz}_v8n_SGD/weights/best.pt')

model.export(format='engine', imgsz=imgsz, half=True)
