from ultralytics import YOLO

imgsz = 160
model = YOLO(r'./yolov8/5_1100dataset_imgsz160_v8n_Adam/weights/best.pt')

model.export(format='rknn', imgsz=imgsz, opset=19)
