import os
import numpy as np
import cv2
from ultralytics import YOLO
import math

model_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\7_400dataset_imgsz640_v8n_SGD\weights\best.engine'
model = YOLO(model_path, task='obb')
coefficient1 = 1.2

img_path = r'E:\desktop\456_test\郭嘉城'
out_img_path = rf'D:\Double-digit-yolo-detection-on-aircraft\output/{cv2.getTickCount()}'
os.makedirs(out_img_path, exist_ok=False)
imgs = os.listdir(img_path)
for img in imgs:
    image = cv2.imread(f'{img_path}/{img}')
    results = model.predict(source=image, imgsz=640, half=True, device='cuda:0',
                            save=False, conf=0.5, iou=0.5, verbose=True)
    r = results[0]
    xywhr = r.obb.xywhr.cpu().tolist()
    xyxy = r.obb.xyxy.cpu().tolist()
    height, width = image.shape[:2]
    cnt = 0
    for (x, y, _, _, r), (left, top, right, bottom) in zip(xywhr, xyxy):
        rad = r
        degree = math.degrees(rad) + 90

        w = right - left
        h = bottom - top

        left -= (w * (coefficient1 - 1) / 2)
        right += (w * (coefficient1 - 1) / 2)
        top -= (h * (coefficient1 - 1) / 2)
        bottom += (h * (coefficient1 - 1) / 2)

        left = left if left >= 0 else 0
        top = top if top >= 0 else 0
        right = right if right <= width else width
        bottom = bottom if bottom <= height else height

        cropped_img = image[int(top):int(bottom), int(left):int(right)]
        print(cropped_img)
        cv2.imshow('0', cv2.resize(cropped_img, (640, 480)))

        rot_mat = cv2.getRotationMatrix2D((x - left, y - top), degree, 0.9)
        rotated_img = cv2.warpAffine(cropped_img, rot_mat, (int(right - left), int(bottom - top)),
                                     flags=cv2.INTER_LANCZOS4)

        cv2.imshow('1', cv2.resize(rotated_img, (640, 480)))
        cv2.imwrite(f'{out_img_path}/{cnt}-{img}', rotated_img)
        cnt += 1
        cv2.waitKey(1000)
