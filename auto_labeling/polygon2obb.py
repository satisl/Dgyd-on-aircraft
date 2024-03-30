import json
import os
import cv2
import numpy as np

json_file1 = r'E:\desktop\test\polygon_json'
json_file2 = r'E:\desktop\test\obb_txt'

classes = ['small colored digit']
os.makedirs(json_file2, exist_ok=True)
for file in os.listdir(json_file1):
    with open(f'{json_file1}/{file}', 'r') as f:
        data = json.load(f)
    width = data['imageWidth']
    height = data['imageHeight']
    shapes = data['shapes']
    with open(f'{json_file2}/{file.rsplit(".", 1)[0]}.txt', 'w') as f:
        for i in range(len(shapes)):
            points = np.array(data['shapes'][i]['points']).astype(int)
            label = data['shapes'][i]['label']
            if label in classes:
                rect = cv2.minAreaRect(points)
                bbox = cv2.boxPoints(rect).astype(int)
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = [(x / width, y / height) for x, y in bbox]
                f.write(f'{classes.index(label)} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n')
            else:
                print(label)
D
with open(f'{json_file2}/classes.txt', mode='w') as f:
    for i in classes:
        f.write(f'{i}\n')
