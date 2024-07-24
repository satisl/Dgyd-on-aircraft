import json
import os
import cv2

img_path = r''
json_path = r''
txt_path = r''

for file in os.listdir(img_path):
    print(file)
    height, width, _ = cv2.imread(f'{img_path}/{file}').shape
    # 从json获取数据
    bboxs = []
    points = []
    with open(f'{json_path}/{file.rsplit(".", 1)[0]}.json', 'r') as f:
        data = json.load(f)
    for shape in data['shapes']:
        if shape['shape_type'] == 'point':
            points.append(shape['points'][0])
        elif shape['shape_type'] == 'rectangle':
            bboxs.append(shape['points'])
        else:
            print('some errors happen')
            raise Exception
    # 将json转换为yolo格式
    lines = []
    for (x1, y1), (x2, y2), (x3, y3), (x4, y4) in bboxs:
        left = min(x1, x2, x3, x4)
        right = max(x1, x2, x3, x4)
        top = min(y1, y2, y3, y4)
        bottom = max(y1, y2, y3, y4)
        for x, y in points:
            if x > left and x < right and y > top and y < bottom:
                c_x = (left + right) / 2
                c_y = (top + bottom) / 2
                w = right - left
                h = bottom - top
                lines.append([0, c_x / width, c_y / height, w / width, h / height, x / width, y / height])
                break

    # 写入label文件
    with open(f'{txt_path}/{file.rsplit(".", 1)[0]}.txt', 'w') as f:
        for line in lines:
            f.write(' '.join([str(_) for _ in line]) + '\n')
