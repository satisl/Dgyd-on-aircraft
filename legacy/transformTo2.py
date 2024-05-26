import os
import xml.etree.ElementTree as ET
import math
from lxml import objectify, etree
import shutil

img_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\2\1/images'
xml_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\2\1/annotations'
save_img_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\2\origin/images'
save_xml_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\2\origin/annotations'

os.makedirs(save_img_path, exist_ok=False)
os.makedirs(save_xml_path, exist_ok=False)
imgs = os.listdir(img_path)

for img in imgs:
    file_name, suffix = img.rsplit('.')

    tree = ET.parse(f'{xml_path}/{file_name}.xml')
    root = tree.getroot()
    # 获取图片长宽
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = size.find('depth').text

    coords = []
    digits = []
    objs = root.findall('object')
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        # 获取[label,x,y,w,h]
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        w = x_max - x_min
        h = y_max - y_min

        if name != 'locater':
            digits.append([name, center_x / width, center_y / height, x_max, x_min, y_max, y_min])
        else:
            coords.append([name, x_max, x_min, y_max, y_min])

    print(digits)
    # 根据各个数字坐标距离，找出最近的两个单位数组成双位数对
    conbined_digits = []
    for i in range(len(digits)):
        min_distance = ['', 1]
        for j in range(len(digits)):
            if i == j:
                continue

            digit1 = digits[i]
            digit2 = digits[j]

            distance = math.dist((digit1[1], digit1[2]), (digit2[1], digit2[2]))
            if distance < min_distance[1]:
                min_distance = [{i, j}, distance]

        if min_distance[0] in conbined_digits:
            continue
        conbined_digits.append(min_distance[0])

    print(conbined_digits)

    # 合并组成双位数对的两个单位数的框为一个双位数对框

    for i in conbined_digits:
        idx = list(i)
        x1max, x1min, y1max, y1min = digits[idx[0]][3:]
        x2max, x2min, y2max, y2min = digits[idx[1]][3:]

        x_max = max(x1max, x2max)
        y_max = max(y1max, y2max)
        x_min = min(x1min, x2min)
        y_min = min(y1min, y2min)

        coords.append(['digits', x_max, x_min, y_max, y_min])

    print(coords)

    # 根据已有信息写入xml文件
    E = objectify.ElementMaker(annotate=False)

    anno_tree = E.annotation(
        E.folder(img_path),
        E.filename(img),
        E.path(os.path.join(img_path, img)),
        E.source(
            E.database('Unknown'),
        ),
        E.size(
            E.width(width),
            E.height(height),
            E.depth(depth)
        ),
        E.segmented(0),
    )

    for i in coords:
        anno_tree.append(
            E.object(
                E.name(i[0]),
                E.pose('Unspecified'),
                E.truncated('0'),
                E.difficult('0'),
                E.bndbox(
                    E.xmin(i[2]),
                    E.ymin(i[4]),
                    E.xmax(i[1]),
                    E.ymax(i[3])
                )
            ))

    etree.ElementTree(anno_tree).write(os.path.join(save_xml_path, f'{file_name}.xml'), pretty_print=True)
    shutil.copyfile(f'{img_path}/{file_name}.{suffix}', f'{save_img_path}/{file_name}.{suffix}')
    # break
