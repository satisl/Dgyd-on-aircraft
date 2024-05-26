import os
import shutil


def parse_polygon(file):
    clss = []
    polygons = []
    with open(file, mode='r') as f:
        for line in f.readlines():
            infos = line.strip('\n').split(' ')
            clss.append(infos[0])
            polygons.append([(float(infos[i]), float(infos[i + 1])) for i in range(1, len(infos), 2)])

    return clss, polygons


def polygon2xywh(polygon):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = polygon
    left = min(x1, x2, x3, x4)
    top = min(y1, y2, y3, y4)
    right = max(x1, x2, x3, x4)
    bottom = max(y1, y2, y3, y4)
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    w = right - left
    h = bottom - top
    return cx, cy, w, h


org_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\7\origin'
dst_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\4\origin'

if __name__ == '__main__':
    img_path1 = f'{org_path}/images'
    img_path2 = f'{dst_path}/images'
    txt_path1 = f'{org_path}/labels'
    txt_path2 = f'{dst_path}/labels'
    os.makedirs(img_path2, exist_ok=False)
    os.makedirs(txt_path2, exist_ok=False)
    shutil.copy(f'{txt_path1}/classes.txt', f'{txt_path2}/classes.txt')

    imgs = os.listdir(img_path1)
    for file in imgs:
        prefix, suffix = file.rsplit('.', 1)
        img_file = f'{prefix}.{suffix}'
        txt_file = f'{prefix}.txt'
        # 复制图片
        shutil.copy(f'{img_path1}/{img_file}', f'{img_path2}/{img_file}')

        # 获取clss,polygons
        clss, polygons = parse_polygon(f'{txt_path1}/{txt_file}')
        with open(f'{txt_path2}/{txt_file}', mode='a') as f:
            for cls, polygon in zip(clss, polygons):
                x, y, w, h = polygon2xywh(polygon)
                f.write(f'{cls} {x} {y} {w} {h}\n')
