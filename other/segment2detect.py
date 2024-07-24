import os
import shutil

import cv2
import numpy as np

org_path = r""
dst_path = r""
img_path1 = f"{org_path}/images"
img_path2 = f"{dst_path}/images"
txt_path1 = f"{org_path}/labels"
txt_path2 = f"{dst_path}/labels"
os.makedirs(img_path2, exist_ok=False)
os.makedirs(txt_path2, exist_ok=False)
shutil.copy(f"{txt_path1}/classes.txt", f"{txt_path2}/classes.txt")

for file in os.listdir(img_path1):
    prefix, suffix = file.rsplit(".", 1)
    img_file = f"{prefix}.{suffix}"
    txt_file = f"{prefix}.txt"
    shutil.copy(f"{img_path1}/{img_file}", f"{img_path2}/{img_file}")
    img = cv2.imread(f"{img_path1}/{img_file}")
    height, width, _ = img.shape

    with open(f"{txt_path1}/{txt_file}", "r") as f:
        lines = f.readlines()
    clss = []
    polygons = []
    for line in lines:
        infos = line.strip("\n").split(" ")
        cls = infos.pop(0)
        polygon = []
        while len(infos) > 0:
            polygon.append((infos.pop(0), infos.pop(0)))
        clss.append(cls)
        polygons.append(polygon)
    print(file)
    with open(f"{txt_path2}/{txt_file}", "w") as f:
        for cls, polygon in zip(clss, polygons):
            if len(polygon) > 0:
                polygon_np = np.array(
                    [(float(x) * width, float(y) * height) for x, y in polygon]
                ).astype(int)
                x, y, w, h = cv2.boundingRect(polygon_np)
                x /= width
                w /= width
                y /= height
                h /= height
                f.write(f"{cls} {x + w / 2} {y + h / 2} {w} {h}\n")
