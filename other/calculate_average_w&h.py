import os
from PIL import Image
import math

img_path = ""
imgs = os.listdir(img_path)

width = 0
height = 0
pixel = 0
for img in imgs:
    if img.rsplit(".", 1)[1] != "jpg":
        continue
    origin_img = Image.open(f"{img_path}/{img}")
    w = origin_img.width
    h = origin_img.height
    width += w
    height += h
    pixel += w * h

print(width / len(imgs), height / len(imgs), math.sqrt(pixel / len(imgs)))
