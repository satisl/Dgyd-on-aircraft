import math
import numpy as np
import cv2
import os

coefficient1 = 1.2  # 靶子截取范围系数
img_path1 = r""
txt_path1 = r""
img_path2 = r""


def crop_and_rotate(xywhr, xyxy, image, width, height, scale=1.0):
    (x, y, rw, rh, r), (left, top, right, bottom) = (xywhr, xyxy)
    # 截取长宽
    w = right - left
    h = bottom - top

    left -= w * (coefficient1 - 1) / 2
    right += w * (coefficient1 - 1) / 2
    top -= h * (coefficient1 - 1) / 2
    bottom += h * (coefficient1 - 1) / 2

    left = left if left >= 0 else 0
    top = top if top >= 0 else 0
    right = right if right <= width else width
    bottom = bottom if bottom <= height else height

    w, h = int(right - left), int(bottom - top)

    # 旋转角度
    rad = r if rw < rh else r - 0.5 * math.pi
    degree = math.degrees(rad)

    nw = (abs(np.sin(rad) * h) + abs(np.cos(rad) * w)) * scale
    nh = (abs(np.cos(rad) * h) + abs(np.sin(rad) * w)) * scale
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), degree, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 图像处理
    cropped_img = image[int(top) : int(bottom), int(left) : int(right)]
    rotated_img = cv2.warpAffine(
        cropped_img,
        rot_mat,
        (int(math.ceil(nw)), int(math.ceil(nh))),
        flags=cv2.INTER_LINEAR,
    )
    return rotated_img


os.makedirs(img_path2, exist_ok=True)

for img_name in os.listdir(img_path1):
    print(img_name)
    img = cv2.imread(f"{img_path1}\{img_name}")
    height, width, _ = img.shape
    prefix, suffix = img_name.rsplit(".", 1)

    with open(f"{txt_path1}\{prefix}.txt", mode="r") as f:
        rotated_imgs = []
        for line in f.readlines():
            cls, x1, y1, x2, y2, x3, y3, x4, y4 = line.strip().split(" ")
            x1, y1, x2, y2, x3, y3, x4, y4 = map(
                float, (x1, y1, x2, y2, x3, y3, x4, y4)
            )

            points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            points *= np.array([width, height])
            points = points.astype(np.float32)

            (x, y), (w, h), angle = cv2.minAreaRect(points)
            xywhr = [x, y, w, h, angle * math.pi / 180]

            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = points
            xyxy = [
                min(x1, x2, x3, x4),
                min(y1, y2, y3, y4),
                max(x1, x2, x3, x4),
                max(y1, y2, y3, y4),
            ]

            rotated_imgs.append(crop_and_rotate(xywhr, xyxy, img, width, height))

        for idx, rotated_img in enumerate(rotated_imgs):
            cv2.imwrite(f"{img_path2}\{prefix}-{idx}.{suffix}", rotated_img)
