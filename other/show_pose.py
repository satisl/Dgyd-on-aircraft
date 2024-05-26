import os
import numpy as np

import cv2


# 生成随机颜色
def generate_random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())


# 读取文件，解析数据
def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            assert len(parts) == 7
            data.append(parts)
    return data


# 画矩形框和点
def draw_rectangles_and_points(image_path, data):
    # 读取图像
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 为每个矩形框和点选择颜色
    colors = [generate_random_color() for _ in range(len(data))]

    for entry, color in zip(data, colors):
        # 解析数据
        id_, x, y, w, h, px, py = entry

        # 绘制矩形框
        x, y, w, h = int(float(x) * width), int(float(y) * height), int(float(w) * width), int(float(h) * height)

        left = x - w // 2
        top = y - h // 2
        right = x + w // 2
        bottom = y + h // 2
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

        # 绘制点
        px, py = int(float(px) * width), int(float(py) * height)
        cv2.circle(image, (px, py), 20, color, -1)

    # 显示图像
    cv2.imshow('Result', cv2.resize(image, (1280, 720)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 主程序
def main():
    img_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\pose\for trainning\images\train'
    txt_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\pose\for trainning\labels\train'
    for file in os.listdir(img_path):
        # 读取数据
        data = read_data_from_file(f'{txt_path}/{file.rsplit(".", 1)[0]}.txt')

        # 绘制矩形框和点
        draw_rectangles_and_points(f'{img_path}/{file}', data)


if __name__ == "__main__":
    main()
