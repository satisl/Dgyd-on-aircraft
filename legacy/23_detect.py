from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os
import math
import time
import numpy as np
import cv2

dst = r'E:\desktop\456_test\100MEDIA'

coefficient1 = 2.5
coefficient2 = 2 * 0.5
imgsz1 = 640
imgsz2 = 96
model_456 = YOLO(f'D:\Double-digit-yolo-detection-on-aircraft\yolov8/2_300dataset_imgsz640_v8n_SGD\weights/best.engine', task='detect')
model_789 = YOLO(f'D:\Double-digit-yolo-detection-on-aircraft\yolov8/3_800dataset_imgsz96_v8n_SGD\weights/best.engine', task='detect')


def detect_456(img):
    results_456 = model_456.predict(source=f'{dst}/{img}',
                                    imgsz=imgsz1, half=True, show_labels=True,
                                    show_conf=True, show_boxes=True,
                                    line_width=1, save=False, conf=0.5, verbose=True)
    r = results_456[0]
    # 获取图片检测相关信息
    xywh = r.boxes.xywh.tolist()
    cls = r.boxes.cls.tolist()

    digits = []
    locaters = []
    for i, j in zip(cls, xywh):
        if i == 0:
            locaters.append(j)
        else:
            digits.append(j)

    return locaters, digits


def from_456_to_789(locaters, digits, image):
    rotated_imgs = []
    xywhs = []
    for i in digits:
        x1, y1, w1, h1 = i
        xywhs.append(i)

        for j in locaters:
            x2, y2 = j[:2]

            if math.dist((x1, y1), (x2, y2)) < min(w1, h1) * coefficient1:
                # 截取靶子

                w1 *= coefficient2
                h1 *= coefficient2

                cropped_img = image[int(y1 - h1):int(y1 + h1), int(x1 - w1):int(x1 + w1)]

                # 旋转靶子
                relative_x = x2 - x1
                relative_y = y2 - y1
                angle_rad = math.atan2(relative_y, relative_x)
                angle_deg = math.degrees(angle_rad) + 90

                w = cropped_img.shape[1]
                h = cropped_img.shape[0]
                nw = (abs(np.sin(angle_rad) * h) + abs(np.cos(angle_rad) * w))
                nh = (abs(np.cos(angle_rad) * h) + abs(np.sin(angle_rad) * w))
                rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle_deg, 1.0)
                rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
                rot_mat[0, 2] += rot_move[0]
                rot_mat[1, 2] += rot_move[1]
                rotated_img = cv2.warpAffine(cropped_img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                                             flags=cv2.INTER_LANCZOS4)
                rotated_imgs.append(rotated_img)

    return rotated_imgs, xywhs


def detect_789(rotated_imgs):
    double_digits = []
    for rotated_img in rotated_imgs:
        # 检测旋转后靶子，具体获取双位数数值
        results = model_789.predict(source=rotated_img,
                                    imgsz=imgsz2, half=True, show_labels=True,
                                    show_conf=True, show_boxes=True,
                                    line_width=1, save=False, conf=0.5, verbose=True)

        r = results[0]
        xywh = r.boxes.xywh.tolist()
        cls = r.boxes.cls.tolist()

        if len(xywh) == 2:
            if xywh[0][0] < xywh[1][0]:
                double_digit = str(int(cls[0])) + str(int(cls[1]))
            else:
                double_digit = str(int(cls[1])) + str(int(cls[0]))

            double_digits.append(double_digit)
    return double_digits


def plot(double_digits, xywhs, image):
    # 在原图片上绘制框并标上具体双位数数值
    annotator = Annotator(image, line_width=10)
    for dg, xywh in zip(double_digits, xywhs):
        x, y, w, h = xywh
        xyxy = [x - w / 2, y - h / 2, x + w / 2, y + w / 2]
        annotator.box_label(xyxy, label=dg)

    return annotator.im


def main():
    imgs = os.listdir(dst)
    now_time = int(time.time())
    times = []
    for img in imgs:
        t1 = time.time()
        # print(img)
        prefix, suffix = img.rsplit('.', 1)
        os.makedirs(f'output/{now_time}/{prefix}', exist_ok=False)
        image = cv2.imread(f'{dst}/{img}')

        t2 = time.time()
        # 识别靶子中三角位置与双位数位置
        locaters, digits = detect_456(img)

        # print(locaters, digits)

        t3 = time.time()
        # 截取靶子中双位数图片，并根据靶子中三角位置与双位数位置计算旋转角度，旋转截取图片，获取旋转后图片以及对应双位数位置
        rotated_imgs, xywhs = from_456_to_789(locaters, digits, image)
        frame = plot(['locater' for i in range(len(locaters))], locaters, image.copy())
        frame = plot(['digit' for i in range(len(digits))], digits, frame)
        cv2.imwrite(f'output/{now_time}/{prefix}_1.{suffix}', frame)

        t4 = time.time()
        # 识别旋转后图片上两单位数的数值，并合并为双位数数值
        double_digits = detect_789(rotated_imgs)

        t5 = time.time()
        # 根据双位数位置以及数值画框并标记数值
        bounded_image = plot(double_digits, xywhs, image)

        t6 = time.time()
        # 图片保存
        cv2.imwrite(f'output/{now_time}/{prefix}.{suffix}', bounded_image)
        cnt = 0
        for rotated_img in rotated_imgs:
            cv2.imwrite(f'output/{now_time}/{prefix}/{prefix}-{cnt}.{suffix}', rotated_img)
            cnt += 1
        t7 = time.time()
        times.append([t1, t2, t3, t4, t5, t6, t7])

    ut = [0, 0, 0, 0, 0, 0, 0]
    for i in times:
        for j in range(len(i) - 1):
            ut[j] += i[j + 1] - i[j]
        ut[-1] += i[-1] - i[0]
    print('\n\n\n')
    for idx, i in enumerate(ut):
        print(idx, i / len(imgs))


if __name__ == '__main__':
    main()
