from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import math
import numpy as np
import cv2

model_456 = YOLO(r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\456_300dataset_imgsz640_v8n_SGD\weights\best.engine', task='detect')
model_789 = YOLO(r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\789_800dataset_imgsz96_v8n_SGD\weights\best.engine', task='detect')
coefficient1 = 2.5
coefficient2 = 2 * 0.5
coefficient3 = 0.005
imgsz1 = 640
imgsz2 = 96
frequence = 1

# cap = cv2.VideoCapture('http://admin:admin@192.168.10.240:8081/')
cap = cv2.VideoCapture(0)


# cap = cv2.VideoCapture(r'E:\desktop\456_test\20231001_125958.mp4')


class Variant:
    width = None
    height = None
    frames_num = 0
    fps_update_before = cv2.getTickCount()
    fps = 0
    detected_digits = [[str(i), 0] for i in range(100)]
    time = []
    times = []
    image_for_concat = None
    image_for_concat_update_before = cv2.getTickCount()


def detect_456(img):
    results_456 = model_456.predict(source=img, imgsz=imgsz1, half=True,
                                    save=False, conf=0.5, verbose=False)
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

                top = int(y1 - h1) if int(y1 - h1) > 0 else 0
                bottom = int(y1 + h1) if int(y1 + h1) < image.shape[0] else image.shape[0]
                left = int(x1 - w1) if int(x1 - w1) > 0 else 0
                right = int(x1 + w1) if int(x1 + w1) < image.shape[1] else image.shape[1]

                cropped_img = image[top:bottom, left:right]

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
        results = model_789.predict(source=rotated_img, imgsz=imgsz2, half=True,
                                    save=False, conf=0.5, verbose=False)

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
    h, w = image.shape[:2]
    annotator = Annotator(image, line_width=int(min(w, h) * coefficient3))
    for dg, xywh in zip(double_digits, xywhs):
        x, y, w, h = xywh
        xyxy = [x - w / 2, y - h / 2, x + w / 2, y + w / 2]
        annotator.box_label(xyxy, label=f'{dg}:{int(w)}*{int(h)}')

    return annotator.im


def text(img, variant, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):
    # 右上角显示已检测双位数次数
    dgs = sorted(variant.detected_digits, reverse=True, key=lambda i: i[1])
    for i, j in enumerate(dgs[:4]):
        dg_text = f'{j[0]}:{j[1]}'
        dg_text_width, dg_text_height = cv2.getTextSize(dg_text, font, font_scale, font_thickness)[0]
        dg_text_x, dg_text_y = int(variant.width - dg_text_width), int((i + 1) * dg_text_height * 2)
        cv2.putText(img, dg_text, (dg_text_x, dg_text_y), font,
                    font_scale,
                    (0, 0, 255),
                    font_thickness)

    # 右上角显示已检测三个双位数的中位数
    dg_text = sorted(dgs[:3], key=lambda i: int(i[0]))[1][0]
    dg_text_width, dg_text_height = cv2.getTextSize(dg_text, font, font_scale, font_thickness)[0]
    dg_text_x, dg_text_y = int(variant.width - dg_text_width), int(5 * dg_text_height * 2)
    cv2.putText(img, dg_text, (dg_text_x, dg_text_y), font,
                font_scale,
                (0, 255, 0),
                font_thickness)

    # 左上角显示FPS
    cv2.putText(img, f'FPS:{variant.fps:.2f}', (10, 30), font, font_scale, (0, 0, 255), font_thickness)

    return img


def concatenate(img, rotated_imgs, variant):
    channel = img.shape[2]
    if variant.image_for_concat is None:
        variant.image_for_concat = np.zeros((variant.height, variant.height // 4, channel), np.uint8)
    else:
        if (cv2.getTickCount() - variant.image_for_concat_update_before) / cv2.getTickFrequency() > 1:
            resized_imgs = [cv2.resize(i, (variant.height // 4, variant.height // 4)) for i in rotated_imgs[:4]]
            if len(resized_imgs) != 0:
                image = resized_imgs[0]
                for i in range(4):
                    if i + 1 < len(resized_imgs):
                        image = np.vstack((image, resized_imgs[i + 1]))
                variant.image_for_concat = np.vstack(
                    (image,
                     np.zeros((variant.height - image.shape[0], variant.height // 4, channel), np.uint8)))
                variant.image_for_concat_update_before = cv2.getTickCount()

    concatenated_image = np.concatenate([img, variant.image_for_concat], axis=1)
    return concatenated_image


def main():
    variant = Variant()

    variant.width, variant.height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(f'output/{cv2.getTickCount()}.avi', fourcc, 100.0,
                          (variant.width + variant.height // 4, variant.height))
    cv2.namedWindow('double_digits_detection', cv2.WINDOW_AUTOSIZE)

    while cap.isOpened():
        # tick1 = cv2.getTickCount()
        # 1 读取摄像头画面
        success, frame = cap.read()
        if success:
            # print((cv2.getTickCount() - tick1) / cv2.getTickFrequency())
            # variant.time.append(cv2.getTickCount())
            # 2 识别靶子中三角位置与双位数位置
            locaters, digits = detect_456(frame)

            # variant.time.append(cv2.getTickCount())
            # 3 截取靶子中双位数图片，并根据靶子中三角位置与双位数位置计算旋转角度，旋转截取图片，获取旋转后图片以及对应双位数位置
            rotated_imgs, xywhs = from_456_to_789(locaters, digits, frame)

            # variant.time.append(cv2.getTickCount())
            # 4 识别旋转后图片上两单位数的数值，并合并为双位数数值
            double_digits = detect_789(rotated_imgs)

            # variant.time.append(cv2.getTickCount())
            # 5 根据双位数位置以及数值画框并标记数值
            bounded_image = plot(double_digits, xywhs, frame)

            # variant.time.append(cv2.getTickCount())
            # 6 附上fps和双位数检测次数及最终中位数
            texted_image = text(bounded_image, variant)

            # variant.time.append(cv2.getTickCount())
            # 7 贴上双位数图片预览图
            concatenated_image = concatenate(texted_image, rotated_imgs, variant)

            # variant.time.append(cv2.getTickCount())
            # 8 检测后图片显示并保存到视频
            cv2.imshow('double_digits_detection', concatenated_image)
            out.write(concatenated_image)

            # variant.time.append(cv2.getTickCount())
            # 9 双位数检测次数计数andFPS计算
            if len(double_digits) != 0:
                for double_digit in double_digits:
                    idx = int(double_digit)
                    variant.detected_digits[idx][1] += 1

            variant.frames_num += 1
            if variant.frames_num > 60:
                variant.fps = variant.frames_num * cv2.getTickFrequency() / (
                        cv2.getTickCount() - variant.fps_update_before)
                variant.frames_num = 0
                variant.fps_update_before = cv2.getTickCount()

            # variant.time.append(cv2.getTickCount())
            # variant.times.append(variant.time)
            # variant.time = []
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    out.release()
    cv2.destroyAllWindows()

    # ut = [0 for i in variant.times[0]]
    # for i in variant.times:
    #     for j in range(len(i) - 1):
    #         ut[j] += i[j + 1] - i[j]
    #     ut[-1] += i[-1] - i[0]
    # print('\n\n\n')
    # for idx, i in enumerate(ut):
    #     print(idx + 1, i / len(variant.times) / cv2.getTickFrequency())
    # print(1 / (ut[-1] / len(variant.times) / cv2.getTickFrequency()))


def update_fps(img, fps, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):
    # 左上角显示FPS
    cv2.putText(img, f'FPS:{fps:.2f}', (10, 30), font, font_scale, (0, 0, 255), font_thickness)
    return img


if __name__ == '__main__':
    main()
