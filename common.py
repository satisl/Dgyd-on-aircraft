import cv2
import time
import numpy as np





def text(img, width, detected_digits, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):
    # 右上角显示已检测双位数次数
    dgs = sorted(detected_digits, reverse=True, key=lambda i: i[1])
    for i, j in enumerate(dgs[:4]):
        dg_text = f'{j[0]}:{j[1]}'
        dg_text_width, dg_text_height = cv2.getTextSize(dg_text, font, font_scale, font_thickness)[0]
        dg_text_x, dg_text_y = int(width - dg_text_width), int((i + 1) * dg_text_height * 2)
        cv2.putText(img, dg_text, (dg_text_x, dg_text_y), font,
                    font_scale,
                    (0, 0, 255),
                    font_thickness)

    # 右上角显示已检测三个双位数的中位数
    dg_text = sorted(dgs[:3], key=lambda i: int(i[0]))[1][0]
    dg_text_width, dg_text_height = cv2.getTextSize(dg_text, font, font_scale, font_thickness)[0]
    dg_text_x, dg_text_y = int(width - dg_text_width), int(5 * dg_text_height * 2)
    cv2.putText(img, dg_text, (dg_text_x, dg_text_y), font,
                font_scale,
                (0, 255, 0),
                font_thickness)

    return img


def concatenate(img, rotated_imgs, height, image_for_concat, image_for_concat_update_before):
    channel = img.shape[2]
    if image_for_concat is None:
        image_for_concat = np.zeros((height, height // 4, channel), np.uint8)
    else:
        if (cv2.getTickCount() - image_for_concat_update_before) \
                / cv2.getTickFrequency() > 1:
            resized_imgs = [cv2.resize(i, (height // 4, height // 4)) for i in rotated_imgs[:4]]
            if len(resized_imgs) != 0:
                image = resized_imgs[0]
                for i in range(4):
                    if i + 1 < len(resized_imgs):
                        image = np.vstack((image, resized_imgs[i + 1]))
                image_for_concat = np.vstack(
                    (image,
                     np.zeros((height - image.shape[0], height // 4, channel), np.uint8)))
                image_for_concat_update_before = cv2.getTickCount()

    concatenated_image = np.concatenate([img, image_for_concat], axis=1)
    return concatenated_image, image_for_concat, image_for_concat_update_before


def detect_456(imgsz, model_456, img, conf, iou):
    results_456 = model_456.predict(source=img, imgsz=imgsz, half=True,
                                    save=False, conf=conf, iou=iou, verbose=False)
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


def detect_789(imgsz, model_789, rotated_imgs, _s, conf, iou):
    double_digits = []
    __s = []
    for rotated_img, _ in zip(rotated_imgs, _s):
        # 检测旋转后靶子，具体获取双位数数值
        results = model_789.predict(source=rotated_img, imgsz=imgsz, half=True,
                                    save=False, conf=conf, iou=iou, verbose=False)

        r = results[0]
        xywh = r.boxes.xywh.tolist()
        cls = r.boxes.cls.tolist()

        if len(xywh) == 2:
            if xywh[0][0] < xywh[1][0]:
                double_digit = str(int(cls[0])) + str(int(cls[1]))
            else:
                double_digit = str(int(cls[1])) + str(int(cls[0]))
            double_digits.append(double_digit)
            __s.append(_)
    return double_digits, __s


def update_fps(img, fps, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):
    # 左上角显示FPS
    cv2.putText(img, f'FPS:{fps:.2f}', (10, 30), font, font_scale, (0, 0, 255), font_thickness)
    return img


def camera(queues, cap, frequence, worker_num, lock):
    time.sleep(1 / frequence)
    for i in range(worker_num):
        success, frame = cap.read()
        if success:
            queues[i].put(frame)

        else:
            cap.release()
            break
    num = 0
    for j in range(worker_num):
        num += queues[j].qsize()
    lock.acquire()
    print('detect_total', num)
    lock.release()
