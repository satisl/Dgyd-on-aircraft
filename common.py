import cv2
import time


def detect_456(imgsz, model_456, img):
    results_456 = model_456.predict(source=img, imgsz=imgsz, half=True,
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


def detect_789(imgsz, model_789, rotated_imgs, _s):
    double_digits = []
    __s = []
    for rotated_img, _ in zip(rotated_imgs, _s):
        # 检测旋转后靶子，具体获取双位数数值
        results = model_789.predict(source=rotated_img, imgsz=imgsz, half=True,
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

