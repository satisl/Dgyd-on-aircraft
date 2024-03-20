import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import math
import numpy as np
import cv2
import threading
import queue

model_456_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\456_300dataset_imgsz640_v8n_SGD\weights\best.engine'
model_789_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\789_800dataset_imgsz96_v8n_SGD\weights\best.engine'
coefficient1 = 2.5
coefficient2 = 2 * 0.5
coefficient3 = 0.005
imgsz1 = 640
imgsz2 = 96


def detect_456(model_456, img):
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

        for j in locaters:
            x2, y2 = j[:2]

            if math.dist((x1, y1), (x2, y2)) < min(w1, h1) * coefficient1:
                # 截取靶子
                xywhs.append(i)
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


def detect_789(model_789, rotated_imgs, xywhs):
    double_digits = []
    xywhs_ = []
    for rotated_img, i in zip(rotated_imgs, xywhs):
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
            xywhs_.append(i)
    return double_digits, xywhs_


def plot(double_digits, xywhs, image):
    # 在原图片上绘制框并标上具体双位数数值
    h, w = image.shape[:2]
    annotator = Annotator(image, line_width=int(min(w, h) * coefficient3))
    for dg, xywh in zip(double_digits, xywhs):
        x, y, w, h = xywh
        xyxy = [x - w / 2, y - h / 2, x + w / 2, y + w / 2]
        annotator.box_label(xyxy, label=f'{dg}:{int(w)}*{int(h)}')

    return annotator.im


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


def main(model_456, model_789, input_frame_queue, output_frame_queue, frequence, lock):
    image_for_concat = None
    image_for_concat_update_before = cv2.getTickCount()

    global flag
    while flag:
        time.sleep(1 / frequence)
        if input_frame_queue.qsize() > 0:

            frame = input_frame_queue.get()
            detected_digits_ = [[str(idx), i] for idx, i in enumerate(detected_digits)]
            height, width = frame.shape[:2]
            # 2 识别靶子中三角位置与双位数位置
            locaters, digits = detect_456(model_456, frame)

            # 3 截取靶子中双位数图片，并根据靶子中三角位置与双位数位置计算旋转角度，旋转截取图片，获取旋转后图片以及对应双位数位置
            rotated_imgs, xywhs = from_456_to_789(locaters, digits, frame)

            # 4 识别旋转后图片上两单位数的数值，并合并为双位数数值
            double_digits, xywhs = detect_789(model_789, rotated_imgs, xywhs)

            # 5 根据双位数位置以及数值画框并标记数值
            bounded_image = plot(double_digits, xywhs, frame)

            # 6 附上双位数检测次数及最终中位数

            texted_image = text(bounded_image, width, detected_digits_)

            # 7 贴上双位数图片预览图
            concatenated_image, image_for_concat, image_for_concat_update_before = \
                concatenate(texted_image,
                            rotated_imgs, height,
                            image_for_concat,
                            image_for_concat_update_before)
            # # 8 检测后图片添加至队列
            output_frame_queue.put(concatenated_image)

            # 9 双位数检测次数计数
            if len(double_digits) != 0:
                lock.acquire()
                for double_digit in double_digits:
                    idx = int(double_digit)
                    detected_digits[idx] += 1
                lock.release()


def camera(input_queues, cap_path, frequence, worker_num, lock):
    cap = cv2.VideoCapture(cap_path)
    global flag
    while cap.isOpened() and flag:
        time.sleep(1 / frequence)
        for i in range(worker_num):
            success, frame = cap.read()
            if success:
                input_queues[i].put(frame)
                num = 0
                for j in range(worker_num):
                    num += input_queues[j].qsize()
                # lock.acquire()
                # # print(f'camera:{i}', input_queues[i].qsize())
                # print('camera_total', num)
                # lock.release()
            else:
                cap.release()
                break


def update_fps(img, fps, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):
    # 左上角显示FPS
    cv2.putText(img, f'FPS:{fps:.2f}', (10, 30), font, font_scale, (0, 0, 255), font_thickness)
    return img


def save(save_frame_queue, cap_path, frequence, lock):
    cap = cv2.VideoCapture(cap_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(f'output/{cv2.getTickCount()}.avi', fourcc, 30,
                          (width + height // 4, height))
    global save_flag
    while save_flag:
        # lock.acquire()
        # print('save', save_frame_queue.qsize())
        # lock.release()

        time.sleep(1 / frequence)
        if save_frame_queue.qsize() > 0:
            out.write(save_frame_queue.get())


if __name__ == '__main__':
    cap_path = 0
    # cap_path = r'E:\desktop\456_test\20231001_125958.mp4'
    frequence = 250
    worker_num = 1

    # 视频流队列
    input_queues = [queue.Queue(maxsize=50) for i in range(worker_num)]
    show_queues = [queue.Queue() for i in range(worker_num)]
    save_frame_queue = queue.Queue()

    flag = True
    lock = threading.Lock()
    # 获取视频帧线程
    t1 = threading.Thread(target=camera, args=(input_queues, cap_path, frequence, worker_num, lock))
    t1.start()

    # ai检测视频帧线程
    detected_digits = [0 for i in range(100)]  # 检测结果储存处
    models = [(YOLO(model_456_path, task='detect'), YOLO(model_789_path, task='detect'))
              for i in range(worker_num)]
    tasks = [threading.Thread(target=main,
                              args=(
                                  i[0], i[1], input_queues[idx], show_queues[idx],
                                  frequence / worker_num, lock
                              ))
             for idx, i in enumerate(models)]
    for i in tasks:
        i.start()

    # 保存视频帧线程
    save_flag = True
    t3 = threading.Thread(target=save, args=(save_frame_queue, cap_path, frequence, lock))
    t3.start()

    # 主进程cv2.imshow窗口
    cv2.namedWindow('0', cv2.WINDOW_AUTOSIZE)
    fps = 0
    frames_num = 0
    fps_update_before = cv2.getTickCount()
    show_flag = True

    while show_flag:
        time.sleep(1 / frequence)
        for i in range(worker_num):
            num = 0
            for j in range(worker_num):
                num += show_queues[j].qsize()
            # lock.acquire()
            # # print(f'show:{i}', show_queues[i].qsize())
            # print('show_total', num)
            # lock.release()

            try:
                frame = show_queues[i].get(timeout=10)
            except queue.Empty:
                show_flag = False
                break
            frames_num += 1
            frame = update_fps(frame, fps)  # 附上fps
            cv2.imshow('0', frame)
            save_frame_queue.put(frame)  # 添加至视频保存队列

            # fps计算
            if frames_num > 60:
                fps = frames_num / ((cv2.getTickCount() - fps_update_before) / cv2.getTickFrequency())
                frames_num = 0
                fps_update_before = cv2.getTickCount()

        # 检测到q，关闭窗口和所有进程
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            show_flag = False
    # 关闭各个线程
    flag = False
    t1.join()
    print('摄像头线程关闭')
    for i in tasks:
        i.join()
    print('ai检测线程关闭')
    while True:
        if save_frame_queue.empty():
            save_flag = False
            t3.join()
            print('视频保存线程关闭')
            break
