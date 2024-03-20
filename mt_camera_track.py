import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import math
import numpy as np
import cv2
import threading
import queue


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
    xyxys = []
    for i in digits:
        x1, y1, w1, h1 = i

        for j in locaters:
            x2, y2 = j[:2]

            if math.dist((x1, y1), (x2, y2)) < min(w1, h1) * coefficient1:
                # 根据数字与三角坐标画最小包围矩形
                w2, h2 = j[2:4]
                xyxy1 = (x1 - w1, y1 - h1, x1 + w1, y1 + h1)
                xyxy2 = (x2 - w2, y2 - h2, x2 + w2, y2 + h2)
                xyxy3 = ((min(xyxy1[0], xyxy2[0]), min(xyxy1[1], xyxy2[1]),
                          max(xyxy1[2], xyxy2[2]), max(xyxy1[3], xyxy2[3])))
                xyxys.append(xyxy3)

                # 截取靶子
                w1 *= coefficient2
                h1 *= coefficient2

                top = int(y1 - h1) if int(y1 - h1) > 0 else 0
                bottom = int(y1 + h1) if int(y1 + h1) < image.shape[0] else image.shape[0]
                left = int(x1 - w1) if int(x1 - w1) > 0 else 0
                right = int(x1 + w1) if int(x1 + w1) < image.shape[1] else image.shape[1]

                cropped_img = image[top:bottom, left:right]

                # 求旋转角度
                relative_x = x2 - x1
                relative_y = y2 - y1
                angle_rad = math.atan2(relative_y, relative_x)
                angle_deg = math.degrees(angle_rad) + 90

                # 旋转图片
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

    return rotated_imgs, xyxys


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


def plot(dg, xyxy, image):
    # 在原图片上绘制框并标上具体双位数数值
    h, w = image.shape[:2]
    annotator = Annotator(image, line_width=int(min(w, h) * coefficient3))
    annotator.box_label(xyxy, label=f'{dg}:{int(xyxy[2] - xyxy[0])}*{int(xyxy[3] - xyxy[1])}')

    return annotator.im


def update_fps(img, fps, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):
    # 左上角显示FPS
    cv2.putText(img, f'FPS:{fps:.2f}', (10, 30), font, font_scale, (0, 0, 255), font_thickness)
    return img


def camera(queues, cap_path, frequence, worker_num, lock):
    cap = cv2.VideoCapture(cap_path)
    global flag
    while cap.isOpened() and flag:
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


def detect(model_456, model_789, queue1, queue2, timeout):
    global flag
    while flag:
        try:
            frame = queue1.get(timeout=timeout)

            # 1 识别靶子中三角位置与双位数位置
            locaters, digits = detect_456(model_456, frame)

            # 2 截取靶子中双位数图片，并根据靶子中三角位置与双位数位置计算旋转角度，旋转截取图片，获取旋转后图片以及对应双位数位置
            rotated_imgs, xyxys = from_456_to_789(locaters, digits, frame)

            # 3 识别旋转后图片上两单位数的数值，并合并为双位数数值
            double_digits, xyxys = detect_789(model_789, rotated_imgs, xyxys)

            # 4 检测后图片添加至队列
            target_bbox = None
            for i, j in zip(double_digits, xyxys):
                if i == target_num:
                    target_bbox = j
                    break
            queue2.put((frame, target_bbox))
        except queue.Empty:
            break


def track(queues, queue_, worker_num, timeout, lock):
    tracker = None
    frames_num = 0
    fps = 0
    fps_update_before = cv2.getTickCount()

    flag1 = True
    global flag
    while flag and flag1:
        num = 0
        for j in range(worker_num):
            num += queues[j].qsize()
        lock.acquire()
        print('track_total', num)
        lock.release()

        for i in range(worker_num):
            try:
                frame, bbox = queues[i].get(timeout=timeout)
                # 对目标靶子画框
                if bbox is not None:
                    tracker = cv2.legacy.TrackerKCF.create()
                    tracker.init(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                    frame = plot(target_num, bbox, frame)

                elif tracker is not None:
                    success, bbox = tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 贴上fps
                if frames_num > 60:
                    fps = frames_num / ((cv2.getTickCount() - fps_update_before) / cv2.getTickFrequency())
                    frames_num = 0
                    fps_update_before = cv2.getTickCount()
                frames_num += 1
                frame = update_fps(frame, fps)

                queue_.put(frame)
            except queue.Empty:
                flag1 = False
                break


def save(queue, cap_path, frequence, lock):
    cap = cv2.VideoCapture(cap_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(f'output/{cv2.getTickCount()}.avi', fourcc, 30,
                          (width, height))
    global save_flag
    while save_flag:
        time.sleep(1 / frequence)
        lock.acquire()
        print('save', queue.qsize())
        lock.release()
        if queue.qsize() > 0:
            out.write(queue.get())
    out.release()


if __name__ == '__main__':
    model_456_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\456_300dataset_imgsz640_v8n_SGD\weights\best.engine'
    model_789_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\789_800dataset_imgsz96_v8n_SGD\weights\best.engine'
    coefficient1 = 2.5
    coefficient2 = 2 * 0.5
    coefficient3 = 0.005
    imgsz1 = 640
    imgsz2 = 96
    target_num = '60'
    # cap_path = 0
    cap_path = r'E:\desktop\456_test\20231001_125958.mp4'
    frequence = 250
    worker_num = 8
    timeout = 10

    # 视频流队列
    detect_queues = [queue.Queue(maxsize=50) for i in range(worker_num)]
    track_queues = [queue.Queue() for i in range(worker_num)]
    show_queue = queue.Queue()
    save_queue = queue.Queue()

    flag = True
    lock = threading.Lock()
    # 获取视频帧线程
    t1 = threading.Thread(target=camera, args=(detect_queues, cap_path, frequence, worker_num, lock))
    t1.start()

    # ai检测视频帧线程
    models = [(YOLO(model_456_path, task='detect'), YOLO(model_789_path, task='detect'))
              for i in range(worker_num)]
    tasks = [threading.Thread(target=detect,
                              args=(
                                  i[0], i[1], detect_queues[idx], track_queues[idx], timeout
                              ))
             for idx, i in enumerate(models)]
    for i in tasks:
        i.start()

    # ai跟踪视频帧线程
    t2 = threading.Thread(target=track, args=(track_queues, show_queue, worker_num, timeout, lock))
    t2.start()

    # 保存视频帧线程
    save_flag = True
    t3 = threading.Thread(target=save, args=(save_queue, cap_path, frequence, lock))
    t3.start()

    # 主进程显示视频帧
    cv2.namedWindow('0', cv2.WINDOW_AUTOSIZE)

    show_flag = True
    while show_flag:
        print('show', show_queue.qsize())
        try:
            frame = show_queue.get(timeout=10)
        except queue.Empty:
            show_flag = False
            break

        cv2.imshow('0', frame)
        save_queue.put(frame)  # 添加至视频保存队列

        # 检测到q，关闭窗口和所有进程
        if cv2.waitKey(1) & 0xFF == ord('q'):
            show_flag = False
            break

    cv2.destroyAllWindows()
    # 关闭各个线程
    flag = False
    t1.join()
    lock.acquire()
    print('摄像头线程关闭')
    lock.release()

    for i in tasks:
        i.join()
    lock.acquire()
    print('ai检测线程关闭')
    lock.release()
    t2.join()
    lock.acquire()
    print('ai追踪线程关闭')
    lock.release()
    while True:
        time.sleep(1 / frequence)
        if save_queue.empty():
            save_flag = False
            t3.join()
            print('视频保存线程关闭')
            break
