import os
from ultralytics import YOLO
import math
import cv2
import threading
import queue
import common
import time

model_2_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\2_400dataset_imgsz640_v8n_SGD\weights\best.engine'
model_3_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\3_1100dataset_imgsz96_v8n_SGD\weights\best.engine'
coefficient1 = 2.5  # 双位数，靶子配对范围系数
coefficient2 = 2 * 0.5  # 双位数截取范围系数
imgsz1 = 640
imgsz2 = 96
conf = 0.5
iou = 0.5
cap_path = r'E:\desktop\456_test\benchmark\quick moving.mp4'
frequence = 20
worker_num = 10
timeout = 3
detected_frames_frequence = 10
save_width, save_height = 640, 480


def camera(queues, cap_path, frequence, worker_num, lock):
    cap = cv2.VideoCapture(cap_path)
    global flag
    while cap.isOpened() and flag:
        common.camera(queues, cap, frequence, worker_num, lock)


def detect(imgsz1, imgsz2, queue1, queue2, lock, detected_frames_frequence, timeout):
    model1 = YOLO(model_2_path, task='detect')
    model2 = YOLO(model_3_path, task='detect')
    global flag, detected_frames_num
    while flag:
        try:
            frame = queue1.get(timeout=timeout)
            # 识别靶子中三角位置与双位数位置
            locaters, digits = common.detect_2(imgsz1, model1, frame, conf, iou)

            # 截取靶子中双位数图片，并根据靶子中三角位置与双位数位置计算旋转角度，旋转截取图片，获取旋转后图片以及对应双位数位置
            rotated_imgs, xywhs, xywhs_ = from_2_to_3(locaters, digits, frame)

            # 对靶子整体画框，检测双位数，三角头检测及配对效果
            xyxys = [(min(x1 - w1, x2 - w2), min(y1 - h1, y2 - h2), max(x1 + w1, x2 + w2), max((y1 + h1, y2 + h2))) for
                     (x1, y1, w1, h1), (x2, y2, w2, h2) in zip(xywhs, xywhs_)]
            for x1, y1, x2, y2 in xyxys:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 识别旋转后图片上两单位数的数值，并合并为双位数数值
            double_digits, xywhs = common.detect_3(imgsz2, model2, rotated_imgs, xywhs, conf, iou)

            # 根据双位数位置以及数值画框并标记数值
            bounded_image = common.plot(double_digits, xywhs, frame)

            # 保存旋转后图片
            flag1 = False
            lock.acquire()
            if len(rotated_imgs) > 0:
                detected_frames_num += 1
                if detected_frames_num > detected_frames_frequence:
                    flag1 = True
                    detected_frames_num = 0
            lock.release()
            if flag1:
                for i in rotated_imgs:
                    cv2.imwrite(f'output/{now_time}/{cv2.getTickCount()}.jpg', i)

            # 检测后图片添加至队列
            queue2.put(cv2.resize(bounded_image, (save_width, save_height)))

            # 双位数检测次数计数
            if len(double_digits) != 0:
                lock.acquire()
                for double_digit in double_digits:
                    idx = int(double_digit)
                    detected_digits[idx][1] += 1
                lock.release()
        except queue.Empty:
            lock.acquire()
            print('ai检测子线程关闭')
            lock.release()
            break


def from_2_to_3(locaters, digits, image):
    rotated_imgs = []
    xywhs = []
    xywhs_ = []
    for i in digits:
        x1, y1, w1, h1 = i

        for j in locaters:
            x2, y2, w2, h2 = j

            if math.dist((x1, y1), (x2, y2)) < min(w1, h1) * coefficient1:
                # 截取靶子
                xywhs.append(i)
                xywhs_.append(j)
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

                rot_mat = cv2.getRotationMatrix2D((x1 - left, y1 - top), angle_deg, 0.8)

                rotated_img = cv2.warpAffine(cropped_img, rot_mat, ((right - left), (bottom - top)),
                                             flags=cv2.INTER_LANCZOS4)
                rotated_imgs.append(rotated_img)

    return rotated_imgs, xywhs, xywhs_


def show(queues, queue1, frequence, worker_num, lock, timeout):
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
                num += queues[j].qsize()
            lock.acquire()
            print('show_total', num)
            lock.release()

            try:
                frame = queues[i].get(timeout=timeout)
            except queue.Empty:
                show_flag = False
                break
            frames_num += 1
            frame = common.text(frame, detected_digits)  # 附上双位数检测次数及最终中位数
            frame = common.update_fps(frame, fps)  # 附上fps
            cv2.imshow('0', frame)
            queue1.put(frame)  # 添加至视频保存队列

            # fps计算
            if frames_num > 60:
                fps = frames_num / ((cv2.getTickCount() - fps_update_before) / cv2.getTickFrequency())
                frames_num = 0
                fps_update_before = cv2.getTickCount()

        # 检测到q，关闭窗口和所有进程
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            show_flag = False


def save(save_frame_queue, lock):
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(f'output/{now_time}/{cv2.getTickCount()}.avi', fourcc, 30,
                          (save_width, save_height))
    global save_flag
    while save_flag:
        # lock.acquire()
        # print('save', save_frame_queue.qsize())
        # lock.release()
        try:
            out.write(save_frame_queue.get(timeout=timeout))
        except queue.Empty:
            break
    out.release()


if __name__ == '__main__':
    # 创建保存文件夹
    now_time = cv2.getTickCount()
    os.makedirs(f'output/{now_time}')

    # 视频流队列
    input_queues = [queue.Queue(maxsize=50) for i in range(worker_num)]
    show_queues = [queue.Queue() for i in range(worker_num)]
    save_queue = queue.Queue()

    flag = True
    lock = threading.Lock()
    # 获取视频帧线程
    t1 = threading.Thread(target=camera, args=(input_queues, cap_path, frequence, worker_num, lock))
    t1.start()

    # ai检测视频帧线程
    detected_frames_num = 0
    detected_digits = [[i, 0] for i in range(100)]  # 检测结果储存处
    tasks = [threading.Thread(target=detect,
                              args=(
                                  imgsz1, imgsz2, input_queues[idx], show_queues[idx], lock,
                                  detected_frames_frequence, timeout
                              )) for idx in range(worker_num)]
    for i in tasks:
        i.start()

    # 保存视频帧线程
    save_flag = True
    t3 = threading.Thread(target=save, args=(save_queue, lock))
    t3.start()

    # 主线程展示视频帧
    show(show_queues, save_queue, frequence, worker_num, lock, timeout)

    # 关闭各个线程
    flag = False
    t1.join()
    print('摄像头线程关闭')
    for i in tasks:
        i.join()
    print('ai检测线程关闭')
    while True:
        if save_queue.empty():
            save_flag = False
            t3.join()
            print('视频保存线程关闭')
            break
