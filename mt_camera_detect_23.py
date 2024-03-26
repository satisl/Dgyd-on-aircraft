import os
import time
from ultralytics import YOLO
import math
import numpy as np
import cv2
import threading
import queue
import common
from common import detect_2, detect_3, update_fps, text

model_2_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\2_400dataset_imgsz640_v8n_SGD\weights\best.engine'
model_3_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\3_1100dataset_imgsz96_v8n_SGD\weights\best.engine'
coefficient1 = 2.5  # 双位数，靶子配对范围系数
coefficient2 = 2 * 0.5  # 双位数截取范围系数
imgsz1 = 640
imgsz2 = 96
conf = 0.5
iou = 0.5
# cap_path = 0
cap_path = r'E:\desktop\456_test\benchmark\quick moving.mp4'
# cap_path = 'rtsp://admin:12345@192.168.10.240:8554/live'
frequence = 250
worker_num = 10
timeout = 3
detected_frames_frequence = 10
save_width, save_height = 640, 480


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

    return rotated_imgs, xywhs, xywhs_


def camera(queues, cap_path, frequence, worker_num, lock):
    cap = cv2.VideoCapture(cap_path)
    global flag
    while cap.isOpened() and flag:
        common.camera(queues, cap, frequence, worker_num, lock)


def main(imgsz1, imgsz2, queue1, queue2, lock, detected_frames_frequence, timeout):
    model1 = YOLO(model_2_path, task='detect')
    model2 = YOLO(model_3_path, task='detect')
    global flag, detected_frames_num
    while flag:
        try:
            frame = queue1.get(timeout=timeout)
            detected_digits_ = [[str(idx), i] for idx, i in enumerate(detected_digits)]
            width = frame.shape[1]
            # 识别靶子中三角位置与双位数位置
            locaters, digits = detect_2(imgsz1, model1, frame, conf, iou)

            # 截取靶子中双位数图片，并根据靶子中三角位置与双位数位置计算旋转角度，旋转截取图片，获取旋转后图片以及对应双位数位置
            rotated_imgs, xywhs, xywhs_ = from_2_to_3(locaters, digits, frame)

            # 识别旋转后图片上两单位数的数值，并合并为双位数数值
            double_digits, xywhs_zipped = detect_3(imgsz2, model2, rotated_imgs, zip(xywhs, xywhs_), conf, iou)

            # 对双位数和三角画框(第一模型检测结果),对靶子整体画框（算法配对结果），debug用
            # frame = common.plot(['locater' for i in range(len(locaters))], locaters, frame)
            # frame = common.plot(['digit' for i in range(len(digits))], digits, frame)
            xyxys = [(min(x1 - w1, x2 - w2), min(y1 - h1, y2 - h2), max(x1 + w1, x2 + w2), max((y1 + h1, y2 + h2))) for
                     (x1, y1, w1, h1), (x2, y2, w2, h2) in xywhs_zipped]
            for x1, y1, x2, y2 in xyxys:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 根据双位数位置以及数值画框并标记数值
            bounded_image = common.plot(double_digits, xywhs, frame)

            # 附上双位数检测次数及最终中位数
            texted_image = text(bounded_image, width, detected_digits_, lock)

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
            queue2.put(cv2.resize(texted_image, (save_width, save_height)))

            # 双位数检测次数计数
            if len(double_digits) != 0:
                lock.acquire()
                for double_digit in double_digits:
                    idx = int(double_digit)
                    detected_digits[idx] += 1
                lock.release()
        except queue.Empty:
            lock.acquire()
            print('ai检测子线程关闭')
            lock.release()
            break


def save(save_frame_queue, lock):
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(f'output/{now_time}/{cv2.getTickCount()}.avi', fourcc, 30,
                          (save_width, save_height))
    global save_flag
    while save_flag:
        lock.acquire()
        print('save', save_frame_queue.qsize())
        lock.release()
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
    detected_digits = [0 for i in range(100)]  # 检测结果储存处
    tasks = [threading.Thread(target=main,
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
    common.show(show_queues, save_queue, frequence, worker_num, lock, timeout)

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
