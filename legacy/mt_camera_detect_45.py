import os
from ultralytics import YOLO
import cv2
import threading
import queue
import common
import time

model_4_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\4_400dataset_imgsz640_v8n_SGD\weights\best.engine'
model_5_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\5_1100dataset_imgsz160_v8n_Adam\weights\best.engine'
coefficient = 0.6  # 截取靶子范围影响系数
coefficient1 = 1.1  # 标记靶子范围影响系数
imgsz1 = 640
imgsz2 = 160
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
    model1 = YOLO(model_4_path, task='detect')
    model2 = YOLO(model_5_path, task='detect')

    global flag, detected_frames_num
    while flag:
        try:
            frame = queue1.get(timeout=timeout)

            # 识别靶子位置
            xywhs = common.detect_4(imgsz1, model1, frame, conf, iou)

            # 截取靶子图片
            cropped_imgs = from_4_to_5(xywhs, frame)

            # 标记靶子位置
            for x, y, w, h in xywhs:
                cv2.rectangle(frame, (int(x - w * coefficient1), int(y - h * coefficient1)),
                              (int(x + w * coefficient1), int(y + h * coefficient1)),
                              (0, 255, 0), 2)

            # 保存靶子图片
            flag1 = False
            lock.acquire()
            if len(cropped_imgs) > 0:
                detected_frames_num += 1
                if detected_frames_num > detected_frames_frequence:
                    flag1 = True
                    detected_frames_num = 0
            lock.release()
            if flag1:
                for i in cropped_imgs:
                    cv2.imwrite(f'output/{now_time}/{cv2.getTickCount()}.jpg', i)

            # 识别旋转后图片上两单位数的数值，并合并为双位数数值
            clss, xywhs = common.detect_5(imgsz2, model2, cropped_imgs, xywhs, conf, iou)

            # 根据双位数位置以及数值画框并标记数值
            bounded_image = common.plot(clss, xywhs, frame)

            queue2.put(cv2.resize(bounded_image, (save_width, save_height)))

            # 双位数检测次数计数
            if len(clss) != 0:
                lock.acquire()
                for cls in clss:
                    idx = int(cls)
                    detected_digits[idx][1] += 1
                lock.release()

        except queue.Empty:
            lock.acquire()
            print('ai检测子线程关闭')
            lock.release()
            break


def from_4_to_5(xywhs, image):
    imgs = []
    for x, y, w, h in xywhs:
        # 截取靶子
        w *= coefficient
        h *= coefficient
        top = int(y - h) if int(y - h) > 0 else 0
        bottom = int(y + h) if int(y + h) < image.shape[0] else image.shape[0]
        left = int(x - w) if int(x - w) > 0 else 0
        right = int(x + w) if int(x + w) < image.shape[1] else image.shape[1]

        cropped_img = image[top:bottom, left:right]
        imgs.append(cropped_img)

    return imgs


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
        lock.acquire()
        print('save', save_frame_queue.qsize())
        lock.release()
        try:
            out.write(save_frame_queue.get(timeout=timeout))
        except queue.Empty:
            break
    out.release()


if __name__ == '__main__':
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
    cv2.destroyAllWindows()
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
