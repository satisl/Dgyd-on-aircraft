import os
from ultralytics import YOLO
import math
import cv2
import threading
import queue
import common
from ultralytics.utils.plotting import Annotator
import time

model_7_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\7_400dataset_imgsz640_v8n_SGD\weights\best.engine'
model_5_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\5_1100dataset_imgsz160_v8n_Adam\weights\best.engine'
coefficient1 = 1.2  # 靶子截取范围系数
imgsz1 = 640
imgsz2 = 160
conf = 0.5
iou = 0.5
cap_path = r'E:\desktop\456_test\benchmark\far far distance 69.mp4'
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


def main(imgsz1, imgsz2, queue1, queue2, lock, detected_frames_frequence, timeout):
    model1 = YOLO(model_7_path, task='obb')
    model2 = YOLO(model_5_path, task='detect')
    global flag, detected_frames_num
    while flag:
        try:
            frame = queue1.get(timeout=timeout)

            # 检测靶子，获取最小外接有向矩形框和最小外接水平矩形框
            xywhrs, xyxys, xyxyxyxys = common.detect_7(imgsz1, model1, frame, conf, iou)
            height, width = frame.shape[:2]

            # 截取并旋转靶子图像
            rotated_imgs = from_7_to_5(xywhrs, xyxys, frame, width, height)

            # 检测靶子检测效果
            for left, top, right, bottom in xyxys:
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # 识别旋转后图片上两单位数的数值，并合并为双位数数值
            double_digits, xyxyxyxys = common.detect_5(imgsz2, model2, rotated_imgs, xyxyxyxys, conf, iou)

            # 根据双位数位置以及数值画框并标记数值
            bounded_image = plot_polygon(double_digits, xyxyxyxys, frame)

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


def from_7_to_5(xywhr, xyxy, image, width, height):
    rotated_imgs = []
    for (x, y, _, _, r), (left, top, right, bottom) in zip(xywhr, xyxy):
        # 截取长宽
        w = right - left
        h = bottom - top
        left -= (w * (coefficient1 - 1) / 2)
        right += (w * (coefficient1 - 1) / 2)
        top -= (h * (coefficient1 - 1) / 2)
        bottom += (h * (coefficient1 - 1) / 2)
        left = left if left >= 0 else 0
        top = top if top >= 0 else 0
        right = right if right <= width else width
        bottom = bottom if bottom <= height else height

        # 旋转角度
        rad = r
        degree = math.degrees(rad) + 90
        rot_mat = cv2.getRotationMatrix2D((x - left, y - top), degree, 0.9)

        # 图像处理
        cropped_img = image[int(top):int(bottom), int(left):int(right)]
        rotated_img = cv2.warpAffine(cropped_img, rot_mat, (int(right - left), int(bottom - top)),
                                     flags=cv2.INTER_LANCZOS4)
        rotated_imgs.append(rotated_img)
    return rotated_imgs


def plot_polygon(double_digits, xyxyxyxys, image, coefficient=0.005):
    # 在原图片上绘制框并标上具体双位数数值
    h, w = image.shape[:2]
    annotator = Annotator(image, line_width=int(min(w, h) * coefficient))
    for dg, xyxyxyxy in zip(double_digits, xyxyxyxys):
        annotator.box_label(xyxyxyxy, label=f'{dg}', rotated=True)

    return annotator.im


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
