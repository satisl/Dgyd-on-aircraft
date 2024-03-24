import time
from ultralytics import YOLO
import cv2
import threading
from ultralytics.utils.plotting import Annotator
import queue
import common

model_4_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\4_300dataset_imgsz640_v8n_SGD\weights\best.pt'
model_5_path = r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\5_700dataset_imgsz160_v8n_Adam\weights\best.pt'
coefficient = 0.6
imgsz1 = 640
imgsz2 = 160
conf = 0.5
iou = 0.5
cap_path = 0
# cap_path = r'E:\desktop\456_test\20231001_125535.mp4'
frequence = 250
worker_num = 1
timeout = 10


def plot(double_digits, xywhs, image, coefficient=0.005):
    # 在原图片上绘制框并标上具体双位数数值
    h, w = image.shape[:2]
    annotator = Annotator(image, line_width=int(min(w, h) * coefficient))
    for dg, xywh in zip(double_digits, xywhs):
        x, y, w, h = xywh
        xyxy = [x - w / 2, y - h / 2, x + w / 2, y + w / 2]
        annotator.box_label(xyxy, label=f'{dg}:{int(w)}*{int(h)}')

    return annotator.im


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


def camera(queues, cap_path, frequence, worker_num, lock):
    cap = cv2.VideoCapture(cap_path)
    global flag
    while cap.isOpened() and flag:
        common.camera(queues, cap, frequence, worker_num, lock)


def main(imgsz1, imgsz2, model_456, model_789, queue1, queue2, lock, timeout):
    global flag
    while flag:
        try:
            frame = queue1.get(timeout=timeout)
            # 识别靶子位置
            xywhs = common.detect_4(imgsz1, model_456, frame, conf, iou)

            # 截取靶子图片
            cropped_imgs = from_4_to_5(xywhs, frame)

            # 识别旋转后图片上两单位数的数值，并合并为双位数数值
            clss = common.detect_5(imgsz2, model_789, cropped_imgs, ['detected' for i in range(len(cropped_imgs))],
                                   conf, iou)

            # 根据双位数位置以及数值画框并标记数值
            bounded_image = plot(clss, xywhs, frame)

            queue2.put(bounded_image)

        except queue.Empty:
            break


def save(save_frame_queue, cap_path, frequence, lock):
    cap = cv2.VideoCapture(cap_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(f'output/{cv2.getTickCount()}.avi', fourcc, 30,
                          (width + height // 4, height))
    global save_flag
    while save_flag:
        lock.acquire()
        print('save', save_frame_queue.qsize())
        lock.release()

        time.sleep(1 / frequence)
        if save_frame_queue.qsize() > 0:
            out.write(save_frame_queue.get())
    out.release()


if __name__ == '__main__':

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
    models = [(YOLO(model_4_path, task='detect'), YOLO(model_5_path, task='detect'))
              for i in range(worker_num)]
    tasks = [threading.Thread(target=main,
                              args=(
                                  imgsz1, imgsz2, i[0], i[1], input_queues[idx], show_queues[idx], lock,
                                  timeout
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
            lock.acquire()
            # print(f'show:{i}', show_queues[i].qsize())
            print('show_total', num)
            lock.release()

            try:
                frame = show_queues[i].get(timeout=timeout)
            except queue.Empty:
                show_flag = False
                break
            frames_num += 1
            frame = common.update_fps(frame, fps)  # 附上fps
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
