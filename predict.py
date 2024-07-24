import math
import os
import queue
import threading
import time
from datetime import datetime
import subprocess
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import sys
import numpy as np

maxsize = 20  # camera线程帧队列最大数量

imgsz1 = 640
model_path1 = (
    r"./yolov8/obb_592dataset_imgsz640_v8n_SGD/weights/best.engine"
)
conf1 = 0.5
iou1 = 0.2

imgsz2 = 160
model_path2 = r"./yolov8/digits_1665dataset_imgsz160_v8n_Adam/weights/best.engine"
conf2 = 0.5
iou2 = 0.5

coefficient1 = 1.2  # 靶子截取范围系数
worker_num = 1  # 同时开启work_num个detect线程

target_saving_per_frame = 100  # 每多少帧检测到靶子便保存靶子图片
send_per_imgs = 100  # 每多少张靶子图片发送一次邮箱

camera_width, camera_height = 3840, 2160  # camera的帧大小
crop_width, crop_height = 2160, 2160  # 裁切后的帧大小
show_width, show_height = 1080, 1080  # show的帧大小
fps_per_frames = 300  # 每几帧计算一次fps
show_mode = "rtmp"  # 预览模式 imshow, rtmp, rtsp, srt
liveUrl = "rtmp://127.0.0.1:1935/video"
rc_mode = "AVBR"
bitrate = "10M"
live_fps = 30  # 直播流fps
save_per_frames = 10  # 每几个帧save一次

count_frequence = 1  # 每几秒显示一次各队列数量

timeout = 5


def letterbox(img, imgsz):

    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(imgsz / shape[0], imgsz / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = imgsz - new_unpad[0], imgsz - new_unpad[1]  # wh padding

    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_NEAREST)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border

    return img


def plot_polygon(img, pointss, texts, coefficient=0.006):
    h, w = img.shape[:2]
    for points, text in zip(pointss, texts):
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(
            img,
            [points],
            isClosed=True,
            color=(255, 0, 0),
            thickness=int(min(h, w) * coefficient),
        )
        cv2.putText(
            img,
            text,
            (np.min(points[:, :, 0]).item(), np.min(points[:, :, 1]).item() - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            min(h, w) * coefficient,
            (0, 0, 255),
            2,
        )

    return img


def camera(queues1, queue2):
    frame_id = 0

    delta_h, delta_w = camera_height - crop_height, camera_width - crop_width

    global flag
    while flag:
        for i in range(worker_num):
            raw_data = sys.stdin.buffer.read(
                camera_width * camera_height * 3 // 2
            )  # 假设视频帧大小为640x480，3个通道
            if not raw_data:
                break

            # 将原始数据转换为图像
            frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                camera_height * 3 // 2, camera_width
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            cropped_frame = frame[
                delta_h // 2 : camera_height - delta_h // 2,  # 修正高度裁剪范围
                delta_w // 2 : camera_width - delta_w // 2,  # 修正宽度裁剪范围
            ]
            queues1[i].put(cropped_frame)
            if frame_id % save_per_frames == 0:
                queue2.put(frame)
            frame_id += 1


class Detect:
    def __init__(self, queue1, queue2):
        self.queue1 = queue1
        self.queue2 = queue2
        self.model1 = YOLO(model_path1, task="obb")
        self.model2 = YOLO(model_path2, task="detect")

    def detect(self):
        global flag, target_num
        while flag:
            try:
                frame = self.queue1.get(timeout=timeout)

                # 检测靶子，获取最小外接有向矩形框和最小外接水平矩形框
                xywhrs, xyxys, xyxyxyxys = self.predict1(frame)

                height, width = frame.shape[:2]
                # 截取并旋转靶子图像
                rotated_imgs = self.crop_and_rotate(xywhrs, xyxys, frame, width, height)

                # 靶子检测结果画框
                for left, top, right, bottom in xyxys:
                    cv2.rectangle(
                        frame,
                        (int(left), int(top)),
                        (int(right), int(bottom)),
                        (0, 255, 0),
                        2,
                    )

                # 识别旋转后图片上两单位数的数值，并合并为双位数数值
                double_digits, xyxyxyxys = self.predict2(rotated_imgs, xyxyxyxys)

                # 保存旋转后图片
                lock.acquire()
                target_num += 1
                if target_num > target_saving_per_frame - 1:
                    save_img_queue.put((double_digits, rotated_imgs))
                    target_num = 0
                lock.release()

                double_digits = [_ for _ in double_digits if _ != "None"]
                # 根据双位数位置以及数值画框并标记数值
                bounded_image = self.plot_polygon(double_digits, xyxyxyxys, frame)

                # 双位数检测次数计数
                if len(double_digits) != 0:
                    lock.acquire()
                    for double_digit in double_digits:
                        idx = int(double_digit)
                        detected_digits[idx][1] += 1
                    lock.release()

                # 缓冲区帧数过多则丢帧
                if self.queue2.qsize() > 40:
                    continue
                # 检测后图片添加至队列
                self.queue2.put(bounded_image)

            except queue.Empty:
                lock.acquire()
                print("detect从camera获取帧超时")
                lock.release()
                time.sleep(1)
                continue

        lock.acquire()
        print("detect子线程关闭")
        lock.release()

    def predict1(self, frame):
        results = self.model1.predict(
            source=frame,
            imgsz=imgsz1,
            half=True,
            device="cuda:0",
            save=False,
            conf=conf1,
            iou=iou1,
            verbose=False,
        )
        r = results[0]
        xywhr = r.obb.xywhr.cpu().tolist()
        xyxy = r.obb.xyxy.cpu().tolist()
        xyxyxyxy = r.obb.xyxyxyxy.cpu().tolist()
        return xywhr, xyxy, xyxyxyxy

    def predict2(self, imgs, xs1):

        xs2 = []
        clss = []
        for img, x in zip(imgs, xs1):
            clss.append("None")
            # 检测旋转后靶子，具体获取双位数数值
            results = self.model2.predict(
                source=img,
                imgsz=imgsz2,
                half=True,
                device="cuda:0",
                save=False,
                conf=conf2,
                iou=iou2,
                verbose=False,
            )

            r = results[0]
            xywhs = r.boxes.xywh.tolist()
            classes = r.boxes.cls.tolist()
            if len(classes) == 3:
                xys = []
                xy = None
                # 拆分单位数和三角头
                for cls, xywh in zip(classes, xywhs):
                    if cls == 10:
                        xy = xywh[:2]
                    else:
                        xys.append((xywh[:2], cls))
                if xy is not None and len(xys) == 2:
                    xs2.append(x)
                    # 根据三角头和双位数相对位置得出双位数数值
                    delta_y = xys[0][0][1] - xys[1][0][1]
                    delta_x = xys[0][0][0] - xys[1][0][0]
                    sita = math.atan2(delta_y, delta_x)

                    x_0 = (xys[0][0][0] - xy[0]) * math.cos(sita) + (
                        xys[0][0][1] - xy[1]
                    ) * math.sin(sita)
                    x_1 = (xys[1][0][0] - xy[0]) * math.cos(sita) + (
                        xys[1][0][1] - xy[1]
                    ) * math.sin(sita)
                    y_0 = (xys[0][0][0] - xy[0]) * math.sin(sita) - (
                        xys[0][0][1] - xy[1]
                    ) * math.cos(sita)

                    if (x_0 - x_1) * y_0 > 0:
                        clss[-1] = f"{int(xys[0][1])}{int(xys[1][1])}"
                    else:
                        clss[-1] = f"{int(xys[1][1])}{int(xys[0][1])}"

        return clss, xs2

    def crop_and_rotate(self, xywhr, xyxy, image, width, height, scale=1.0):
        rotated_imgs = []
        for (x, y, rw, rh, r), (left, top, right, bottom) in zip(xywhr, xyxy):
            # 截取长宽
            w = right - left
            h = bottom - top

            left -= w * (coefficient1 - 1) / 2
            right += w * (coefficient1 - 1) / 2
            top -= h * (coefficient1 - 1) / 2
            bottom += h * (coefficient1 - 1) / 2

            left = left if left >= 0 else 0
            top = top if top >= 0 else 0
            right = right if right <= width else width
            bottom = bottom if bottom <= height else height

            w, h = int(right - left), int(bottom - top)

            # 旋转角度
            rad = r if rw < rh else r - 0.5 * math.pi
            degree = math.degrees(rad)

            nw = (abs(np.sin(rad) * h) + abs(np.cos(rad) * w)) * scale
            nh = (abs(np.cos(rad) * h) + abs(np.sin(rad) * w)) * scale
            rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), degree, scale)
            rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
            rot_mat[0, 2] += rot_move[0]
            rot_mat[1, 2] += rot_move[1]
            # 图像处理
            cropped_img = image[int(top) : int(bottom), int(left) : int(right)]
            rotated_img = cv2.warpAffine(
                cropped_img,
                rot_mat,
                (int(math.ceil(nw)), int(math.ceil(nh))),
                flags=cv2.INTER_LINEAR,
            )
            rotated_imgs.append(rotated_img)
        return rotated_imgs

    def plot_polygon(self, double_digits, xyxyxyxys, image):
        # 在原图片上绘制框并标上具体双位数数值
        texts = [f"{dg}" for dg in double_digits]
        img = plot_polygon(image, xyxyxyxys, texts, 0.001)

        return img


class SaveImg:
    def __init__(self, queue1):
        self.queue1 = queue1

    def save_disk(self):
        global flag
        while flag:
            try:
                t = time.time()
                double_digits, rotated_imgs = self.queue1.get(timeout=timeout)
                for idx, (digit, img) in enumerate(zip(double_digits, rotated_imgs)):
                    cv2.imwrite(
                        f"./output/{now_time}/{t}-{idx}-[{digit}].jpg",
                        letterbox(img, imgsz2),
                    )
            except queue.Empty:
                lock.acquire()
                print("save_img从detect2获取帧超时")
                lock.release()
                time.sleep(1)
                continue
        lock.acquire()
        print("save_img子线程关闭")
        lock.release()


class Show:
    def __init__(self, queues, mode):
        self.queues = queues
        self.mode = mode
        self.frames_num = 0

    def show(self):
        time1 = time.time()
        self.push_live()
        lock.acquire()
        print(f"total_fps:{self.frames_num / (time.time() - time1)}")
        print("show主线程关闭")
        lock.release()

    def push_live(self):
        if self.mode == "rtsp":
            live_format = "rtsp"
        elif self.mode == "rtmp":
            live_format = "flv"
        elif self.mode == "srt":
            live_format = "mpegts"
        else:
            live_format = ""
            lock.acquire()
            print("推流格式设置不正确")
            lock.release()

        # 推流到指定rtsp地址
        command = [
            "ffmpeg",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "{}x{}".format(show_width, show_height),
            "-r",
            str(live_fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-pixel_format",
            "yuv420p",
            "-b:v",
            bitrate,
            "-rc_mode",
            rc_mode,
            "-f",
            live_format,
            liveUrl,
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as p:
            fps = 0
            frames_num = 0
            fps_update_before = time.time()
            show_flag = True

            while show_flag:
                for i in range(worker_num):
                    try:
                        frame = self.queues[i].get(timeout=timeout)
                        frame = cv2.resize(
                            frame,
                            (show_width, show_height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    except queue.Empty:
                        show_flag = False
                        break
                    frames_num += 1
                    self.frames_num += 1
                    # frame = self.text(frame)  # 附上双位数检测次数及最终中位数
                    frame = self.update_fps(frame, fps)  # 附上fps
                    p.stdin.write(frame.tobytes())  # 写入ffmpeg

                # fps计算
                if frames_num > fps_per_frames:
                    fps = frames_num / (time.time() - fps_update_before)
                    frames_num = 0
                    fps_update_before = time.time()

    def update_fps(
        self, img, fps, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2
    ):
        # 左上角显示FPS
        cv2.putText(
            img,
            f"FPS:{fps:.2f}",
            (10, 30),
            font,
            font_scale,
            (0, 0, 255),
            font_thickness,
        )
        return img


def save_video(save_frame_queue):
    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    save_fps = live_fps // save_per_frames
    out = cv2.VideoWriter(
        f"./output/{now_time}/{time.time()}.avi",
        fourcc,
        save_fps,
        (camera_width, camera_height),
    )
    while True:
        try:
            frame = save_frame_queue.get(timeout=timeout)
            out.write(frame)
        except queue.Empty:
            break
    out.release()

    with open(f"./output/{now_time}/detected_num.txt", mode="w") as f:
        lock.acquire()
        dgs = sorted(detected_digits, reverse=True, key=lambda i: i[1])
        lock.release()

        for digit, number in dgs:
            f.write(f"{digit}\t{number}\n")


class Count:
    def __init__(self, frequence, queues=tuple(), queuess=tuple()):
        self.frequence = frequence
        self.queues = queues
        self.queuess = queuess

    def count(self):
        global flag
        while flag:
            time.sleep(self.frequence)
            lock.acquire()
            for name, queue in self.queues:
                print(name, queue.qsize())
            for name, queues in self.queuess:
                print(name, self.count_queues(queues) / len(queues))
            print("\n")
            lock.release()

    def count_queues(self, queues):
        num = 0
        for _ in queues:
            num += _.qsize()
        return num


if __name__ == "__main__":
    # 创建保存文件夹
    now_time = cv2.getTickCount()
    os.makedirs(f"./output/{now_time}")

    # 视频流队列
    detect_queues = [queue.Queue(maxsize=maxsize) for _ in range(worker_num)]
    show_queues = [queue.Queue() for _ in range(worker_num)]
    save_img_queue = queue.Queue()
    save_video_queue = queue.Queue()
    # 检测结果储存处
    target_num = 0
    detected_digits = [[i, 0] for i in range(100)]

    flag = True
    lock = threading.Lock()
    # camera线程
    t1 = threading.Thread(target=camera, args=(detect_queues, save_video_queue))
    # detect线程
    tasks1 = [
        threading.Thread(target=Detect(detect_queues[idx], show_queues[idx]).detect)
        for idx in range(worker_num)
    ]
    # save_img线程
    t3 = threading.Thread(target=SaveImg(save_img_queue).save_disk, args=tuple())
    # save_video线程
    save_flag = True
    t4 = threading.Thread(target=save_video, args=(save_video_queue,))
    # count线程
    t5 = threading.Thread(
        target=Count(
            count_frequence,
            (
                ("save_img", save_img_queue),
                ("save_video", save_video_queue),
            ),
            (
                ("detect", detect_queues),
                ("show", show_queues),
            ),
        ).count
    )

    # 开启子线程
    t1.start()
    for i in tasks1:
        i.start()
    t3.start()
    t4.start()
    t5.start()

    # 主线程展示视频帧
    Show(show_queues, show_mode).show()

    # show线程结束后，关闭各个线程
    flag = False
    t5.join()
    print("count线程关闭")
    t1.join()
    print("camera线程关闭")
    for i in tasks1:
        i.join()
    print("detect线程关闭")
    t3.join()
    print("save_img线程关闭")
    t4.join()
    print("save_video线程关闭")
