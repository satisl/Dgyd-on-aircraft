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

cap_path = './benchmark/record2.mp4'
camera_frequence = 1 / 150  # 每几秒读取work_num帧
maxsize = 20  # camera线程帧队列最大数量

model_7_path = './yolov8/obb_400dataset_imgsz640_v8n_SGD/weights/best.engine'
model_5_path = './yolov8/5_1100dataset_imgsz160_v8n_Adam/weights/best.engine'
imgsz1 = 640
imgsz2 = 160
conf = 0.5
iou = 0.5
coefficient1 = 1.2  # 靶子截取范围系数
worker_num = 8  # 同时开启work_num个detect线程

target_saving_per_frame = 1  # 每多少帧检测到靶子便保存靶子图片
save_img_mode = 'disk'  # 靶子图片保存模式 disk or email
mail_mode = 'netease'  # 邮箱设置
send_per_imgs = 100  # 每多少张靶子图片发送一次邮箱

save_width, save_height = 640, 360  # show和save的帧大小
fps_per_frames = 300  # 每几帧计算一次fps
show_mode = 'imshow'  # 预览模式 imshow or rtmp

liveUrl = ""
preset = 'slow'
live_fps = 90  # 直播流fps

count_frequence = 1  # 每几秒显示一次各队列数量

timeout = 6


def camera(queues, path, frequence):
    cap = cv2.VideoCapture(path)
    global flag
    while cap.isOpened() and flag:
        time.sleep(frequence)
        for i in range(worker_num):
            success, frame = cap.read()
            if success:
                queues[i].put(frame)
            else:
                cap.release()
                break
    # import numpy as np
    # frame = cv2.imdecode(np.fromfile(r'', dtype=np.uint8),
    #                      cv2.IMREAD_COLOR)
    # while flag:
    #     time.sleep(frequence)
    #     for i in range(worker_num):
    #         queues[i].put(frame.copy())


class Detect:
    def __init__(self, queue1, queue2):
        self.queue1 = queue1
        self.queue2 = queue2
        self.model1 = YOLO(model_7_path, task='obb')
        self.model2 = YOLO(model_5_path, task='detect')

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
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

                # 识别旋转后图片上两单位数的数值，并合并为双位数数值
                double_digits, xyxyxyxys = self.predict2(rotated_imgs, xyxyxyxys)

                # 保存旋转后图片
                lock.acquire()
                target_num += 1
                if target_num > target_saving_per_frame:
                    save_img_queue.put((double_digits, rotated_imgs))
                    target_num = 0
                lock.release()

                double_digits = [_ for _ in double_digits if _ != 'None']
                # 根据双位数位置以及数值画框并标记数值
                bounded_image = self.plot_polygon(double_digits, xyxyxyxys, frame)

                # 检测后图片添加至队列
                self.queue2.put(cv2.resize(bounded_image, (save_width, save_height)))

                # 双位数检测次数计数
                if len(double_digits) != 0:
                    lock.acquire()
                    for double_digit in double_digits:
                        idx = int(double_digit)
                        detected_digits[idx][1] += 1
                    lock.release()

            except queue.Empty:
                lock.acquire()
                print('detect从camera获取帧超时')
                lock.release()
                continue
        lock.acquire()
        print('detect子线程关闭')
        lock.release()

    def predict1(self, frame):
        results = self.model1.predict(source=frame, imgsz=imgsz1, half=True, device='cuda:0',
                                      save=False, conf=conf, iou=iou, verbose=False)
        r = results[0]
        xywhr = r.obb.xywhr.cpu().tolist()
        xyxy = r.obb.xyxy.cpu().tolist()
        xyxyxyxy = r.obb.xyxyxyxy.cpu().tolist()
        return xywhr, xyxy, xyxyxyxy

    def predict2(self, imgs, _s):

        __s = []
        clss = []
        for idx, (img, _) in enumerate(zip(imgs, _s)):
            clss.append('None')
            # 检测旋转后靶子，具体获取双位数数值
            results = self.model2.predict(source=img, imgsz=imgsz2, half=True, device='cuda:0',
                                          save=False, conf=conf, iou=iou, verbose=False)

            r = results[0]
            xywh = r.boxes.xywh.tolist()
            cls = r.boxes.cls.tolist()

            if len(xywh) == 3:
                xys = []
                xy = None
                # 拆分单位数和三角头
                for i, j in zip(cls, xywh):
                    if i == 10:
                        xy = j[:2]
                    else:
                        xys.append((j[:2], i))

                if xy is not None and len(xys) == 2:
                    __s.append(_)
                    # 根据三角头和双位数相对位置得出双位数数值
                    delta_y = xys[0][0][1] - xys[1][0][1]
                    delta_x = xys[0][0][0] - xys[1][0][0]
                    sita = math.atan2(delta_y, delta_x)

                    x_0 = (xys[0][0][0] - xy[0]) * math.cos(sita) + (xys[0][0][1] - xy[1]) * math.sin(sita)
                    x_1 = (xys[1][0][0] - xy[0]) * math.cos(sita) + (xys[1][0][1] - xy[1]) * math.sin(sita)
                    y_0 = (xys[0][0][0] - xy[0]) * math.sin(sita) - (xys[0][0][1] - xy[1]) * math.cos(sita)

                    if (x_0 - x_1) * y_0 > 0:
                        clss[-1] = f'{int(xys[0][1])}{int(xys[1][1])}'
                    else:
                        clss[-1] = f'{int(xys[1][1])}{int(xys[0][1])}'

        return clss, __s

    def crop_and_rotate(self, xywhr, xyxy, image, width, height):
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

    def plot_polygon(self, double_digits, xyxyxyxys, image, coefficient=0.005):
        # 在原图片上绘制框并标上具体双位数数值
        h, w = image.shape[:2]
        annotator = Annotator(image, line_width=int(min(w, h) * coefficient))
        for dg, xyxyxyxy in zip(double_digits, xyxyxyxys):
            annotator.box_label(xyxyxyxy, label=f'{dg}', rotated=True)

        return annotator.im


class SaveImg:
    def __init__(self, queue1):
        self.queue1 = queue1

    def save(self, mode):
        if mode == 'disk':
            self.save_disk()
        elif mode == 'email':
            self.save_email()

    def save_disk(self):
        global flag
        while flag:
            try:
                t = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                double_digits, rotated_imgs = self.queue1.get(timeout=timeout)
                for idx, (digit, img) in enumerate(zip(double_digits, rotated_imgs)):
                    cv2.imwrite(f'output/{now_time}/{t}-{idx}-[{digit}].jpg', img)
            except queue.Empty:
                lock.acquire()
                print('save_img从detect2获取帧超时')
                lock.release()
                continue
        lock.acquire()
        print('save_img子线程关闭')
        lock.release()

    def save_email(self):
        import smtplib
        if mail_mode == 'netease':
            from_email = ''
            to_email = ''
            smtp_server = 'smtp.163.com'
            smtp_port = 25
            key = os.environ.get('smtp_netease_key')
        else:
            from_email = ''
            to_email = ''
            smtp_server = 'smtp.qq.com'
            smtp_port = 587
            key = os.environ.get('smtp_qq_key')

        infos = []
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(from_email, key)
            global flag
            while flag:
                try:
                    digits, imgs = self.queue1.get(timeout=timeout)
                    all_None = all(digit == 'None' for digit in digits)  # 筛选出更有可能的靶标截图
                    if not all_None:
                        infos.extend([_ for _ in zip(digits, imgs)])
                    if len(infos) > send_per_imgs:
                        msg = self.generate_email(infos, from_email, to_email)
                        server.sendmail(from_email, to_email, msg.as_string())
                        infos = []
                except queue.Empty:
                    lock.acquire()
                    print('save_img从detect2获取帧超时')
                    lock.release()
                    continue
            if len(infos) > 0:
                msg = self.generate_email(infos, from_email, to_email)
                server.sendmail(from_email, to_email, msg.as_string())
            lock.acquire()
            print('save_img子线程关闭')
            lock.release()

    def generate_email(self, infos, from_email, to_email):
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.image import MIMEImage
        html = '''<html>
                    <head>
                    <style>
                    /* 设置图片容器的样式 */
                    .image-container {
                            display: flex; /* 使用 flex 布局 */
                            flex-wrap: wrap; /* 自动换行 */
                            justify-content: center; /* 水平居中 */
                    }

                    /* 设置每张图片的样式 */
                            .image-container img {
                            width: [target_width]px; /* 设置图片宽度 */
                            height: [target_width]px; /* 高度自适应 */
                            margin: 5px; /* 图片间距 */
                    }
                    </style>
                    </head>
                    <body>
                    <div class="image-container">'''.replace('[target_width]', str(imgsz2))

        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        lock.acquire()
        middle_dg = sorted(sorted(detected_digits, reverse=True, key=lambda i: i[1])[:3], key=lambda i: i[0])[1][0]
        lock.release()
        msg['Subject'] = f'中位数:{middle_dg} {datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        pic_inline = ''
        for idx, (digit, img) in enumerate(infos):
            # 改变图片尺寸
            resized_img = cv2.resize(img, (imgsz2, imgsz2))
            _, img_data = cv2.imencode('.jpg', resized_img)

            # 添加图片附件
            image = MIMEImage(img_data.tobytes())
            image.add_header('Content-ID', f'<image{idx}.jpg>')
            msg.attach(image)
            tmp_pic_inline = f'<div><p>{digit}</p><img src="cid:image{idx}.jpg" alt="image{idx}.jpg"></div>'
            pic_inline += tmp_pic_inline
        html = html + pic_inline + '</div></body></html>'
        content = MIMEText(html, 'html', 'utf-8')
        msg.attach(content)
        return msg


def transit(queues1, queue2):
    while flag:
        for i in range(worker_num):
            try:
                frame = queues1[i].get(timeout=timeout)
            except queue.Empty:
                lock.acquire()
                print('transit从detect获取帧超时')
                lock.release()
                continue
            queue2.put(frame)


class Show:
    def __init__(self, queue1, queue2, mode):
        self.queue1 = queue1
        self.queue2 = queue2
        self.mode = mode
        self.frames_num = 0

    def show(self):
        time1 = time.time()
        if self.mode == 'imshow':
            self.imshow()
        elif self.mode in ('rtsp', 'rtmp'):
            self.push_live()
        lock.acquire()
        print(f'total_fps:{self.frames_num / (time.time() - time1)}')
        print('show主线程关闭')
        lock.release()

    def imshow(self):
        # 主进程cv2.imshow窗口
        cv2.namedWindow('0', cv2.WINDOW_AUTOSIZE)
        fps = 0
        frames_num = 0
        fps_update_before = cv2.getTickCount()
        show_flag = True

        while show_flag:
            try:
                frame = self.queue1.get(timeout=timeout)
            except queue.Empty:
                break
            frames_num += 1
            self.frames_num += 1
            frame = self.text(frame)  # 附上双位数检测次数及最终中位数
            frame = self.update_fps(frame, fps)  # 附上fps
            cv2.imshow('0', frame)
            self.queue2.put(frame)  # 添加至视频保存队列

            # fps计算
            if frames_num > fps_per_frames:
                fps = frames_num / ((cv2.getTickCount() - fps_update_before) / cv2.getTickFrequency())
                frames_num = 0
                fps_update_before = cv2.getTickCount()

            # 检测到q，关闭窗口和所有进程
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                show_flag = False

    def push_live(self):
        if self.mode == 'rtsp':
            live_format = 'rtsp'
        elif self.mode == 'rtmp':
            live_format = 'flv'
        else:
            live_format = ''
            lock.acquire()
            print('推流格式设置不正确')
            lock.release()

        # 推流到指定rtsp地址
        command = ['ffmpeg',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', "{}x{}".format(save_width, save_height),
                   '-r', str(live_fps),
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', preset,
                   '-f', live_format,
                   liveUrl]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as p:
            fps = 0
            frames_num = 0
            fps_update_before = cv2.getTickCount()
            show_flag = True

            while show_flag:
                try:
                    frame = self.queue1.get(timeout=timeout)
                except queue.Empty:
                    break
                frames_num += 1
                self.frames_num += 1
                frame = self.text(frame)  # 附上双位数检测次数及最终中位数
                frame = self.update_fps(frame, fps)  # 附上fps
                p.stdin.write(frame.tobytes())
                self.queue2.put(frame)  # 添加至视频保存队列

                # fps计算
                if frames_num > fps_per_frames:
                    fps = frames_num / ((cv2.getTickCount() - fps_update_before) / cv2.getTickFrequency())
                    frames_num = 0
                    fps_update_before = cv2.getTickCount()

    def text(self, img, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.5, font_thickness=2):
        # 右上角显示已检测双位数次数
        lock.acquire()
        dgs = sorted(detected_digits, reverse=True, key=lambda i: i[1])
        lock.release()
        width = img.shape[1]
        for i, j in enumerate(dgs[:4]):
            dg_text = f'{j[0]}:{j[1]}'
            dg_text_width, dg_text_height = cv2.getTextSize(dg_text, font, font_scale, font_thickness)[0]
            dg_text_x, dg_text_y = int(width - dg_text_width), int((i + 1) * dg_text_height * 2)
            cv2.putText(img, dg_text, (dg_text_x, dg_text_y), font,
                        font_scale,
                        (0, 0, 255),
                        font_thickness)

        # 右上角显示已检测三个双位数的中位数
        dg_text = str(sorted(dgs[:3], key=lambda i: i[0])[1][0])
        dg_text_width, dg_text_height = cv2.getTextSize(dg_text, font, font_scale, font_thickness)[0]
        dg_text_x, dg_text_y = int(width - dg_text_width), int(5 * dg_text_height * 2)
        cv2.putText(img, dg_text, (dg_text_x, dg_text_y), font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness)
        return img

    def update_fps(self, img, fps, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):
        # 左上角显示FPS
        cv2.putText(img, f'FPS:{fps:.2f}', (10, 30), font, font_scale, (0, 0, 255), font_thickness)
        return img


def save_video(save_frame_queue):
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(f'output/{now_time}/{cv2.getTickCount()}.avi', fourcc, live_fps,
                          (save_width, save_height))
    global save_flag
    while save_flag:
        try:
            out.write(save_frame_queue.get(timeout=timeout))
        except queue.Empty:
            break
    out.release()

    with open(f'output/{now_time}/detected_num.txt', mode='w') as f:
        lock.acquire()
        dgs = sorted(detected_digits, reverse=True, key=lambda i: i[1])
        lock.release()

        for digit, number in dgs:
            f.write(f'{digit}\t{number}\n')


class Count:
    def __init__(self, a, b, c, d, e, f):
        self.queues1 = a
        self.queues2 = b
        self.queue3 = c
        self.queue4 = d
        self.queue5 = e
        self.frequence = f

    def count(self):
        global flag
        while flag:
            time.sleep(self.frequence)
            nums = [self.count_queues(_) for _ in (self.queues1, self.queues2)]
            nums.append(self.queue3.qsize())
            nums.append(self.queue4.qsize())
            nums.append(self.queue5.qsize())
            lock.acquire()
            print('camera->detect', nums[0] / worker_num)
            print('detect->transit', nums[1])
            print('detect->save_img', nums[2])
            print('transit->show', nums[3])
            print('show->save_video', nums[4])
            print('\n')
            lock.release()

    def count_queues(self, queues):
        num = 0
        for _ in queues:
            num += _.qsize()
        return num


if __name__ == '__main__':
    # 创建保存文件夹
    now_time = cv2.getTickCount()
    os.makedirs(f'output/{now_time}')

    # 视频流队列
    detect_queues = [queue.Queue(maxsize=maxsize) for i in range(worker_num)]
    transit_queues = [queue.Queue() for i in range(worker_num)]
    show_queue = queue.Queue()
    save_img_queue = queue.Queue()
    save_video_queue = queue.Queue()
    # 检测结果储存处
    target_num = 0
    detected_digits = [[i, 0] for i in range(100)]

    flag = True
    lock = threading.Lock()
    # camera线程
    t1 = threading.Thread(target=camera, args=(detect_queues, cap_path, camera_frequence))
    # detect线程
    tasks1 = [threading.Thread(target=Detect(detect_queues[idx], transit_queues[idx]).detect) for idx in
              range(worker_num)]
    # save_img线程
    t3 = threading.Thread(target=SaveImg(save_img_queue).save, args=(save_img_mode,))
    # transit线程
    t2 = threading.Thread(target=transit, args=(transit_queues, show_queue))
    # save_video线程
    save_flag = True
    t4 = threading.Thread(target=save_video, args=(save_video_queue,))
    # count线程
    t5 = threading.Thread(
        target=Count(detect_queues, transit_queues, save_img_queue, show_queue, save_video_queue,
                     count_frequence).count)

    # 开启子线程
    t1.start()
    for i in tasks1:
        i.start()
    t3.start()
    t2.start()
    t4.start()
    t5.start()

    # 主线程展示视频帧
    Show(show_queue, save_video_queue, show_mode).show()

    # show线程结束后，关闭各个线程
    flag = False
    t5.join()
    print('count线程关闭')
    t1.join()
    print('camera线程关闭')
    for i in tasks1:
        i.join()
    print('detect线程关闭')
    t3.join()
    print('save_img线程关闭')
    t2.join()
    print('transit线程关闭')
    while True:
        if save_video_queue.empty():
            save_flag = False
            t4.join()
            print('save_video线程关闭')
            break
