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
camera_frequence = 1 / 60  # 每几秒读取work_num_1帧

model_7_path = './yolov8/detect_400dataset_imgsz640_v8n_SGD/weights/best.engine'
model_5_path = './yolov8/5_1100dataset_imgsz160_v8n_Adam/weights/best.engine'
imgsz1 = 640
imgsz2 = 160
conf = 0.5
iou = 0.5
coefficient1 = 0.6  # 靶子截取范围系数
worker_num_1 = 1  # 同时开启work_num_1个detect1线程
worker_num_2 = 1  # 同时开启work_num_2个detect2线程

target_saving_per_frame = 1  # 每多少帧检测到靶子便保存靶子图片
save_img_mode = 'email'  # 靶子图片保存模式 disk or email
mail_mode = 'netease'  # 邮箱设置
send_per_imgs = 100  # 每多少张靶子图片发送一次邮箱

save_width, save_height = 640, 360  # show和save的帧大小
fps_per_frames = 60  # 每几帧计算一次fps
show_mode = 'imshow'  # 预览模式 imshow or rtmp

liveUrl = ""
preset = 'medium'
live_fps = 30  # 直播流fps

count_frequence = 1  # 每几秒显示一次各队列数量

timeout = 3  # 线程多少秒内无法获取传来的帧，结束线程


def camera(queues, path, frequence, worker_num):
    global model1_is_ready, model2_is_ready
    while not model1_is_ready or not model2_is_ready:
        time.sleep(1)

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


class Detect1(object):
    def detect(self, queue1, queue2, timeout):
        model1 = YOLO(model_7_path, task='detect')

        global model1_is_ready
        model1_is_ready = True

        global flag
        while flag:
            try:
                frame = queue1.get(timeout=timeout)

                # 检测靶子，获取最小外接有向矩形框和最小外接水平矩形框
                xywhs = self.predict(imgsz1, model1, frame, conf, iou)
                height, width = frame.shape[:2]

                # 截取并旋转靶子图像
                cropped_imgs = self.crop(xywhs, frame, width, height)

                # 靶子检测结果画框
                for x, y, w, h in xywhs:
                    cv2.rectangle(frame, (int(x - w * coefficient1), int(y - h * coefficient1)),
                                  (int(x + w * coefficient1), int(y + h * coefficient1)),
                                  (0, 255, 0), 2)

                queue2.put((frame, xywhs, cropped_imgs))

            except queue.Empty:
                lock.acquire()
                print('detect1子线程关闭')
                lock.release()
                break

    def predict(self, imgsz, model, img, conf, iou):
        results = model.predict(source=img, imgsz=imgsz, half=True, device='cuda:0',
                                save=False, conf=conf, iou=iou, verbose=False)
        r = results[0]
        xywhs = r.boxes.xywh.tolist()
        return xywhs

    def crop(self, xywh, image, width, height):
        cropped_imgs = []
        for x, y, w, h in xywh:
            # 截取长宽
            left = x - w * coefficient1
            top = y - h * coefficient1
            right = x + w * coefficient1
            bottom = y + h * coefficient1
            left = left if left >= 0 else 0
            top = top if top >= 0 else 0
            right = right if right <= width else width
            bottom = bottom if bottom <= height else height

            # 图像处理
            cropped_img = image[int(top):int(bottom), int(left):int(right)].copy()
            cropped_imgs.append(cropped_img)
        return cropped_imgs


def transit(queues1, queues2, timout):
    global model1_is_ready, model2_is_ready
    while not model1_is_ready or not model2_is_ready:
        time.sleep(1)

    num = 0
    global flag
    while flag:
        try:
            frame, xyxyxyxys, rotated_imgs = queues1[num % worker_num_1].get(timeout=timout)
            queues2[num % worker_num_2].put((frame, xyxyxyxys, rotated_imgs))
            num += 1
        except queue.Empty:
            lock.acquire()
            print('transit子线程关闭')
            lock.release()
            break


class Detect2(object):
    def detect(self, queue1, queue2, queue3, timeout):
        model2 = YOLO(model_5_path, task='detect')

        global model2_is_ready
        model2_is_ready = True

        global flag, target_num
        while flag:
            try:
                frame, xywhs, cropped_imgs = queue1.get(timeout=timeout)
                # 识别旋转后图片上两单位数的数值，并合并为双位数数值
                double_digits, xywhs = self.predict(imgsz2, model2, cropped_imgs, xywhs, conf, iou)

                # 保存旋转后图片
                lock.acquire()
                if len(cropped_imgs) > 0:
                    target_num += 1
                    if target_num > target_saving_per_frame:
                        queue3.put((double_digits, cropped_imgs))
                        target_num = 0
                lock.release()

                double_digits = [_ for _ in double_digits if _ != 'None']
                # 根据双位数位置以及数值画框并标记数值
                bounded_image = self.plot_bbox(double_digits, xywhs, frame)

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
                print('detect2子线程关闭')
                lock.release()
                break

    def predict(self, imgsz, model, imgs, _s, conf, iou):

        __s = []
        clss = []
        for idx, (img, _) in enumerate(zip(imgs, _s)):
            clss.append('None')
            # 检测旋转后靶子，具体获取双位数数值
            results = model.predict(source=img, imgsz=imgsz, half=True, device='cuda:0',
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
                        clss[-1] = (f'{int(xys[0][1])}{int(xys[1][1])}')
                    else:
                        clss[-1] = (f'{int(xys[1][1])}{int(xys[0][1])}')

        return clss, __s

    def plot_bbox(self, double_digits, xywhs, image, coefficient=0.005):
        # 在原图片上绘制框并标上具体双位数数值
        h, w = image.shape[:2]
        annotator = Annotator(image, line_width=int(min(w, h) * coefficient))
        for dg, xywh in zip(double_digits, xywhs):
            x, y, w, h = xywh
            xyxy = [x - w / 2, y - h / 2, x + w / 2, y + w / 2]
            annotator.box_label(xyxy, label=f'{dg}:{int(w)}*{int(h)}')

        return annotator.im


class SaveImg(object):
    def save(self, mode, queue1, timeout):
        global model1_is_ready, model2_is_ready
        while not model1_is_ready or not model2_is_ready:
            time.sleep(1)

        timeout += target_saving_per_frame / live_fps
        if mode == 'disk':
            self.save_disk(queue1, timeout)
        elif mode == 'email':
            self.save_email(queue1, timeout)

    def save_disk(self, queue1, timeout):
        global flag
        while flag:
            try:
                t = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                double_digits, rotated_imgs = queue1.get(timeout=timeout)
                for idx, (digit, img) in enumerate(zip(double_digits, rotated_imgs)):
                    cv2.imwrite(f'output/{now_time}/{t}-{idx}-[{digit}].jpg', img)
            except queue.Empty:
                lock.acquire()
                print('save_img子线程关闭')
                lock.release()
                break

    def save_email(self, queue1, timeout):
        import smtplib
        if mail_mode == 'netease':
            from_email = 'm15363428961@163.com'
            to_email = 'm15363428961@163.com'
            smtp_server = 'smtp.163.com'
            smtp_port = 25
            key = os.environ.get('smtp_netease_key')
        else:
            from_email = '1336458913@qq.com'
            to_email = '1336458913@qq.com'
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
                    digits, imgs = queue1.get(timeout=timeout)
                    all_None = all(digit == 'None' for digit in digits)
                    if not all_None:
                        infos.extend([_ for _ in zip(digits, imgs)])
                    if len(infos) > send_per_imgs:
                        msg = self.generate_email(infos, from_email, to_email)
                        server.sendmail(from_email, to_email, msg.as_string())
                        infos = []
                except queue.Empty:
                    lock.acquire()
                    print('save_img子线程关闭')
                    lock.release()
                    break
            if len(infos) > 0:
                msg = self.generate_email(infos, from_email, to_email)
                server.sendmail(from_email, to_email, msg.as_string())

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
        middle_dg = sorted(sorted(detected_digits, reverse=True, key=lambda i: i[1])[:3], key=lambda i: i[0])[1][0]
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


class Show(object):
    def show(self, mode, queues, queue1, worker_num, timeout):
        global model1_is_ready, model2_is_ready
        while not model1_is_ready or not model2_is_ready:
            time.sleep(1)

        if mode == 'imshow':
            self.imshow(queues, queue1, worker_num, timeout)
        elif mode in ('rtsp', 'rtmp'):
            self.push_live(mode, queues, queue1, worker_num, timeout)

    def imshow(self, queues, queue1, worker_num, timeout):
        # 主进程cv2.imshow窗口
        cv2.namedWindow('0', cv2.WINDOW_AUTOSIZE)
        fps = 0
        frames_num = 0
        fps_update_before = cv2.getTickCount()
        show_flag = True

        while show_flag:
            for i in range(worker_num):
                try:
                    frame = queues[i].get(timeout=timeout)
                except queue.Empty:
                    show_flag = False
                    break
                frames_num += 1
                frame = self.text(frame, detected_digits)  # 附上双位数检测次数及最终中位数
                frame = self.update_fps(frame, fps)  # 附上fps
                cv2.imshow('0', frame)
                queue1.put(frame)  # 添加至视频保存队列

                # fps计算
                if frames_num > fps_per_frames:
                    fps = frames_num / ((cv2.getTickCount() - fps_update_before) / cv2.getTickFrequency())
                    frames_num = 0
                    fps_update_before = cv2.getTickCount()

            # 检测到q，关闭窗口和所有进程
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                show_flag = False

    def push_live(self, mode, queues, queue1, worker_num, timeout):
        if mode == 'rtsp':
            live_format = 'rtsp'
        elif mode == 'rtmp':
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
                for i in range(worker_num):
                    try:
                        frame = queues[i].get(timeout=timeout)
                    except queue.Empty:
                        show_flag = False
                        break
                    frames_num += 1
                    frame = self.text(frame, detected_digits)  # 附上双位数检测次数及最终中位数
                    frame = self.update_fps(frame, fps)  # 附上fps
                    p.stdin.write(frame.tobytes())
                    queue1.put(frame)  # 添加至视频保存队列

                    # fps计算
                    if frames_num > fps_per_frames:
                        fps = frames_num / ((cv2.getTickCount() - fps_update_before) / cv2.getTickFrequency())
                        frames_num = 0
                        fps_update_before = cv2.getTickCount()

                # 检测到q，关闭窗口和所有进程
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    show_flag = False

    def text(self, img, detected_digits, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.5, font_thickness=2):
        # 右上角显示已检测双位数次数
        dgs = sorted(detected_digits, reverse=True, key=lambda i: i[1])
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


def save_video(save_frame_queue, timeout):
    global model1_is_ready, model2_is_ready
    while not model1_is_ready or not model2_is_ready:
        time.sleep(1)

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
        for digit, number in sorted(detected_digits, reverse=True, key=lambda i: i[1]):
            f.write(f'{digit}\t{number}\n')


class Count(object):
    def count(self, queues1, queues2, queues3, queues4, queue5, queue6, frequence):

        global flag
        while flag:
            time.sleep(frequence)
            nums = [self.count_queues(_) for _ in (queues1, queues2, queues3, queues4)]
            num4 = queue5.qsize()
            num5 = queue6.qsize()
            lock.acquire()
            print('input_queues', nums[0])
            print('transit_queues', nums[1])
            print('input_queues1', nums[2])
            print('show_queues', nums[3])
            print('save_img_queue', num4)
            print('save_queue', num5)
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
    input_queues = [queue.Queue(maxsize=10) for i in range(worker_num_1)]
    transit_queues = [queue.Queue() for i in range(worker_num_1)]
    input_queues1 = [queue.Queue() for i in range(worker_num_2)]
    show_queues = [queue.Queue() for i in range(worker_num_2)]
    save_img_queue = queue.Queue()
    save_video_queue = queue.Queue()

    flag = True
    lock = threading.Lock()
    model1_is_ready = False
    model2_is_ready = False
    # camera线程
    t1 = threading.Thread(target=camera, args=(input_queues, cap_path, camera_frequence, worker_num_1))
    t1.start()

    # detect1线程
    tasks1 = [threading.Thread(target=Detect1().detect,
                               args=(
                                   input_queues[idx], transit_queues[idx], timeout
                               )) for idx in range(worker_num_1)]
    for i in tasks1:
        i.start()

    # transit线程
    t2 = threading.Thread(target=transit, args=(transit_queues, input_queues1, timeout))
    t2.start()

    # detect2线程
    target_num = 0
    detected_digits = [[i, 0] for i in range(100)]  # 检测结果储存处
    tasks2 = [threading.Thread(target=Detect2().detect,
                               args=(
                                   input_queues1[idx], show_queues[idx], save_img_queue, timeout
                               )) for idx in range(worker_num_2)]
    for i in tasks2:
        i.start()
    # save_img线程
    t3 = threading.Thread(target=SaveImg().save, args=(save_img_mode, save_img_queue, timeout))
    t3.start()

    # save_video线程
    save_flag = True
    t4 = threading.Thread(target=save_video, args=(save_video_queue, timeout))
    t4.start()

    # count线程
    t5 = threading.Thread(target=Count().count,
                          args=(
                              input_queues, transit_queues, input_queues1, show_queues, save_img_queue,
                              save_video_queue,
                              count_frequence))
    t5.start()

    # 主线程展示视频帧
    Show().show(show_mode, show_queues, save_video_queue, worker_num_2, timeout)

    # 关闭各个线程
    flag = False
    t5.join()
    print('count线程关闭')
    t1.join()
    print('camera线程关闭')
    for i in tasks1:
        i.join()
    print('detect1线程关闭')
    t2.join()
    print('transit线程关闭')
    for i in tasks2:
        i.join()
    print('detect2线程关闭')
    t3.join()
    print('save_img线程关闭')
    while True:
        if save_video_queue.empty():
            save_flag = False
            t4.join()
            print('save_video线程关闭')
            break
