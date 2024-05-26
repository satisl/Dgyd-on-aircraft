from ultralytics import YOLO
import os
import math
import time

dst = r'E:\python\yolov8\datasets\123\for training\images\train'
model = YOLO(r'E:\python\yolov8\yolov8\123_imgsz1280_v8n_SGD(overfit)\weights\best.pt')

# imgs = os.listdir(dst)
imgs = ['00038_1.jpg', '00042_6.jpg']

coefficient1 = 1.5
coefficient2 = 3.5

for img in imgs:
    t1 = time.time()
    results = model.predict(source=f'{dst}/{img}',
                            imgsz=1280, half=True, show_labels=True,
                            show_conf=True, show_boxes=True,
                            line_width=1, save=False, conf=0.5)

    for r in results:
        xywh = r.boxes.xywh.tolist()
        cls = r.boxes.cls.tolist()

        digits = []
        locaters = []
        num = len(cls)
        for i in range(num):
            idx1 = i

            # print(cls[idx1], type(cls[idx1]))

            if cls[idx1] != 10:
                for j in range(num - i - 1):
                    idx2 = i + j + 1
                    xy1, xy2 = xywh[idx1][:2], xywh[idx2][:2]

                    distance = math.dist(xy1, xy2)
                    if distance <= coefficient1 * min(xywh[idx1][2], xywh[idx1][3]):
                        # print(cls[idx1], cls[idx2])

                        digits.append([(cls[idx1], xy1), (cls[idx2], xy2)])
            else:
                locaters.append((cls[idx1], xywh[idx1][:2]))

        double_digits = []
        for digit in digits:
            # print(digit)

            center_xy = [(digit[0][1][0] + digit[1][1][0]) / 2, (digit[0][1][1] + digit[1][1][1]) / 2]
            distance = math.dist(digit[0][1], digit[1][1])

            # print('______')
            for locater in locaters:
                # print(locater)

                if math.dist(center_xy, locater[1]) <= coefficient2 * distance:
                    double_digits.append([digit[0], digit[1], locater])
                    locaters.remove(locater)
                    break

        for i in double_digits:
            # print(i)
            xy0, xy1, xy2 = i[0][1], i[1][1], i[2][1]
            sita = math.atan((xy0[1] - xy1[1]) / (xy0[0] - xy1[0]))
            # print(sita)
            x_0 = (xy0[0] - xy2[0]) * math.cos(sita) - (xy0[1] - xy2[1]) * math.sin(sita)
            x_1 = (xy1[0] - xy2[0]) * math.cos(sita) - (xy1[1] - xy2[1]) * math.sin(sita)
            y_0 = (xy0[0] - xy2[0]) * math.sin(sita) + (xy0[1] - xy2[1]) * math.cos(sita)

            if (x_0 - x_1) * y_0 < 0:
                print(i[0][0], i[1][0])
            else:
                print(i[1][0], i[0][0])
    print(time.time() - t1)
    # break
