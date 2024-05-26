import cv2
from ultralytics.utils.plotting import Annotator

if __name__ == '__main__':
    coefficient3 = 0.005
    path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\5\for trainning\images\train\00000-0_1.jpg'
    imgsz = 160

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (imgsz, imgsz))
    x, y, w, h = cv2.selectROI(img)
    height, width = img.shape[:2]
    annotator = Annotator(img, line_width=int(min(width, height) * coefficient3))
    annotator.box_label((x, y, x + w, y + h), label=f'{int(w)}*{int(h)}')
    cv2.imshow('0', annotator.im)
    cv2.waitKey(10000)
