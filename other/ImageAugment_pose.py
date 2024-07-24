import os
import random
import threading
import imgaug as ia
import imgaug.augmenters as iaa
import cv2


def parse_polygon_and_keypoint(file, height, width):
    clss = []
    polygons = []
    keypoints = []
    with open(file, mode='r') as f:
        for line in f.readlines():
            infos = line.strip('\n').split(' ')
            clss.append(infos[0])
            (x, y), (w, h), (px, py) = [(float(infos[i]) * width, float(infos[i + 1]) * height) for i in
                                        range(1, len(infos), 2)]
            polygons.append(
                [(x - w / 2, y - h / 2), (x - w / 2, y + h / 2), (x + w / 2, y + h / 2), (x + w / 2, y - h / 2)])
            keypoints.append(
                (px, py)
            )
    return clss, polygons, keypoints


def main(files, img_path1, img_path2, txt_path1, txt_path2, lock):
    for file in files:
        prefix, suffix = file.rsplit('.', 1)
        img_file = f'{img_path1}/{prefix}.{suffix}'
        txt_file = f'{txt_path1}/{prefix}.txt'

        img = cv2.imread(img_file)
        height, width = img.shape[:2]
        clss, polygons, keypoints = parse_polygon_and_keypoint(txt_file, height, width)
        polygons = [ia.Polygon(polygon, label=cls) for polygon, cls in zip(polygons, clss)]
        cnt = 0
        while cnt < aug_num:
            img_aug, polygons_aug, keypoints_aug = seq(images=[img], polygons=[polygons], keypoints=[keypoints])
            with open(f'{txt_path2}/{prefix}-{cnt}.txt', mode='w') as f:
                for polygon, (px, py) in zip(polygons_aug[0], keypoints_aug[0]):
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = [[x / width, y / height] for (x, y) in polygon.exterior]
                    left = min(x1, x2, x3, x4)
                    right = max(x1, x2, x3, x4)
                    top = min(y1, y2, y3, y4)
                    bottom = max(y1, y2, y3, y4)
                    x = (right + left) / 2
                    y = (bottom + top) / 2
                    w = right - left
                    h = bottom - top
                    px /= width
                    py /= height
                    f.write(f'{polygon.label} {x} {y} {w} {h} {px} {py}\n')

            cv2.imwrite(f'{img_path2}/{prefix}-{cnt}.{suffix}', img_aug[0])

            lock.acquire()
            print(file, cnt)
            lock.release()
            cnt += 1


ia.seed(42)
seq = iaa.Sequential([
    iaa.SomeOf((0, 4), [
        # 噪声
        iaa.OneOf([
            iaa.AdditiveGaussianNoise(scale=random.uniform(0.02, 0.1) * 255, per_channel=True),
            iaa.AdditiveLaplaceNoise(scale=random.uniform(0.02, 0.1) * 255, per_channel=True),
            iaa.SaltAndPepper(p=(0, 0.1))
        ]),
        # 色彩改变
        iaa.OneOf([
            # 亮度改变
            iaa.Multiply((0.5, 1.5)),
            # hsv改变
            iaa.MultiplyBrightness((0.8, 1.2)),
            iaa.MultiplySaturation((0.5, 1.5)),
            iaa.MultiplyHue((0.5, 1.5)),
            iaa.MultiplyHueAndSaturation((0.5, 1.5)),
            # 灰度图
            iaa.Grayscale(alpha=(0.0, 1.0))
        ]),
        # 模糊
        iaa.OneOf([
            iaa.AverageBlur(k=(0, 5)),
            iaa.GaussianBlur(sigma=(0, 2)),
            iaa.MedianBlur(k=(1, 6)),
            iaa.MotionBlur(k=3, angle=[-45, 45])
        ]),
        # 几何变换
        iaa.OneOf([
            iaa.Affine(scale=(0.5, 1.0)),
        ])
    ])
])

org_path = r''
dst_path = r''
os.makedirs(dst_path, exist_ok=False)

aug_num = 10
worker_num = 8
lock = threading.Lock()

tasks = ['train', 'val', 'test']
for task in tasks:
    img_path1 = f'{org_path}/images/{task}'
    img_path2 = f'{dst_path}/images/{task}'
    txt_path1 = f'{org_path}/labels/{task}'
    txt_path2 = f'{dst_path}/labels/{task}'
    os.makedirs(img_path2, exist_ok=False)
    os.makedirs(txt_path2, exist_ok=False)

    files = os.listdir(img_path1)
    files.sort()

    split_num = len(files) // worker_num
    ts = [threading.Thread(target=main, args=(
        files[i * split_num:(i + 1) * split_num], img_path1, img_path2, txt_path1, txt_path2, lock))
          for i in
          range(worker_num)]
    for t in ts:
        t.start()
    if worker_num * split_num < len(files):
        t1 = threading.Thread(target=main, args=(
            files[worker_num * split_num:], img_path1, img_path2, txt_path1, txt_path2, lock))
        t1.start()
        t1.join()
    for t in ts:
        t.join()

    # # 效果预览
    # main(files, img_path1, img_path2, txt_path1, txt_path2, lock)
    # posi = ia.PolygonsOnImage(polygons_aug[0], shape=img_aug[0].shape)
    # img_with_polys = posi.draw_on_image(
    #     img_aug[0], alpha_points=0, alpha_face=0.5, color_lines=(255, 0, 0)
    # )
    # kypi = ia.KeypointsOnImage.from_xy_array(keypoints_aug[0], shape=img_aug[0].shape)
    # img_with_kps = kypi.draw_on_image(img_with_polys, color=(0, 0, 255), size=10)
    # cv2.imshow('0', cv2.resize(img_with_kps, (640, 480)))
    # cv2.waitKey(1)
