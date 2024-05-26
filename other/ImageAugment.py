import os
import random
import threading
import imgaug as ia
import imgaug.augmenters as iaa
import cv2


def main(files, img_path1, img_path2, lock):
    for file in files:
        prefix, suffix = file.rsplit(".", 1)
        img_file = f"{img_path1}/{prefix}.{suffix}"

        img = cv2.imread(img_file)
        cnt = 0
        while cnt < aug_num:
            img_aug = seq(images=[img])

            cv2.imwrite(f"{img_path2}/{prefix}-{cnt}.{suffix}", img_aug[0])

            lock.acquire()
            print(file, cnt)
            lock.release()
            cnt += 1


ia.seed(42)
seq = iaa.Sequential(
    [
        iaa.SomeOf(
            (0, 4),
            [
                # 噪声
                iaa.OneOf(
                    [
                        iaa.AdditiveGaussianNoise(
                            scale=random.uniform(0.02, 0.1) * 255, per_channel=True
                        ),
                        iaa.AdditiveLaplaceNoise(
                            scale=random.uniform(0.02, 0.1) * 255, per_channel=True
                        ),
                        iaa.SaltAndPepper(p=(0, 0.1)),
                    ]
                ),
                # 色彩改变
                iaa.OneOf(
                    [
                        # 亮度改变
                        iaa.Multiply((0.5, 1.5)),
                        # hsv改变
                        iaa.MultiplyBrightness((0.8, 1.2)),
                        iaa.MultiplySaturation((0.5, 1.5)),
                        iaa.MultiplyHue((0.5, 1.5)),
                        iaa.MultiplyHueAndSaturation((0.5, 1.5)),
                        # 灰度图
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                    ]
                ),
                # 模糊
                iaa.OneOf(
                    [
                        iaa.AverageBlur(k=(0, 5)),
                        iaa.GaussianBlur(sigma=(0, 2)),
                        iaa.MedianBlur(k=(1, 5)),
                        iaa.MotionBlur(k=3, angle=[-45, 45]),
                    ]
                ),
                # 几何变换
                iaa.SomeOf(
                    (1, 2),
                    [
                        iaa.OneOf(
                            [
                                iaa.Sequential(
                                    [
                                        iaa.Affine(scale=(0.5, 0.7)),
                                        iaa.SomeOf(
                                            (0, 3),
                                            [
                                                iaa.Affine(shear=(-45, 45)),
                                                iaa.Affine(rotate=(-45, 45)),
                                                iaa.Affine(
                                                    translate_percent={
                                                        "x": (-0.2, 0.2),
                                                        "y": (-0.2, 0.2),
                                                    }
                                                ),
                                            ],
                                        ),
                                    ]
                                ),
                                iaa.Affine(scale=(0.7, 1.0)),
                            ]
                        ),
                        iaa.Rot90((0, 3), keep_size=False),
                    ],
                ),
            ],
        ),
    ],
)

org_path = r"D:\Double-digit-yolo-detection-on-aircraft\datasets\obb\background splited"
dst_path = (
    r"D:\Double-digit-yolo-detection-on-aircraft\datasets\obb\background augmented"
)
os.makedirs(dst_path, exist_ok=False)

aug_num = 10
worker_num = 8
lock = threading.Lock()

tasks = ["train", "val", "test"]
for task in tasks:
    img_path1 = f"{org_path}/images/{task}"
    img_path2 = f"{dst_path}/images/{task}"
    os.makedirs(img_path2, exist_ok=False)

    files = os.listdir(img_path1)
    files.sort()

    split_num = len(files) // worker_num
    ts = [
        threading.Thread(
            target=main,
            args=(
                files[i * split_num : (i + 1) * split_num],
                img_path1,
                img_path2,
                lock,
            ),
        )
        for i in range(worker_num)
    ]
    for t in ts:
        t.start()
    if worker_num * split_num < len(files):
        t1 = threading.Thread(
            target=main,
            args=(files[worker_num * split_num :], img_path1, img_path2, lock),
        )
        t1.start()
        t1.join()
    for t in ts:
        t.join()
