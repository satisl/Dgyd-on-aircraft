# encoding='UTF-8'
# author: pureyang
# TIME: 2019/8/22 下午5:15
# Description:数据图片的重命名

# 能实现文件的重命名,注意修改文件的路径

import os
import shutil

img_path = r"D:\Double-digit-yolo-detection-on-aircraft\datasets\obb\background\images"
save_img_path = (
    r"D:\Double-digit-yolo-detection-on-aircraft\datasets\obb\background\images1"
)
os.makedirs(save_img_path, exist_ok=False)

file_names = os.listdir(img_path)
c = 582
# 随机获取一张图片的格式
f_first = file_names[0]

suffix = f_first.split(".")[-1]  # 图片文件的后缀
# shutil.copyfile(f'{txt_path}/classes.txt', f'{save_txt_path}/classes.txt')
for file in file_names:
    shutil.copyfile(
        os.path.join(img_path, file),
        os.path.join(save_img_path, "{:0>5d}.{}".format(c, file.rsplit(".")[1])),
    )

    c += 1
