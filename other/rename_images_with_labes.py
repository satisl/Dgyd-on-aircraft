import os
import shutil

img_path = r""
txt_path = r""
save_img_path = r""
save_txt_path = r""
os.makedirs(save_img_path, exist_ok=False)
os.makedirs(save_txt_path, exist_ok=False)

file_names = os.listdir(img_path)
c = 1182
# 随机获取一张图片的格式
f_first = file_names[0]

suffix = f_first.split(".")[-1]  # 图片文件的后缀
shutil.copyfile(f"{txt_path}/classes.txt", f"{save_txt_path}/classes.txt")
for file in file_names:
    shutil.copyfile(
        os.path.join(img_path, file),
        os.path.join(save_img_path, "{:0>5d}.{}".format(c, file.rsplit(".")[1])),
    )
    shutil.copyfile(
        os.path.join(txt_path, file.rsplit(".")[0] + ".txt"),
        os.path.join(save_txt_path, "{:0>5d}.{}".format(c, "txt")),
    )

    c += 1
