import os
import random
import shutil

random.seed(42)

org_path = r"D:\Double-digit-yolo-detection-on-aircraft\datasets\digits\origin1"
dst_path = r"D:\Double-digit-yolo-detection-on-aircraft\datasets\digits\splited1"

train_proportion = 0.7
val_proportion = 0.2
files = os.listdir(f"{org_path}/images")

random.shuffle(files)
train_num = int(len(files) * train_proportion)
val_num = int(len(files) * val_proportion)

three_files = dict()
three_files["train"] = files[:train_num]
three_files["val"] = files[train_num : train_num + val_num]
three_files["test"] = files[train_num + val_num :]

for i in ["train", "val", "test"]:
    os.makedirs(f"{dst_path}/images/{i}", exist_ok=False)
    os.makedirs(f"{dst_path}/labels/{i}", exist_ok=False)

    for file in three_files[i]:
        prefix = file.rsplit(".")[0]
        shutil.copyfile(f"{org_path}/images/" + file, f"{dst_path}/images/{i}/" + file)
        shutil.copyfile(
            f"{org_path}/labels/" + prefix + ".txt",
            f"{dst_path}/labels/{i}/" + prefix + ".txt",
        )
