import os
import shutil

remove_list = """IMG_3532
IMG_3539
IMG_3531
"""

img_path = r''
xml_path = r''
save_img_path = r''
save_xml_path = r''

remove_list = [i for i in remove_list.split('\n')]

imgs = os.listdir(img_path)

for idx, img in enumerate(imgs):
    prefix, suffix = img.rsplit('.', 1)
    if img.rsplit('.', 1)[0] not in remove_list:
        shutil.copyfile(f'{img_path}/{prefix}.{suffix}', f'{save_img_path}/{prefix}.{suffix}')
        shutil.copyfile(f'{xml_path}/{prefix}.json', f'{save_xml_path}/{prefix}.json')
