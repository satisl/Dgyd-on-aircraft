import xml.etree.ElementTree as ET
import math
from os import getcwd
import os


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = math.fabs(box[1] - box[0])
    h = math.fabs(box[3] - box[2])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(in_file, out_file):
    xml_text = in_file.read()
    root = ET.fromstring(xml_text)
    in_file.close()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls = cls if cls != 'locater' else 'locaters'
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    out_file.close()


wd = getcwd()

if __name__ == '__main__':
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'locaters']

    path = r'E:\desktop\picture_croped'
    os.makedirs(f'{path}\\labels', exist_ok=False)

    # for i in ['train', 'val', 'test']:
    #
    #     os.makedirs(f'{path}/labels/{i}', exist_ok=False)
    #
    #     for xml_path in os.listdir(f'{path}/annotations/{i}'):
    #         xml_name = xml_path.split('\\')[-1]
    #
    #         in_file = open(f"{path}/annotations/{i}/{xml_name[:-3]}xml", 'r')  # xml文件路径
    #         out_file = open(f"{path}/labels/{i}/{xml_name[:-3]}txt", 'w')  # 转换后的txt文件存放路径
    #
    #         convert_annotation(in_file, out_file)
    #
    #     with open(f'{path}/labels/{i}/classes.txt', mode='w', encoding='utf-8') as f:
    #         for j in classes:
    #             f.write(j + '\n')

    for xml_path in os.listdir(f'{path}/annotations'):
        xml_name = xml_path.split('\\')[-1]

        in_file = open(f"{path}/annotations/{xml_name[:-3]}xml", 'r')  # xml文件路径
        out_file = open(f"{path}/labels/{xml_name[:-3]}txt", 'w')  # 转换后的txt文件存放路径

        convert_annotation(in_file, out_file)

    with open(f'{path}/labels/classes.txt', mode='w', encoding='utf-8') as f:
        for j in classes:
            f.write(j + '\n')
