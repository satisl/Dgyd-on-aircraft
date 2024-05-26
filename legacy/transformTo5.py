import os
import math
from lxml import objectify, etree
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def save_xml(file_name, save_folder, img_info, height, width, channel, bboxs_info):
    '''
    :param file_name:文件名
    :param save_folder:#保存的xml文件的结果
    :param height:图片的信息
    :param width:图片的宽度
    :param channel:通道
    :return:
    '''
    folder_name, img_name = img_info  # 得到图片的信息

    E = objectify.ElementMaker(annotate=False)

    anno_tree = E.annotation(
        E.folder(folder_name),
        E.filename(img_name),
        E.path(os.path.join(folder_name, img_name)),
        E.source(
            E.database('Unknown'),
        ),
        E.size(
            E.width(width),
            E.height(height),
            E.depth(channel)
        ),
        E.segmented(0),
    )

    labels, bboxs = bboxs_info  # 得到边框和标签信息
    for label, box in zip(labels, bboxs):
        anno_tree.append(
            E.object(
                E.name(label),
                E.pose('Unspecified'),
                E.truncated('0'),
                E.difficult('0'),
                E.bndbox(
                    E.xmin(box[0]),
                    E.ymin(box[1]),
                    E.xmax(box[2]),
                    E.ymax(box[3])
                )
            ))

    etree.ElementTree(anno_tree).write(os.path.join(save_folder, file_name), pretty_print=True)


def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_img_bbox(img, bboxes, angle=5., scale=1.):
    '''
    参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
    输入:
        img:图像array,(h,w,c)
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        angle:旋转角度
        scale:默认1
    输出:
        rot_img:旋转后的图像array
        rot_bboxes:旋转后的boundingbox坐标list
    '''
    # ---------------------- 旋转图像 ----------------------
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 仿射变换
    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    # ---------------------- 矫正bbox坐标 ----------------------
    # rot_mat是最终的旋转矩阵
    # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
    rot_bboxes = list()
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx + rw
        ry_max = ry + rh
        # 加入list中
        rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

    return rot_img, rot_bboxes


img_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\5/4/images'
xml_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\5/4/annotations'
xml_path1 = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\5/1/annotations'
save_img_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\5/origin1/images'
save_xml_path = r'D:\Double-digit-yolo-detection-on-aircraft\datasets\5/origin1/annotations'

coefficient1 = 2.5 * 2
coefficient2 = 1
coefficient3 = 0.1

if __name__ == '__main__':
    os.makedirs(save_img_path, exist_ok=False)
    os.makedirs(save_xml_path, exist_ok=False)

    imgs = os.listdir(img_path)
    for img in imgs:
        # 获取locaters,digits
        prefix, suffix = img.rsplit('.', 1)
        tree = ET.parse(f'{xml_path}/{prefix}.xml')
        root = tree.getroot()

        targets = []
        objs = root.findall('object')
        for ix, obj in enumerate(objs):
            name = obj.find('name').text
            box = obj.find('bndbox')
            x_min = int(box[0].text)
            y_min = int(box[1].text)
            x_max = int(box[2].text)
            y_max = int(box[3].text)

            width = int((x_max - x_min) * coefficient3)
            height = int((y_max - y_min) * coefficient3)
            targets.append((x_min - width, y_min - height, x_max + width, y_max + height))

        # 获取第一版locaters和digits
        tree = ET.parse(f'{xml_path1}/{prefix}.xml')
        root = tree.getroot()

        locaters = []
        digits = []

        objs = root.findall('object')
        for ix, obj in enumerate(objs):
            name = obj.find('name').text
            box = obj.find('bndbox')
            x_min = int(box[0].text)
            y_min = int(box[1].text)
            x_max = int(box[2].text)
            y_max = int(box[3].text)

            if name == 'locater':
                locaters.append([x_min, y_min, x_max, y_max])
            else:
                digits.append([x_min, y_min, x_max, y_max, name])

        # 转换第一版locater和digits坐标
        size = root.find('size')
        cnt = 0
        image = cv2.imread(f'{img_path}/{img}')

        print(img)
        for left, top, right, bottem in targets:
            labels = []
            bboxs = []
            left = left if left > 0 else 0
            top = top if top > 0 else 0

            print(left, top, right, bottem)
            for x1, y1, x2, y2 in locaters:
                if x1 > left and y1 > top and x2 < right and y2 < bottem:
                    labels.append('locater')
                    bboxs.append((x1 - left, y1 - top, x2 - left, y2 - top))
            for x1, y1, x2, y2, name in digits:
                if x1 > left and y1 > top and x2 < right and y2 < bottem:
                    labels.append(name)
                    bboxs.append((x1 - left, y1 - top, x2 - left, y2 - top))
            cv2.imwrite(f'{save_img_path}/{prefix}-{cnt}.{suffix}', image[top:bottem, left:right])
            save_xml(f'{prefix}-{cnt}.xml', save_xml_path,
                     (save_img_path, f'{prefix}-{cnt}.{suffix}'),
                     bottem - top, right - left, size[2].text, (labels, bboxs))
            cnt += 1
