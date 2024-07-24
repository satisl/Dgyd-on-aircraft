import cv2
import os
import numpy as np


def gamma_correction(frame, gamma=1.0):
    frame_normalized = frame / 255.0
    corrected_frame = np.power(frame_normalized, gamma)
    corrected_frame = (corrected_frame * 255).astype(np.uint8)
    return corrected_frame


input_folder = r""
output_folder = r""
gamma_value = 10  # 调整gamma值以控制亮度

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        frame = cv2.imread(image_path)

        # 应用伽马校正
        gamma_corrected_frame = gamma_correction(frame, gamma=gamma_value)

        # 保存处理后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, gamma_corrected_frame)

print("图像处理完成。")
