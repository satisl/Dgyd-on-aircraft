import cv2
import os


def extract_frames(video_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频文件是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    while True:
        cap.grab()
        # 每 30 帧保存一帧
        if frame_count % 5 == 0:
            # 读取视频帧
            ret, frame = cap.retrieve()
            # 检查是否成功读取帧
            if not ret:
                break
            # 构造输出文件路径
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            # 保存帧到文件
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {frame_count} to {frame_path}")

        frame_count += 1

    # 释放视频对象
    cap.release()


if __name__ == "__main__":
    # 视频文件路径
    video_path = r"E:\desktop\新建文件夹\1715821238.7073598.avi"
    # 输出文件夹路径
    output_folder = r"E:\desktop\frame"

    # 提取帧
    extract_frames(video_path, output_folder)
