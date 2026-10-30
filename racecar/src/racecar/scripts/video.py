#!/usr/bin/env python3
import cv2
def record_video(output_file, duration, fps=30, width=640, height=480):
    """
    录制视频并保存为MP4格式

    参数:
        output_file (str): 输出视频文件名（例如 "output.mp4"）
        duration (int): 录制时长（秒）
        fps (int): 视频帧率，默认为30
        width (int): 视频宽度，默认为640
        height (int): 视频高度，默认为480
    """
    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4视频编码器
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"开始录制视频，时长为{duration}秒，保存为{output_file}")
    start_time = cv2.getTickCount()  # 获取开始时间

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        # 显示实时画面
        cv2.imshow('Recording', frame)

        # 写入帧到视频文件
        out.write(frame)

        # 检查是否达到录制时长
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time >= duration:
            break

        # 按下 'q' 键提前退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户提前终止录制")
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("视频录制完成")

if __name__ == '__main__':
    # 设置录制参数
    output_file = 'output_AB.mp4'  # 输出文件名
    duration = 300000  # 录制时长（秒）
    record_video(output_file, duration)
