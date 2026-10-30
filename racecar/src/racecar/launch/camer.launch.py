import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess

def generate_launch_description():
    # 声明摄像头设备参数
    video_device_arg = DeclareLaunchArgument(
        'video_device', default_value='/dev/video0',
        description='Video device path for the USB camera'
    )

    # 启动 usb_cam 节点（原始图发到 image_raw_raw）
    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        parameters=[{
            'video_device': '/dev/video0',  # 默认设备路径
            'image_width': 640,            # 图像宽度
            'image_height': 480,           # 图像高度
            'pixel_format': 'yuyv',        # 修改为 yuyv 格式
            'camera_frame_id': 'camera'    # 摄像头帧 ID
        }],
        remappings=[
            ('image_raw', 'image_raw_raw'),
        ],
        output='screen'
    )

    # 直接跑源码脚本，不依赖 install/lib
    flip_script = os.path.expanduser('~/racecar/src/racecar/scripts/image_flip.py')
    image_flip_node = ExecuteProcess(
        cmd=['python3', flip_script],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(video_device_arg)
    ld.add_action(usb_cam_node)
    ld.add_action(image_flip_node)

    return ld

#查看摄像头路径：ls /dev/video*
# 安装usb_cam：
#     sudo apt update
#     sudo apt install ros-humble-usb-cam
# ros2 topic list
#显示画面：ros2 run rqt_image_view rqt_image_view
