#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImageFlip(Node):
    def __init__(self):
        super().__init__('image_flip')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/image_raw_raw', self.cb, 10)
        self.pub = self.create_publisher(Image, '/image_raw', 10)
        self.get_logger().info('image_flip: /image_raw_raw -> rotate 180 -> /image_raw')

    def cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # 摄像头整机装反：旋转 180°（上下+左右一起校正）
        # 若只要上下翻转、左右不变，改为：out = cv2.flip(img, 0)
        out = cv2.rotate(img, cv2.ROTATE_180)
        out_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageFlip()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
