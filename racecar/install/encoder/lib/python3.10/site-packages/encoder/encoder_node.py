#!/usr/bin/env python3
# coding: UTF-8

import struct

import rclpy
from rclpy.node import Node
import serial

from nav_msgs.msg import Odometry

# 轮周长 (m)，与车上标定一致
C = 227.45 / 1000.0
# 单圈分辨率（10bit）
RESOLUTION = 1024.0


def modbus_crc16(data: bytes) -> int:
    """规格书 5.5 CRC16 (Modbus)。"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def parse_holding_register_i16(response: bytes, slave_addr: int = 0x01):
    """
    解析 0x03 读 1 个保持寄存器的应答：
    ADR | 03 | 02 | DATA_H | DATA_L | CRC_L | CRC_H
    返回有符号 16 位整数；帧非法时返回 None。
    """
    if len(response) < 7:
        return None
    frame = response[:7]
    if frame[0] != slave_addr or frame[1] != 0x03 or frame[2] != 0x02:
        return None
    crc_calc = modbus_crc16(frame[:5])
    crc_recv = frame[5] | (frame[6] << 8)
    if crc_calc != crc_recv:
        return None
    value = struct.unpack('>h', frame[3:5])[0]
    return float(value)


class EncoderNode(Node):
    def __init__(self):
        super().__init__('encoder_vel')
        self.pub = self.create_publisher(Odometry, 'encoder', 10)

        self.port = self.declare_parameter('serial_port', '/dev/encoder').value
        self.baud = self.declare_parameter('baud_rate', 57600).value
        self.k = self.declare_parameter('k', 1.0).value
        # 规格书寄存器 0x000A：角速度采样时间，出厂默认 100ms
        self.sample_time = self.declare_parameter('sample_time', 0.1).value
        self.slave_addr = self.declare_parameter('slave_addr', 1).value

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05,
        )
        self.get_logger().info(
            f'Serial {self.port} @ {self.baud} open={self.ser.is_open}, '
            f'sample_time={self.sample_time}s'
        )

        # 50Hz 轮询；角速度含义由编码器采样时间决定，与轮询周期无关
        self.timer = self.create_timer(0.02, self.timer_callback)

        # 读角速度寄存器 0x0003，数量 1（规格书 5.4.3）
        self.cmd_read_angle_vel = bytes.fromhex('01 03 00 03 00 01 74 0A')

    def read_angle_velocity(self):
        self.ser.reset_input_buffer()
        self.ser.write(self.cmd_read_angle_vel)
        response = self.ser.read(7)
        return parse_holding_register_i16(response, self.slave_addr)

    def timer_callback(self):
        angle_v = self.read_angle_velocity()
        if angle_v is None:
            self.get_logger().warn('Invalid Modbus response from encoder', throttle_duration_sec=1.0)
            return

        # 规格书：线速度(m/s) = 角速度值 * C / (分辨率 * 采样时间_s)
        # 3.57 / 0.25 / 1.03 为原有机械/标定系数，保留可调 k
        Vel = (
            1.36 * angle_v * C / RESOLUTION / self.sample_time
            * self.k * 0.25 * 1.03
        )

        pub_vel = Odometry()
        pub_vel.header.frame_id = 'odom'
        pub_vel.child_frame_id = 'base_footprint'
        pub_vel.header.stamp = self.get_clock().now().to_msg()
        pub_vel.twist.twist.linear.x = float(Vel)
        self.pub.publish(pub_vel)


def main(args=None):
    rclpy.init(args=args)
    node = EncoderNode()
    try:
        rclpy.spin(node)
    finally:
        if node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
