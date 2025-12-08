#!/usr/local/share/pynq-venv/bin/python3

import os
import numpy as np
from datetime import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

def read_bin_file(bin_path):
    """读取KITTI的bin点云文件，返回N×4 numpy数组"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def pointcloud2_msg(points, frame_id='velodyne'):
    """把numpy点云转换成sensor_msgs/PointCloud2消息"""
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id

    msg.height = 1
    msg.width = points.shape[0]

    # 定义点字段，x,y,z, intensity各4字节float32
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    msg.is_bigendian = False
    msg.point_step = 16  # 每个点16字节(4float32)
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True

    # numpy数据转bytes，注意按照msg.point_step排列
    msg._data = points.astype(np.float32).tobytes()

    return msg

def kitti_timestamp_to_ros_time(ts_str):
    """
    KITTI时间戳字符串格式示例：'2011-09-30 13:02:46.223600000'
    转换为builtin_interfaces.msg.Time（秒+纳秒）
    """

    dt = datetime.strptime(ts_str[:26], "%Y-%m-%d %H:%M:%S.%f")

    timestamp_secs = dt.timestamp()
    sec = int(timestamp_secs)
    nanosec = int((timestamp_secs - sec) * 1e9)
    ros_time = Time(sec=sec, nanosec=nanosec)
    return ros_time

class KittiPointCloudPublisher(Node):
    def __init__(self, data_path, timestamps_file, frame_id='velodyne', publish_frequency=10.0):
        super().__init__('kitti_pointcloud_publisher')
        self.pub = self.create_publisher(PointCloud2, '/velodyne_points', 10)

        self.data_path = data_path
        self.frame_id = frame_id
        self.publish_frequency = publish_frequency

        self.files = sorted(os.listdir(data_path))

        with open(timestamps_file, 'r') as f:
            self.timestamps = [line.strip() for line in f.readlines()]
        assert len(self.files) == len(self.timestamps), \
            f"点云文件和时间戳数量不匹配: {len(self.files)} vs {len(self.timestamps)}"

        self.index = 0
        self.timer = self.create_timer(1.0 / publish_frequency, self.timer_callback)

    def timer_callback(self):
        if self.index >= len(self.files):
            self.get_logger().info('All pointcloud frames published, shutting down.')
            rclpy.shutdown()
            return

        file_path = os.path.join(self.data_path, self.files[self.index])
        points = read_bin_file(file_path)
        msg = pointcloud2_msg(points, frame_id=self.frame_id)

        ros_time = kitti_timestamp_to_ros_time(self.timestamps[self.index])
        msg.header.stamp = ros_time

        self.pub.publish(msg)
        self.get_logger().info(f'Published frame {self.index + 1}/{len(self.files)}: {self.files[self.index]} with timestamp {self.timestamps[self.index]}')
        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    kitti_velodyne_dir = "/home/ubuntu/FPT_Competition/LO-SAM-ROS2/2011_09_30_drive_0027_sync/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data"
    timestamps_file = "/home/ubuntu/FPT_Competition/LO-SAM-ROS2/2011_09_30_drive_0027_sync/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/timestamps.txt"

    node = KittiPointCloudPublisher(kitti_velodyne_dir, timestamps_file, frame_id='velodyne', publish_frequency=1)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
