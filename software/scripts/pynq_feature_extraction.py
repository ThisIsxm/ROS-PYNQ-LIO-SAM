#!/usr/local/share/pynq-venv/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import struct
import time
import os
from pynq import Overlay, allocate
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from lo_sam.msg import FeatureIndicesRaw, LidarClouds
import sensor_msgs_py.point_cloud2 as pc2

# ==================== 常量定义 ====================
N_SCANS = 64
Horizon_SCAN = 1800
MAX_FPGA_POINTS = N_SCANS * Horizon_SCAN  # 115200
EXTRACT_SEGMENT = 6
FLAT_SIZE = 4
LESS_SHARP_SIZE = 20
SHARP_SIZE = 2

INPUT_BUFFER_DEPTH = MAX_FPGA_POINTS 
OUTPUT_CLOUD_SIZE = MAX_FPGA_POINTS
FLAT_BUFFER_DEPTH = EXTRACT_SEGMENT * N_SCANS * FLAT_SIZE
LESSSHARP_BUFFER_DEPTH = EXTRACT_SEGMENT * N_SCANS * LESS_SHARP_SIZE
SHARP_BUFFER_DEPTH = EXTRACT_SEGMENT * N_SCANS * SHARP_SIZE

AP_START_REG = 0x00
AP_DONE_MASK = 0x02
POLL_SLEEP_MS = 0.00005  # 50微秒
FPGA_TIMEOUT = 5.0

# ==================== 预定义 PointCloud2 字段 ====================
POINTCLOUD2_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]

# ==================== 工具函数 ====================
def float_to_int(float_val: float) -> int:
    return struct.unpack('I', struct.pack('f', float_val))[0]


class PYNQFeatureExtractorNode(Node):
    
    def __init__(self):
        super().__init__('pynq_feature_extractor_node')

        # PYNQ 相关对象
        self.overlay = None
        self.fpga_feature_extract = None
        self.reg_map = None
        
        # DMA 缓冲区
        self.input_buffer = None
        self.point_buffer = None
        self.range_buffer = None
        self.lessflat_buffer = None
        self.ground_buffer = None
        self.flat_buffer = None
        self.lesssharp_buffer = None
        self.sharp_buffer = None
        
        # ========== 零拷贝优化：预分配内存视图 ==========
        self.input_view_2d = None           # (N, 4) 2D 视图，用于直接写入
        self.flat_view = None               # 特征索引视图
        self.sharp_view = None
        self.lesssharp_view = None
        
        # 预分配输出缓冲区列表（批量 invalidate）
        self.output_buffers_to_invalidate = []
        
        # 只预分配 FeatureIndicesRaw 消息
        self.flat_raw_msg = FeatureIndicesRaw()
        self.sharp_raw_msg = FeatureIndicesRaw()
        self.lesssharp_raw_msg = FeatureIndicesRaw()
        
        # 预分配 PointCloud2 消息（复用消息对象）
        self.cloud_msg = PointCloud2()
        self.cloud_msg.height = 1
        self.cloud_msg.is_bigendian = False
        self.cloud_msg.is_dense = True
        self.cloud_msg.fields = POINTCLOUD2_FIELDS
        self.cloud_msg.point_step = 16

        # 初始化 PYNQ
        self._init_pynq()

        # ROS 2 Publishers，全部使用新消息 FeatureIndicesRaw
        self.fpga_output_cloud_pub = self.create_publisher(
            PointCloud2,
            '/feature/fpga_output_cloud',
            100
        )

        self.fpga_output_flat_pub = self.create_publisher(
            FeatureIndicesRaw,
            '/feature/fpga_output_flat',
            100
        )

        self.fpga_output_sharp_pub = self.create_publisher(
            FeatureIndicesRaw,
            '/feature/fpga_output_sharp',
            100
        )

        self.fpga_output_lesssharp_pub = self.create_publisher(
            FeatureIndicesRaw,
            '/feature/fpga_output_lesssharp',
            100
        )

        # ROS 2 Subscribers
        self.lidar_clouds_sub = self.create_subscription(
            LidarClouds,
            'feature/reordered_cloud',
            self.lidar_clouds_callback,
            100
        )

        self.get_logger().info('Node initialized and ready.')


    def _init_pynq(self):

        try:
            package_share_dir = get_package_share_directory('lo_sam')
            bitstream_path = os.path.join(package_share_dir, 'bitstreams', 'feature_extraction.bit')
            
            self.get_logger().info(f"Loading PYNQ overlay from: {bitstream_path}")
            self.overlay = Overlay(bitstream_path)
            self.fpga_feature_extract = self.overlay.fpga_feature_extract_0
            self.reg_map = self.fpga_feature_extract.register_map
            self.get_logger().info("PYNQ Overlay loaded successfully.")

            # 定义数据类型
            point_dtype = np.dtype([
                ('x', np.float32), ('y', np.float32), 
                ('z', np.float32), ('intensity', np.float32)
            ])
            index_dtype = np.dtype([('scan_id', np.int16), ('column_id', np.int16)])
            
            # 分配 DMA 缓冲区
            self.input_buffer = allocate(shape=(INPUT_BUFFER_DEPTH,), dtype=point_dtype)
            self.point_buffer = allocate(shape=(OUTPUT_CLOUD_SIZE,), dtype=point_dtype)
            self.range_buffer = allocate(shape=(OUTPUT_CLOUD_SIZE,), dtype=np.float32)
            self.lessflat_buffer = allocate(shape=(OUTPUT_CLOUD_SIZE,), dtype=np.uint8) 
            self.ground_buffer = allocate(shape=(OUTPUT_CLOUD_SIZE,), dtype=np.uint8) 
            self.flat_buffer = allocate(shape=(FLAT_BUFFER_DEPTH,), dtype=index_dtype)
            self.lesssharp_buffer = allocate(shape=(LESSSHARP_BUFFER_DEPTH,), dtype=index_dtype)
            self.sharp_buffer = allocate(shape=(SHARP_BUFFER_DEPTH,), dtype=index_dtype)
            
            # ========== 创建所有必要的内存视图（一次性创建，永久复用） ==========
            # 输入缓冲区视图：将结构化数组转为 (N, 4) 的 float32 视图
            self.input_view_2d = self.input_buffer.view(np.float32).reshape(-1, 4)
            
            # 输出缓冲区视图
            self.flat_view = self.flat_buffer
            self.sharp_view = self.sharp_buffer
            self.lesssharp_view = self.lesssharp_buffer
            
            # 预定义需要 invalidate 的缓冲区列表
            self.output_buffers_to_invalidate = [
                self.point_buffer,
                self.range_buffer,
                self.lessflat_buffer,
                self.ground_buffer,
                self.flat_buffer,
                self.lesssharp_buffer,
                self.sharp_buffer
            ]      

            # 批量设置寄存器地址
            self._set_buffer_addresses()

            self.get_logger().info("PL initialized successfully.")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize PL: {e}")
            self._free_dma_buffers()
            rclpy.shutdown()

    def _set_buffer_addresses(self):

        buffers = [
            ('laserCloudInArray', self.input_buffer),
            ('rangeimage_point', self.point_buffer),
            ('rangeimage_range', self.range_buffer),
            ('rangeimage_lessflat', self.lessflat_buffer),
            ('rangeimage_ground', self.ground_buffer),
            ('rangeimage_flat', self.flat_buffer),
            ('rangeimage_lesssharp', self.lesssharp_buffer),
            ('rangeimage_sharp', self.sharp_buffer),
        ]
        
        for name, buffer in buffers:
            addr = buffer.physical_address
            setattr(self.reg_map, f"{name}_1", addr & 0xFFFFFFFF)
            setattr(self.reg_map, f"{name}_2", addr >> 32)
        
        self.get_logger().info("DMA buffer addresses set to FPGA registers.")

    def _free_dma_buffers(self):

        buffers = [
            self.input_buffer, self.point_buffer, self.range_buffer,
            self.lessflat_buffer, self.ground_buffer, self.flat_buffer,
            self.lesssharp_buffer, self.sharp_buffer
        ]
        
        for buf in buffers:
            if buf is not None:
                buf.freebuffer()
        
        self.get_logger().info("All DMA buffers freed.")

    def lidar_clouds_callback(self, msg: LidarClouds):

        start_time = time.perf_counter()

        # 检查数据有效性
        raw_data_len = len(msg.lidar_clouds.data)
        if raw_data_len == 0:
            self.get_logger().warn("Empty point cloud. Skipping.")
            return

        # # ========== Step 1: 数据填充 (Buffer View & Memcpy) ==========
        t0 = time.perf_counter()
        
        incoming_floats = np.frombuffer(msg.lidar_clouds.data, dtype=np.float32)
        
        num_points_in_msg = incoming_floats.shape[0] // 4
        incoming_points = incoming_floats[:num_points_in_msg*4].reshape(num_points_in_msg, 4)

        cloud_size = min(num_points_in_msg, MAX_FPGA_POINTS)

        self.input_view_2d[:cloud_size] = incoming_points[:cloud_size]
        self.input_buffer.flush()
        
        t1 = time.perf_counter()

        # ========== Step 2: 启动 FPGA 运算 ==========
        start_ori_int = float_to_int(msg.start_ori)
        end_ori_int = float_to_int(msg.end_ori)

        # 批量写入寄存器
        self.reg_map.cloudSize = cloud_size
        self.reg_map.start_point_ori = start_ori_int
        self.reg_map.end_point_ori = end_ori_int
        self.fpga_feature_extract.write(AP_START_REG, 1)

        # ========== Step 3: 轮询 FPGA 完成状态 ==========
        poll_start = time.perf_counter()
        reg_val = self.fpga_feature_extract.read(AP_START_REG)
        while (reg_val & AP_DONE_MASK) == 0:
            if time.perf_counter() - poll_start > FPGA_TIMEOUT:
                raise RuntimeError(f"FPGA timeout after {FPGA_TIMEOUT}s")
            time.sleep(POLL_SLEEP_MS)
            reg_val = self.fpga_feature_extract.read(AP_START_REG)
        
        t2 = time.perf_counter()

        # ========== Step 4: 批量 invalidate 输出缓冲区 ==========
        for buf in self.output_buffers_to_invalidate:
            buf.invalidate()
        
        t3 = time.perf_counter()

        # ========== Step 5: 发布结果 ==========
        self._publish_results(msg.header, cloud_size)
        
        end_time = time.perf_counter()
        
        # 性能日志
        self.get_logger().info(
            f"数据填充: {(t1-t0)*1000:.2f}ms | "
            f"FPGA执行: {(t2-t1)*1000:.2f}ms | "
            f"消息发布: {(end_time-t3)*1000:.2f}ms | "
            f"节点总耗时: {(end_time-start_time)*1000:.2f}ms"
        )

    def _publish_results(self, input_header: Header, cloud_size: int):
        
        # ========== 1. 发布 PointCloud2 ==========
        self.cloud_msg.header = input_header
        self.cloud_msg.width = cloud_size
        self.cloud_msg.row_step = 16 * cloud_size
        
        self.cloud_msg._data = self.point_buffer[:cloud_size].tobytes()
        
        self.fpga_output_cloud_pub.publish(self.cloud_msg)

        # ========== 2. 发布特征索引 ==========
        def publish_indices_raw(buffer_view, msg_obj, publisher):
            valid_mask = (buffer_view['scan_id'] != 0) | (buffer_view['column_id'] != 0)
            valid_items = buffer_view[valid_mask]

            count = len(valid_items)
            msg_obj.header = input_header
            msg_obj.count = count
            msg_obj.data = valid_items.tobytes()

            publisher.publish(msg_obj)

        publish_indices_raw(self.flat_view, self.flat_raw_msg, self.fpga_output_flat_pub)
        publish_indices_raw(self.sharp_view, self.sharp_raw_msg, self.fpga_output_sharp_pub)
        publish_indices_raw(self.lesssharp_view, self.lesssharp_raw_msg, self.fpga_output_lesssharp_pub)

        self.get_logger().info(
            f"Published: {cloud_size} points, "
            f"{self.flat_raw_msg.count} flat, "
            f"{self.sharp_raw_msg.count} sharp, "
            f"{self.lesssharp_raw_msg.count} lesssharp"
        )

def main(args=None):
    rclpy.init(args=args)
    node = PYNQFeatureExtractorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down PYNQ Feature Extractor Node.")
        node._free_dma_buffers()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
