import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取包目录
    lo_sam_share = get_package_share_directory('lo_sam')
    
    # 参数文件
    lidar_config = os.path.join(lo_sam_share, 'config', 'VLP16.yaml')
    rviz_config = os.path.join(lo_sam_share, 'rviz_cfg', 'LO-SAM.rviz')
    
    # rviz 是否启动的参数
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to launch RViz'
    )
    
    # 主节点
    nodes = [
        Node(
            package='lo_sam',
            executable='feature_extract',   
            name='feature_extract',
            output='screen',
            parameters=[lidar_config]
        ),
        Node(
            package='lo_sam',
            executable='pldataprocess',   
            name='pldataprocess',
            output='screen',
        ),
        Node(
            package='lo_sam',
            executable='laserOdometry',
            name='laserOdometry',
            output='screen'
        ),
        Node(
            package='lo_sam',
            executable='map_optimization',
            name='map_optimization',
            output='screen'
        ),
        Node(
            package='lo_sam',
            executable='pynq_feature_extraction.py',
            name='pynq_feature_extraction',
            output='screen'
        ),
        Node(
            package='lo_sam',
            executable='kitti_pointcloud_publisher.py',
            name='kitti_pointcloud_publisher',
            output='screen'
        )
    ]
    
    # RViz 节点（按参数控制是否启动）
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config],
        output='screen'
    )
    
    # 按参数控制rviz启动
    rviz_group = GroupAction([rviz_node])
    
    from launch.conditions import IfCondition

    return LaunchDescription([
        rviz_arg,
        *nodes,
        GroupAction([rviz_node], condition=IfCondition(LaunchConfiguration('rviz')))
    ])
