"""
Launch file for Camera 3D Perception Stack.

Starts:
  1. Perception node (YOLOv8 + Depth + Tracking)
  2. Static TF publisher (map -> base_link)
  3. RViz2 with saved config

Run: ros2 launch launch/perception_launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    rviz_config = os.path.join(project_dir, 'configs', 'perception.rviz')
    perception_script = os.path.join(project_dir, 'src', 'ros2_node', 'perception_node.py')
    
    # 1. Static TF: map -> base_link
    tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_base',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
    )
    
    # 2. Perception node (delayed 2s to let TF start first)
    perception_node = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', perception_script],
                cwd=project_dir,
                output='screen',
                shell=False
            )
        ]
    )
    
    # 3. RViz2 (delayed 3s to let perception start)
    rviz_node = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_config],
                output='log'
            )
        ]
    )
    
    return LaunchDescription([
        tf_node,
        perception_node,
        rviz_node,
    ])
