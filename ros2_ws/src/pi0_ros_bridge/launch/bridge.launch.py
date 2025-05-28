from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pi0_ros_bridge',
            executable='pi0_ros_node',
            name='pi0_ros_bridge',
            output='screen'
        )
    ])