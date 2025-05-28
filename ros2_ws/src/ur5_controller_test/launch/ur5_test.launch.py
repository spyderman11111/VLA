from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ur5_controller_test',
            executable='joint_test_publisher',
            name='ur5_test',
            output='screen'
        )
    ])
