from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pi0_ur5_grasp',
            executable='pi0_grasp_node',
            name='pi0_grasp_node',
            output='screen',
            parameters=[
                {'prompt': 'pick up the cube'}  
            ]
        )
    ])
