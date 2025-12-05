from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='enhanced_detector',
            executable='detector_node',
            name='enhanced_detector',
            output='screen',
            emulate_tty=True,
            parameters=[
                # Add any parameters here if needed
            ]
        )
    ])