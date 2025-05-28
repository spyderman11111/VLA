import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class UR5Test(Node):
    def __init__(self):
        super().__init__('ur5_test')
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.timer = self.create_timer(1.0, self.send_command)

    def send_command(self):
        msg = JointTrajectory()
        msg.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        point = JointTrajectoryPoint()
        point.positions = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]
        point.time_from_start.sec = 2
        msg.points.append(point)
        self.pub.publish(msg)