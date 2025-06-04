import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class UR5InitPose(Node):
    def __init__(self):
        super().__init__('ur5_init_pose_sender')

        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        self.timer = self.create_timer(1.0, self.send_trajectory)

    def send_trajectory(self):
        joint_names = [
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'shoulder_pan_joint'
        ]

        home_positions = [
            -1.495906178151266,
            0.18928194046020508,
            -0.2825863997088831,
            -1.5792220274554651,
            0.03916294872760773,
            1.5943816900253296
        ]

        msg = JointTrajectory()
        msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = home_positions
        point.time_from_start.sec = 2  

        msg.points.append(point)
        self.publisher_.publish(msg)
        self.get_logger().info("new initial position command has been published")
        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = UR5InitPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
