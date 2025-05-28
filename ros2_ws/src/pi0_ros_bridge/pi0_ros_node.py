import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import numpy as np
import cv2

class Pi0ControlNode(Node):
    def __init__(self):
        super().__init__('pi0_controller')
        self.bridge = CvBridge()

        # 订阅 UR5 关节状态 + 摄像头图像
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(Image, '/wrist_camera/image_raw', self.image_callback, 10)

        # 发布 UR5 控制指令
        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

        # 定时执行主逻辑
        self.create_timer(0.1, self.tick)

        self.joint_state = None
        self.wrist_rgb = None
        self.pi0_model = self.load_model()

    def joint_callback(self, msg: JointState):
        self.joint_state = np.array(msg.position)

    def image_callback(self, msg: Image):
        self.wrist_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def load_model(self):
        from openpi.policies import policy_config
        from openpi.training import config as pi0_config
        from openpi.shared import download
        cfg = pi0_config.get_config("pi0_fast_droid")
        ckpt = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")
        return policy_config.create_trained_policy(cfg, ckpt)

    def tick(self):
        if self.joint_state is None or self.wrist_rgb is None:
            return

        # 打包 pi0 输入
        inputs = {
            "state": self.joint_state,
            "image": {
                "left_wrist_0_rgb": self.wrist_rgb,
            },
            "prompt": "pick up the red cup",
        }

        # Pi0 推理
        result = self.pi0_model.infer(inputs)
        action = result["actions"][-1]  # 拿最后一帧动作

        # 发布控制指令
        msg = JointTrajectory()
        msg.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", ..., "gripper_joint"]
        point = JointTrajectoryPoint()
        point.positions = action.tolist()
        msg.points.append(point)
        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = Pi0ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
