import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2


class Pi0ControlNode(Node):
    def __init__(self):
        super().__init__('pi0_controller')
        self.bridge = CvBridge()

        # 订阅 UR5 关节状态 + 摄像头图像 + prompt 输入
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(Image, '/wrist_camera/image_raw', self.image_callback, 10)
        self.create_subscription(String, '/pi0_prompt', self.prompt_callback, 10)

        # 发布控制指令
        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

        # 定时执行推理
        self.timer = self.create_timer(0.1, self.tick)

        # 初始状态
        self.joint_state = None
        self.wrist_rgb = None
        self.prompt = "pick up the red cup"

        # 加载 pi0 模型
        self.pi0_model = self.load_model()
        self.get_logger().info("Pi0 controller initialized with default prompt.")

    def joint_callback(self, msg: JointState):
        # 提取并按顺序排列 UR5 六个关节
        ur5_joints = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        name2pos = dict(zip(msg.name, msg.position))
        try:
            self.joint_state = np.array([name2pos[name] for name in ur5_joints])
        except KeyError:
            self.get_logger().warn("Joint state message missing expected joint names!")

    def image_callback(self, msg: Image):
        self.wrist_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def prompt_callback(self, msg: String):
        self.prompt = msg.data
        self.get_logger().info(f"[Prompt updated] {self.prompt}")

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
            "image_mask": {
                "left_wrist_0_rgb": np.True_,
            },
            "prompt": self.prompt,
        }

        # 推理并提取最后一帧动作
        try:
            result = self.pi0_model.infer(inputs)
            action = result["actions"][-1]
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        # 构造控制指令
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        point = JointTrajectoryPoint()
        point.positions = action.tolist()
        point.time_from_start = Duration(sec=1)

        msg.points.append(point)
        self.publisher.publish(msg)
        self.get_logger().info(f"Published joint command for prompt: {self.prompt}")

def main():
    rclpy.init()
    node = Pi0ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
