import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import numpy as np
import cv2
from .input_adapter import UR5Inputs
from .output_adapter import UR5Outputs

class Pi0Node(Node):
    def __init__(self):
        super().__init__('pi0_ros_bridge')
        self.bridge = CvBridge()
        self.joint_state = None
        self.base_rgb = None
        self.wrist_rgb = None
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.img_sub = self.create_subscription(Image, '/camera/color/image_raw', self.img_callback, 10)
        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.timer = self.create_timer(0.1, self.tick)
        self.model = self.load_model()

    def load_model(self):
        # Dummy function: replace with real load_model_from_config()
        class Dummy:
            def __call__(self, x):
                return {"actions": np.random.randn(1, 7)}
        return Dummy()

    def joint_callback(self, msg):
        self.joint_state = msg

    def img_callback(self, msg):
        self.base_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.wrist_rgb = self.base_rgb  # for demo, use same image

    def tick(self):
        if self.joint_state is None or self.base_rgb is None:
            return
        joints = np.array(self.joint_state.position[:6])
        gripper = np.array([0.0])
        adapter = UR5Inputs(action_dim=7)
        input_dict = adapter({
            "joints": joints,
            "gripper": gripper,
            "base_rgb": self.base_rgb,
            "wrist_rgb": self.wrist_rgb,
            "prompt": "pick up the object"
        })
        output = self.model(input_dict)
        decoded = UR5Outputs()(output)
        self.publish(decoded["actions"][0])

    def publish(self, action):
        msg = JointTrajectory()
        msg.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        point = JointTrajectoryPoint()
        point.positions = action[:6].tolist()
        point.time_from_start.sec = 1
        msg.points.append(point)
        self.publisher.publish(msg)
