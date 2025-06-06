import sys
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import cv2


from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import PolicyFeature, FeatureType

import torch

dummy_stats = {
    "observation.state": {
        "mean": torch.zeros(14),
        "std": torch.ones(14)
    },
    "action": {
        "mean": torch.zeros(14),
        "std": torch.ones(14)
    }
}

class SmolVLAGraspNode(Node):
    def __init__(self):
        super().__init__('smolvla_grasp_node')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = SmolVLAPolicy.from_pretrained(
            "lerobot/smolvla_base",
            dataset_stats=dummy_stats
        ).to(self.device)

        self.policy.config.input_features = {
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "goal": PolicyFeature(type=FeatureType.ENV, shape=(48,))
        }
        self.policy.config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))
        }

        prompt = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input("Prompt: ")
        tokenizer = self.policy.language_tokenizer
        tokenized = tokenizer(prompt, max_length=48, padding='max_length', truncation=True, return_tensors='pt')
        self.prompt_tensor = tokenized["input_ids"].squeeze(0).to(self.device)

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_joints = None
        self.processed = False

        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_cb, 10)
        self.pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        self.home_names = [
            'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
            'wrist_2_joint', 'wrist_3_joint', 'shoulder_pan_joint'
        ]
        self.home_positions = [-1.4959, 0.1893, -0.2826, -1.5792, 0.0392, 1.5944]

        self.go_home()
        self.get_logger().info("Waiting for sensor input...")

    def go_home(self):
        msg = JointTrajectory()
        msg.joint_names = self.home_names
        point = JointTrajectoryPoint()
        point.positions = self.home_positions
        point.velocities = [0.3] * len(self.home_positions)
        point.time_from_start = Duration(sec=8)
        msg.points.append(point)
        self.pub.publish(msg)
        self.get_logger().info("Going to initial position...")

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            img = cv2.resize(img, (512, 512))
            img = (img.astype(np.float32) / 127.5) - 1.0
            img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            self.latest_image = torch.tensor(img, dtype=torch.float32, device=self.device)
            self.try_infer()
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def joint_cb(self, msg):
        joints = np.array(msg.position, dtype=np.float32)
        padded = np.concatenate([joints, np.zeros(14 - len(joints))], dtype=np.float32)
        self.latest_joints = torch.tensor(padded, dtype=torch.float32, device=self.device)
        self.try_infer()

    def try_infer(self):
        if self.latest_image is None or self.latest_joints is None or self.processed:
            return
        self.processed = True

        model_input = {
            "observation.image": self.latest_image.unsqueeze(0),
            "observation.state": self.latest_joints.unsqueeze(0),
            "goal": self.prompt_tensor.unsqueeze(0)
        }

        try:
            with torch.no_grad():
                out = self.policy(model_input)["action"]
            joints = out[0][:6].cpu().numpy()
            self.send_traj(joints)
            self.get_logger().info(f"Predicted action: {joints.tolist()}")
            self.create_timer(4.0, self.reset_and_shutdown)
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            rclpy.shutdown()

    def send_traj(self, positions):
        msg = JointTrajectory()
        msg.joint_names = self.home_names
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.velocities = [0.2] * len(positions)
        point.time_from_start = Duration(sec=10)
        msg.points.append(point)
        self.pub.publish(msg)
        self.get_logger().info("Action published.")

    def reset_and_shutdown(self):
        self.go_home()
        self.get_logger().info("Returning to home. Shutting down.")
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = SmolVLAGraspNode()
    rclpy.spin(node)
