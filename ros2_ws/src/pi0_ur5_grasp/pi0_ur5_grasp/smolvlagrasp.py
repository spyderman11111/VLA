import sys
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import PolicyFeature, FeatureType

import torch

class SimpleGraspNode(Node):
    def __init__(self):
        super().__init__('simple_grasp_node')

        # Load policy
        self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

        # Configure input/output features
        self.policy.config.input_features = {
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "goal": PolicyFeature(type=FeatureType.ENV, shape=(48,))  # token ids
        }
        self.policy.config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))
        }

        # Get prompt and tokenize it
        prompt = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input("Prompt: ")
        tokenizer = self.policy.language_tokenizer
        tokenized = tokenizer(prompt, max_length=48, padding='max_length', truncation=True, return_tensors='pt')
        self.prompt_tensor = tokenized["input_ids"].squeeze(0).numpy().astype(np.float32)

        # Setup ROS I/O
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_cb, 10)
        self.pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        self.latest_image = None
        self.latest_joints = None
        self.processed = False

        # Home joint config
        self.home_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
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
        point.time_from_start = Duration(sec=2)
        msg.points.append(point)
        self.pub.publish(msg)
        self.get_logger().info("Going to initial position...")

    def image_cb(self, msg):
        h, w = msg.height, msg.width
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, -1))
        if msg.encoding.lower() == 'bgr8':
            img = img[:, :, ::-1]
        self.latest_image = np.transpose(img.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))
        self.try_infer()

    def joint_cb(self, msg):
        self.latest_joints = np.array(msg.position, dtype=np.float32)
        self.try_infer()

    def try_infer(self):
        if self.latest_image is None or self.latest_joints is None or self.processed:
            return
        self.processed = True

        # Prepare state vector
        state = np.concatenate([self.latest_joints, np.zeros(14 - len(self.latest_joints))], dtype=np.float32)

        # Model input dict
        model_input = {
            "observation.image": self.latest_image,
            "observation.state": state,
            "goal": self.prompt_tensor  # Already padded to shape (48,)
        }

        try:
            out = self.policy.select_action(model_input)
            joints = np.asarray(out)[0][:6]  # UR5 only takes first 6
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
        point.velocities = [0.3] * len(positions)
        point.time_from_start = Duration(sec=2)
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
    node = SimpleGraspNode()
    rclpy.spin(node)
