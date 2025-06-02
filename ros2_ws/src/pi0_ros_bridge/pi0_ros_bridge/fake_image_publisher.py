import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class FakeImagePublisher(Node):
    def __init__(self):
        super().__init__('fake_image_publisher')
        self.publisher = self.create_publisher(Image, '/wrist_camera/image_raw', 10)
        self.bridge = CvBridge()

        # 你可以替换为你自己的测试图像路径
        self.image_path = os.path.expanduser("~/test_images/test.jpg")

        self.timer = self.create_timer(1.0, self.publish_image)
        self.get_logger().info(f"Publishing image from: {self.image_path}")

    def publish_image(self):
        if not os.path.exists(self.image_path):
            self.get_logger().warn("Image not found.")
            return

        img = cv2.imread(self.image_path)
        if img is None:
            self.get_logger().warn("Failed to load image.")
            return

        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.publisher.publish(msg)
        self.get_logger().info("Published test image.")

def main():
    rclpy.init()
    node = FakeImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
