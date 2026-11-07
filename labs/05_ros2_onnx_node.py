"""Minimal ROS 2 ONNX inference node (tutorial policy uses 3-D Pendulum observation).
Requires ROS 2 Python environment, onnxruntime, std_msgs.
Usage: ros2 run ... if packaged, OR python labs/05_ros2_onnx_node.py
Input: /rl_observation Float32MultiArray [cos(theta), sin(theta), angular_velocity]
Output: /rl_action Float32MultiArray [torque]
NOT compatible with the real MicroDuck 61-in/14-out contract.
"""
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class PolicyNode(Node):
    def __init__(self):
        super().__init__('onnx_policy_tutorial')
        self.session = ort.InferenceSession('pendulum_actor.onnx', providers=['CPUExecutionProvider'])
        self.pub = self.create_publisher(Float32MultiArray, '/rl_action', 10)
        self.sub = self.create_subscription(Float32MultiArray, '/rl_observation', self.observe, 10)
        self.get_logger().info('Waiting for observation arrays of length 3')
    def observe(self, msg):
        if len(msg.data) != 3:
            self.get_logger().warning('Expected exactly 3 observation values'); return
        obs = np.asarray(msg.data, dtype=np.float32).reshape(1, 3)
        out = self.session.run(None, {'observations': obs})[0].ravel()
        if not np.all(np.isfinite(out)):
            self.get_logger().error('Non-finite action, dropping output'); return
        action = Float32MultiArray(); action.data = out.astype(float).tolist()
        self.pub.publish(action)

def main():
    rclpy.init(); node = PolicyNode()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()
if __name__ == '__main__': main()
