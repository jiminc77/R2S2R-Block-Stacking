from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros_bridge")

import rospy
from geometry_msgs.msg import PoseArray

class PoseListener:
    """Subscribes to /block_poses and latches the first valid message or waits."""
    def __init__(self, topic="/block_poses"):
        self.topic = topic
        self.poses = None

    def wait_for_poses(self, timeout=10.0):
        print(f"[ROS] Waiting for {self.topic}...")
        try:
            msg = rospy.wait_for_message(self.topic, PoseArray, timeout=timeout)
            return self._process_msg(msg)
        except rospy.ROSException:
            print("[ROS] Timeout waiting for poses.")
            return None

    def _process_msg(self, msg):
        poses = []
        for p in msg.poses:
            # Return combined [x, y, z, w, x, y, z]
            poses.append([
                p.position.x, p.position.y, p.position.z,
                p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z
            ])
        return poses
