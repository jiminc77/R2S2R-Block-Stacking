#!/usr/bin/env python3

import subprocess
import sys
import time
import json
import rospy
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String


class StackingPipeline:
    """Manages sequential execution of scanner and policy search"""

    def __init__(self):
        rospy.init_node('stacking_pipeline', anonymous=True)
        self.scan_complete = False
        self.policy_complete = False
        self.max_retries = 3
        self.scanner_proc = None
        self.policy_proc = None
        self.policy_start_time = 0
        self.workspace_root = '/home/ailab/workspace/catkin_ws/src/digital_twin'
        rospy.on_shutdown(self.shutdown)

    def run_scanner(self, attempt=1):
        """Execute ArUco scanner with retry logic"""
        rospy.loginfo(f"[Scanner] Attempt {attempt}/{self.max_retries}")
        self.scan_complete = False
        
        sub = rospy.Subscriber('/block_poses', PoseArray, self._scan_callback)
        self.scanner_proc = subprocess.Popen(
            ['python3', 'scripts/aruco_multiview_scanner.py'],
            cwd=self.workspace_root,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        timeout = rospy.Time.now() + rospy.Duration(300) # 5 min timeout
        rate = rospy.Rate(2)

        while not self.scan_complete and not rospy.is_shutdown():
            if rospy.Time.now() > timeout:
                rospy.logwarn("[Scanner] Timeout reached.")
                self.scanner_proc.terminate()
                self.scanner_proc.wait(timeout=10)
                break

            if self.scanner_proc.poll() is not None:
                rospy.logwarn(f"[Scanner] Process exited code: {self.scanner_proc.returncode}")
                break
            
            rate.sleep()

        sub.unregister()

        if not self.scan_complete:
            if attempt < self.max_retries:
                rospy.logwarn("[Scanner] Retrying in 2s...")
                time.sleep(2)
                return self.run_scanner(attempt + 1)
            else:
                rospy.logerr("[Scanner] Failed after max retries.")
                return False

        rospy.loginfo("[Scanner] Complete.")
        return True

    def _scan_callback(self, msg):
        if len(msg.poses) > 0 and not self.scan_complete:
            rospy.loginfo(f"[Scanner] Detected {len(msg.poses)} blocks.")
            self.scan_complete = True

    def run_policy_search(self):
        """Execute Isaac Sim optimal stacking policy search inside Docker container"""
        rospy.loginfo("[Policy Search] Launching Docker container...")
        self.policy_complete = False
        self.policy_start_time = time.time()

        sub = rospy.Subscriber('/optimal_stacking/result', String, self._policy_callback)

        cmd = [
            'docker', 'exec',
            '-w', '/workspace/catkin_ws/src/digital_twin',
            'isaac-sim',
            'bash', '-c',
            'ln -sf /isaac-sim/kit/python/bin/python3 /usr/bin/python3 && '
            'source /isaac-sim/exts/omni.isaac.ros_bridge/noetic/setup.bash && '
            '/isaac-sim/python.sh scripts/optimal_stacking/run_policy_search.py --num_envs 4096 --headless'
        ]

        self.policy_proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        
        # 10 minute timeout
        timeout = rospy.Time.now() + rospy.Duration(600)
        rate = rospy.Rate(1)
        success = False

        while not self.policy_complete and not rospy.is_shutdown():
            if rospy.Time.now() > timeout:
                rospy.logerr("[Policy Search] Timeout.")
                self.policy_proc.terminate()
                break

            if self.policy_proc.poll() is not None:
                if self.policy_proc.returncode != 0:
                     rospy.logwarn(f"[Policy Search] Exited with error: {self.policy_proc.returncode}")
                     break
                # Only warn if it exits unexpectedly without result
                rospy.logwarn("[Policy Search] Process exited unexpectedly.")
                break

            rate.sleep()

        if self.policy_complete:
            rospy.loginfo("[Policy Search] Result received.")
            success = True
        else:
            if self.policy_proc.poll() is None:
                self.policy_proc.terminate()

        sub.unregister()
        return success

    def _policy_callback(self, msg):
        if not self.policy_complete:
            try:
                data = json.loads(msg.data)
                if data.get('generated_at', 0) > self.policy_start_time:
                    self.policy_complete = True
            except Exception as e:
                rospy.logwarn(f"[Policy Search] Parse error: {e}")

    def shutdown(self):
        rospy.loginfo("Shutting down pipeline...")
        if self.scanner_proc and self.scanner_proc.poll() is None:
            self.scanner_proc.terminate()
        if self.policy_proc and self.policy_proc.poll() is None:
            self.policy_proc.terminate()

    def execute(self):
        rospy.loginfo("--- Starting Stacking Pipeline ---")

        # Phase 1: Scanner
        if not self.run_scanner():
            return False

        # Phase 2: Policy Search
        if not self.run_policy_search():
            return False

        rospy.loginfo("--- Pipeline Complete ---")
        return True


def main():
    try:
        pipeline = StackingPipeline()
        if pipeline.execute():
            rospy.loginfo("Node kept alive for distributed ROS system. Press Ctrl+C to exit.")
            rospy.spin()
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        rospy.logerr(f"Pipeline error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import traceback
    main()
