#!/usr/bin/env python3
import yaml
import sys
import rospy
import tf2_ros
import geometry_msgs.msg

def publish_handeye_tf():
    rospy.init_node('handeye_tf_publisher')
    
    # Get config file path
    config_file = rospy.get_param('~config_file', '')
    if not config_file:
        rospy.logerr("No config_file parameter provided!")
        sys.exit(1)
        
    rospy.loginfo(f"Loading Hand-Eye Calibration from: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            calib_data = yaml.safe_load(f)
            
        trans_data = calib_data['transformation']
        params = calib_data['parameters']
        
        # Frames
        parent_frame = params['robot_effector_frame']   # e.g., tool0
        child_frame = params['tracking_base_frame']     # e.g., camera_color_optical_frame
        
        # Transform
        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = trans_data['x']
        t.transform.translation.y = trans_data['y']
        t.transform.translation.z = trans_data['z']
        t.transform.rotation.x = trans_data['qx']
        t.transform.rotation.y = trans_data['qy']
        t.transform.rotation.z = trans_data['qz']
        t.transform.rotation.w = trans_data['qw']
        
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # Publish static transform
        rospy.loginfo(f"Broadcasting static transform: {parent_frame} -> {child_frame}")
        broadcaster.sendTransform(t)
        
        rospy.spin()
        
    except Exception as e:
        rospy.logerr(f"Failed to load/publish calibration: {e}")
        sys.exit(1)

if __name__ == '__main__':
    publish_handeye_tf()
