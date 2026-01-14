#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

def get_current_pose():
    rospy.init_node('get_robot_pose', anonymous=True)
    
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    rospy.sleep(1.0)  # Wait for TF to populate
    
    try:
        # Get transform from base_link to tool0 (end-effector)
        transform = tf_buffer.lookup_transform('base_link', 'tool0', rospy.Time(0), rospy.Duration(2.0))
        
        pos = transform.transform.translation
        rot = transform.transform.rotation
        
        print("\n" + "="*50)
        print("현재 로봇 End-Effector 위치 (base_link 기준)")
        print("="*50)
        print(f"Position:")
        print(f"  X: {pos.x:.4f} m")
        print(f"  Y: {pos.y:.4f} m")
        print(f"  Z: {pos.z:.4f} m")
        print(f"\nOrientation (Quaternion):")
        print(f"  x: {rot.x:.4f}")
        print(f"  y: {rot.y:.4f}")
        print(f"  z: {rot.z:.4f}")
        print(f"  w: {rot.w:.4f}")
        print("="*50 + "\n")
        
        # Return as dict for potential use
        return {
            'position': [pos.x, pos.y, pos.z],
            'orientation': [rot.x, rot.y, rot.z, rot.w]
        }
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print(f"\n❌ TF 조회 실패: {e}")
        print("  - vision_pc.launch가 실행 중인지 확인하세요")
        print("  - 로봇이 연결되어 있는지 확인하세요\n")
        return None

if __name__ == '__main__':
    try:
        get_current_pose()
    except rospy.ROSInterruptException:
        pass
