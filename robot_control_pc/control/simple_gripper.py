#!/usr/bin/env python3
import sys
import argparse
from robotiq_gripper import RobotiqGripperUSB

def main():
    parser = argparse.ArgumentParser(description="Simple Robotiq Gripper Control")
    parser.add_argument('command', choices=['open', 'close', 'status'], help="Command to execute")
    parser.add_argument('--port', default='/dev/ttyUSB0', help="Serial port (default: /dev/ttyUSB0)")
    args = parser.parse_args()

    print(f"🔌 Connecting to {args.port}...")
    gripper = RobotiqGripperUSB(portname=args.port)
    
    if not gripper.connect():
        sys.exit(1)

    if not gripper.activate():
        print("❌ Failed to activate")
        sys.exit(1)

    if args.command == 'open':
        print("👐 Opening...")
        print("✅ Done" if gripper.open() else "❌ Failed")

    elif args.command == 'close':
        print("✊ Closing...")
        print("✅ Done" if gripper.close() else "❌ Failed")

    elif args.command == 'status':
        status = gripper.get_status()
        if status:
            pos_m = 0.140 * (1.0 - (status['position'] / 255.0))
            print(f"📊 Position: {status['position']} ({pos_m:.3f}m)")
            print(f"   Active: {status['activated']}, GTO: {status['go_to']}, Fault: {status['fault']}")
        else:
            print("❌ Failed to get status")

    gripper.disconnect()

if __name__ == "__main__":
    main()
