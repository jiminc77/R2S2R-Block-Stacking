#!/bin/bash
CONTROL_IP="172.27.190.160"
VISION_IP="172.27.190.242"
WORKSPACE_PATH="$HOME/workspace/catkin_ws"

export ROS_MASTER_URI=http://$CONTROL_IP:11311
export ROS_IP=$VISION_IP
source $WORKSPACE_PATH/devel/setup.bash

echo "✅ [Current Terminal] Network settings applied immediately."

if grep -q "ROS_MASTER_URI=http://$CONTROL_IP:11311" ~/.bashrc; then
    echo "ℹ️  [.bashrc] Settings already exist. Skipping backup."
else
    echo "" >> ~/.bashrc
    echo "# --- ROS DIGITAL TWIN NETWORK SETUP ---" >> ~/.bashrc
    echo "export ROS_MASTER_URI=http://$CONTROL_IP:11311" >> ~/.bashrc
    echo "export ROS_IP=$VISION_IP" >> ~/.bashrc
    echo "source $WORKSPACE_PATH/devel/setup.bash" >> ~/.bashrc
    echo "# --------------------------------------" >> ~/.bashrc
    echo "✅ [.bashrc] Settings saved for future terminals."
fi

echo "----------------------------------------"
echo "ROS_MASTER_URI: $ROS_MASTER_URI"
echo "ROS_IP        : $ROS_IP"
echo "----------------------------------------"