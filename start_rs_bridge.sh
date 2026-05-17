#!/bin/bash
# Start RealSense camera via ROS2 + frame bridge for inference
# Run this on the HOST before starting the inference client
set -e
source /opt/ros/jazzy/setup.bash

echo "Starting RealSense camera node (color only, 640x480@30fps)..."
ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true \
    enable_depth:=false \
    enable_infra:=false \
    enable_infra1:=false \
    enable_infra2:=false \
    rgb_camera.color_profile:=640x480x30 \
    rgb_camera.enable_auto_exposure:=true &
RS_PID=$!
sleep 3

echo "Starting frame bridge (writes to /tmp/inference_diag/rs_top.npy)..."
python3 ~/src/inference/rs_bridge.py &
BRIDGE_PID=$!

echo "RealSense bridge running (RS node PID=$RS_PID, bridge PID=$BRIDGE_PID)"
echo "Press Ctrl+C to stop"

trap "kill $RS_PID $BRIDGE_PID 2>/dev/null; exit 0" INT TERM
wait
