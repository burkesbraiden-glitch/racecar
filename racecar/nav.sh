#!/bin/bash

# Function to handle termination
terminate() {
  echo "终止所有后台进程..."
  kill "$Run_car" "$Run_nav" 2>/dev/null
  wait "$Run_car" "$Run_nav" 2>/dev/null
  echo "所有进程已终止。"
}

# Trap Ctrl+C (SIGINT)
trap terminate SIGINT


# 启动 ROS 2 launch 文件

echo "启动 ROS 2 launch 文件: Run_car.launch.py"
ros2 launch racecar Run_car.launch.py &
Run_car=$!

# 等待5秒
sleep 10


# 启动 nav2 节点
 echo "启动 nav2 节点"
 ros2 launch racecar Run_nav.launch.py map:=/home/davinci-mini/racecar/src/racecar/map/2026_8_4_map.yaml &
 Run_nav=$!

echo "所有节点已启动"

#Wait for all background processes to complete
wait $Run_car  $Run_nav