#!/usr/bin/env bash
# Replay a route recorded by first_lap_mapping.sh.
# Usage: bash second_lap.sh [map_name] [nav_delay_seconds] [max_speed_mps] [sign_arm_progress] [extra launch args...]

# A finished second-lap launch intentionally keeps publishing a stop command.
# Refuse a second copy instead of allowing two managers to fight each other.
if ros2 node list 2>/dev/null | grep -Fxq '/second_lap_manager'; then
  echo "错误：检测到旧的 /second_lap_manager 仍在运行。"
  echo "请回到旧 second_lap.sh 终端按 Ctrl+C，确认它完全退出后再重试。"
  exit 2
fi

MAP_NAME="${1:-${MAP_NAME:-first_lap_map}}"
if [ "$#" -ge 1 ]; then shift; fi

NAVIGATION_START_DELAY="${1:-${NAVIGATION_START_DELAY:-5}}"
if [ "$#" -ge 1 ]; then shift; fi

MAXIMUM_LINEAR_SPEED="${1:-${MAXIMUM_LINEAR_SPEED:-0.55}}"
if [ "$#" -ge 1 ]; then shift; fi

SIGN_ARM_PROGRESS="${1:-${SIGN_ARM_PROGRESS:-0.70}}"
if [ "$#" -ge 1 ]; then shift; fi

MAP_DIR="${MAP_DIR:-/home/davinci-mini/racecar/src/racecar/map}"
ARTIFACT_STEM="${MAP_NAME%_map}"

exec ros2 launch racecar second_lap.launch.py \
  map:="$MAP_DIR/$MAP_NAME.yaml" \
  route_file:="$MAP_DIR/${ARTIFACT_STEM}_route.yaml" \
  pose_file:="$MAP_DIR/${ARTIFACT_STEM}_pose.yaml" \
  navigation_start_delay:="$NAVIGATION_START_DELAY" \
  maximum_linear_speed:="$MAXIMUM_LINEAR_SPEED" \
  sign_arm_progress:="$SIGN_ARM_PROGRESS" \
  "$@"
