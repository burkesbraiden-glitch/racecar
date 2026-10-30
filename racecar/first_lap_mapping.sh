#!/usr/bin/env bash
# Run one timed mapping lap and save the resulting map automatically.
# Usage: bash first_lap_mapping.sh [lap_seconds] [map_name] [tracking_delay_seconds] [route_spacing_metres] [extra launch args...]

LAP_DURATION="${1:-${LAP_DURATION:-20}}"
if [ "$#" -ge 1 ]; then shift; fi

MAP_NAME="${1:-${MAP_NAME:-first_lap_$(date +%Y%m%d_%H%M%S)}}"
if [ "$#" -ge 1 ]; then shift; fi

TRACKING_START_DELAY="${1:-${TRACKING_START_DELAY:-12}}"
if [ "$#" -ge 1 ]; then shift; fi

ROUTE_SAMPLE_DISTANCE="${ROUTE_SAMPLE_DISTANCE:-0.25}"
if [ "$#" -ge 1 ] && [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  ROUTE_SAMPLE_DISTANCE="$1"
  shift
fi

exec ros2 launch racecar first_lap_mapping.launch.py \
  lap_duration:="$LAP_DURATION" \
  map_name:="$MAP_NAME" \
  tracking_start_delay:="$TRACKING_START_DELAY" \
  route_sample_distance:="$ROUTE_SAMPLE_DISTANCE" \
  "$@"
