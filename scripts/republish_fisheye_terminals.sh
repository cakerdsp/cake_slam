#!/usr/bin/env bash
set -euo pipefail

LEFT_CMD='rosrun image_transport republish compressed in:=/fisheye/left/image_raw raw out:=/fisheye/left/image_raw'
RIGHT_CMD='rosrun image_transport republish compressed in:=/fisheye/right/image_raw raw out:=/fisheye/right/image_raw'

run_in_terminal() {
  local title="$1"
  local cmd="$2"

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --title="$title" -- bash -lc "$cmd; exec bash"
  elif command -v konsole >/dev/null 2>&1; then
    konsole --new-tab --title "$title" -e bash -lc "$cmd; exec bash"
  elif command -v xterm >/dev/null 2>&1; then
    xterm -T "$title" -e bash -lc "$cmd; exec bash" &
  else
    echo "No supported terminal found. Install gnome-terminal, konsole, or xterm." >&2
    exit 1
  fi
}

if ! command -v rosrun >/dev/null 2>&1; then
  echo "rosrun not found. Please source your ROS environment first, for example:" >&2
  echo "  source /opt/ros/noetic/setup.bash" >&2
  echo "  source ~/catkin_ws/devel/setup.bash" >&2
  exit 1
fi

run_in_terminal "fisheye left compressed -> raw" "$LEFT_CMD"
run_in_terminal "fisheye right compressed -> raw" "$RIGHT_CMD"

echo "Started fisheye compressed-to-raw republish terminals."
