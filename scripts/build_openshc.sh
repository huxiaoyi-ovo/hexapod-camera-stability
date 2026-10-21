#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${HEXAPOD_OPENSHC_BUILD_DIR:-$ROOT/data/raw/openshc_build}"
ROS_SETUP="/opt/ros/noetic/setup.bash"

if [[ ! -r "$ROS_SETUP" ]]; then
  echo "ROS Noetic is required at $ROS_SETUP; no packages are installed by this script." >&2
  exit 2
fi
source "$ROS_SETUP"
if ! command -v catkin_make >/dev/null 2>&1; then
  echo "catkin_make is unavailable in the existing ROS Noetic environment." >&2
  exit 2
fi

mkdir -p "$BUILD_DIR/src"
if [[ ! -e "$BUILD_DIR/src/CMakeLists.txt" ]]; then
  ln -s /opt/ros/noetic/share/catkin/cmake/toplevel.cmake "$BUILD_DIR/src/CMakeLists.txt"
fi
ln -sfn "$ROOT/third_party/openshc" "$BUILD_DIR/src/syropod_highlevel_controller"
ln -sfn "$ROOT/integration/openshc" "$BUILD_DIR/src/openshc_adapter"
catkin_make -C "$BUILD_DIR" -j"${HEXAPOD_OPENSHC_JOBS:-2}" -DCMAKE_BUILD_TYPE=Release
test -x "$BUILD_DIR/devel/lib/openshc_adapter/openshc_bridge"
echo "OpenSHC bridge built: $BUILD_DIR/devel/lib/openshc_adapter/openshc_bridge"
