#!/usr/bin/env bash
# image_capture.sh — Launch the ROS2 image capture viewer
# Usage: ./image_capture.sh [--help]
# Press '1' in the display window to capture RGB + Depth images.

set -e

# ─── Configurable defaults ────────────────────────────────────────────────────
RGB_TOPIC="/zed/zed_node/left/color/rect/image"
DEPTH_TOPIC="/zed/zed_node/depth/depth_registered"
CAMERA_INFO_TOPIC="/zed/zed_node/left/color/rect/camera_info"
OUTPUT_DIR="./captured_images"
DEPTH_SCALE="1000.0"          # depth raw unit → meters divisor (1000 = mm→m)
WINDOW_WIDTH="640"
WINDOW_HEIGHT="360"
QOS_RELIABILITY="best_effort" # "reliable" or "best_effort"

# ─── Paths ────────────────────────────────────────────────────────────────────
# Cross-shell compatible script directory detection
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  # Running in bash
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  # Running in zsh
  SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
  # Fallback for other shells
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

PYTHON_SCRIPT="${SCRIPT_DIR}/image_capture.py"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Auto-detect ROS2 distro and shell-appropriate setup file
ROS2_SETUP=""
CURRENT_SHELL="$(basename "${SHELL:-/bin/bash}")"
if [[ "${CURRENT_SHELL}" == "zsh" ]]; then
  SETUP_FILE="setup.zsh"
else
  SETUP_FILE="setup.bash"
fi
for distro in rolling jazzy iron humble galactic foxy; do
  if [[ -f "/opt/ros/${distro}/${SETUP_FILE}" ]]; then
    ROS2_SETUP="/opt/ros/${distro}/${SETUP_FILE}"
    break
  fi
done
if [[ -z "${ROS2_SETUP}" ]]; then
  echo "[ERROR] No ROS2 installation found in /opt/ros/"
  exit 1
fi

# ─── Help ─────────────────────────────────────────────────────────────────────
if [[ "${1}" == "--help" || "${1}" == "-h" ]]; then
  cat <<EOF
Usage: $(basename "$0") [options]

Options (override defaults):
  --rgb-topic          RGB image topic       (default: ${RGB_TOPIC})
  --depth-topic        Depth image topic     (default: ${DEPTH_TOPIC})
  --camera-info-topic  CameraInfo topic      (default: ${CAMERA_INFO_TOPIC})
  --output-dir         Save directory        (default: ${OUTPUT_DIR})
  --depth-scale        Depth scale factor    (default: ${DEPTH_SCALE})
  --window-width       Window width px       (default: ${WINDOW_WIDTH})
  --window-height      Window height px      (default: ${WINDOW_HEIGHT})
  --qos-reliability    reliable|best_effort  (default: ${QOS_RELIABILITY})
  -h, --help           Show this help

Keyboard shortcuts in viewer:
  1      Capture current RGB + Depth + CameraInfo to OUTPUT_DIR
  q/ESC  Quit

Test with a ROS bag:
  ros2 bag play ~/Project/my_ros2_tools/data/ros_bag/zed_left_bag/ --loop &
  ./$(basename "$0")
EOF
  exit 0
fi

# ─── Validate environment ─────────────────────────────────────────────────────
if [[ ! -f "${ROS2_SETUP}" ]]; then
  echo "[ERROR] ROS2 setup not found at ${ROS2_SETUP}"
  exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "[ERROR] Python script not found: ${PYTHON_SCRIPT}"
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[ERROR] Virtual environment not found: ${VENV_DIR}"
  echo "        Create it with: python3 -m venv .venv && .venv/bin/pip install opencv-python numpy"
  exit 1
fi

# ─── Source ROS2 ──────────────────────────────────────────────────────────────
echo "[INFO] Sourcing ROS2: ${ROS2_SETUP}"
source "${ROS2_SETUP}"

# ─── Activate venv and inject ROS2 Python paths ───────────────────────────────
echo "[INFO] Activating venv: ${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Inject ROS2 Python packages into PYTHONPATH so the venv Python can find rclpy/cv_bridge
ROS2_PYTHON_PATH="/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages"
export PYTHONPATH="${ROS2_PYTHON_PATH}:${PYTHONPATH:-}"

# ─── Parse any extra CLI overrides passed to this script ──────────────────────
# Forward unrecognized args directly to Python (allows --rgb-topic etc. at call site)
EXTRA_ARGS=("$@")

# ─── Launch ───────────────────────────────────────────────────────────────────
echo "[INFO] Starting image capture viewer..."
echo "       RGB topic       : ${RGB_TOPIC}"
echo "       Depth topic     : ${DEPTH_TOPIC}"
echo "       CameraInfo topic: ${CAMERA_INFO_TOPIC}"
echo "       Output dir      : ${OUTPUT_DIR}"
echo ""

exec python3 "${PYTHON_SCRIPT}" \
  --rgb-topic         "${RGB_TOPIC}" \
  --depth-topic       "${DEPTH_TOPIC}" \
  --camera-info-topic "${CAMERA_INFO_TOPIC}" \
  --output-dir        "${OUTPUT_DIR}" \
  --depth-scale       "${DEPTH_SCALE}" \
  --window-width      "${WINDOW_WIDTH}" \
  --window-height     "${WINDOW_HEIGHT}" \
  --qos-reliability   "${QOS_RELIABILITY}" \
  "${EXTRA_ARGS[@]}"
