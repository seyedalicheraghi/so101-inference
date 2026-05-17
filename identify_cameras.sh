#!/usr/bin/env bash
# identify_cameras.sh -- interactively map front/top/wrist cameras to /dev/videoN.
#
# Assumes the "top" camera is an Intel RealSense (auto-detected via v4l2 card
# name). Front and wrist are plain USB webcams you can physically unplug.
#
# Output: the exact ./run_client.sh invocation with --cam-front/--cam-top/--cam-wrist.
#
# Requires: v4l-utils  (sudo apt install -y v4l-utils)

set -euo pipefail

if ! command -v v4l2-ctl >/dev/null; then
  echo "ERROR: v4l2-ctl not found." >&2
  echo "Install with: sudo apt install -y v4l-utils" >&2
  exit 1
fi

snapshot() { ls /dev/video* 2>/dev/null | sort; }

realsense_nodes() {
  local out=()
  for v in /dev/video*; do
    [[ -e "$v" ]] || continue
    if v4l2-ctl --device "$v" --info 2>/dev/null | grep -qi "realsense"; then
      out+=("$v")
    fi
  done
  printf "%s\n" "${out[@]:-}"
}

capture_node() {
  local lowest="" lowest_num=99999 n
  for v in "$@"; do
    [[ -n "$v" ]] || continue
    n="${v##*/video}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    if (( n < lowest_num )); then lowest_num=$n; lowest=$v; fi
  done
  echo "$lowest"
}

removed() {
  comm -23 \
    <(printf "%s\n" $1 | sort -u) \
    <(printf "%s\n" $2 | sort -u)
}

idx_of() { echo "${1##*/video}"; }

identify() {
  local label="$1"
  echo                                                                  >&2
  echo ">>> Unplug the **$label** camera USB cable now, then press Enter." >&2
  read -r _
  sleep 1
  local after gone node
  after=$(snapshot)
  gone=$(removed "$BEFORE_ALL" "$after")
  if [[ -z "$gone" ]]; then
    echo "ERROR: nothing disappeared from /dev/video*." >&2
    echo "       Either you did not unplug it, or it is shared with another bus." >&2
    return 1
  fi
  node=$(capture_node $gone)
  echo "    detected $label = $node  (all vanished nodes: $(echo $gone | tr '\n' ' '))" >&2
  echo                                                                  >&2
  echo ">>> Plug the $label camera back in, then press Enter."           >&2
  read -r _
  for _i in 1 2 3 4 5; do
    sleep 2
    [[ "$(snapshot)" == "$BEFORE_ALL" ]] && break
  done
  if [[ "$(snapshot)" != "$BEFORE_ALL" ]]; then
    echo "    WARN: not all video nodes returned yet (continuing anyway)" >&2
  fi
  echo "$node"
}

echo "=== Current /dev/video* nodes ==="
snapshot | sed "s/^/  /"

RS_LIST=$(realsense_nodes)
if [[ -z "$RS_LIST" ]]; then
  echo
  echo "WARN: no RealSense detected. The TOP camera will be left UNSET (?)." >&2
  TOP=""
else
  # prefer the YUYV/MJPG/RGB (color) node among RealSense nodes; OpenCV cannot decode Z16/GREY
  TOP=""
  for v in $RS_LIST; do
    fmts=$(v4l2-ctl --device="$v" --list-formats 2>/dev/null)
    if echo "$fmts" | grep -qE "YUYV|MJPG|RGB"; then TOP="$v"; break; fi
  done
  [[ -z "$TOP" ]] && TOP=$(capture_node $RS_LIST)
  echo
  echo "RealSense detected for TOP:"
  echo "  capture node : $TOP"
  echo "  all RS nodes : $(echo $RS_LIST | tr '\n' ' ')"
fi

BEFORE_ALL=$(snapshot | tr '\n' ' ')

WRIST=$(identify "WRIST") || exit 1
FRONT=$(identify "FRONT") || exit 1

echo
echo "===================== RESULT ====================="
printf "  wrist : %-12s\n" "$WRIST"
printf "  front : %-12s\n" "$FRONT"
printf "  top   : %-12s  (Intel RealSense)\n" "${TOP:-???}"
echo
echo "Run with:"
echo "  cd ~/src/inference"
if [[ -n "$TOP" ]]; then
  printf "  ./run_client.sh --cam-front %s --cam-top %s --cam-wrist %s\n" \
    "$(idx_of "$FRONT")" "$(idx_of "$TOP")" "$(idx_of "$WRIST")"
else
  printf "  ./run_client.sh --cam-front %s --cam-wrist %s --cam-top <UNKNOWN>\n" \
    "$(idx_of "$FRONT")" "$(idx_of "$WRIST")"
fi
