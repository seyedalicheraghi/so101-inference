#!/usr/bin/env bash
# Build the openpi SO-101 inference client image.
# Uses docker buildkit's --build-context so we don't have to copy openpi-client
# into this folder.
set -euo pipefail
cd "$(dirname "$0")"

OPENPI_DIR="${OPENPI_DIR:-$HOME/src/openpi}"
TAG="${TAG:-openpi-client:so101}"

if [[ ! -d "$OPENPI_DIR/packages/openpi-client" ]]; then
  echo "ERROR: openpi-client package not found at $OPENPI_DIR/packages/openpi-client" >&2
  exit 1
fi

echo "Building $TAG"
echo "  openpi-client = $OPENPI_DIR/packages/openpi-client"
DOCKER_BUILDKIT=1 docker build \
  -t "$TAG" \
  --build-context "openpi-client=$OPENPI_DIR/packages/openpi-client" \
  .
echo
echo "Done. Image: $TAG"
echo "Run with: ./run_client.sh ..."
