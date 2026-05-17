# SO-101 inference client. Talks to the openpi policy server over a websocket
# and drives the SO-101 follower (USB serial) + 3 USB cameras (V4L2).
#
# No lerobot dependency - we use feetech-servo-sdk directly to read/write the
# 6 STS3215 motors. That keeps the image small (~200 MB) and the build fast.
# The trained model expects positions in degrees relative to the per-joint
# homing offset; pass --calibration <file> at runtime to match the data.

FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# openpi-client comes from the openpi repo via a docker build-context.
COPY --from=openpi-client . /opt/openpi-client/
RUN pip install /opt/openpi-client

# Direct Feetech motor driver (no lerobot, no torch).
RUN pip install \
        "numpy<2" \
        "opencv-python-headless<5" \
        "pyserial>=3.5" \
        "feetech-servo-sdk" "typing_extensions" "websockets" "msgpack"

WORKDIR /app
COPY run_so101.py /app/run_so101.py
COPY calibrate.py /app/calibrate.py
COPY home.py /app/home.py
COPY save_pose.py /app/save_pose.py

ENTRYPOINT ["python", "/app/run_so101.py"]
