"""TRT-backed websocket policy server for SO-101.

Loads the converted PyTorch SafeTensors checkpoint, then monkey-patches the
model's sample_actions with a TensorRT engine. Serves the same websocket
protocol as scripts/serve_policy.py, so the existing run_so101 client works
unchanged.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys

logging.basicConfig(level=logging.INFO, force=True)

# Apply transformers + gemma patches before importing anything that touches them.
import subprocess
subprocess.run(
    ["bash", "-c",
     "cp -r /opt/openpi-train/src/openpi/models_pytorch/transformers_replace/* "
     "/usr/local/lib/python3.12/dist-packages/transformers/"],
    check=True,
)
subprocess.run(
    ["python", "/opt/openpi-train/openpi_on_thor/patches/apply_gemma_fixes.py"],
    check=True,
)

# Patch load_pytorch to handle tied-weight / dtype mismatches across container versions.
import safetensors.torch as _st
import openpi.models_pytorch.pi0_pytorch as _pi0pt

def _load_pytorch_patched(self, train_config, weight_path: str):
    model = _pi0pt.PI0Pytorch(config=train_config.model)
    state_dict = _st.load_file(weight_path)
    model.load_state_dict(state_dict, strict=False)
    return model

import openpi.models.model as _model_mod
for _cls in vars(_model_mod).values():
    if isinstance(_cls, type) and hasattr(_cls, "load_pytorch"):
        _cls.load_pytorch = _load_pytorch_patched

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from openpi_on_thor.trt_model_forward import setup_pi0_tensorrt_engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_so101_low_mem_finetune")
    p.add_argument("--checkpoint-dir", required=True,
                   help="PyTorch SafeTensors checkpoint dir")
    p.add_argument("--engine-path", required=True,
                   help="Path to TensorRT .engine file")
    p.add_argument("--default-prompt", default=None)
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    cfg = _config.get_config(args.config)
    logging.info("Loading PyTorch policy from %s", args.checkpoint_dir)
    policy = policy_config.create_trained_policy(
        cfg, args.checkpoint_dir, default_prompt=args.default_prompt,
    )

    logging.info("Wrapping policy with TensorRT engine: %s", args.engine_path)
    policy = setup_pi0_tensorrt_engine(policy, args.engine_path)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Serving on ws://%s:%d (host %s)", local_ip, args.port, hostname)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
