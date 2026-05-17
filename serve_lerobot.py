"""LeRobot-backed websocket policy server for SO-101 (PyTorch, no TRT).

Mirrors the inference pipeline from ros2_collection/so101_ros2/inference/run_inference.py:
  preprocessor → policy.predict_action_chunk → postprocessor

Returns {"actions": float32 [H, 6]} in LeRobot-normalized space (-100..100 body, 0..100 gripper).
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    force=True)
log = logging.getLogger("serve_lerobot")


# ─── Tokenizer patch (must run before any processor loading) ─────────────────
def _patch_paligemma_tokenizer() -> None:
    """Redirect AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224') to
    a local SentencePiece model so we don't need HuggingFace auth."""
    from pathlib import Path
    import os
    spm_candidates = [
        Path(os.path.expanduser("~/.cache/openpi/big_vision/paligemma_tokenizer.model")),
        Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model"),
    ]
    spm_path = next((p for p in spm_candidates if p.exists()), None)
    if spm_path is None:
        log.warning("No local paligemma_tokenizer.model found; AutoTokenizer may hit gated repo.")
        return
    log.info("Using local SentencePiece tokenizer: %s", spm_path)

    import transformers
    from transformers import GemmaTokenizer

    _orig = transformers.AutoTokenizer.from_pretrained

    def _patched(name_or_path, *args, **kwargs):
        s = str(name_or_path)
        if "paligemma" in s.lower() and not Path(s).exists():
            log.info("AutoTokenizer: redirecting %r -> local GemmaTokenizer", s)
            return GemmaTokenizer(vocab_file=str(spm_path))
        return _orig(name_or_path, *args, **kwargs)

    transformers.AutoTokenizer.from_pretrained = _patched


# ─── Policy + processor loading (matches reference run_inference.py) ─────────
def load_policy_and_processors(checkpoint_dir: str, device: str = "cuda"):
    """Load PI05Policy + pre/post processors using the factory function."""
    from pathlib import Path
    log.info("Importing lerobot ...")
    import lerobot
    log.info("  lerobot version: %s", getattr(lerobot, "__version__", "?"))

    ckpt = Path(checkpoint_dir).absolute()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    # Patch tokenizer BEFORE processor loading
    _patch_paligemma_tokenizer()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    # Load policy (same as reference)
    log.info("Loading PI05Policy from %s ...", ckpt)
    config = PreTrainedConfig.from_pretrained(ckpt)
    config.device = device
    policy = PI05Policy.from_pretrained(ckpt, config=config)
    policy = policy.to(device, dtype=torch.bfloat16)
    policy.eval()
    policy.reset()
    n_params = sum(p.numel() for p in policy.parameters()) / 1e9
    log.info("Policy: %s (%.2fB params) device=%s", type(policy).__name__, n_params, device)

    # Load processors via factory (registers pi05-specific steps)
    log.info("Loading preprocessor/postprocessor via make_pre_post_processors ...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    log.info("Preprocessor steps: %s",
             [type(s).__name__ for s in preprocessor.steps])
    log.info("Postprocessor steps: %s",
             [type(s).__name__ for s in postprocessor.steps])

    return policy, preprocessor, postprocessor


# ─── Observation builder (matches reference build_observation) ────────────────
def build_observation(obs: dict, device: str) -> dict:
    """Convert websocket obs dict to the format preprocessor expects.

    Reference build_observation returns:
        observation.images.{front,top,wrist}: (1, 3, H, W) float32 in [0, 1]
        observation.state: (1, 6) float32
        task: [str]
    """
    result: dict = {}

    # Images: uint8 (H, W, 3) RGB → float32 (1, 3, H, W) ∈ [0, 1]
    for key in ("observation.images.front", "observation.images.top", "observation.images.wrist"):
        if key in obs:
            arr = np.asarray(obs[key], dtype=np.uint8)
            t = torch.from_numpy(arr).to(device=device, dtype=torch.float32) / 255.0
            result[key] = t.permute(2, 0, 1).unsqueeze(0)

    # State: (6,) float32 → (1, 6) float32
    state = np.asarray(obs["observation.state"], dtype=np.float32)
    result["observation.state"] = (
        torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)
    )

    # Task: str → [str]
    task = obs.get("task", "") or ""
    result["task"] = [task]

    return result


# ─── Policy adapter for openpi websocket server ──────────────────────────────
class LeRobotPolicyAdapter:
    """Duck-typed policy with .infer(obs) and .metadata."""

    def __init__(self, policy, preprocessor, postprocessor, device: str, default_prompt: str):
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = device
        self._default_prompt = default_prompt
        self._infer_count = 0
        self.metadata = {
            "backend": "lerobot-pytorch",
            "device": device,
            "default_prompt": default_prompt,
        }

    @torch.inference_mode()
    def infer(self, obs: dict) -> dict:
        t0 = time.time()
        self._infer_count += 1

        # Ensure task is set
        if "task" not in obs and "prompt" in obs:
            obs["task"] = obs["prompt"]
        if not obs.get("task"):
            obs["task"] = self._default_prompt

        # Build observation dict (with batch dim, matching reference)
        observation = build_observation(obs, self._device)

        # preprocessor → policy → postprocessor (same as reference)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batch = self._preprocessor(observation)
            actions = self._policy.predict_action_chunk(batch)
            actions = self._postprocessor(actions)

        # actions: Tensor (1, H, 6) in LeRobot-normalized space
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().float().numpy()
        actions = np.asarray(actions, dtype=np.float32)

        # Normalize shape to [H, 6]
        if actions.ndim == 3:
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions[None, :]

        dt_ms = (time.time() - t0) * 1e3

        # Verbose diagnostic on first few inferences
        if self._infer_count <= 3:
            state = np.asarray(obs.get("observation.state", []), dtype=np.float32)
            log.info("─── DIAGNOSTIC infer #%d ───", self._infer_count)
            log.info("  state (input): %s", np.round(state, 2).tolist())
            log.info("  task: %r", obs.get("task", "")[:80])
            for key in ("observation.images.front", "observation.images.top", "observation.images.wrist"):
                if key in obs:
                    img = np.asarray(obs[key])
                    log.info("  %s: shape=%s dtype=%s mean=%.1f",
                             key.split(".")[-1], img.shape, img.dtype, img.mean())
            log.info("  actions[0] (first step): %s", np.round(actions[0], 2).tolist())
            log.info("  actions[24] (mid step):  %s", np.round(actions[min(24, len(actions)-1)], 2).tolist())
            log.info("  actions[-1] (last step): %s", np.round(actions[-1], 2).tolist())
            log.info("  action range: min=%s max=%s",
                     np.round(actions.min(axis=0), 2).tolist(),
                     np.round(actions.max(axis=0), 2).tolist())
            log.info("  dt=%.0fms", dt_ms)
            log.info("───────────────────────────")
        else:
            log.info("infer: actions=%s dt=%.0fms task=%r",
                     actions.shape, dt_ms,
                     obs.get("task", "")[:50])

        return {"actions": actions}


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True,
                   help="Path to .../pretrained_model directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--default-prompt",
                   default="Pick up the white box and place it in the white target area.")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    log.info("PyTorch %s  CUDA available: %s  device: %s",
             torch.__version__, torch.cuda.is_available(), args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    policy, preprocessor, postprocessor = load_policy_and_processors(
        args.checkpoint_dir, args.device
    )

    adapter = LeRobotPolicyAdapter(
        policy, preprocessor, postprocessor, args.device, args.default_prompt,
    )

    from openpi.serving import websocket_policy_server

    host = socket.gethostname()
    try:
        ip = socket.gethostbyname(host)
    except Exception:
        ip = "?"
    log.info("Serving on ws://0.0.0.0:%d (host=%s ip=%s)", args.port, host, ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=adapter, host="0.0.0.0", port=args.port, metadata=adapter.metadata,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
