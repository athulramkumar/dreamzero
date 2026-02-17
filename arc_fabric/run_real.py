#!/usr/bin/env python3
"""Arc Fabric — Real example with live DreamZero model.

Demonstrates checkpoint, fork, and rewind against the actual DreamZero
model running on 2× H100 GPUs with distributed inference.

Usage:
    cd /workspace/dreamzero/dreamzero
    conda run -n dreamzero torchrun --standalone --nproc_per_node=2 \
        -m arc_fabric.run_real \
        --model-path checkpoints/DreamZero-DROID \
        --output-dir outputs

What it does (rank 0):
    1. Load model, create StateManager + TrajectoryTree
    2. Generate a 4-chunk "trunk" with prompt A
    3. Checkpoint the trunk
    4. Fork 1: continue 3 chunks with prompt B (different task)
    5. Fork 2: continue 3 chunks with prompt A but different seed
    6. Fork 3: continue 3 chunks with prompt C
    7. Simulate failure on fork 3 → rewind 2 steps → retry
    8. Decode all latents → save videos to --output-dir
    9. Print trajectory tree JSON
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pickle
import sys
import time

import cv2
import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch
from torch.distributed.device_mesh import init_device_mesh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("arc_fabric.run_real")

# ---------------------------------------------------------------------------
# Input data helpers (same frame schedule as test_client / eval_suite)
# ---------------------------------------------------------------------------
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug_image")

# AR_droid camera key → video filename
CAMERA_FILES = {
    "video.exterior_image_1_left": "exterior_image_1_left.mp4",
    "video.exterior_image_2_left": "exterior_image_2_left.mp4",
    "video.wrist_image_left": "wrist_image_left.mp4",
}
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24


def load_all_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames from {video_path}")
    return np.stack(frames, axis=0)


def load_camera_frames() -> dict[str, np.ndarray]:
    camera_frames = {}
    for cam_key, fname in CAMERA_FILES.items():
        path = os.path.join(VIDEO_DIR, fname)
        camera_frames[cam_key] = load_all_frames(path)
        logger.info(f"Loaded {cam_key}: {camera_frames[cam_key].shape}")
    return camera_frames


def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks = []
    current_frame = 23
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def make_obs(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    prompt: str,
    session_id: str,
) -> dict:
    """Build observation in AR_droid format (what GrootSimPolicy expects).

    Keys use ``video.`` and ``state.`` prefixes, matching the transform pipeline.
    """
    obs: dict = {}
    for cam_key, all_frames in camera_frames.items():
        sel = all_frames[frame_indices]  # (T, H, W, 3) or (H, W, 3) if len==1
        obs[cam_key] = sel  # T H W C uint8

    obs["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)
    obs["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)
    obs["annotation.language.action_text"] = prompt
    return obs


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def broadcast_batch_to_workers(obs: dict) -> None:
    serialized = pickle.dumps(obs)
    size_tensor = torch.tensor([len(serialized)], dtype=torch.int64, device="cuda")
    dist.broadcast(size_tensor, src=0)
    data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
    dist.broadcast(data_tensor, src=0)


def receive_batch_from_rank0() -> Batch:
    size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
    dist.broadcast(size_tensor, src=0)
    data_size = size_tensor.item()
    data_tensor = torch.zeros(data_size, dtype=torch.uint8, device="cuda")
    dist.broadcast(data_tensor, src=0)
    obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
    return Batch(obs=obs)


# ---------------------------------------------------------------------------
# Video decode & save
# ---------------------------------------------------------------------------

def decode_and_save_video(
    video_chunks: list[torch.Tensor],
    policy,
    output_path: str,
    fps: int = 5,
) -> list[np.ndarray]:
    """Decode accumulated latent chunks through the VAE and save as MP4."""
    if not video_chunks:
        logger.warning("No video chunks to decode")
        return []

    ah = policy.trained_model.action_head
    cat = torch.cat(video_chunks, dim=2)
    frames_tensor = ah.vae.decode(
        cat,
        tiled=ah.tiled,
        tile_size=(ah.tile_size_height, ah.tile_size_width),
        tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
    )
    frames_tensor = rearrange(frames_tensor, "B C T H W -> B T H W C")
    frames_tensor = frames_tensor[0]
    frames_np = ((frames_tensor.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frame_list = [frames_np[i] for i in range(frames_np.shape[0])]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frame_list, fps=fps, codec="libx264")
    logger.info(f"Saved {len(frame_list)} frames to {output_path}")
    return frame_list


# ---------------------------------------------------------------------------
# Single inference step (coordinated across ranks)
# ---------------------------------------------------------------------------

def do_inference_step(
    policy,
    obs: dict,
    signal_group: dist.ProcessGroup,
) -> tuple:
    """Run one forward pass on rank 0, coordinating with all workers."""
    # Signal workers: 0 = continue
    signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
    dist.broadcast(signal_tensor, src=0, group=signal_group)

    broadcast_batch_to_workers(obs)
    batch = Batch(obs=obs)

    dist.barrier()
    with torch.no_grad():
        result_batch, video_pred = policy.lazy_joint_forward_causal(batch)
    dist.barrier()

    return result_batch, video_pred


# ---------------------------------------------------------------------------
# Worker loop (non-rank-0)
# ---------------------------------------------------------------------------

def worker_loop(
    policy,
    signal_group: dist.ProcessGroup,
) -> None:
    """Worker loop for rank != 0 — participates in distributed forward passes."""
    logger.info(f"Worker loop started for rank {dist.get_rank()}")
    signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")

    while True:
        try:
            dist.broadcast(signal_tensor, src=0, group=signal_group)
            signal = signal_tensor.item()

            if signal == 1:
                logger.info("Worker received shutdown signal")
                break
            if signal == 2:
                continue

            batch = receive_batch_from_rank0()
            dist.barrier()
            with torch.no_grad():
                policy.lazy_joint_forward_causal(batch)
            dist.barrier()
        except Exception as e:
            logger.error(f"Worker error: {e}")
            break


# ---------------------------------------------------------------------------
# Main Arc Fabric pipeline (rank 0)
# ---------------------------------------------------------------------------

def run_arc_fabric(
    policy,
    signal_group: dist.ProcessGroup,
    output_dir: str,
    trunk_chunks: int = 4,
    fork_chunks: int = 3,
) -> None:
    """Full Arc Fabric pipeline: trunk → checkpoint → 3 forks → rewind → retry."""

    # Import Arc Fabric components
    from arc_fabric.manager import StateManager
    from arc_fabric.tree import TrajectoryTree

    tree = TrajectoryTree()
    state_manager = StateManager(policy.trained_model, max_checkpoints=20)

    os.makedirs(output_dir, exist_ok=True)

    # ── Load input frames ────────────────────────────────────────────
    camera_frames = load_camera_frames()
    total_frames = min(v.shape[0] for v in camera_frames.values())
    schedule = build_frame_schedule(total_frames, trunk_chunks + fork_chunks + 5)
    logger.info(f"Frame schedule: {len(schedule)} chunks, total input frames={total_frames}")

    # ── Prompts for demonstration ────────────────────────────────────
    PROMPT_A = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
    PROMPT_B = "Pick up the cup and place it to the left"
    PROMPT_C = "Push the pan to the right side of the stove"

    # ==================================================================
    # PHASE 1: Trunk generation (shared prefix)
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Generating trunk (%d chunks, prompt A)", trunk_chunks)
    logger.info("=" * 60)

    trunk_videos: list[torch.Tensor] = []
    trunk_session_id = "trunk-session"
    chunk_idx = 0

    # Initial single-frame step
    obs = make_obs(camera_frames, [0], PROMPT_A, trunk_session_id)
    result, video_pred = do_inference_step(policy, obs, signal_group)
    trunk_videos.append(video_pred)
    logger.info(f"  Trunk initial: video_pred shape={video_pred.shape}")

    # Multi-frame chunks
    for ci in range(trunk_chunks):
        if ci >= len(schedule):
            break
        obs = make_obs(camera_frames, schedule[ci], PROMPT_A, trunk_session_id)
        result, video_pred = do_inference_step(policy, obs, signal_group)
        trunk_videos.append(video_pred)
        chunk_idx = ci + 1
        logger.info(f"  Trunk chunk {ci}: video_pred shape={video_pred.shape}")

    # Checkpoint the trunk
    trunk_cp_id = state_manager.checkpoint(
        session_id=trunk_session_id,
        prompt_text=PROMPT_A,
        action_history=[],
        frame_index=chunk_idx,
    )
    tree.add_node(
        checkpoint_id=trunk_cp_id,
        session_id=trunk_session_id,
        frame_index=chunk_idx,
        prompt=PROMPT_A,
        metadata={"phase": "trunk"},
    )
    logger.info(f"  Trunk checkpoint: {trunk_cp_id} at frame {chunk_idx}")
    logger.info(f"  StateManager: {state_manager}")

    # Save trunk video
    trunk_frames = decode_and_save_video(
        trunk_videos, policy,
        os.path.join(output_dir, "trunk.mp4"),
    )

    # ==================================================================
    # PHASE 2: Fork 1 — different prompt (PROMPT_B)
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Fork 1 — prompt B (%d chunks)", fork_chunks)
    logger.info("=" * 60)

    # Restore trunk state
    state_manager.restore(trunk_cp_id)
    fork1_videos = list(trunk_videos)  # copy trunk prefix
    fork1_session_id = "fork-1-prompt-b"

    for ci in range(fork_chunks):
        si = trunk_chunks + ci
        if si >= len(schedule):
            break
        obs = make_obs(camera_frames, schedule[si], PROMPT_B, fork1_session_id)
        result, video_pred = do_inference_step(policy, obs, signal_group)
        fork1_videos.append(video_pred)
        logger.info(f"  Fork 1 chunk {ci}: video_pred shape={video_pred.shape}")

    fork1_cp_id = state_manager.checkpoint(
        session_id=fork1_session_id,
        prompt_text=PROMPT_B,
        parent_checkpoint_id=trunk_cp_id,
        frame_index=chunk_idx + fork_chunks,
    )
    tree.add_node(
        checkpoint_id=fork1_cp_id,
        session_id=fork1_session_id,
        frame_index=chunk_idx + fork_chunks,
        prompt=PROMPT_B,
        parent_id=trunk_cp_id,
        metadata={"phase": "fork1-prompt-b"},
    )
    decode_and_save_video(
        fork1_videos, policy,
        os.path.join(output_dir, "fork1_prompt_b.mp4"),
    )

    # ==================================================================
    # PHASE 3: Fork 2 — same prompt A, different seed
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Fork 2 — prompt A, seed=999 (%d chunks)", fork_chunks)
    logger.info("=" * 60)

    state_manager.restore(trunk_cp_id)
    policy.trained_model.action_head.seed = 999
    fork2_videos = list(trunk_videos)
    fork2_session_id = "fork-2-seed-999"

    for ci in range(fork_chunks):
        si = trunk_chunks + ci
        if si >= len(schedule):
            break
        obs = make_obs(camera_frames, schedule[si], PROMPT_A, fork2_session_id)
        result, video_pred = do_inference_step(policy, obs, signal_group)
        fork2_videos.append(video_pred)
        logger.info(f"  Fork 2 chunk {ci}: video_pred shape={video_pred.shape}")

    fork2_cp_id = state_manager.checkpoint(
        session_id=fork2_session_id,
        prompt_text=PROMPT_A,
        parent_checkpoint_id=trunk_cp_id,
        frame_index=chunk_idx + fork_chunks,
    )
    tree.add_node(
        checkpoint_id=fork2_cp_id,
        session_id=fork2_session_id,
        frame_index=chunk_idx + fork_chunks,
        prompt=PROMPT_A,
        parent_id=trunk_cp_id,
        metadata={"phase": "fork2-seed-999", "seed": 999},
    )
    decode_and_save_video(
        fork2_videos, policy,
        os.path.join(output_dir, "fork2_seed_999.mp4"),
    )

    # ==================================================================
    # PHASE 4: Fork 3 — prompt C → simulate failure → rewind → retry
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PHASE 4: Fork 3 — prompt C → failure → rewind → retry")
    logger.info("=" * 60)

    state_manager.restore(trunk_cp_id)
    policy.trained_model.action_head.seed = 42
    fork3_videos = list(trunk_videos)
    fork3_session_id = "fork-3-prompt-c"

    fork3_checkpoints = []

    for ci in range(fork_chunks):
        si = trunk_chunks + ci
        if si >= len(schedule):
            break
        obs = make_obs(camera_frames, schedule[si], PROMPT_C, fork3_session_id)
        result, video_pred = do_inference_step(policy, obs, signal_group)
        fork3_videos.append(video_pred)

        # Checkpoint each step
        cp_id = state_manager.checkpoint(
            session_id=fork3_session_id,
            prompt_text=PROMPT_C,
            parent_checkpoint_id=fork3_checkpoints[-1] if fork3_checkpoints else trunk_cp_id,
            frame_index=chunk_idx + ci + 1,
        )
        fork3_checkpoints.append(cp_id)
        tree.add_node(
            checkpoint_id=cp_id,
            session_id=fork3_session_id,
            frame_index=chunk_idx + ci + 1,
            prompt=PROMPT_C,
            parent_id=fork3_checkpoints[-2] if len(fork3_checkpoints) >= 2 else trunk_cp_id,
            metadata={"phase": "fork3-prompt-c", "step": ci},
        )
        logger.info(f"  Fork 3 chunk {ci}: checkpoint {cp_id}")

    # Save the "failed" fork 3 video
    decode_and_save_video(
        fork3_videos, policy,
        os.path.join(output_dir, "fork3_prompt_c_FAILED.mp4"),
    )

    # ── Rewind 2 steps ───────────────────────────────────────────────
    logger.info("  Fork 3: SIMULATED FAILURE — rewinding 2 steps")
    rewind_steps = min(2, len(fork3_checkpoints))
    if rewind_steps > 0:
        rewind_target_idx = len(fork3_checkpoints) - rewind_steps - 1
        if rewind_target_idx < 0:
            rewind_target_id = trunk_cp_id
        else:
            rewind_target_id = fork3_checkpoints[rewind_target_idx]

        state_manager.restore(rewind_target_id)
        logger.info(f"  Rewound to checkpoint: {rewind_target_id}")

        # Retry with modified prompt
        PROMPT_C_RETRY = "Gently push the pan to the left side of the stove"
        retry_videos = list(trunk_videos)
        # Re-add the pre-rewind chunks
        for vi in range(max(0, rewind_target_idx + 1)):
            retry_videos.append(fork3_videos[len(trunk_videos) + vi])

        retry_session_id = "fork-3-retry"
        for ci in range(fork_chunks):
            si = trunk_chunks + max(0, rewind_target_idx + 1) + ci
            if si >= len(schedule):
                break
            obs = make_obs(camera_frames, schedule[si], PROMPT_C_RETRY, retry_session_id)
            result, video_pred = do_inference_step(policy, obs, signal_group)
            retry_videos.append(video_pred)
            logger.info(f"  Retry chunk {ci}: video_pred shape={video_pred.shape}")

        retry_cp_id = state_manager.checkpoint(
            session_id=retry_session_id,
            prompt_text=PROMPT_C_RETRY,
            parent_checkpoint_id=rewind_target_id,
            frame_index=chunk_idx + max(0, rewind_target_idx + 1) + fork_chunks,
        )
        tree.add_node(
            checkpoint_id=retry_cp_id,
            session_id=retry_session_id,
            frame_index=chunk_idx + max(0, rewind_target_idx + 1) + fork_chunks,
            prompt=PROMPT_C_RETRY,
            parent_id=rewind_target_id,
            metadata={"phase": "fork3-retry", "rewind_from": fork3_checkpoints[-1]},
        )
        decode_and_save_video(
            retry_videos, policy,
            os.path.join(output_dir, "fork3_retry.mp4"),
        )

    # ==================================================================
    # PHASE 5: Summary
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PHASE 5: Summary")
    logger.info("=" * 60)

    tree_json = tree.to_d3_tree()
    tree_path = os.path.join(output_dir, "trajectory_tree.json")
    with open(tree_path, "w") as f:
        json.dump(tree_json, f, indent=2, default=str)
    logger.info(f"Trajectory tree saved to {tree_path}")

    flat_tree = tree.to_dict()
    flat_path = os.path.join(output_dir, "trajectory_flat.json")
    with open(flat_path, "w") as f:
        json.dump(flat_tree, f, indent=2, default=str)

    logger.info(f"Total checkpoints stored: {len(state_manager.checkpoints)}")
    logger.info(f"Total memory: {state_manager.total_memory_bytes() / 1e9:.2f} GB")
    logger.info(f"Trajectory tree nodes: {len(tree)}")
    logger.info(f"Fork points: {len(tree.get_fork_points())}")
    logger.info(f"Leaves: {len(tree.get_leaves())}")

    videos_produced = [
        f for f in os.listdir(output_dir) if f.endswith(".mp4")
    ]
    logger.info(f"Videos saved to {output_dir}/:")
    for v in sorted(videos_produced):
        vpath = os.path.join(output_dir, v)
        logger.info(f"  {v}  ({os.path.getsize(vpath) / 1024:.0f} KB)")

    logger.info("Arc Fabric real example complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Arc Fabric — real example")
    parser.add_argument(
        "--model-path",
        default="checkpoints/DreamZero-DROID",
        help="Path to DreamZero checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save generated videos",
    )
    parser.add_argument("--trunk-chunks", type=int, default=4)
    parser.add_argument("--fork-chunks", type=int, default=3)
    args = parser.parse_args()

    # ── Distributed init ─────────────────────────────────────────────
    os.environ["ENABLE_DIT_CACHE"] = "true"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ip",))

    signal_group = dist.new_group(
        backend="gloo",
        timeout=datetime.timedelta(seconds=50000),
    )

    # ── Load model ───────────────────────────────────────────────────
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    logger.info(f"Rank {rank}: Loading model from {args.model_path}")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=args.model_path,
        device="cuda",
        device_mesh=device_mesh,
    )
    logger.info(f"Rank {rank}: Model loaded")

    # ── Run ───────────────────────────────────────────────────────────
    if rank == 0:
        try:
            run_arc_fabric(
                policy,
                signal_group,
                output_dir=args.output_dir,
                trunk_chunks=args.trunk_chunks,
                fork_chunks=args.fork_chunks,
            )
        finally:
            # Signal workers to shut down
            signal_tensor = torch.ones(1, dtype=torch.int32, device="cpu")
            dist.broadcast(signal_tensor, src=0, group=signal_group)
    else:
        worker_loop(policy, signal_group)

    dist.destroy_process_group()
    logger.info(f"Rank {rank}: Done")


if __name__ == "__main__":
    main()
