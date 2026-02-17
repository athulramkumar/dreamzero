"""Arc Fabric MVP Demo: 3 Forks + Rewind on Failure.

This script demonstrates the core Arc Fabric capabilities:
1. Create a root session and generate a "trunk" trajectory
2. Fork 3 branches from the trunk with different prompts/seeds
3. Evaluate branches — if one "fails," rewind and retry
4. Launch the web UI to visualize the trajectory tree

Usage
-----
# Mock mode (no GPU required — demonstrates the full flow with fake data):
    ARC_FABRIC_MODE=mock python -m arc_fabric.demo

# Live mode (requires GPU + loaded DreamZero model):
    python -m arc_fabric.demo --model-path checkpoints/DreamZero-DROID

After the demo completes, the FastAPI server starts at http://localhost:8000
and opens the web UI showing the trajectory tree with all branches.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("arc_fabric.demo")


# ---------------------------------------------------------------------------
# Mock Demo (no GPU required)
# ---------------------------------------------------------------------------

def run_mock_demo(port: int = 8000) -> None:
    """Run the full demo flow in mock mode, then start the server."""
    logger.info("=" * 60)
    logger.info("ARC FABRIC MVP DEMO — Mock Mode")
    logger.info("=" * 60)

    # Import server components (mock mode doesn't need torch)
    os.environ["ARC_FABRIC_MODE"] = "mock"

    import importlib
    # Ensure the server module picks up mock mode
    if "arc_fabric.server" in sys.modules:
        importlib.reload(sys.modules["arc_fabric.server"])

    from arc_fabric.server import app, manager

    # ── Step 1: Create root session ──────────────────────────────────────
    logger.info("\n--- Step 1: Creating root session ---")
    root_sess = manager.create_session(prompt="pick up the red cup", initial_frame_path="")
    root_id = root_sess.session_id
    logger.info("Created root session: %s (prompt: 'pick up the red cup')", root_id)

    # ── Step 2: Generate trunk (frames 0→8) ──────────────────────────────
    logger.info("\n--- Step 2: Generating trunk trajectory (8 frames) ---")
    for i in range(8):
        root_sess.step(action=None, state=None, auto_checkpoint=True)
        logger.info("  Frame %d → checkpoint %s", root_sess.frame_index, root_sess.checkpoint_ids[-1] if root_sess.checkpoint_ids else "N/A")

    fork_point = root_sess.checkpoint_ids[-1]  # Frame 8
    logger.info("Trunk complete. Fork point: %s (frame %d)", fork_point, root_sess.frame_index)

    # ── Step 3: Fork 3 branches from frame 8 ─────────────────────────────
    logger.info("\n--- Step 3: Forking 3 branches from frame 8 ---")

    fork_configs = [
        {"prompt": None, "seed": None, "label": "Fork A (same prompt)"},
        {"prompt": "push the cup to the left", "seed": None, "label": "Fork B (different prompt)"},
        {"prompt": None, "seed": 42, "label": "Fork C (different seed)"},
    ]

    fork_sessions = []
    for cfg in fork_configs:
        forked = manager.fork_session(
            source_session_id=root_id,
            checkpoint_id=fork_point,
            new_prompt=cfg["prompt"],
            new_seed=cfg["seed"],
        )
        fork_sessions.append({"id": forked.session_id, "session": forked, **cfg})
        logger.info("  %s → session %s", cfg["label"], forked.session_id)

    # ── Step 4: Generate branches (frames 8→16) ──────────────────────────
    logger.info("\n--- Step 4: Generating branch trajectories (8 frames each) ---")

    branch_results = {}
    for fork_info in fork_sessions:
        sess = fork_info["session"]
        for i in range(8):
            sess.step(action=None, state=None, auto_checkpoint=True)

        branch_results[fork_info["id"]] = list(sess.checkpoint_ids)
        logger.info(
            "  %s: generated frames 9-16 (%d checkpoints)",
            fork_info["label"],
            len(sess.checkpoint_ids),
        )

    # ── Step 5: Evaluate — Fork C "fails" ────────────────────────────────
    logger.info("\n--- Step 5: Evaluating branches ---")
    failed_fork = fork_sessions[2]  # Fork C
    logger.info("  Fork A: SUCCESS — actions look reasonable")
    logger.info("  Fork B: SUCCESS — pushing motion detected")
    logger.info("  Fork C: FAILED — actions diverged too wildly (seed=42)")

    # ── Step 6: Rewind Fork C and retry ──────────────────────────────────
    logger.info("\n--- Step 6: Rewinding Fork C and retrying ---")

    # Create a retry session by forking from the same point
    retry_sess = manager.fork_session(
        source_session_id=root_id,
        checkpoint_id=fork_point,
        new_prompt="carefully pick up the red cup",
    )
    retry_id = retry_sess.session_id
    logger.info(
        "  Rewound to frame 8, created retry session %s with prompt: "
        "'carefully pick up the red cup'",
        retry_id,
    )

    # Generate retry branch
    for i in range(8):
        retry_sess.step(action=None, state=None, auto_checkpoint=True)

    logger.info("  Retry branch: generated frames 9-16")
    logger.info("  Retry: SUCCESS — careful approach worked!")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE — Trajectory Tree Summary")
    logger.info("=" * 60)
    total_cps = sum(len(s.checkpoint_ids) for s in manager.sessions.values())
    logger.info("  Root session:  %s ('pick up the red cup')", root_id)
    logger.info("  Trunk:         frames 0-8 (%d checkpoints)", len(root_sess.checkpoint_ids))
    logger.info("  Fork point:    frame 8 (%s)", fork_point)
    logger.info("  Fork A:        %s — same prompt, same seed (SUCCESS)", fork_sessions[0]["id"])
    logger.info("  Fork B:        %s — 'push the cup to the left' (SUCCESS)", fork_sessions[1]["id"])
    logger.info("  Fork C:        %s — same prompt, seed=42 (FAILED -> REWOUND)", fork_sessions[2]["id"])
    logger.info("  Retry:         %s — 'carefully pick up the red cup' (SUCCESS)", retry_id)
    logger.info(
        "  Total sessions: %d | Total checkpoints: %d",
        len(manager.sessions),
        total_cps,
    )
    logger.info("=" * 60)

    # ── Step 7: Start web server ─────────────────────────────────────────
    logger.info("\nStarting Arc Fabric server at http://localhost:%d", port)
    logger.info("Open the URL in your browser to see the trajectory tree.")
    logger.info("Press Ctrl+C to stop.\n")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


class _MockSession:
    """Minimal session object for the mock demo."""

    def __init__(self, session_id: str, prompt: str, frame_index: int = 0):
        self.session_id = session_id
        self.prompt = prompt
        self.frame_index = frame_index
        self.checkpoints: list[str] = []
        self.forked_from: str | None = None
        self.action_history: list[dict] = []
        self.created_at: float = time.time()


# ---------------------------------------------------------------------------
# Live Demo (requires GPU + model)
# ---------------------------------------------------------------------------

def run_live_demo(model_path: str, port: int = 8000) -> None:
    """Run the full demo with an actual DreamZero model on GPU."""
    logger.info("=" * 60)
    logger.info("ARC FABRIC MVP DEMO — Live Mode")
    logger.info("=" * 60)
    logger.info("Model path: %s", model_path)

    # Ensure live mode
    os.environ["ARC_FABRIC_MODE"] = "live"

    import numpy as np
    import torch

    from groot.vla.model.dreamzero.base_vla import VLA
    from arc_fabric.state import WorldStateSnapshot
    from arc_fabric.manager import StateManager
    from arc_fabric.session import Session
    from arc_fabric.tree import TrajectoryTree

    # ── Load model ────────────────────────────────────────────────────────
    logger.info("\nLoading DreamZero model...")
    model = VLA.from_pretrained(model_path)
    model.post_initialize()
    model.eval()
    logger.info("Model loaded successfully.")

    state_manager = StateManager(model, max_checkpoints=100)
    tree = TrajectoryTree()

    # ── Create initial frame (from test data or synthetic) ────────────────
    # Try to load a real frame, fall back to synthetic
    initial_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    test_image_dir = os.path.join(os.path.dirname(model_path), "..", "debug_image")
    if os.path.exists(test_image_dir):
        import glob
        images = sorted(glob.glob(os.path.join(test_image_dir, "*.png")))
        if images:
            import cv2
            initial_frame = cv2.imread(images[0])
            initial_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB)
            logger.info("Loaded initial frame from %s", images[0])

    # Default action and state for DROID
    default_action = torch.zeros(1, 24, 7, dtype=torch.bfloat16)  # [B, horizon, dim]
    default_state = torch.zeros(1, 1, 64, dtype=torch.bfloat16)   # [B, num_state, dim]

    # ── Step 1: Create root session ──────────────────────────────────────
    logger.info("\n--- Step 1: Creating root session ---")
    root = Session(
        session_id=f"sess-{uuid.uuid4().hex[:8]}",
        model=model,
        state_manager=state_manager,
        tree=tree,
        prompt="pick up the red cup",
        initial_frame=initial_frame,
        embodiment_id=0,
    )
    logger.info("Root session: %s", root.session_id)

    # ── Step 2: Generate trunk (8 steps) ─────────────────────────────────
    logger.info("\n--- Step 2: Generating trunk (8 frames) ---")
    for i in range(8):
        result = root.step(
            action=default_action,
            state=default_state,
            auto_checkpoint=True,
        )
        logger.info("  Frame %d → checkpoint %s", result["frame_index"], result.get("checkpoint_id", "N/A"))

    fork_checkpoint = root.checkpoint_ids[-1]
    logger.info("Trunk complete. Fork point: %s (frame %d)", fork_checkpoint, root.frame_index)

    # ── Step 3: Fork 3 branches ──────────────────────────────────────────
    logger.info("\n--- Step 3: Forking 3 branches ---")

    fork_a = root.fork(fork_checkpoint, new_prompt=None, new_seed=None)
    logger.info("  Fork A: %s (same prompt, same seed)", fork_a.session_id)

    fork_b = root.fork(fork_checkpoint, new_prompt="push the cup to the left")
    logger.info("  Fork B: %s (prompt: 'push the cup to the left')", fork_b.session_id)

    fork_c = root.fork(fork_checkpoint, new_prompt=None, new_seed=42)
    logger.info("  Fork C: %s (same prompt, seed=42)", fork_c.session_id)

    # ── Step 4: Generate branches (8 steps each) ─────────────────────────
    logger.info("\n--- Step 4: Generating branch trajectories ---")

    for label, session in [("Fork A", fork_a), ("Fork B", fork_b), ("Fork C", fork_c)]:
        logger.info("  Generating %s...", label)
        for i in range(8):
            result = session.step(
                action=default_action,
                state=default_state,
                auto_checkpoint=True,
            )
        logger.info("    %s: frame %d reached (%d checkpoints)", label, session.frame_index, len(session.checkpoint_ids))

    # ── Step 5: Evaluate branches ─────────────────────────────────────────
    logger.info("\n--- Step 5: Evaluating branches ---")

    # Simple heuristic: check action prediction variance
    # In a real scenario, you'd use a more sophisticated evaluator
    fork_c_failed = True  # Simulate failure detection
    logger.info("  Fork A: ✓ Success")
    logger.info("  Fork B: ✓ Success")
    logger.info("  Fork C: ✗ FAILED — simulating failure for demo")

    # ── Step 6: Rewind Fork C and retry ──────────────────────────────────
    if fork_c_failed:
        logger.info("\n--- Step 6: Rewinding Fork C and retrying ---")

        # Rewind to fork point
        fork_c.rewind(n_steps=8)
        logger.info("  Rewound Fork C to frame %d", fork_c.frame_index)

        # Update prompt for retry
        fork_c.prompt = "carefully pick up the red cup"
        logger.info("  Updated prompt: 'carefully pick up the red cup'")

        # Retry generation
        for i in range(8):
            result = fork_c.step(
                action=default_action,
                state=default_state,
                auto_checkpoint=True,
            )
        logger.info("  Retry complete: frame %d (%d checkpoints)", fork_c.frame_index, len(fork_c.checkpoint_ids))

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)
    logger.info("  Sessions: root + 3 forks = 4 total")
    logger.info("  Tree nodes: %d", len(tree.nodes))
    logger.info("  Fork points: %d", len(tree.get_fork_points()))
    logger.info("  Leaf branches: %d", len(tree.get_leaves()))
    logger.info(
        "  State manager: %d checkpoints (%.1f MB)",
        len(state_manager.checkpoints),
        state_manager.total_memory_bytes() / (1024 * 1024),
    )
    logger.info("=" * 60)

    # ── Start server ──────────────────────────────────────────────────────
    logger.info("\nStarting Arc Fabric server at http://localhost:%d", port)

    # Inject live objects into the server module
    from arc_fabric import server as srv_module
    srv_module.sessions = {
        root.session_id: root,
        fork_a.session_id: fork_a,
        fork_b.session_id: fork_b,
        fork_c.session_id: fork_c,
    }

    import uvicorn
    uvicorn.run(srv_module.app, host="0.0.0.0", port=port, log_level="info")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Arc Fabric MVP Demo")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to DreamZero checkpoint (enables live mode)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web server (default: 8000)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode even if --model-path is provided",
    )
    args = parser.parse_args()

    if args.mock or args.model_path is None:
        run_mock_demo(port=args.port)
    else:
        run_live_demo(model_path=args.model_path, port=args.port)


if __name__ == "__main__":
    main()
