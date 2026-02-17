# Arc Fabric Architecture

> Golden reference document for all agents working on the Arc Fabric platform.
> Design principle: "Treat latents, timesteps, and control signals as first-class state."

---

## Overview

Arc Fabric is a state management platform for DreamZero, enabling:
- **Checkpointing** world state at any frame
- **Forking** from checkpoints with different actions/prompts
- **Rewinding** to previous states when a trajectory fails
- **Visualizing** trajectory trees and comparing divergent branches

Target: 2×H100 GPUs, max 5 concurrent sessions.

---

## Architecture Layers

```
┌────────────────────────────────────────────────────────────┐
│                    Web UI (Layer 3)                         │
│   arc_fabric/ui/index.html                                 │
│   - Trajectory tree visualization (D3.js)                  │
│   - Side-by-side video comparison (up to 3)                │
│   - Control panel: create, fork, rewind, step              │
│   - Timeline with branch points                            │
├────────────────────────────────────────────────────────────┤
│                  API Server (Layer 2)                       │
│   arc_fabric/server.py                                     │
│   - FastAPI REST + WebSocket                               │
│   - Session lifecycle management                           │
│   - Frame streaming                                        │
├────────────────────────────────────────────────────────────┤
│               State Manager SDK (Layer 1)                  │
│   arc_fabric/state.py     - WorldStateSnapshot dataclass   │
│   arc_fabric/manager.py   - StateManager checkpoint/restore│
│   arc_fabric/session.py   - Session lifecycle wrapper      │
│   arc_fabric/tree.py      - TrajectoryTree data structure  │
├────────────────────────────────────────────────────────────┤
│               DreamZero Model (existing)                   │
│   groot/vla/model/dreamzero/                               │
│   - WANPolicyHead (action head with KV caches)             │
│   - CausalWanModel (diffusion transformer)                 │
│   - WanVideoVAE (encoder/decoder)                          │
└────────────────────────────────────────────────────────────┘
```

---

## Layer 1: State Manager SDK

### File: `arc_fabric/state.py`

```python
@dataclass
class WorldStateSnapshot:
    """Complete, serializable snapshot of DreamZero generation state."""

    # Identity
    checkpoint_id: str              # UUID
    session_id: str                 # Parent session
    frame_index: int                # Which frame this checkpoint is at
    timestamp: float                # Wall clock time

    # KV Caches (the big ones - offloaded to CPU)
    kv_cache: list[torch.Tensor]            # 32 layers × [2, B, seq, 40, 128]
    kv_cache_neg: list[torch.Tensor]        # 32 layers × [2, B, seq, 40, 128]
    crossattn_cache: list[torch.Tensor]     # 32 layers × [2, B, 512, 40, 128]
    crossattn_cache_neg: list[torch.Tensor] # 32 layers × [2, B, 512, 40, 128]

    # Accumulated latents
    latent_video: torch.Tensor      # [B, T_accum, 16, H/8, W/8] on CPU

    # Cached embeddings
    clip_feas: torch.Tensor         # [B, 257, 1280]
    ys: torch.Tensor                # [B, 20, T, H/8, W/8]
    prompt_embs: list[torch.Tensor] # Encoded text prompts

    # Tracking
    current_start_frame: int
    language: torch.Tensor | None   # Cached text tokens
    seed: int

    # Metadata
    prompt_text: str                # Human-readable prompt
    action_history: list[dict]      # Actions taken up to this point
    parent_checkpoint_id: str | None  # For tree lineage

    def to_cpu(self) -> 'WorldStateSnapshot': ...
    def to_gpu(self, device: str = 'cuda') -> 'WorldStateSnapshot': ...
    def size_bytes(self) -> int: ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> 'WorldStateSnapshot': ...
```

### File: `arc_fabric/manager.py`

```python
class StateManager:
    """Manages checkpointing and restoring of DreamZero world state."""

    def __init__(self, model: VLA, max_checkpoints: int = 50):
        self.model = model
        self.action_head = model.action_head  # WANPolicyHead
        self.checkpoints: dict[str, WorldStateSnapshot] = {}

    def extract_state(self, session_id: str, prompt_text: str = "",
                      action_history: list = None,
                      parent_checkpoint_id: str = None) -> WorldStateSnapshot:
        """Extract current model state into a CPU-offloaded snapshot."""
        # Deep copy all KV caches to CPU
        # Deep copy latent_video, clip_feas, ys to CPU
        # Record current_start_frame, language, seed
        ...

    def restore_state(self, snapshot: WorldStateSnapshot) -> None:
        """Restore model state from a snapshot (move tensors back to GPU)."""
        # Copy KV caches back to GPU
        # Restore current_start_frame, language
        # Set clip_feas, ys
        ...

    def checkpoint(self, session_id: str, **kwargs) -> str:
        """Snapshot current state → returns checkpoint_id."""
        snapshot = self.extract_state(session_id, **kwargs)
        self.checkpoints[snapshot.checkpoint_id] = snapshot
        return snapshot.checkpoint_id

    def restore(self, checkpoint_id: str) -> WorldStateSnapshot:
        """Restore model to a previously checkpointed state."""
        snapshot = self.checkpoints[checkpoint_id]
        self.restore_state(snapshot)
        return snapshot

    def delete_checkpoint(self, checkpoint_id: str) -> None: ...
    def list_checkpoints(self, session_id: str = None) -> list[str]: ...
```

### File: `arc_fabric/session.py`

```python
class Session:
    """A single generation session. Wraps model + state for one trajectory."""

    def __init__(self, session_id: str, model: VLA, state_manager: StateManager,
                 prompt: str, initial_frame: np.ndarray, embodiment_id: int = 0):
        self.session_id = session_id
        self.model = model
        self.state_manager = state_manager
        self.prompt = prompt
        self.frame_index = 0
        self.action_history = []
        self.checkpoint_ids = []          # Ordered list of checkpoints
        self.latent_video = None          # Accumulated latents
        self.decoded_frames = []          # Optional: decoded RGB frames

    def step(self, action: np.ndarray, state: np.ndarray,
             auto_checkpoint: bool = True) -> dict:
        """Advance one generation step with the given action."""
        # 1. Prepare inputs
        # 2. Call model.lazy_joint_video_action_causal()
        # 3. Store latent_video output
        # 4. Optionally checkpoint
        # 5. Return action_pred, video_pred, frame_index
        ...

    def checkpoint(self) -> str:
        """Save current state. Returns checkpoint_id."""
        ...

    def rewind(self, n_steps: int = 1) -> str:
        """Go back n steps by restoring the appropriate checkpoint."""
        # Find checkpoint n steps back
        # Restore state
        # Trim action_history and checkpoint_ids
        ...

    def get_decoded_frame(self, frame_index: int) -> np.ndarray:
        """Decode a latent frame to RGB using the VAE."""
        ...

    def get_trajectory_video(self) -> np.ndarray:
        """Decode all accumulated latents to video."""
        ...

    def to_dict(self) -> dict:
        """Serialize session metadata for API."""
        ...
```

### File: `arc_fabric/tree.py`

```python
@dataclass
class TreeNode:
    """A node in the trajectory tree."""
    checkpoint_id: str
    session_id: str
    frame_index: int
    prompt: str
    action: dict | None        # Action that led to this node
    parent_id: str | None      # Parent node checkpoint_id
    children_ids: list[str]    # Child node checkpoint_ids
    metadata: dict             # Extra info (timing, success, etc.)

class TrajectoryTree:
    """Tree structure for managing branching trajectories."""

    def __init__(self):
        self.nodes: dict[str, TreeNode] = {}
        self.root_id: str | None = None

    def add_node(self, checkpoint_id: str, session_id: str,
                 frame_index: int, prompt: str,
                 action: dict = None, parent_id: str = None,
                 metadata: dict = None) -> TreeNode: ...

    def get_node(self, checkpoint_id: str) -> TreeNode: ...
    def get_children(self, checkpoint_id: str) -> list[TreeNode]: ...
    def get_branch(self, checkpoint_id: str) -> list[TreeNode]: ...
    def get_lineage(self, checkpoint_id: str) -> list[TreeNode]: ...

    def to_dict(self) -> dict:
        """Serialize tree for JSON/API/UI."""
        ...

    def to_d3_tree(self) -> dict:
        """Convert to D3.js-compatible nested tree format."""
        ...
```

---

## Layer 2: API Server

### File: `arc_fabric/server.py`

FastAPI application with these endpoints:

```
Session Management:
  POST   /api/sessions                 → Create session (prompt + initial frame)
  GET    /api/sessions                 → List all active sessions
  GET    /api/sessions/{id}            → Session info + tree
  DELETE /api/sessions/{id}            → Teardown session

Generation:
  POST   /api/sessions/{id}/step       → Advance with action/state
  POST   /api/sessions/{id}/checkpoint → Snapshot current state
  POST   /api/sessions/{id}/rewind     → Go back N steps

Forking:
  POST   /api/sessions/{id}/fork       → Fork from checkpoint
    Body: { checkpoint_id, new_prompt?, new_seed? }
    Returns: new session_id

Data:
  GET    /api/sessions/{id}/frames/{n} → Decoded frame as PNG
  GET    /api/sessions/{id}/video      → Video of trajectory (MP4)
  GET    /api/sessions/{id}/tree       → Full trajectory tree JSON
  GET    /api/sessions/{id}/latents/{n}→ Raw latent tensor

Static:
  GET    /                             → Web UI (index.html)
  GET    /static/*                     → Static assets

WebSocket:
  WS     /ws/sessions/{id}/stream      → Real-time frame stream
```

### Key Implementation Details

- **Single model instance** shared across all sessions
- **Session swap**: when switching active session, offload current KV caches to CPU, load target's
- **Max 5 sessions** enforced at creation time
- **Auto-checkpoint** every step (configurable)
- **Video encoding**: decode latents through VAE, encode as MP4 using imageio

---

## Layer 3: Web UI

### File: `arc_fabric/ui/index.html`

Single-page application (vanilla HTML/JS/CSS + D3.js) with three panels:

```
┌──────────────────┬───────────────────────────────┬──────────────┐
│                  │                               │              │
│  Trajectory Tree │   Video Comparison Panel      │   Control    │
│  (D3.js)         │   (up to 3 side-by-side)      │   Panel      │
│                  │                               │              │
│  ● Root          │   ┌─────┐ ┌─────┐ ┌─────┐    │  [New Sess]  │
│  ├── Fork A      │   │ V1  │ │ V2  │ │ V3  │    │  [Step]      │
│  ├── Fork B      │   │     │ │     │ │     │    │  [Fork]      │
│  └── Fork C      │   └─────┘ └─────┘ └─────┘    │  [Rewind]    │
│                  │                               │  [Compare]   │
│  Click node to   │   Timeline scrubber with      │              │
│  select branch   │   divergence point marker     │  Prompt:     │
│                  │                               │  [________]  │
└──────────────────┴───────────────────────────────┴──────────────┘
```

### UI Features
1. **Tree View**: Interactive D3.js tree. Click nodes to select. Color-coded by session.
2. **Video Panel**: Up to 3 decoded videos side-by-side. Auto-plays from divergence point.
3. **Control Panel**: Create/fork/rewind/step. Action input (or auto-generate).
4. **Timeline**: Horizontal timeline showing common trunk and branches.

---

## MVP Demo Flow

### `arc_fabric/demo.py`

```
DEMO: "3 Forks with Rewind on Failure"

1. SETUP
   - Load DreamZero model
   - Load initial frame from test data
   - Create root session with prompt "pick up the red cup"

2. TRUNK GENERATION (frames 0-8)
   - Step forward 8 frames with default actions
   - Auto-checkpoint at each step
   - This builds the "trunk" of the tree

3. FORK at frame 8
   - Fork A: "pick up the red cup" (same prompt, same seed)
   - Fork B: "push the cup to the left" (different prompt)
   - Fork C: "pick up the red cup" (same prompt, different seed=42)

4. BRANCH GENERATION (frames 8-16, all 3 forks)
   - Step each fork forward 8 frames
   - Auto-checkpoint at each step
   - Collect action predictions from each branch

5. EVALUATE BRANCHES
   - Simple heuristic: check if action magnitudes are reasonable
   - If any branch has actions that diverge too wildly → mark as "failed"

6. REWIND on failure
   - For the "failed" branch, rewind to frame 8
   - Try again with a modified prompt: "carefully pick up the red cup"
   - Step forward 8 more frames

7. VISUALIZATION
   - Start the FastAPI server
   - Open the web UI
   - Display the trajectory tree with all branches
   - Show 3 side-by-side videos comparing the branches
```

---

## File Layout

```
arc_fabric/
├── __init__.py
├── CODEBASE.md          # Codebase understanding (reference)
├── ARCHITECTURE.md      # This file (golden reference for agents)
├── state.py             # WorldStateSnapshot dataclass
├── manager.py           # StateManager (checkpoint/restore)
├── session.py           # Session lifecycle
├── tree.py              # TrajectoryTree data structure
├── server.py            # FastAPI REST + WebSocket server
├── demo.py              # MVP demo script
└── ui/
    ├── index.html       # Single-page web UI
    └── static/          # CSS, JS, assets
```

---

## Implementation Rules for Agents

1. **Import from `arc_fabric`**: All Arc Fabric code lives in the `arc_fabric/` package.
2. **Don't modify DreamZero core**: Access model state through public attributes (`model.action_head.kv_cache1`, etc.). No monkey-patching.
3. **CPU offload always**: When checkpointing, `.cpu().clone()` all tensors. When restoring, `.cuda()`.
4. **Deep copy KV caches**: Use `[layer.clone() for layer in cache]`. Never hold references to GPU tensors in checkpoints.
5. **One active session on GPU**: Swap sessions by offloading/loading KV caches.
6. **Deterministic noise**: Use the model's seed (1140) or allow override. Store seed in snapshot.
7. **Tree is append-only**: Nodes are never deleted from the tree (even after rewind).
8. **JSON-serializable metadata**: All metadata in TreeNode must be JSON-safe.
9. **Use `latent_video` parameter**: When restoring state, pass the latent video through `lazy_joint_video_action_causal()` using the existing `latent_video` parameter.
10. **Intercept resets**: After restoring a checkpoint, manually set `current_start_frame`, `language`, `clip_feas`, `ys` to bypass automatic reset logic.

---

## Dependencies

Add to `pyproject.toml`:
```
fastapi>=0.100.0
uvicorn>=0.20.0
python-multipart>=0.0.6
```

The web UI uses only CDN-hosted libraries (D3.js, no build step required).
