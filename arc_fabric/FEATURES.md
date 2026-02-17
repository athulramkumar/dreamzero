# Arc Fabric — Features & Capabilities

Arc Fabric is a state-management platform built on top of **DreamZero**, a
diffusion-based world model for robotic manipulation.  It treats the
high-dimensional latent states produced by DreamZero's autoregressive
generation as **first-class, checkpointable objects** — enabling
capabilities that are impossible when the model is treated as a
black-box video generator.

---

## Core Concepts

### Latent State as First-Class Object

Every frame DreamZero generates is produced from a rich internal state:

| Component | Shape (typical) | Description |
|---|---|---|
| Self-attention KV caches | `32 × [2, B, seq, 40, 128]` bf16 | Per-layer memory of all past tokens (image, action, state) |
| Cross-attention KV caches | `32 × [2, B, 512, 40, 128]` bf16 | Cached text-conditioned attention |
| CLIP features | `[B, 257, 1280]` | Image encoder output for visual conditioning |
| VAE conditioning (ys) | `[B, 20, T, H/8, W/8]` | Noise schedule conditioning signal |
| Latent video | `[B, 16, T, H/8, W/8]` bf16 | Accumulated denoised latent frames |
| Language embedding | Tensor | Cached tokenised prompt (reset-detection key) |
| `current_start_frame` | int | Autoregressive position counter |
| Seed | int | Deterministic noise seed |

Arc Fabric captures **all** of this into a single `WorldStateSnapshot` and
manages its lifecycle on CPU/GPU/disk.

---

## Feature 1 — Checkpoint / Save World State

**What it does:**  Snapshot the complete internal state of the world model at
any autoregressive frame `N`.

**How it works:**
1. `StateManager.extract_state()` deep-copies every KV cache tensor, cached
   embedding, and tracking scalar from `WANPolicyHead` to CPU.
2. The snapshot is stored in an in-memory `OrderedDict` with LRU eviction.
3. Snapshots can also be serialised to disk with `WorldStateSnapshot.save()`.

**Why it matters:**
- A checkpoint is ~4 GB (dominated by KV caches), while the equivalent
  decoded video frames would be ~100× larger.
- The checkpoint is a *complete, minimal, replayable* representation — it
  contains everything needed to resume generation exactly.
- No need to re-run the VAE encoder or re-process historical frames.

**API:**
```python
checkpoint_id = state_manager.checkpoint(
    session_id="sess-1",
    prompt_text="pick up the red cup",
)
```

---

## Feature 2 — Resume from Checkpoint

**What it does:**  Restore the model to a previously saved state and continue
generation from that exact point.

**How it works:**
1. `StateManager.restore(checkpoint_id)` loads the snapshot.
2. All KV caches are deep-copied back to GPU.
3. `current_start_frame` and `language` are explicitly set to prevent the
   model's reset-detection logic from zeroing the caches.
4. Generation continues as if no interruption occurred.

**Why it matters:**
- Enables **pause / resume** for long-running generation.
- Foundation for fork and rewind (both build on restore).
- Deterministic: same seed + same state = same output.

**API:**
```python
snapshot = state_manager.restore(checkpoint_id)
# model is now at frame N — call step() to continue
```

---

## Feature 3 — Fork (Branch from Checkpoint)

**What it does:**  Create a new, independent generation branch from any
existing checkpoint — with a different prompt, seed, or action sequence.

**How it works:**
1. The current session is checkpointed (so its state is safe).
2. The target checkpoint is restored onto the shared model.
3. A new `Session` is created, inheriting the restored state.
4. Optionally, the prompt or seed is overridden.
5. The `TrajectoryTree` records the fork point and parent lineage.

**Why it matters:**
- **Speculative planning:** a robot can explore "what happens if I push
  left vs. right?" from the same world state.
- **Prompt ablation:** test how different language instructions affect
  the same scene.
- **Seed ablation:** sample multiple futures under the same instruction.
- Forks share all computation up to the fork point — only the divergent
  future is generated.

**API:**
```python
forked_session = session.fork(
    checkpoint_id=cp_id,
    new_prompt="push the cup to the left",
    new_seed=42,
)
```

---

## Feature 4 — Rewind (Go Back N Steps)

**What it does:**  Roll the model state back by N autoregressive steps.

**How it works:**
1. `Session.rewind(n_steps)` looks up the checkpoint from `n_steps` ago.
2. That checkpoint is restored onto the model.
3. The session's action history and checkpoint list are trimmed.
4. Generation can continue from the earlier point.

**Why it matters:**
- **Failure recovery:** if the model's predicted trajectory diverges into
  an undesirable state (collision, drop, etc.), the agent rewinds and
  retries with a different action.
- Much cheaper than re-generating from scratch — only the failed segment
  is re-computed.

**API:**
```python
restored_id = session.rewind(n_steps=3)
# model is now 3 steps earlier — try a different action
```

---

## Feature 5 — Trajectory Tree

**What it does:**  Tracks the full branching history of all sessions as a
directed tree.

**How it works:**
- Each checkpoint becomes a `TreeNode` with parent/children links.
- Fork operations create new branches; rewinds re-root from earlier nodes.
- The tree is serialisable to JSON and to a D3.js-compatible format for
  web visualisation.

**Why it matters:**
- **Audit trail:** every decision point and branch is recorded.
- **Visualisation:** operators and researchers can see the exploration
  space at a glance.
- **RL integration:** the tree structure maps directly to a search tree
  that an RL agent can navigate.

**API:**
```python
tree.add_node(checkpoint_id, session_id, frame_index, ...)
d3_json = tree.to_d3_tree()   # for the web UI
lineage = tree.get_lineage(checkpoint_id)  # root → ... → node
```

---

## Feature 6 — Latent Compression & Transport

**What it does:**  Latent states are orders of magnitude smaller than decoded
pixels, making them efficient for storage and transmission.

| Representation | Size per frame (approx.) |
|---|---|
| Decoded RGB (180×320×3, uint8) | 173 KB |
| Latent (16×23×40, bf16) | 29 KB |
| Full KV-cache checkpoint | ~4 GB (entire history) |

**Why it matters:**
- **Edge-to-cloud:** a robot on the edge can transmit latents to a cloud
  server for analysis or fork exploration.
- **Archival:** entire simulation trajectories can be stored in a fraction
  of the space required for video.
- **Replay:** latents can be decoded to video at any time via the VAE
  decoder — no need to re-run the full model.

---

## Feature 7 — Latent Space Analysis / Visualisation

**What it does:**  Decode any accumulated latent frame to RGB pixels on demand.

**How it works:**
1. `Session.get_decoded_frame(frame_index)` extracts a single latent slice.
2. The slice is passed through the VAE decoder (`WanVideoVAE.decode()`).
3. The result is an RGB uint8 `[H, W, C]` numpy array.

**Why it matters:**
- **Debugging:** inspect what the model "sees" at any checkpoint.
- **Comparison:** visually diff two forked trajectories frame-by-frame.
- **No full decode needed:** only the frames of interest are decoded,
  saving GPU time.

**API:**
```python
rgb_frame = session.get_decoded_frame(frame_index=5)
# rgb_frame is a (H, W, 3) uint8 numpy array
```

---

## Feature 8 — Hybrid Control + Learning

**What it does:**  Inject human-controlled or learned modifications at the
latent level.

**How it works:**
- **Prompt override:** fork with a new language instruction to steer the
  model's predicted future.
- **Seed override:** sample a different stochastic future from the same
  state.
- **Action override:** supply custom action tensors to drive the model in a
  specific direction.
- **Latent injection:** pass a modified `latent_video` tensor to bypass
  the VAE encoder entirely, allowing direct latent-space editing.

**Why it matters:**
- Researchers can train secondary models on latent trajectories
  (e.g. "which latent configurations lead to successful grasps?").
- Operators can manually steer generation by editing prompts mid-rollout.
- RL agents can use the fork/rewind interface as a search tree for
  planning.

---

## Architecture Summary

```
┌──────────────────────────────────────┐
│            Web UI (D3.js)            │  ← Trajectory tree + video comparison
├──────────────────────────────────────┤
│        REST / WebSocket API          │  ← FastAPI server, mock or live mode
├──────────────────────────────────────┤
│        Arc Fabric Python SDK         │
│  ┌────────────┐  ┌────────────────┐  │
│  │ Session    │  │ StateManager   │  │
│  │ .step()    │  │ .checkpoint()  │  │
│  │ .fork()    │  │ .restore()     │  │
│  │ .rewind()  │  │ .extract_state │  │
│  └────────────┘  └────────────────┘  │
│  ┌────────────┐  ┌────────────────┐  │
│  │ Trajectory │  │ WorldState     │  │
│  │ Tree       │  │ Snapshot       │  │
│  └────────────┘  └────────────────┘  │
├──────────────────────────────────────┤
│         DreamZero VLA Model          │
│  WANPolicyHead ← KV caches, VAE,    │
│  text encoder, CLIP, scheduler       │
└──────────────────────────────────────┘
```

---

## Hardware Requirements

- **Minimum:** 1× NVIDIA H100 80 GB (single-GPU, reduced batch)
- **Recommended:** 2× NVIDIA H100 80 GB (distributed inference via
  `torchrun`)
- **Memory budget (2× H100):**
  - Model weights: ~45 GB (split across GPUs)
  - Active KV caches: ~4.2 GB per session
  - Checkpoint storage: CPU RAM (~4 GB per checkpoint, LRU-evicted)
  - Max concurrent sessions: 5 (with headroom for VAE decode)
