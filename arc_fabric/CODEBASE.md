# DreamZero Codebase Deep Dive

> This document captures the complete understanding of the DreamZero codebase
> as needed by Arc Fabric's state management layer. Last updated: 2026-02-16.

---

## 1. Project Structure

```
dreamzero/
├── groot/vla/                          # Core VLA model
│   ├── model/dreamzero/                # DreamZero-specific components
│   │   ├── base_vla.py                 # VLA wrapper (VLAConfig, VLA class)
│   │   ├── action_head/
│   │   │   └── wan_flow_matching_action_tf_efficient.py  # WANPolicyHead (main inference engine)
│   │   └── modules/
│   │       ├── wan_video_dit_action_casual_chunk.py      # CausalWanModel (diffusion transformer)
│   │       ├── wan_video_vae.py                          # WanVideoVAE (encoder/decoder)
│   │       ├── flow_match_scheduler.py                   # FlowMatchScheduler (training)
│   │       ├── flow_unipc_multistep_scheduler.py         # FlowUniPCMultistepScheduler (inference)
│   │       ├── wan_video_text_encoder.py                 # T5 text encoder
│   │       ├── wan2_1_attention.py                       # AttentionModule (flash attention)
│   │       ├── wan2_1_submodule.py                       # RoPE, norms, submodules
│   │       └── vram_management.py                        # VRAM offloading utilities
│   ├── model/n1_5/                     # Base model components
│   │   ├── sim_policy.py               # Policy wrapper for simulation
│   │   └── modules/action_encoder.py   # SinusoidalPositionalEncoding, swish
│   └── data/                           # Dataset loading and transforms
├── evals/                              # Evaluation suite
│   ├── eval_suite.py                   # Main evaluation script
│   ├── test_cases.json                 # 297+ test cases
│   ├── analyze_actions.py              # Post-hoc trajectory analysis
│   └── gallery_data/                   # Gallery tasks (267 total)
├── eval_utils/                         # Inference utilities
│   ├── policy_server.py                # WebSocket policy server
│   ├── policy_client.py                # WebSocket policy client
│   └── run_sim_eval.py                 # Simulation evaluation runner
├── socket_test_optimized_AR.py         # Main real-world inference server
├── test_client_AR.py                   # Test client for AR server
├── checkpoints/                        # Model checkpoints (gitignored)
│   └── DreamZero-DROID/                # Main checkpoint (~50GB)
├── scripts/
│   ├── setup_env.sh                    # Environment setup
│   └── download_weights.sh             # Weight download from HF
├── pyproject.toml                      # Dependencies (PyTorch 2.8, flash-attn, etc.)
├── README.md                           # Project overview
└── SETUP.md                            # Setup guide
```

---

## 2. Model Architecture Hierarchy

```
VLA (base_vla.py)
├── backbone                            # Lightweight, mostly pass-through
└── action_head: WANPolicyHead
    ├── text_encoder (T5-based)         # Frozen, encodes text → [B, 512, 4096]
    ├── image_encoder (CLIP-based)      # Frozen, encodes image → clip_feas [B, 257, 1280]
    ├── vae: WanVideoVAE                # Frozen, video ↔ latent [B, 16, T, H/8, W/8]
    ├── scheduler: FlowMatchScheduler   # Training scheduler
    └── model: CausalWanModel           # Trainable diffusion transformer
        ├── patch_embedding             # Conv3d(16, 2048, kernel=(1,2,2))
        ├── text_embedding              # Linear(4096, 2048)
        ├── time_embedding              # Sinusoidal → MLP
        ├── time_projection             # SiLU → Linear → [B, F, 6, 2048]
        ├── img_emb: MLPProj            # CLIP features → 2048-dim
        ├── action_encoder              # MultiEmbodimentActionEncoder
        ├── state_encoder               # CategorySpecificMLP
        ├── action_decoder              # CategorySpecificMLP
        ├── blocks (×32)                # CausalWanAttentionBlock
        │   ├── self_attn               # CausalWanSelfAttention (flash attention)
        │   ├── cross_attn              # T5 cross-attention
        │   └── ffn                     # GELU feed-forward
        ├── head: CausalHead            # Final prediction head
        ├── freqs                       # 3D RoPE frequencies (temporal, height, width)
        ├── freqs_action                # 1D RoPE for action tokens
        └── freqs_state                 # 1D RoPE for state tokens
```

---

## 3. Latent Flow: End-to-End Generation Pipeline

### 3.1 Entry Point: `WANPolicyHead.lazy_joint_video_action()`

This is the main autoregressive generation method (line 913 of `wan_flow_matching_action_tf_efficient.py`).

### 3.2 Step-by-step Flow

```
Input: video frames [B, T, H, W, C] uint8 + text tokens + state features

1. PREPROCESS
   video → rearrange → normalize → [B, C, T, H, W] bfloat16

2. RESET DETECTION (lines 952-965)
   - language changed? → reset current_start_frame = 0
   - single frame input? → reset
   - current_start_frame >= local_attn_size? → reset

3. TEXT ENCODING (line 973)
   text_tokens → T5 → prompt_embs [B, 512, 4096]
   (When cfg_scale != 1.0, also encode negative prompt)

4. IMAGE ENCODING (lines 986-989, first frame only when start_frame==0)
   image → CLIP → clip_feas [B, 257, 1280]
   image → VAE → ys [B, 20, T, H/8, W/8] (conditioning signal)
   These are CACHED in self.clip_feas and self.ys

5. VAE ENCODING (lines 997-1022, when start_frame != 0)
   Reference frames → VAE.encode() → image latent [B, 16, T', H/8, W/8]
   OR: use latent_video parameter directly (the hook for Arc Fabric!)

6. NOISE GENERATION (lines 1025-1026)
   noise_obs = randn([B, 16, num_frame_per_block, H/8, W/8])
   noise_action = randn([B, action_horizon, action_dim])
   Seed is deterministic (self.seed = 1140)

7. KV CACHE INIT (lines 1035-1047, when start_frame == 0)
   Create fresh kv_cache1, kv_cache_neg: 32 layers × [2, B, 0, 40, 128]
   Create crossattn_cache, crossattn_cache_neg: 32 layers × [2, B, 512, 40, 128]

8. CONDITIONING PASS (lines 1062-1082, when start_frame == 0)
   Run first image through diffusion model with timestep=0
   → Populates KV cache with conditioning frame context
   → current_start_frame += 1

9. REFERENCE FRAME PASS (lines 1086-1109, when start_frame != 1)
   Encode reference latents into KV cache
   → Updates KV cache with latest context

10. DENOISING LOOP (lines 1118-1224, 16 steps)
    For each step i in scheduler.timesteps:
      a. Check DIT cache: should_run_model(i)?
         - If yes: run _run_diffusion_steps() → flow predictions
         - If no: reuse previous predictions
      b. flow_pred = uncond + cfg_scale * (cond - uncond)
      c. sample_scheduler.step() → update noisy_input (video)
      d. sample_scheduler_action.step() → update noisy_input_action

11. OUTPUT ASSEMBLY (lines 1226-1232)
    latents = denoised video
    latents_action = denoised actions
    If start_frame == 1: prepend conditioning image latent
    current_start_frame += num_frame_per_block

12. RETURN
    BatchFeature(action_pred=latents_action, video_pred=output)
```

---

## 4. State That Must Be Checkpointed

### 4.1 KV Caches (Largest Component)

```python
# Self-attention KV cache: 32 layers
self.kv_cache1: list[Tensor]      # Each: [2, B, seq_len, 40, 128]
self.kv_cache_neg: list[Tensor]    # Each: [2, B, seq_len, 40, 128]

# Cross-attention KV cache: 32 layers
self.crossattn_cache: list[Tensor]     # Each: [2, B, 512, 40, 128]
self.crossattn_cache_neg: list[Tensor] # Each: [2, B, 512, 40, 128]
```

**Size estimate**: For a sequence of 8 frames at bfloat16:
- Self-attn per layer: `2 × 1 × (8 × 440) × 40 × 128 × 2 bytes ≈ 57 MB`
- Self-attn total: `57 MB × 32 layers × 2 (pos+neg) ≈ 3.6 GB`
- Cross-attn per layer: `2 × 1 × 512 × 40 × 128 × 2 bytes ≈ 10 MB`
- Cross-attn total: `10 MB × 32 layers × 2 ≈ 640 MB`
- **Total KV caches: ~4.2 GB**

### 4.2 Cached Embeddings

```python
self.clip_feas: Tensor   # [B, 257, 1280] ≈ 1.3 MB
self.ys: Tensor          # [B, 20, T, H/8, W/8] ≈ 20-50 MB
```

### 4.3 Tracking State

```python
self.current_start_frame: int     # Position in autoregressive sequence
self.language: Tensor             # Cached text tokens (for reset detection)
```

### 4.4 Accumulated Latents

```python
output: Tensor  # [B, T_accum, 16, H/8, W/8] - grows each step
                # This is the latent_video that can be passed back
```

### 4.5 Noise State

```python
self.seed: int = 1140             # Deterministic noise generation
# Note: FlowUniPCMultistepScheduler is reconstructible from params
```

---

## 5. Key Hooks for Arc Fabric

### 5.1 `latent_video` Parameter (line 913, 997-998)

The `lazy_joint_video_action()` method already accepts `latent_video` as an optional parameter.
When `current_start_frame != 0`, if `latent_video` is provided, it's used directly instead of
re-encoding video through the VAE. **This is the primary injection point for restoring state.**

### 5.2 KV Cache Access

KV caches are stored as instance variables on `WANPolicyHead`:
- `self.kv_cache1`, `self.kv_cache_neg`
- `self.crossattn_cache`, `self.crossattn_cache_neg`

They can be read/written between generation steps. Deep copies must be used for forking.

### 5.3 State Reset Conditions (lines 952-965)

State resets happen when:
1. `self.language` changes (different text prompt)
2. `videos.shape[2] == 1` (single frame input)
3. `current_start_frame >= local_attn_size`
4. `self.language is None` (first call)

**Arc Fabric must intercept these resets** when restoring from checkpoint.

### 5.4 Frame Sequence Length

```python
frame_seqlen = int(height * width / 4)  # For DROID 320×240: 440 tokens/frame
# For 480×640: 880 tokens/frame (AgiBot)
```

### 5.5 Action Encoding

```python
self.model.action_encoder(action, timestep_action, embodiment_id)
# action: [B, action_horizon, action_dim]  (action_horizon=24, action_dim=7 for DROID)
self.model.state_encoder(state, embodiment_id)
# state: [B, num_state, state_dim]
```

---

## 6. Attention Architecture Details

### 6.1 Token Structure per Block

```
[first_image: frame_seqlen tokens]
[image_block_0: num_frame_per_block × frame_seqlen tokens]
[image_block_1: ...]
...
[action_block_0: num_action_per_block tokens]  (num_action_per_block=32)
[action_block_1: ...]
...
[state_block_0: num_state_per_block tokens]    (num_state_per_block=1)
[state_block_1: ...]
```

### 6.2 Attention Pattern

- **Image blocks**: attend to first_image + previous image blocks (within local window) + same-index action + same-index state
- **Action blocks**: attend to first_image + previous image blocks + same-index action + same-index state
- **State blocks**: self-attention only (conditioning)
- **First image**: self-attention only (conditioning)

### 6.3 KV Cache Growth

During autoregressive generation, new frame tokens are appended to the KV cache.
When `kv_cache.shape[1] > max_attention_size`, it's truncated to `max_attention_size`.

```python
max_attention_size = local_attn_size * frame_seqlen  # or 21 * frame_seqlen if global
```

---

## 7. Scheduler Details

### 7.1 Training: `FlowMatchScheduler`

- 1000 timestep buckets
- Flow matching with sigma shift=5
- Beta distribution sampling for noise levels

### 7.2 Inference: `FlowUniPCMultistepScheduler`

- Default 16 inference steps
- Shift=5 (sigma shift for importance weighting)
- UniPC multistep method
- Creates new instance per generation call (stateless, reconstructible)

### 7.3 Decoupled Inference (optional)

- Video schedule: sigmas rescaled to end at `video_inference_final_noise` instead of 0
- Action schedule: always fully denoises (1000→0)
- This means video stays partially noisy while actions are clean

---

## 8. DIT Caching (Compute Optimization)

The `should_run_model()` method (line 883) implements DIT step caching:
- Not all 16 denoising steps actually run the full transformer
- A mask (e.g., `[T,T,T,F,F,F,T,F,F,F,T,F,F,T,T,T]`) determines which steps run
- Skipped steps reuse the previous flow prediction
- Dynamic scheduling uses cosine similarity between consecutive predictions
- This reduces compute from 16 to ~8 full DiT forward passes

---

## 9. Configuration

### 9.1 Key Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_inference_steps` | 16 | Denoising steps per frame |
| `num_frame_per_block` | 1 | Frames per autoregressive step |
| `action_horizon` | 24 | Action sequence length |
| `action_dim` | varies | 7 for DROID, varies per embodiment |
| `local_attn_size` | varies | Attention window (-1 = global) |
| `cfg_scale` | 5.0 | Classifier-free guidance scale |
| `seed` | 1140 | Deterministic noise seed |
| `num_layers` | 32 | Transformer blocks |
| `dim` | 2048 | Hidden dimension |
| `num_heads` | 40 (head_dim=128) | Attention heads |
| `frame_seqlen` | 440 (DROID) | Tokens per latent frame |

### 9.2 Embodiment Support

Multi-embodiment via `CategorySpecificMLP` and `CategorySpecificLinear`:
- DROID: action_dim=7, state_dim varies
- AgiBot: different dims
- YAM: different dims
- Embodiment ID selects the weight matrices

---

## 10. Memory Footprint on 2×H100

| Component | Size | Location |
|-----------|------|----------|
| Model weights (shared) | ~25 GB | GPU |
| Active session KV caches | ~4.2 GB | GPU |
| Active session latent video | ~50 MB | GPU |
| Active session embeddings | ~50 MB | GPU |
| Inactive checkpoint (offloaded) | ~4.3 GB | CPU RAM |
| T5/CLIP encoders (frozen) | ~5 GB | GPU |
| VAE (frozen) | ~2 GB | GPU |
| **Total per active session** | **~4.3 GB** | GPU |
| **Total model footprint** | **~32 GB** | GPU |
| **Budget for sessions (2×80GB)** | **~128 GB free** | GPU |

This means we can fit ~5 active sessions or 1 active + 20+ offloaded checkpoints.
