"""WorldStateSnapshot: complete, serializable snapshot of DreamZero generation state.

This dataclass captures every piece of mutable state needed to resume or fork
a DreamZero autoregressive generation from an arbitrary frame.  Tensors are
stored on CPU by default (created via ``to_cpu()`` during checkpointing) and
can be moved back to GPU with ``to_gpu()`` before restoring into the model.
"""

from __future__ import annotations

import copy
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def _tensors_to_device(
    tensors: list[torch.Tensor] | torch.Tensor | None,
    device: torch.device | str,
) -> list[torch.Tensor] | torch.Tensor | None:
    """Deep-copy a tensor or list of tensors to *device*."""
    if tensors is None:
        return None
    if isinstance(tensors, list):
        return [t.to(device=device).clone() for t in tensors]
    return tensors.to(device=device).clone()


def _tensor_size_bytes(t: torch.Tensor) -> int:
    return t.nelement() * t.element_size()


@dataclass
class WorldStateSnapshot:
    """Complete, serializable snapshot of DreamZero generation state.

    Attributes
    ----------
    checkpoint_id : str
        UUID that uniquely identifies this checkpoint.
    session_id : str
        ID of the parent session that created this checkpoint.
    frame_index : int
        Which autoregressive frame this checkpoint corresponds to.
    timestamp : float
        Wall-clock time (``time.time()``) when the checkpoint was taken.
    kv_cache, kv_cache_neg : list[torch.Tensor]
        Self-attention KV caches for positive / negative CFG branches.
        32 tensors (one per transformer layer), each shaped
        ``[2, B, seq_len, 40, 128]`` in ``bfloat16``.
    crossattn_cache, crossattn_cache_neg : list[torch.Tensor]
        Cross-attention KV caches.  32 tensors, each shaped
        ``[2, B, 512, 40, 128]`` in ``bfloat16``.
    latent_video : torch.Tensor
        Accumulated latent frames ``[B, T_accum, 16, H/8, W/8]``.
    clip_feas : torch.Tensor
        CLIP image features ``[B, 257, 1280]``.
    ys : torch.Tensor
        VAE conditioning signal ``[B, 20, T, H/8, W/8]``.
    prompt_embs : list[torch.Tensor]
        Encoded text-prompt embeddings (one per CFG branch).
    current_start_frame : int
        Autoregressive position counter inside ``WANPolicyHead``.
    language : torch.Tensor | None
        Cached text token IDs used for reset detection.
    seed : int
        Deterministic noise seed.
    prompt_text : str
        Human-readable text prompt.
    action_history : list[dict]
        Actions taken up to this checkpoint.
    parent_checkpoint_id : str | None
        Checkpoint from which this one was forked (tree lineage).
    """

    # ── identity ──────────────────────────────────────────────────────────
    checkpoint_id: str
    session_id: str
    frame_index: int
    timestamp: float

    # ── self-attention KV caches (32 layers each) ─────────────────────────
    kv_cache: list[torch.Tensor]
    kv_cache_neg: list[torch.Tensor]

    # ── cross-attention KV caches (32 layers each) ────────────────────────
    crossattn_cache: list[torch.Tensor]
    crossattn_cache_neg: list[torch.Tensor]

    # ── accumulated latents ───────────────────────────────────────────────
    latent_video: torch.Tensor

    # ── cached embeddings ─────────────────────────────────────────────────
    clip_feas: torch.Tensor
    ys: torch.Tensor
    prompt_embs: list[torch.Tensor]

    # ── tracking state ────────────────────────────────────────────────────
    current_start_frame: int
    language: torch.Tensor | None
    seed: int

    # ── metadata ──────────────────────────────────────────────────────────
    prompt_text: str
    action_history: list[dict[str, Any]] = field(default_factory=list)
    parent_checkpoint_id: str | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Device helpers
    # ──────────────────────────────────────────────────────────────────────

    def to_cpu(self) -> WorldStateSnapshot:
        """Return a **new** snapshot with every tensor deep-copied to CPU."""
        return WorldStateSnapshot(
            checkpoint_id=self.checkpoint_id,
            session_id=self.session_id,
            frame_index=self.frame_index,
            timestamp=self.timestamp,
            kv_cache=_tensors_to_device(self.kv_cache, "cpu"),
            kv_cache_neg=_tensors_to_device(self.kv_cache_neg, "cpu"),
            crossattn_cache=_tensors_to_device(self.crossattn_cache, "cpu"),
            crossattn_cache_neg=_tensors_to_device(self.crossattn_cache_neg, "cpu"),
            latent_video=_tensors_to_device(self.latent_video, "cpu"),
            clip_feas=_tensors_to_device(self.clip_feas, "cpu"),
            ys=_tensors_to_device(self.ys, "cpu"),
            prompt_embs=_tensors_to_device(self.prompt_embs, "cpu"),
            current_start_frame=self.current_start_frame,
            language=_tensors_to_device(self.language, "cpu"),
            seed=self.seed,
            prompt_text=self.prompt_text,
            action_history=copy.deepcopy(self.action_history),
            parent_checkpoint_id=self.parent_checkpoint_id,
        )

    def to_gpu(self, device: str = "cuda") -> WorldStateSnapshot:
        """Return a **new** snapshot with every tensor deep-copied to *device*."""
        return WorldStateSnapshot(
            checkpoint_id=self.checkpoint_id,
            session_id=self.session_id,
            frame_index=self.frame_index,
            timestamp=self.timestamp,
            kv_cache=_tensors_to_device(self.kv_cache, device),
            kv_cache_neg=_tensors_to_device(self.kv_cache_neg, device),
            crossattn_cache=_tensors_to_device(self.crossattn_cache, device),
            crossattn_cache_neg=_tensors_to_device(self.crossattn_cache_neg, device),
            latent_video=_tensors_to_device(self.latent_video, device),
            clip_feas=_tensors_to_device(self.clip_feas, device),
            ys=_tensors_to_device(self.ys, device),
            prompt_embs=_tensors_to_device(self.prompt_embs, device),
            current_start_frame=self.current_start_frame,
            language=_tensors_to_device(self.language, device),
            seed=self.seed,
            prompt_text=self.prompt_text,
            action_history=copy.deepcopy(self.action_history),
            parent_checkpoint_id=self.parent_checkpoint_id,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Size estimation
    # ──────────────────────────────────────────────────────────────────────

    def size_bytes(self) -> int:
        """Estimate the total memory consumed by tensor data in this snapshot."""
        total = 0

        for cache_list in (
            self.kv_cache,
            self.kv_cache_neg,
            self.crossattn_cache,
            self.crossattn_cache_neg,
            self.prompt_embs,
        ):
            if cache_list is not None:
                for t in cache_list:
                    total += _tensor_size_bytes(t)

        for tensor in (self.latent_video, self.clip_feas, self.ys, self.language):
            if tensor is not None:
                total += _tensor_size_bytes(tensor)

        return total

    # ──────────────────────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str | os.PathLike) -> None:
        """Persist this snapshot to disk via :func:`torch.save`.

        All tensors are moved to CPU before saving so that the checkpoint is
        device-agnostic.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cpu_snap = self.to_cpu()
        payload: dict[str, Any] = {
            "checkpoint_id": cpu_snap.checkpoint_id,
            "session_id": cpu_snap.session_id,
            "frame_index": cpu_snap.frame_index,
            "timestamp": cpu_snap.timestamp,
            "kv_cache": cpu_snap.kv_cache,
            "kv_cache_neg": cpu_snap.kv_cache_neg,
            "crossattn_cache": cpu_snap.crossattn_cache,
            "crossattn_cache_neg": cpu_snap.crossattn_cache_neg,
            "latent_video": cpu_snap.latent_video,
            "clip_feas": cpu_snap.clip_feas,
            "ys": cpu_snap.ys,
            "prompt_embs": cpu_snap.prompt_embs,
            "current_start_frame": cpu_snap.current_start_frame,
            "language": cpu_snap.language,
            "seed": cpu_snap.seed,
            "prompt_text": cpu_snap.prompt_text,
            "action_history": cpu_snap.action_history,
            "parent_checkpoint_id": cpu_snap.parent_checkpoint_id,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> WorldStateSnapshot:
        """Deserialise a snapshot from *path*.

        The returned snapshot has all tensors on CPU; call ``to_gpu()`` before
        restoring into the model.
        """
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            checkpoint_id=payload["checkpoint_id"],
            session_id=payload["session_id"],
            frame_index=payload["frame_index"],
            timestamp=payload["timestamp"],
            kv_cache=payload["kv_cache"],
            kv_cache_neg=payload["kv_cache_neg"],
            crossattn_cache=payload["crossattn_cache"],
            crossattn_cache_neg=payload["crossattn_cache_neg"],
            latent_video=payload["latent_video"],
            clip_feas=payload["clip_feas"],
            ys=payload["ys"],
            prompt_embs=payload["prompt_embs"],
            current_start_frame=payload["current_start_frame"],
            language=payload["language"],
            seed=payload["seed"],
            prompt_text=payload["prompt_text"],
            action_history=payload["action_history"],
            parent_checkpoint_id=payload["parent_checkpoint_id"],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def new_id() -> str:
        """Generate a fresh UUID string suitable for *checkpoint_id*."""
        return str(uuid.uuid4())

    def __repr__(self) -> str:
        size_mb = self.size_bytes() / (1024 * 1024)
        return (
            f"WorldStateSnapshot("
            f"id={self.checkpoint_id!r}, "
            f"session={self.session_id!r}, "
            f"frame={self.frame_index}, "
            f"start_frame={self.current_start_frame}, "
            f"size={size_mb:.1f} MB, "
            f"parent={self.parent_checkpoint_id!r})"
        )
