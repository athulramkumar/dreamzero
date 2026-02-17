"""StateManager: checkpoint / restore bridge between DreamZero and Arc Fabric.

This module reads and writes the mutable state living on
``WANPolicyHead`` (the VLA's action head) so that generation can be
paused, forked, and rewound at any autoregressive frame.

Usage
-----
>>> manager = StateManager(model, max_checkpoints=50)
>>> cid = manager.checkpoint(session_id="sess-1", prompt_text="pick up cup")
>>> # … run a few more steps …
>>> manager.restore(cid)  # rewind back
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

import torch

from .state import WorldStateSnapshot

logger = logging.getLogger(__name__)


class StateManager:
    """Manages checkpointing and restoring of DreamZero world state.

    Parameters
    ----------
    model : VLA
        A fully-initialised VLA instance whose ``action_head`` is a
        ``WANPolicyHead``.
    max_checkpoints : int
        Maximum number of snapshots to keep in memory.  When the limit is
        exceeded the oldest checkpoint (by insertion order) is evicted.
    """

    def __init__(self, model: Any, max_checkpoints: int = 50) -> None:
        self.model = model
        self.action_head = model.action_head  # WANPolicyHead
        self.max_checkpoints = max_checkpoints

        # OrderedDict preserves insertion order for LRU eviction.
        self.checkpoints: OrderedDict[str, WorldStateSnapshot] = OrderedDict()

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @property
    def _device(self) -> torch.device:
        """Best-effort device detection from the action head."""
        if hasattr(self.action_head, "_device"):
            return torch.device(self.action_head._device)
        return torch.device("cuda")

    @staticmethod
    def _clone_cache_to_cpu(
        cache: list[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        """Deep-copy a list of KV-cache tensors to CPU.

        Returns an empty list when *cache* is ``None`` (fresh model with no
        generation yet).
        """
        if cache is None:
            return []
        return [t.detach().cpu().clone() for t in cache]

    @staticmethod
    def _clone_tensor_to_cpu(
        tensor: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Deep-copy a single tensor to CPU, or return ``None``."""
        if tensor is None:
            return None
        return tensor.detach().cpu().clone()

    @staticmethod
    def _restore_cache_to_device(
        cache: list[torch.Tensor],
        device: torch.device | str,
    ) -> list[torch.Tensor]:
        """Deep-copy a list of CPU tensors back to *device*."""
        return [t.to(device=device).clone() for t in cache]

    @staticmethod
    def _restore_tensor_to_device(
        tensor: torch.Tensor | None,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.to(device=device).clone()

    def _evict_if_needed(self) -> None:
        """Evict the oldest checkpoint when the capacity limit is reached."""
        while len(self.checkpoints) > self.max_checkpoints:
            evicted_id, evicted_snap = self.checkpoints.popitem(last=False)
            logger.info(
                "Evicted checkpoint %s (session=%s, frame=%d) — "
                "%d checkpoints remain",
                evicted_id,
                evicted_snap.session_id,
                evicted_snap.frame_index,
                len(self.checkpoints),
            )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def extract_state(
        self,
        session_id: str,
        prompt_text: str = "",
        action_history: list[dict[str, Any]] | None = None,
        parent_checkpoint_id: str | None = None,
        *,
        latent_video: torch.Tensor | None = None,
        prompt_embs: list[torch.Tensor] | None = None,
        frame_index: int | None = None,
    ) -> WorldStateSnapshot:
        """Extract the current model state into a CPU-offloaded snapshot.

        Reads the four KV caches (``kv_cache1``, ``kv_cache_neg``,
        ``crossattn_cache``, ``crossattn_cache_neg``) plus the cached
        embeddings and tracking scalars from ``self.action_head``.

        Parameters
        ----------
        session_id : str
            Owning session identifier.
        prompt_text : str
            Human-readable text prompt for metadata.
        action_history : list[dict] | None
            Actions taken so far (deep-copied into the snapshot).
        parent_checkpoint_id : str | None
            For tree lineage — the checkpoint this one descends from.
        latent_video : torch.Tensor | None
            Accumulated latent video tensor (not stored on the action head —
            the caller, typically ``Session``, must supply it).
        prompt_embs : list[torch.Tensor] | None
            Pre-encoded text embeddings.  These are computed per call inside
            ``lazy_joint_video_action`` and are **not** cached on the action
            head, so the caller may optionally supply them.
        frame_index : int | None
            Explicit frame index.  When ``None`` the current value of
            ``action_head.current_start_frame`` is used.
        """
        head = self.action_head

        # Deep-copy KV caches → CPU
        kv_cache = self._clone_cache_to_cpu(head.kv_cache1)
        kv_cache_neg = self._clone_cache_to_cpu(head.kv_cache_neg)
        crossattn_cache = self._clone_cache_to_cpu(head.crossattn_cache)
        crossattn_cache_neg = self._clone_cache_to_cpu(head.crossattn_cache_neg)

        # Deep-copy embeddings → CPU
        clip_feas = self._clone_tensor_to_cpu(head.clip_feas)
        ys = self._clone_tensor_to_cpu(head.ys)
        language = self._clone_tensor_to_cpu(head.language)

        # Latent video and prompt embeddings are not stored on the action head;
        # accept them from the caller and deep-copy to CPU.
        latent_video_cpu = self._clone_tensor_to_cpu(latent_video)
        prompt_embs_cpu: list[torch.Tensor] = (
            [t.detach().cpu().clone() for t in prompt_embs]
            if prompt_embs is not None
            else []
        )

        current_start_frame: int = head.current_start_frame
        seed: int = head.seed

        snapshot = WorldStateSnapshot(
            checkpoint_id=WorldStateSnapshot.new_id(),
            session_id=session_id,
            frame_index=frame_index if frame_index is not None else current_start_frame,
            timestamp=time.time(),
            kv_cache=kv_cache,
            kv_cache_neg=kv_cache_neg,
            crossattn_cache=crossattn_cache,
            crossattn_cache_neg=crossattn_cache_neg,
            latent_video=latent_video_cpu,
            clip_feas=clip_feas,
            ys=ys,
            prompt_embs=prompt_embs_cpu,
            current_start_frame=current_start_frame,
            language=language,
            seed=seed,
            prompt_text=prompt_text,
            action_history=list(action_history) if action_history is not None else [],
            parent_checkpoint_id=parent_checkpoint_id,
        )

        logger.info(
            "Extracted snapshot %s (session=%s, frame=%d, %.1f MB)",
            snapshot.checkpoint_id,
            session_id,
            snapshot.frame_index,
            snapshot.size_bytes() / (1024 * 1024),
        )
        return snapshot

    def restore_state(self, snapshot: WorldStateSnapshot) -> None:
        """Restore model state from a snapshot (move tensors back to GPU).

        This overwrites the live KV caches and cached embeddings on
        ``self.action_head`` with deep copies from the snapshot.

        The snapshot itself is **not** mutated — its tensors remain on
        whatever device they were on (typically CPU).

        After restoring, ``current_start_frame`` and ``language`` are set
        so that the reset-detection logic in ``lazy_joint_video_action``
        does **not** trigger an unwanted reset.
        """
        device = self._device
        head = self.action_head

        # Restore self-attention KV caches
        head.kv_cache1 = self._restore_cache_to_device(snapshot.kv_cache, device)
        head.kv_cache_neg = self._restore_cache_to_device(snapshot.kv_cache_neg, device)

        # Restore cross-attention KV caches
        head.crossattn_cache = self._restore_cache_to_device(
            snapshot.crossattn_cache, device,
        )
        head.crossattn_cache_neg = self._restore_cache_to_device(
            snapshot.crossattn_cache_neg, device,
        )

        # Restore cached embeddings
        head.clip_feas = self._restore_tensor_to_device(snapshot.clip_feas, device)
        head.ys = self._restore_tensor_to_device(snapshot.ys, device)

        # Restore tracking state — this prevents the reset logic from
        # zeroing current_start_frame on the next call.
        head.current_start_frame = snapshot.current_start_frame
        head.language = self._restore_tensor_to_device(snapshot.language, device)

        logger.info(
            "Restored snapshot %s → device=%s (session=%s, frame=%d)",
            snapshot.checkpoint_id,
            device,
            snapshot.session_id,
            snapshot.frame_index,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Convenience wrappers
    # ──────────────────────────────────────────────────────────────────────

    def checkpoint(self, session_id: str, **kwargs: Any) -> str:
        """Snapshot the current model state and store it.

        All *kwargs* are forwarded to :meth:`extract_state`.

        Returns
        -------
        str
            The ``checkpoint_id`` of the newly created snapshot.
        """
        snapshot = self.extract_state(session_id, **kwargs)
        self.checkpoints[snapshot.checkpoint_id] = snapshot
        self._evict_if_needed()
        return snapshot.checkpoint_id

    def restore(self, checkpoint_id: str) -> WorldStateSnapshot:
        """Restore the model to a previously checkpointed state.

        Parameters
        ----------
        checkpoint_id : str
            Must be a key present in ``self.checkpoints``.

        Returns
        -------
        WorldStateSnapshot
            The snapshot that was restored (still CPU-resident).

        Raises
        ------
        KeyError
            If *checkpoint_id* is not found.
        """
        if checkpoint_id not in self.checkpoints:
            raise KeyError(
                f"Checkpoint {checkpoint_id!r} not found. "
                f"Available: {list(self.checkpoints.keys())}"
            )
        snapshot = self.checkpoints[checkpoint_id]
        self.restore_state(snapshot)
        return snapshot

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Remove a checkpoint from the in-memory store.

        Raises :class:`KeyError` if *checkpoint_id* does not exist.
        """
        if checkpoint_id not in self.checkpoints:
            raise KeyError(f"Checkpoint {checkpoint_id!r} not found.")
        del self.checkpoints[checkpoint_id]
        logger.info("Deleted checkpoint %s", checkpoint_id)

    def list_checkpoints(self, session_id: str | None = None) -> list[str]:
        """Return checkpoint IDs, optionally filtered by *session_id*.

        The list is ordered oldest-first (insertion order).
        """
        if session_id is None:
            return list(self.checkpoints.keys())
        return [
            cid
            for cid, snap in self.checkpoints.items()
            if snap.session_id == session_id
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def total_memory_bytes(self) -> int:
        """Sum of ``size_bytes()`` across all stored checkpoints."""
        return sum(snap.size_bytes() for snap in self.checkpoints.values())

    def __repr__(self) -> str:
        total_mb = self.total_memory_bytes() / (1024 * 1024)
        return (
            f"StateManager(checkpoints={len(self.checkpoints)}, "
            f"max={self.max_checkpoints}, "
            f"total={total_mb:.1f} MB)"
        )
