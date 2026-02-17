"""Session lifecycle management for Arc Fabric.

A Session wraps a shared DreamZero VLA model with checkpoint/fork/rewind
capabilities.  The model is **not** owned by the session — multiple sessions
share a single model instance on GPU, swapping their state in and out via
the StateManager.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from arc_fabric.tree import TrajectoryTree

    from groot.vla.model.dreamzero.base_vla import VLA


class Session:
    """A single generation session: one trajectory through the world model.

    Parameters
    ----------
    session_id:
        Unique identifier for this session.
    model:
        Shared VLA model instance (lives on GPU).
    state_manager:
        StateManager used for checkpoint/restore of model state.
    tree:
        Shared TrajectoryTree that tracks nodes across *all* sessions.
    prompt:
        Text prompt for this session's generation.
    initial_frame:
        The first RGB frame as a uint8 numpy array ``[H, W, C]``.
    embodiment_id:
        Robot embodiment id (selects action/state encoder weights).
    """

    def __init__(
        self,
        session_id: str,
        model: "VLA",
        state_manager: Any,  # StateManager — imported at runtime to avoid cycles
        tree: "TrajectoryTree",
        prompt: str,
        initial_frame: np.ndarray,
        embodiment_id: int = 0,
    ) -> None:
        self.session_id = session_id
        self.model = model
        self.state_manager = state_manager
        self.tree = tree

        self.prompt = prompt
        self.initial_frame = initial_frame  # [H, W, C] uint8
        self.embodiment_id = embodiment_id

        self.frame_index: int = 0
        self.action_history: list[dict] = []
        self.checkpoint_ids: list[str] = []

        # Accumulated latent video tensor — grows each step.
        # Shape: [B, T_accum, 16, H/8, W/8] after transposition from model output.
        self.latent_video: torch.Tensor | None = None

        # Most recent model outputs (kept for convenience / downstream use).
        self._last_action_pred: torch.Tensor | None = None
        self._last_video_pred: torch.Tensor | None = None

        self.created_at: float = time.time()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def step(
        self,
        action: torch.Tensor,
        state: torch.Tensor,
        auto_checkpoint: bool = True,
    ) -> dict[str, Any]:
        """Advance one generation step with the given action and state.

        Parameters
        ----------
        action:
            Action tensor ``[B, action_horizon, action_dim]``.
        state:
            State tensor ``[B, num_state, state_dim]``.
        auto_checkpoint:
            If True (default), automatically checkpoint after the step.

        Returns
        -------
        dict with keys ``action_pred``, ``video_pred``, ``frame_index``,
        ``checkpoint_id`` (or None if auto_checkpoint is False).
        """
        # Build the image input for the model.
        # On the very first step we use the initial_frame; on subsequent
        # steps we pass the accumulated latent video directly via
        # ``latent_video`` so the model skips re-encoding through the VAE.
        images = self._build_image_input()

        # Tokenize / encode the prompt through the model's backbone prep.
        # The model expects ``text`` and ``text_attention_mask`` as produced
        # by its tokenizer — we pass them through from a prior encoding or
        # let the model handle raw tokens.  Here we construct the full
        # input dict expected by ``model.lazy_joint_video_action_causal``.
        input_dict = self._build_model_input(images, action, state)

        # Determine whether to inject latent_video.
        latent_video_for_model: torch.Tensor | None = None
        if self.frame_index > 0 and self.latent_video is not None:
            # Model expects [B, 16, T, H/8, W/8] — our accumulator is
            # [B, T, 16, H/8, W/8] so transpose channels and time.
            latent_video_for_model = self.latent_video.transpose(1, 2)

        output = self.model.lazy_joint_video_action_causal(
            input_dict, latent_video=latent_video_for_model
        )

        action_pred = output["action_pred"]
        video_pred = output["video_pred"]

        # ``video_pred`` comes back as [B, 16, T_out, H/8, W/8] from the
        # model.  Transpose to [B, T_out, 16, H/8, W/8] for accumulation.
        video_pred_bt = video_pred.transpose(1, 2)

        # Accumulate latents.
        if self.latent_video is None:
            self.latent_video = video_pred_bt.cpu()
        else:
            self.latent_video = torch.cat(
                [self.latent_video, video_pred_bt[:, -self.model.action_head.num_frame_per_block:].cpu()],
                dim=1,
            )

        self._last_action_pred = action_pred
        self._last_video_pred = video_pred

        # Record action.
        action_record = _tensor_to_serializable(action)
        self.action_history.append(action_record)

        self.frame_index += 1

        # Checkpoint & tree bookkeeping.
        checkpoint_id: str | None = None
        if auto_checkpoint:
            checkpoint_id = self.checkpoint()

            parent_id = self.checkpoint_ids[-2] if len(self.checkpoint_ids) >= 2 else None
            self.tree.add_node(
                checkpoint_id=checkpoint_id,
                session_id=self.session_id,
                frame_index=self.frame_index,
                prompt=self.prompt,
                action=action_record,
                parent_id=parent_id,
                metadata={
                    "step_time": time.time(),
                    "auto_checkpoint": True,
                },
            )

        return {
            "action_pred": action_pred,
            "video_pred": video_pred,
            "frame_index": self.frame_index,
            "checkpoint_id": checkpoint_id,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def checkpoint(self) -> str:
        """Snapshot the current model state.  Returns the checkpoint_id."""
        checkpoint_id = self.state_manager.checkpoint(
            session_id=self.session_id,
            prompt_text=self.prompt,
            action_history=list(self.action_history),
            parent_checkpoint_id=(
                self.checkpoint_ids[-1] if self.checkpoint_ids else None
            ),
            latent_video=self.latent_video,
            frame_index=self.frame_index,
        )
        self.checkpoint_ids.append(checkpoint_id)
        return checkpoint_id

    # ------------------------------------------------------------------
    # Rewind
    # ------------------------------------------------------------------

    def rewind(self, n_steps: int = 1) -> str:
        """Go back *n_steps* by restoring the appropriate checkpoint.

        Trims action_history and checkpoint_ids to that point, resets
        frame_index, and restores model state.

        Raises
        ------
        ValueError
            If there are fewer checkpoints than *n_steps*.

        Returns
        -------
        The restored checkpoint_id.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        if len(self.checkpoint_ids) < n_steps:
            raise ValueError(
                f"Cannot rewind {n_steps} steps: only "
                f"{len(self.checkpoint_ids)} checkpoint(s) available"
            )

        # Target index in the checkpoint list.
        target_idx = len(self.checkpoint_ids) - n_steps - 1

        if target_idx < 0:
            # Rewinding to *before* the first checkpoint — restore to the
            # very first checkpoint and reset to its state.
            target_idx = 0

        target_checkpoint_id = self.checkpoint_ids[target_idx]
        snapshot = self.state_manager.restore(target_checkpoint_id)

        # Trim histories to the target point.
        self.checkpoint_ids = self.checkpoint_ids[: target_idx + 1]
        self.action_history = self.action_history[: target_idx]

        # Reset frame tracking from the snapshot.
        self.frame_index = snapshot.frame_index

        # Restore latent video from the snapshot.
        if snapshot.latent_video is not None:
            self.latent_video = snapshot.latent_video.cpu()
        else:
            self.latent_video = None

        return target_checkpoint_id

    # ------------------------------------------------------------------
    # Forking
    # ------------------------------------------------------------------

    def fork(
        self,
        checkpoint_id: str,
        new_prompt: str | None = None,
        new_seed: int | None = None,
    ) -> "Session":
        """Fork a new session from a specific checkpoint.

        1. Checkpoint the *current* session (so no state is lost).
        2. Restore the target checkpoint into the model.
        3. Create a brand-new ``Session`` that starts from that point.

        Parameters
        ----------
        checkpoint_id:
            Which checkpoint to fork from.
        new_prompt:
            Optional new prompt for the forked session.
        new_seed:
            Optional new random seed.

        Returns
        -------
        A new ``Session`` instance sharing the same model and tree.
        """
        # 1. Save current session state so it isn't lost.
        if self.frame_index > 0:
            self.checkpoint()

        # 2. Restore the target checkpoint onto the model.
        snapshot = self.state_manager.restore(checkpoint_id)

        # 3. Optionally override the seed.
        if new_seed is not None:
            self.model.action_head.seed = new_seed

        # 4. Optionally override the prompt — clear cached language so the
        #    model re-encodes on the next step.
        prompt = new_prompt if new_prompt is not None else snapshot.prompt_text
        if new_prompt is not None:
            self.model.action_head.language = None

        # 5. Build the new session.
        new_session_id = str(uuid.uuid4())
        forked = Session(
            session_id=new_session_id,
            model=self.model,
            state_manager=self.state_manager,
            tree=self.tree,
            prompt=prompt,
            initial_frame=self.initial_frame,
            embodiment_id=self.embodiment_id,
        )

        # Carry over state from the snapshot.
        forked.frame_index = snapshot.frame_index
        forked.action_history = list(snapshot.action_history)
        forked.checkpoint_ids = [checkpoint_id]

        if snapshot.latent_video is not None:
            forked.latent_video = snapshot.latent_video.cpu()

        return forked

    # ------------------------------------------------------------------
    # Frame decoding
    # ------------------------------------------------------------------

    def get_decoded_frame(self, frame_index: int) -> np.ndarray:
        """Decode a single latent frame through the VAE and return RGB uint8.

        Parameters
        ----------
        frame_index:
            Which accumulated frame to decode (0-based).

        Returns
        -------
        ``np.ndarray`` with shape ``[H, W, C]`` and dtype ``uint8``.

        Raises
        ------
        ValueError
            If no latent video has been accumulated yet or index is out of
            range.
        """
        if self.latent_video is None:
            raise ValueError("No latent video accumulated yet — run step() first")

        total_frames = self.latent_video.shape[1]
        if frame_index < 0 or frame_index >= total_frames:
            raise ValueError(
                f"frame_index {frame_index} out of range [0, {total_frames})"
            )

        # Extract a single frame: [B, 1, C_latent, H/8, W/8]
        latent_frame = self.latent_video[:, frame_index : frame_index + 1]
        # VAE expects [B, C, T, H/8, W/8]
        latent_frame_gpu = latent_frame.transpose(1, 2).to(
            device=self.model.device, dtype=torch.bfloat16
        )

        vae = self.model.action_head.vae
        with torch.no_grad():
            decoded = vae.decode(
                latent_frame_gpu,
                tiled=self.model.action_head.tiled,
                tile_size=(
                    self.model.action_head.tile_size_height,
                    self.model.action_head.tile_size_width,
                ),
                tile_stride=(
                    self.model.action_head.tile_stride_height,
                    self.model.action_head.tile_stride_width,
                ),
            )

        # decoded: [B, C, T, H, W] in [-1, 1]
        frame = decoded[0, :, 0]  # [C, H, W]
        frame = (frame.clamp(-1, 1).float() + 1.0) / 2.0 * 255.0
        frame = frame.byte().permute(1, 2, 0).cpu().numpy()  # [H, W, C] uint8
        return frame

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable session metadata."""
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "embodiment_id": self.embodiment_id,
            "frame_index": self.frame_index,
            "num_checkpoints": len(self.checkpoint_ids),
            "checkpoint_ids": list(self.checkpoint_ids),
            "action_history": list(self.action_history),
            "latent_frames": (
                self.latent_video.shape[1] if self.latent_video is not None else 0
            ),
            "created_at": self.created_at,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_image_input(self) -> np.ndarray:
        """Prepare the image array expected by the model.

        On the first step (``frame_index == 0``) this is the ``initial_frame``
        reshaped to ``[B, T, H, W, C]`` uint8.  On subsequent steps the model
        receives the latent video directly via the ``latent_video`` parameter
        and the image input is still needed for shape/preprocessing but won't
        be VAE-encoded.
        """
        if self.frame_index == 0:
            # First step: single initial frame → [1, 1, H, W, C]
            return self.initial_frame[np.newaxis, np.newaxis, ...]
        else:
            # Subsequent steps: the model uses `latent_video` directly.
            # We still pass the initial frame as a single-frame image so that
            # shape-dependent preprocessing in prepare_input can run, but the
            # heavy VAE encoding is skipped.
            return self.initial_frame[np.newaxis, np.newaxis, ...]

    def _build_model_input(
        self,
        images: np.ndarray,
        action: torch.Tensor,
        state: torch.Tensor,
    ) -> dict[str, Any]:
        """Assemble the input dict for ``model.lazy_joint_video_action_causal``.

        The model (via ``prepare_input``) expects:
        - ``images``: uint8 numpy ``[B, T, H, W, C]``
        - ``text``, ``text_attention_mask``, ``text_negative``,
          ``text_attention_mask_negative``: tokenized prompt tensors
        - ``state``: ``[B, num_state, state_dim]``
        - ``embodiment_id``: ``[B]``
        - ``action`` (optional): ``[B, horizon, dim]``
        """
        device = self.model.device

        # Tokenize the prompt using the text encoder's tokenizer (T5).
        text_inputs = _tokenize_prompt(
            self.model.action_head.text_encoder,
            self.prompt,
            device=device,
        )

        embodiment_id = torch.tensor(
            [self.embodiment_id], dtype=torch.long, device=device
        )

        input_dict: dict[str, Any] = {
            "images": images,
            **text_inputs,
            "state": state,
            "embodiment_id": embodiment_id,
        }

        if action is not None:
            input_dict["action"] = action

        return input_dict

    def __repr__(self) -> str:
        return (
            f"Session(id={self.session_id!r}, prompt={self.prompt!r}, "
            f"frame={self.frame_index})"
        )


# ======================================================================
# Module-level helpers
# ======================================================================

def _tokenize_prompt(
    text_encoder: Any,
    prompt: str,
    device: torch.device | str = "cuda",
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """Tokenize a text prompt for the T5-based text encoder.

    Returns a dict with ``text``, ``text_attention_mask``,
    ``text_negative``, and ``text_attention_mask_negative`` — the
    keys expected by ``WANPolicyHead.prepare_input / _prepare_text_inputs``.
    """
    tokenizer = text_encoder.tokenizer

    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    text = tokens.input_ids.to(device)
    text_attention_mask = tokens.attention_mask.to(device)

    # Negative prompt: empty string (unconditional for CFG).
    neg_tokens = tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    text_negative = neg_tokens.input_ids.to(device)
    text_attention_mask_negative = neg_tokens.attention_mask.to(device)

    return {
        "text": text,
        "text_attention_mask": text_attention_mask,
        "text_negative": text_negative,
        "text_attention_mask_negative": text_attention_mask_negative,
    }


def _tensor_to_serializable(t: torch.Tensor | np.ndarray | Any) -> Any:
    """Convert a tensor (or nested structure) to a JSON-safe value."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().tolist()
    if isinstance(t, np.ndarray):
        return t.tolist()
    if isinstance(t, dict):
        return {k: _tensor_to_serializable(v) for k, v in t.items()}
    if isinstance(t, (list, tuple)):
        return [_tensor_to_serializable(v) for v in t]
    return t
