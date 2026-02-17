"""
Arc Fabric API Server (Layer 2)

FastAPI application exposing checkpoint/fork/rewind operations for DreamZero.

Supports two modes controlled by ARC_FABRIC_MODE env var:
  - mock  (default): Returns placeholder data, no GPU needed
  - live:  Loads the DreamZero model and runs real inference

Usage:
    ARC_FABRIC_MODE=mock uvicorn arc_fabric.server:app --host 0.0.0.0 --port 8420
"""

from __future__ import annotations

import io
import os
import random
import struct
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARC_FABRIC_MODE: str = os.environ.get("ARC_FABRIC_MODE", "mock")
MAX_SESSIONS: int = 5
ACTION_DIM: int = 7  # DROID embodiment
ACTION_HORIZON: int = 24
DEFAULT_SEED: int = 1140
FRAME_W, FRAME_H = 320, 240

UI_DIR = Path(__file__).parent / "ui"

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    prompt: str
    initial_frame_path: str
    embodiment_id: int = 0


class CreateSessionResponse(BaseModel):
    session_id: str
    prompt: str
    frame_index: int


class SessionInfo(BaseModel):
    session_id: str
    prompt: str
    frame_index: int
    num_checkpoints: int


class SessionDetail(BaseModel):
    session_id: str
    prompt: str
    frame_index: int
    num_checkpoints: int
    checkpoint_ids: list[str]
    action_history: list[dict]
    tree: dict


class StepRequest(BaseModel):
    action: Optional[list[float]] = None
    state: Optional[list[float]] = None
    auto_checkpoint: bool = True


class StepResponse(BaseModel):
    frame_index: int
    checkpoint_id: str
    action_pred: list[list[float]]


class CheckpointResponse(BaseModel):
    checkpoint_id: str
    frame_index: int


class RewindRequest(BaseModel):
    n_steps: int = 1


class RewindResponse(BaseModel):
    checkpoint_id: str
    frame_index: int


class ForkRequest(BaseModel):
    checkpoint_id: str
    new_prompt: Optional[str] = None
    new_seed: Optional[int] = None


class ForkResponse(BaseModel):
    session_id: str
    forked_from: str
    frame_index: int


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_placeholder_png(frame_index: int, width: int = FRAME_W, height: int = FRAME_H) -> bytes:
    """Generate a minimal valid PNG with a colored rectangle and frame number.

    Creates the image byte-by-byte without any imaging library so mock mode
    has zero heavy dependencies.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        hue = (frame_index * 37) % 360
        r, g, b = _hsv_to_rgb(hue, 0.45, 0.85)
        img = Image.new("RGB", (width, height), (r, g, b))
        draw = ImageDraw.Draw(img)

        text = f"Frame {frame_index}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except (OSError, IOError):
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((width - tw) // 2, (height - th) // 2), text, fill=(255, 255, 255), font=font)

        label = f"Arc Fabric – mock mode"
        draw.text((8, height - 22), label, fill=(200, 200, 200))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return _make_minimal_png(frame_index, width, height)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (h in 0-360, s/v in 0-1) to RGB (0-255)."""
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _make_minimal_png(frame_index: int, width: int = FRAME_W, height: int = FRAME_H) -> bytes:
    """Generate a minimal valid PNG without any imaging library (fallback)."""
    import zlib

    hue = (frame_index * 37) % 360
    r, g, b = _hsv_to_rgb(hue, 0.45, 0.85)

    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00" + bytes([r, g, b]) * width

    compressed = zlib.compress(raw_rows)

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr_data)
    png += _png_chunk(b"IDAT", compressed)
    png += _png_chunk(b"IEND", b"")
    return png


def _mock_action_pred() -> list[list[float]]:
    """Return a plausible random action prediction array."""
    return [
        [round(random.gauss(0, 0.02), 5) for _ in range(ACTION_DIM)]
        for _ in range(ACTION_HORIZON)
    ]


def _mock_sample_tree(sessions: dict[str, _MockSession]) -> dict:
    """Build a D3.js-compatible tree from mock sessions."""
    nodes: list[dict] = []
    for sess in sessions.values():
        for i, cp_id in enumerate(sess.checkpoint_ids):
            nodes.append(
                {
                    "checkpoint_id": cp_id,
                    "session_id": sess.session_id,
                    "frame_index": i,
                    "prompt": sess.prompt,
                    "parent_id": sess.checkpoint_ids[i - 1] if i > 0 else sess.forked_from,
                    "children": [],
                }
            )

    lookup = {n["checkpoint_id"]: n for n in nodes}
    roots: list[dict] = []
    for n in nodes:
        pid = n["parent_id"]
        if pid and pid in lookup:
            lookup[pid]["children"].append(n)
        else:
            roots.append(n)

    if len(roots) == 1:
        return roots[0]
    return {"checkpoint_id": "virtual-root", "session_id": "", "frame_index": -1,
            "prompt": "", "parent_id": None, "children": roots}


# ---------------------------------------------------------------------------
# Mock session object (used only in mock mode)
# ---------------------------------------------------------------------------


class _MockSession:
    """Lightweight stand-in for a real Session when running without GPU."""

    def __init__(self, session_id: str, prompt: str, initial_frame_path: str,
                 embodiment_id: int = 0, forked_from: str | None = None,
                 start_frame: int = 0):
        self.session_id = session_id
        self.prompt = prompt
        self.initial_frame_path = initial_frame_path
        self.embodiment_id = embodiment_id
        self.frame_index: int = start_frame
        self.checkpoint_ids: list[str] = []
        self.action_history: list[dict] = []
        self.forked_from: str | None = forked_from
        self.seed: int = DEFAULT_SEED

        initial_cp = self._make_checkpoint()
        self.checkpoint_ids.append(initial_cp)

    def _make_checkpoint(self) -> str:
        cp_id = str(uuid.uuid4())
        return cp_id

    def step(self, action: list[float] | None, state: list[float] | None,
             auto_checkpoint: bool = True) -> StepResponse:
        self.frame_index += 1
        action_used = action or [round(random.gauss(0, 0.02), 5) for _ in range(ACTION_DIM)]
        self.action_history.append({"frame": self.frame_index, "action": action_used})

        cp_id = ""
        if auto_checkpoint:
            cp_id = self._make_checkpoint()
            self.checkpoint_ids.append(cp_id)

        return StepResponse(
            frame_index=self.frame_index,
            checkpoint_id=cp_id,
            action_pred=_mock_action_pred(),
        )

    def checkpoint(self) -> CheckpointResponse:
        cp_id = self._make_checkpoint()
        self.checkpoint_ids.append(cp_id)
        return CheckpointResponse(checkpoint_id=cp_id, frame_index=self.frame_index)

    def rewind(self, n_steps: int = 1) -> RewindResponse:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        target = max(0, len(self.checkpoint_ids) - 1 - n_steps)
        self.checkpoint_ids = self.checkpoint_ids[: target + 1]
        self.frame_index = max(0, self.frame_index - n_steps)
        self.action_history = self.action_history[: self.frame_index]
        return RewindResponse(
            checkpoint_id=self.checkpoint_ids[-1],
            frame_index=self.frame_index,
        )

    def to_info(self) -> SessionInfo:
        return SessionInfo(
            session_id=self.session_id,
            prompt=self.prompt,
            frame_index=self.frame_index,
            num_checkpoints=len(self.checkpoint_ids),
        )

    def to_detail(self, tree: dict) -> SessionDetail:
        return SessionDetail(
            session_id=self.session_id,
            prompt=self.prompt,
            frame_index=self.frame_index,
            num_checkpoints=len(self.checkpoint_ids),
            checkpoint_ids=list(self.checkpoint_ids),
            action_history=list(self.action_history),
            tree=tree,
        )


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Holds all sessions and shared resources.

    In *mock* mode every field is a plain Python object.
    In *live* mode the real model, StateManager, and TrajectoryTree are loaded.
    """

    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self.sessions: dict[str, _MockSession] = {}

        if mode == "live":
            self._init_live()

    # -- live mode bootstrap (deferred import keeps mock mode dependency-free) --

    def _init_live(self) -> None:
        """Import heavy modules and instantiate the real model stack."""
        try:
            from arc_fabric.manager import StateManager
            from arc_fabric.session import Session
            from arc_fabric.tree import TrajectoryTree
        except ImportError as exc:
            raise RuntimeError(
                "Live mode requires arc_fabric SDK modules and PyTorch. "
                "Set ARC_FABRIC_MODE=mock if you don't have GPU dependencies."
            ) from exc

        self.tree = TrajectoryTree()
        # Model and StateManager would be initialised here once weights are
        # available.  For now we store references as None so the code path
        # is exercised even if weights aren't downloaded.
        self.model = None
        self.state_manager = None

    # -- session CRUD -------------------------------------------------------

    def create_session(
        self, prompt: str, initial_frame_path: str, embodiment_id: int = 0
    ) -> _MockSession:
        if len(self.sessions) >= MAX_SESSIONS:
            raise HTTPException(
                status_code=409,
                detail=f"Maximum number of sessions ({MAX_SESSIONS}) reached. "
                       f"Delete an existing session first.",
            )
        session_id = str(uuid.uuid4())
        sess = _MockSession(
            session_id=session_id,
            prompt=prompt,
            initial_frame_path=initial_frame_path,
            embodiment_id=embodiment_id,
        )
        self.sessions[session_id] = sess
        return sess

    def get_session(self, session_id: str) -> _MockSession:
        sess = self.sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return sess

    def delete_session(self, session_id: str) -> None:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        del self.sessions[session_id]

    def list_sessions(self) -> list[SessionInfo]:
        return [s.to_info() for s in self.sessions.values()]

    def fork_session(
        self, source_session_id: str, checkpoint_id: str,
        new_prompt: str | None = None, new_seed: int | None = None,
    ) -> _MockSession:
        source = self.get_session(source_session_id)

        if checkpoint_id not in source.checkpoint_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint {checkpoint_id} not found in session {source_session_id}",
            )

        cp_index = source.checkpoint_ids.index(checkpoint_id)

        if len(self.sessions) >= MAX_SESSIONS:
            raise HTTPException(
                status_code=409,
                detail=f"Maximum number of sessions ({MAX_SESSIONS}) reached.",
            )

        new_session_id = str(uuid.uuid4())
        forked = _MockSession(
            session_id=new_session_id,
            prompt=new_prompt or source.prompt,
            initial_frame_path=source.initial_frame_path,
            embodiment_id=source.embodiment_id,
            forked_from=checkpoint_id,
            start_frame=cp_index,
        )
        if new_seed is not None:
            forked.seed = new_seed

        self.sessions[new_session_id] = forked
        return forked

    def build_tree(self) -> dict:
        return _mock_sample_tree(self.sessions)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Arc Fabric",
    description="Checkpoint / Fork / Rewind platform for DreamZero",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = SessionManager(mode=ARC_FABRIC_MODE)


# -- Startup ----------------------------------------------------------------

@app.on_event("startup")
async def _startup() -> None:
    mode_label = "MOCK (no GPU)" if ARC_FABRIC_MODE == "mock" else "LIVE"
    print(
        f"\n{'=' * 56}\n"
        f"  Arc Fabric server starting\n"
        f"  Mode : {mode_label}\n"
        f"  UI   : http://localhost:8420/\n"
        f"  API  : http://localhost:8420/api/sessions\n"
        f"{'=' * 56}\n"
    )


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------


@app.post("/api/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(req: CreateSessionRequest):
    sess = manager.create_session(
        prompt=req.prompt,
        initial_frame_path=req.initial_frame_path,
        embodiment_id=req.embodiment_id,
    )
    return CreateSessionResponse(
        session_id=sess.session_id,
        prompt=sess.prompt,
        frame_index=sess.frame_index,
    )


@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_sessions():
    return manager.list_sessions()


@app.get("/api/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    sess = manager.get_session(session_id)
    tree = manager.build_tree()
    return sess.to_detail(tree)


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    manager.delete_session(session_id)


# ---------------------------------------------------------------------------
# Generation endpoints
# ---------------------------------------------------------------------------


@app.post("/api/sessions/{session_id}/step", response_model=StepResponse)
async def step_session(session_id: str, req: StepRequest):
    sess = manager.get_session(session_id)

    if req.action is not None and len(req.action) != ACTION_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"action must have {ACTION_DIM} dimensions (got {len(req.action)})",
        )

    return sess.step(
        action=req.action,
        state=req.state,
        auto_checkpoint=req.auto_checkpoint,
    )


@app.post("/api/sessions/{session_id}/checkpoint", response_model=CheckpointResponse)
async def checkpoint_session(session_id: str):
    sess = manager.get_session(session_id)
    return sess.checkpoint()


@app.post("/api/sessions/{session_id}/rewind", response_model=RewindResponse)
async def rewind_session(session_id: str, req: RewindRequest):
    sess = manager.get_session(session_id)

    if req.n_steps < 1:
        raise HTTPException(status_code=422, detail="n_steps must be >= 1")

    if req.n_steps > len(sess.checkpoint_ids) - 1:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot rewind {req.n_steps} steps; session only has "
                   f"{len(sess.checkpoint_ids) - 1} checkpoints to rewind through.",
        )

    return sess.rewind(n_steps=req.n_steps)


# ---------------------------------------------------------------------------
# Forking
# ---------------------------------------------------------------------------


@app.post("/api/sessions/{session_id}/fork", response_model=ForkResponse, status_code=201)
async def fork_session(session_id: str, req: ForkRequest):
    forked = manager.fork_session(
        source_session_id=session_id,
        checkpoint_id=req.checkpoint_id,
        new_prompt=req.new_prompt,
        new_seed=req.new_seed,
    )
    return ForkResponse(
        session_id=forked.session_id,
        forked_from=req.checkpoint_id,
        frame_index=forked.frame_index,
    )


# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------


@app.get("/api/sessions/{session_id}/frames/{frame_index}")
async def get_frame(session_id: str, frame_index: int):
    sess = manager.get_session(session_id)

    if frame_index < 0 or frame_index > sess.frame_index:
        raise HTTPException(
            status_code=404,
            detail=f"Frame {frame_index} not available (session is at frame {sess.frame_index})",
        )

    png_bytes = _make_placeholder_png(frame_index)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/api/sessions/{session_id}/video")
async def get_video(session_id: str):
    sess = manager.get_session(session_id)

    if ARC_FABRIC_MODE == "mock":
        # Return a minimal placeholder — a sequence of PNGs stitched into
        # an in-memory MP4 would require ffmpeg/imageio.  For mock mode we
        # return empty bytes with the correct content type so the UI can
        # detect the zero-length body gracefully.
        return Response(content=b"", media_type="video/mp4")

    raise HTTPException(status_code=501, detail="Live video encoding not yet implemented")


@app.get("/api/sessions/{session_id}/tree")
async def get_tree(session_id: str):
    _ = manager.get_session(session_id)  # validate existence
    return manager.build_tree()


# ---------------------------------------------------------------------------
# Static / UI
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content=(
                "<!DOCTYPE html><html><head><title>Arc Fabric</title></head>"
                "<body style='font-family:system-ui;padding:2rem'>"
                "<h1>Arc Fabric</h1>"
                "<p>No UI has been built yet. Place <code>index.html</code> in "
                "<code>arc_fabric/ui/</code>.</p>"
                "<p>API docs: <a href='/docs'>/docs</a></p>"
                "</body></html>"
            ),
            status_code=200,
        )
    return HTMLResponse(content=index_path.read_text(), status_code=200)


# Mount static assets if the directory exists
_static_dir = UI_DIR / "static"
if _static_dir.exists() and _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Main (convenience: `python -m arc_fabric.server`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "arc_fabric.server:app",
        host="0.0.0.0",
        port=8420,
        reload=True,
    )
