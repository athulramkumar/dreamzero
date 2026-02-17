#!/usr/bin/env python3
"""Comprehensive evaluation suite for DreamZero-DROID.

Runs structured test cases against a live server, measures action outputs,
and produces a detailed report including:
  - Per-prompt action statistics (shape, range, smoothness)
  - Language sensitivity analysis (cosine similarity between paired prompts)
  - Timing benchmarks per chunk
  - Optional action trajectory plots

Usage:
    # Run all test categories:
    python evals/eval_suite.py --port 5000

    # Run specific category:
    python evals/eval_suite.py --port 5000 --category seen_tasks

    # Run a single test case by ID:
    python evals/eval_suite.py --port 5000 --case-id seen_01

    # Run with action plots:
    python evals/eval_suite.py --port 5000 --plot
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_utils.policy_server as policy_server
from eval_utils.policy_client import WebsocketClientPolicy

VIDEO_DIR = Path(__file__).resolve().parent.parent / "debug_image"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}

RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24


@dataclass
class ChunkResult:
    chunk_idx: int
    frame_indices: list
    actions: np.ndarray
    latency_s: float


@dataclass
class EvalResult:
    case_id: str
    prompt: str
    category: str
    session_id: str
    chunks: list = field(default_factory=list)
    total_time_s: float = 0.0
    error: str | None = None

    @property
    def all_actions(self) -> np.ndarray:
        if not self.chunks:
            return np.empty((0, 8))
        return np.concatenate([c.actions for c in self.chunks], axis=0)

    @property
    def mean_chunk_latency(self) -> float:
        if not self.chunks:
            return 0.0
        return np.mean([c.latency_s for c in self.chunks])

    def action_stats(self) -> dict:
        a = self.all_actions
        if a.size == 0:
            return {}
        diffs = np.diff(a, axis=0)
        smoothness = np.mean(np.linalg.norm(diffs, axis=1)) if len(diffs) > 0 else 0.0
        return {
            "shape": list(a.shape),
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "joint_ranges": [float(a[:, j].max() - a[:, j].min()) for j in range(a.shape[1])],
            "smoothness_l2": float(smoothness),
            "mean_chunk_latency_s": float(self.mean_chunk_latency),
            "total_time_s": float(self.total_time_s),
            "num_chunks": len(self.chunks),
        }


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
        path = VIDEO_DIR / fname
        camera_frames[cam_key] = load_all_frames(str(path))
        logging.info(f"  Loaded {cam_key}: {camera_frames[cam_key].shape}")
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


def make_obs(camera_frames, frame_indices, prompt, session_id):
    obs = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        if len(frame_indices) == 1:
            selected = selected[0]
        obs[cam_key] = selected
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def run_single_eval(
    client: WebsocketClientPolicy,
    camera_frames: dict[str, np.ndarray],
    case_id: str,
    prompt: str,
    category: str,
    num_chunks: int = 15,
) -> EvalResult:
    session_id = str(uuid.uuid4())
    result = EvalResult(
        case_id=case_id,
        prompt=prompt,
        category=category,
        session_id=session_id,
    )
    total_frames = min(v.shape[0] for v in camera_frames.values())
    schedule = build_frame_schedule(total_frames, num_chunks)
    t_total_start = time.perf_counter()

    try:
        # Initial frame
        obs = make_obs(camera_frames, [0], prompt, session_id)
        t0 = time.perf_counter()
        actions = client.infer(obs)
        dt = time.perf_counter() - t0
        if isinstance(actions, dict):
            joint = actions.get("action.joint_position", np.zeros((24, 7)))
            gripper = actions.get("action.gripper_position", np.zeros((24, 1)))
            if isinstance(joint, np.ndarray) and isinstance(gripper, np.ndarray):
                if gripper.ndim == 1:
                    gripper = gripper.reshape(-1, 1)
                actions = np.concatenate([joint, gripper], axis=-1)
            else:
                actions = np.zeros((24, 8))
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        result.chunks.append(ChunkResult(
            chunk_idx=-1, frame_indices=[0], actions=actions, latency_s=dt
        ))

        # Subsequent chunks
        for ci, frame_indices in enumerate(schedule):
            obs = make_obs(camera_frames, frame_indices, prompt, session_id)
            t0 = time.perf_counter()
            actions = client.infer(obs)
            dt = time.perf_counter() - t0
            if isinstance(actions, dict):
                joint = actions.get("action.joint_position", np.zeros((24, 7)))
                gripper = actions.get("action.gripper_position", np.zeros((24, 1)))
                if isinstance(joint, np.ndarray) and isinstance(gripper, np.ndarray):
                    if gripper.ndim == 1:
                        gripper = gripper.reshape(-1, 1)
                    actions = np.concatenate([joint, gripper], axis=-1)
                else:
                    actions = np.zeros((24, 8))
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            result.chunks.append(ChunkResult(
                chunk_idx=ci, frame_indices=frame_indices, actions=actions, latency_s=dt
            ))

        # Trigger reset to save video
        client.reset({})

    except Exception as e:
        result.error = str(e)
        logging.error(f"  ERROR in {case_id}: {e}")

    result.total_time_s = time.perf_counter() - t_total_start
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    min_len = min(len(a), len(b))
    return float(np.linalg.norm(a[:min_len] - b[:min_len]))


def analyze_language_sensitivity(results: dict[str, EvalResult], test_cases: dict) -> list[dict]:
    analysis = []
    cases = test_cases.get("language_sensitivity", {}).get("cases", [])
    paired_done = set()
    for case in cases:
        cid = case["id"]
        pid = case.get("paired_with")
        if not pid or cid in paired_done or pid in paired_done:
            continue
        if cid not in results or pid not in results:
            continue
        a1 = results[cid].all_actions
        a2 = results[pid].all_actions
        if a1.size == 0 or a2.size == 0:
            continue
        min_len = min(len(a1), len(a2))
        cos_sim = cosine_similarity(a1[:min_len], a2[:min_len])
        l2_dist = l2_distance(a1, a2)
        per_joint_diff = np.mean(np.abs(a1[:min_len] - a2[:min_len]), axis=0).tolist()
        analysis.append({
            "pair": [cid, pid],
            "prompts": [results[cid].prompt, results[pid].prompt],
            "expected_difference": case.get("expected_difference", "unknown"),
            "cosine_similarity": cos_sim,
            "l2_distance": l2_dist,
            "per_joint_mean_abs_diff": per_joint_diff,
            "language_sensitive": cos_sim < 0.95,
        })
        paired_done.add(cid)
        paired_done.add(pid)
    return analysis


def generate_report(
    all_results: dict[str, EvalResult],
    lang_analysis: list[dict],
    test_cases: dict,
) -> dict:
    report = {
        "summary": {
            "total_cases": len(all_results),
            "successful": sum(1 for r in all_results.values() if r.error is None),
            "failed": sum(1 for r in all_results.values() if r.error is not None),
        },
        "timing": {},
        "per_case": {},
        "language_sensitivity": lang_analysis,
        "action_range_check": [],
    }

    latencies = []
    for cid, result in all_results.items():
        stats = result.action_stats()
        report["per_case"][cid] = {
            "prompt": result.prompt,
            "category": result.category,
            "error": result.error,
            **stats,
        }
        if result.error is None:
            latencies.append(result.mean_chunk_latency)
            a = result.all_actions
            if a.size > 0:
                joints_in_range = bool(np.all(np.abs(a[:, :7]) < 10.0))
                gripper_in_range = bool(np.all((a[:, 7] >= -1.0) & (a[:, 7] <= 2.0)))
                report["action_range_check"].append({
                    "case_id": cid,
                    "joints_in_range": joints_in_range,
                    "gripper_in_range": gripper_in_range,
                    "joint_min": float(a[:, :7].min()),
                    "joint_max": float(a[:, :7].max()),
                    "gripper_min": float(a[:, 7].min()),
                    "gripper_max": float(a[:, 7].max()),
                })

    if latencies:
        report["timing"] = {
            "mean_chunk_latency_s": float(np.mean(latencies)),
            "std_chunk_latency_s": float(np.std(latencies)),
            "min_chunk_latency_s": float(np.min(latencies)),
            "max_chunk_latency_s": float(np.max(latencies)),
        }

    return report


def plot_action_trajectories(all_results: dict[str, EvalResult], output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "Gripper"]

    for cid, result in all_results.items():
        if result.error is not None:
            continue
        actions = result.all_actions
        if actions.size == 0:
            continue

        fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"[{cid}] {result.prompt[:80]}...", fontsize=10)
        for j in range(8):
            ax = axes[j // 2, j % 2]
            ax.plot(actions[:, j], linewidth=0.8)
            ax.set_ylabel(joint_names[j], fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel("Timestep")
        axes[-1, 1].set_xlabel("Timestep")
        plt.tight_layout()
        fig.savefig(output_dir / f"{cid}_trajectory.png", dpi=120)
        plt.close(fig)
        logging.info(f"  Saved plot: {cid}_trajectory.png")

    # Overlay plot for language sensitivity pairs
    lang_pairs = []
    for cid, result in all_results.items():
        if result.category == "language_sensitivity":
            lang_pairs.append((cid, result))

    if len(lang_pairs) >= 2:
        for i in range(0, len(lang_pairs) - 1, 2):
            cid_a, res_a = lang_pairs[i]
            cid_b, res_b = lang_pairs[i + 1]
            a1 = res_a.all_actions
            a2 = res_b.all_actions
            if a1.size == 0 or a2.size == 0:
                continue
            min_len = min(len(a1), len(a2))

            fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
            fig.suptitle(
                f"Language Sensitivity: '{res_a.prompt[:40]}' vs '{res_b.prompt[:40]}'",
                fontsize=9,
            )
            for j in range(8):
                ax = axes[j // 2, j % 2]
                ax.plot(a1[:min_len, j], linewidth=0.8, label=cid_a, alpha=0.8)
                ax.plot(a2[:min_len, j], linewidth=0.8, label=cid_b, alpha=0.8, linestyle="--")
                ax.set_ylabel(joint_names[j], fontsize=8)
                ax.grid(True, alpha=0.3)
                if j == 0:
                    ax.legend(fontsize=7)
            axes[-1, 0].set_xlabel("Timestep")
            axes[-1, 1].set_xlabel("Timestep")
            plt.tight_layout()
            fig.savefig(output_dir / f"lang_compare_{cid_a}_vs_{cid_b}.png", dpi=120)
            plt.close(fig)


def print_report_summary(report: dict):
    s = report["summary"]
    print("\n" + "=" * 70)
    print(f"  DREAMZERO-DROID EVALUATION REPORT")
    print("=" * 70)
    print(f"  Total cases: {s['total_cases']}  |  Passed: {s['successful']}  |  Failed: {s['failed']}")

    t = report.get("timing", {})
    if t:
        print(f"\n  Timing (per chunk):")
        print(f"    Mean: {t['mean_chunk_latency_s']:.3f}s  |  Std: {t['std_chunk_latency_s']:.3f}s")
        print(f"    Min:  {t['min_chunk_latency_s']:.3f}s  |  Max: {t['max_chunk_latency_s']:.3f}s")

    print(f"\n  Per-Case Results:")
    print(f"  {'ID':<12} {'Category':<22} {'Actions':<12} {'Range':<20} {'Latency':<10} {'Smooth':<8}")
    print(f"  {'-'*12} {'-'*22} {'-'*12} {'-'*20} {'-'*10} {'-'*8}")
    for cid, info in report["per_case"].items():
        if info.get("error"):
            print(f"  {cid:<12} {'ERROR':<22} {info['error'][:40]}")
            continue
        shape = info.get("shape", [0, 0])
        rng = f"[{info.get('min', 0):.3f}, {info.get('max', 0):.3f}]"
        lat = f"{info.get('mean_chunk_latency_s', 0):.3f}s"
        smooth = f"{info.get('smoothness_l2', 0):.4f}"
        print(f"  {cid:<12} {info.get('category', '?'):<22} {str(shape):<12} {rng:<20} {lat:<10} {smooth:<8}")

    lang = report.get("language_sensitivity", [])
    if lang:
        print(f"\n  Language Sensitivity Analysis:")
        print(f"  {'Pair':<22} {'CosSim':<10} {'L2 Dist':<12} {'Sensitive?':<12}")
        print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*12}")
        for entry in lang:
            pair_str = f"{entry['pair'][0]} vs {entry['pair'][1]}"
            sensitive = "YES" if entry["language_sensitive"] else "NO"
            print(f"  {pair_str:<22} {entry['cosine_similarity']:.4f}    {entry['l2_distance']:.4f}      {sensitive}")

    ranges = report.get("action_range_check", [])
    if ranges:
        all_ok = all(r["joints_in_range"] and r["gripper_in_range"] for r in ranges)
        print(f"\n  Action Range Checks: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
        for r in ranges:
            if not r["joints_in_range"] or not r["gripper_in_range"]:
                print(f"    WARN {r['case_id']}: joints=[{r['joint_min']:.3f},{r['joint_max']:.3f}] "
                      f"gripper=[{r['gripper_min']:.3f},{r['gripper_max']:.3f}]")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DreamZero-DROID Evaluation Suite")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--category", default=None,
                        help="Run specific category: seen_tasks, unseen_verb_tasks, language_sensitivity, robustness")
    parser.add_argument("--case-id", default=None, help="Run a single test case by ID")
    parser.add_argument("--num-chunks", type=int, default=15)
    parser.add_argument("--plot", action="store_true", help="Generate action trajectory plots")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_cases_path = Path(__file__).parent / "test_cases.json"
    with open(test_cases_path) as f:
        test_cases = json.load(f)

    # Collect cases to run
    cases_to_run = []
    # Discover all categories dynamically from test_cases.json
    categories = [k for k in test_cases.keys() if not k.startswith("_") and isinstance(test_cases[k], dict)]

    for cat in categories:
        if args.category and cat != args.category:
            continue
        cat_data = test_cases.get(cat, {})
        for case in cat_data.get("cases", []):
            if args.case_id and case["id"] != args.case_id:
                continue
            cases_to_run.append({
                **case,
                "category": cat,
            })

    if not cases_to_run:
        print("No test cases matched filters.")
        return

    logging.info(f"Loading camera frames from {VIDEO_DIR}...")
    camera_frames = load_camera_frames()

    logging.info(f"Connecting to server at {args.host}:{args.port}...")
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")

    all_results: dict[str, EvalResult] = {}

    logging.info(f"\nRunning {len(cases_to_run)} test cases...\n")

    for i, case in enumerate(cases_to_run):
        cid = case["id"]
        prompt = case["prompt"]
        cat = case["category"]
        logging.info(f"[{i+1}/{len(cases_to_run)}] {cid} ({cat}): \"{prompt[:60]}...\"")

        result = run_single_eval(
            client=client,
            camera_frames=camera_frames,
            case_id=cid,
            prompt=prompt,
            category=cat,
            num_chunks=args.num_chunks,
        )
        all_results[cid] = result

        stats = result.action_stats()
        if result.error:
            logging.info(f"  FAILED: {result.error}")
        else:
            logging.info(f"  OK: {stats.get('shape')} range=[{stats.get('min', 0):.3f},{stats.get('max', 0):.3f}] "
                        f"latency={stats.get('mean_chunk_latency_s', 0):.3f}s")

    # Analysis
    lang_analysis = analyze_language_sensitivity(all_results, test_cases)
    report = generate_report(all_results, lang_analysis, test_cases)

    # Output
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    report_path = out_dir / f"eval_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Report saved to {report_path}")

    # Save raw action arrays
    actions_dir = out_dir / f"actions_{timestamp}"
    actions_dir.mkdir(exist_ok=True)
    for cid, result in all_results.items():
        if result.error is None:
            np.save(actions_dir / f"{cid}.npy", result.all_actions)

    if args.plot:
        plot_dir = out_dir / f"plots_{timestamp}"
        logging.info(f"Generating plots to {plot_dir}...")
        plot_action_trajectories(all_results, plot_dir)

    print_report_summary(report)


if __name__ == "__main__":
    main()
