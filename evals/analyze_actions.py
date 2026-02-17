#!/usr/bin/env python3
"""Post-hoc analysis of saved eval results.

Loads .npy action files from a previous eval_suite.py run and produces
comparative analysis, including:
  - Cross-prompt action similarity matrix
  - Per-joint variance analysis
  - Action smoothness comparison
  - Trajectory divergence over time

Usage:
    python evals/analyze_actions.py --actions-dir evals/results/actions_YYYYMMDD_HHMMSS
    python evals/analyze_actions.py --actions-dir evals/results/actions_YYYYMMDD_HHMMSS --plot
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np


def load_actions(actions_dir: Path) -> dict[str, np.ndarray]:
    actions = {}
    for f in sorted(actions_dir.glob("*.npy")):
        actions[f.stem] = np.load(f)
    return actions


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.flatten(), b.flatten()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(np.dot(a_flat, b_flat) / denom) if denom > 1e-12 else 0.0


def compute_similarity_matrix(actions: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    ids = sorted(actions.keys())
    n = len(ids)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = actions[ids[i]], actions[ids[j]]
            min_len = min(len(a), len(b))
            matrix[i, j] = cosine_similarity(a[:min_len], b[:min_len])
    return ids, matrix


def compute_per_joint_variance(actions: dict[str, np.ndarray]) -> dict:
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "Gripper"]
    result = {}
    for cid, a in actions.items():
        result[cid] = {joint_names[j]: float(np.var(a[:, j])) for j in range(min(8, a.shape[1]))}
    return result


def compute_smoothness(actions: dict[str, np.ndarray]) -> dict:
    result = {}
    for cid, a in actions.items():
        if len(a) < 2:
            result[cid] = {"l2": 0.0, "max_jerk": 0.0}
            continue
        diffs = np.diff(a, axis=0)
        l2 = float(np.mean(np.linalg.norm(diffs, axis=1)))
        if len(a) > 2:
            accel = np.diff(diffs, axis=0)
            max_jerk = float(np.max(np.abs(accel)))
        else:
            max_jerk = 0.0
        result[cid] = {"l2_smoothness": l2, "max_jerk": max_jerk}
    return result


def compute_temporal_divergence(
    actions: dict[str, np.ndarray], reference_id: str
) -> dict[str, list[float]]:
    if reference_id not in actions:
        return {}
    ref = actions[reference_id]
    result = {}
    for cid, a in actions.items():
        if cid == reference_id:
            continue
        min_len = min(len(ref), len(a))
        divergence = np.linalg.norm(ref[:min_len] - a[:min_len], axis=1).tolist()
        result[cid] = divergence
    return result


def print_similarity_matrix(ids: list[str], matrix: np.ndarray):
    print("\nAction Cosine Similarity Matrix:")
    header = f"{'':>12}" + "".join(f"{cid:>12}" for cid in ids)
    print(header)
    for i, cid in enumerate(ids):
        row = f"{cid:>12}" + "".join(f"{matrix[i, j]:>12.4f}" for j in range(len(ids)))
        print(row)


def plot_analysis(actions: dict, actions_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available, skipping plots")
        return

    plot_dir = actions_dir.parent / (actions_dir.name.replace("actions_", "analysis_plots_"))
    plot_dir.mkdir(exist_ok=True)

    # Similarity heatmap
    ids, matrix = compute_similarity_matrix(actions)
    if len(ids) > 1:
        fig, ax = plt.subplots(figsize=(max(8, len(ids)), max(6, len(ids) * 0.7)))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ids, fontsize=7)
        plt.colorbar(im, ax=ax, label="Cosine Similarity")
        ax.set_title("Cross-Prompt Action Similarity")
        plt.tight_layout()
        fig.savefig(plot_dir / "similarity_matrix.png", dpi=120)
        plt.close(fig)
        logging.info(f"Saved similarity_matrix.png")

    # Per-joint variance comparison
    variance = compute_per_joint_variance(actions)
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "Grip"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(8)
    width = 0.8 / max(1, len(variance))
    for i, (cid, jv) in enumerate(sorted(variance.items())):
        vals = [jv.get(jn, 0) for jn in ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "Gripper"]]
        ax.bar(x + i * width, vals, width, label=cid, alpha=0.7)
    ax.set_xticks(x + width * len(variance) / 2)
    ax.set_xticklabels(joint_names)
    ax.set_ylabel("Variance")
    ax.set_title("Per-Joint Action Variance by Test Case")
    ax.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    fig.savefig(plot_dir / "per_joint_variance.png", dpi=120)
    plt.close(fig)
    logging.info(f"Saved per_joint_variance.png")

    # Temporal divergence from first case
    ref_id = sorted(actions.keys())[0]
    divergence = compute_temporal_divergence(actions, ref_id)
    if divergence:
        fig, ax = plt.subplots(figsize=(12, 5))
        for cid, div in sorted(divergence.items()):
            ax.plot(div, label=cid, alpha=0.7, linewidth=0.8)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(f"L2 Distance from {ref_id}")
        ax.set_title(f"Temporal Divergence (reference: {ref_id})")
        ax.legend(fontsize=6, ncol=3)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(plot_dir / "temporal_divergence.png", dpi=120)
        plt.close(fig)
        logging.info(f"Saved temporal_divergence.png")

    print(f"\nPlots saved to {plot_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze DreamZero eval action outputs")
    parser.add_argument("--actions-dir", required=True, help="Directory with .npy action files")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--reference", default=None, help="Reference case ID for divergence analysis")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    actions_dir = Path(args.actions_dir)
    actions = load_actions(actions_dir)
    logging.info(f"Loaded {len(actions)} action trajectories")

    if not actions:
        print("No .npy files found.")
        return

    # Summary stats
    print("\nAction Trajectory Summary:")
    print(f"{'ID':<14} {'Shape':<14} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10} {'Smooth':<10}")
    print("-" * 78)
    smoothness = compute_smoothness(actions)
    for cid in sorted(actions.keys()):
        a = actions[cid]
        sm = smoothness[cid]["l2_smoothness"]
        print(f"{cid:<14} {str(a.shape):<14} {a.min():<10.4f} {a.max():<10.4f} "
              f"{a.mean():<10.4f} {a.std():<10.4f} {sm:<10.4f}")

    # Similarity matrix
    ids, matrix = compute_similarity_matrix(actions)
    print_similarity_matrix(ids, matrix)

    # Smoothness
    print("\nSmoothness Analysis:")
    print(f"{'ID':<14} {'L2 Smoothness':<16} {'Max Jerk':<12}")
    print("-" * 42)
    for cid in sorted(smoothness.keys()):
        s = smoothness[cid]
        print(f"{cid:<14} {s['l2_smoothness']:<16.6f} {s.get('max_jerk', 0):<12.6f}")

    if args.plot:
        plot_analysis(actions, actions_dir)

    # Save analysis JSON
    analysis = {
        "similarity_matrix": {"ids": ids, "matrix": matrix.tolist()},
        "smoothness": smoothness,
        "per_joint_variance": compute_per_joint_variance(actions),
    }
    analysis_path = actions_dir.parent / f"analysis_{actions_dir.name.replace('actions_', '')}.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
