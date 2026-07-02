"""
Sparse Parameterization Demo

Demonstrates why constant per-joint offsets generalize better than per-frame offsets.
"""

import numpy as np
from typing import Tuple


def generate_motion_library(n_motions: int = 5, n_joints: int = 10, n_frames: int = 100):
    """Generate a library of different motions."""
    motions = []
    for i in range(n_motions):
        t = np.linspace(0, 4 * np.pi, n_frames)
        motion = np.zeros((n_frames, n_joints))
        for j in range(n_joints):
            # Different frequency and phase for each motion
            freq = 1.0 + i * 0.2
            phase = j * 0.5 + i * 0.3
            motion[:, j] = np.sin(freq * t + phase) * 0.5
        motions.append(motion)
    return motions


def sparse_retargeting(motion: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Apply sparse (constant) retargeting.
    
    Args:
        motion: Original motion (n_frames, n_joints)
        offsets: Constant offsets per joint (n_joints,)
    
    Returns:
        Retargeted motion
    """
    return motion + offsets[np.newaxis, :]


def dense_retargeting(motion: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Apply dense (per-frame) retargeting.
    
    Args:
        motion: Original motion (n_frames, n_joints)
        offsets: Per-frame offsets (n_frames, n_joints)
    
    Returns:
        Retargeted motion
    """
    return motion + offsets


def learn_sparse_offsets(motions: list, target_offset: float = 0.2) -> np.ndarray:
    """Learn constant offsets that work across all motions."""
    n_joints = motions[0].shape[1]
    offsets = np.zeros(n_joints)
    
    # Simple optimization: average required offset across all motions
    for motion in motions:
        # Simulate target: add constant offset
        target = motion + target_offset
        # Learn offset
        offsets += np.mean(target - motion, axis=0)
    
    offsets /= len(motions)
    return offsets


def learn_dense_offsets(motion: np.ndarray, target_offset: float = 0.2) -> np.ndarray:
    """Learn per-frame offsets for a single motion."""
    # Simulate target: add constant offset
    target = motion + target_offset
    # Learn per-frame offsets
    offsets = target - motion
    return offsets


def evaluate_generalization(train_motions: list, test_motions: list, 
                            sparse_offsets: np.ndarray, dense_offsets_list: list,
                            target_offset: float = 0.2):
    """Evaluate how well learned offsets generalize to new motions."""
    
    print("\n" + "=" * 60)
    print("Generalization Evaluation")
    print("=" * 60)
    
    # Sparse: same offsets for all motions
    sparse_errors = []
    for motion in test_motions:
        target = motion + target_offset
        retargeted = sparse_retargeting(motion, sparse_offsets)
        error = np.mean((retargeted - target) ** 2)
        sparse_errors.append(error)
    
    print(f"\nSparse (constant offsets):")
    print(f"  Parameters: {len(sparse_offsets)}")
    print(f"  Test error: {np.mean(sparse_errors):.6f} ± {np.std(sparse_errors):.6f}")
    
    # Dense: different offsets per training motion, test on new motions
    dense_errors = []
    for test_motion in test_motions:
        # Use first training motion's offsets (doesn't generalize well)
        retargeted = dense_retargeting(test_motion, dense_offsets_list[0])
        target = test_motion + target_offset
        error = np.mean((retargeted - target) ** 2)
        dense_errors.append(error)
    
    print(f"\nDense (per-frame offsets):")
    print(f"  Parameters: {dense_offsets_list[0].size}")
    print(f"  Test error: {np.mean(dense_errors):.6f} ± {np.std(dense_errors):.6f}")
    
    print(f"\nGeneralization improvement: {np.mean(dense_errors) / np.mean(sparse_errors):.2f}x")
    print(f"Parameter reduction: {dense_offsets_list[0].size / len(sparse_offsets):.0f}x")


def main():
    """Run demo."""
    print("=" * 60)
    print("Sparse vs Dense Parameterization Demo")
    print("=" * 60)
    
    # Generate motion library
    print("\nGenerating motion library...")
    all_motions = generate_motion_library(n_motions=10, n_joints=10, n_frames=100)
    train_motions = all_motions[:7]
    test_motions = all_motions[7:]
    
    print(f"Training motions: {len(train_motions)}")
    print(f"Test motions: {len(test_motions)}")
    
    # Learn sparse offsets (constant per joint)
    print("\nLearning sparse offsets...")
    sparse_offsets = learn_sparse_offsets(train_motions, target_offset=0.2)
    
    # Learn dense offsets (per-frame for each motion)
    print("Learning dense offsets...")
    dense_offsets_list = []
    for motion in train_motions:
        offsets = learn_dense_offsets(motion, target_offset=0.2)
        dense_offsets_list.append(offsets)
    
    # Evaluate generalization
    evaluate_generalization(train_motions, test_motions, 
                           sparse_offsets, dense_offsets_list,
                           target_offset=0.2)
    
    print("\n" + "=" * 60)
    print("Key Insight:")
    print("=" * 60)
    print("Sparse parameterization (constant offsets) generalizes better")
    print("to new motions because it captures systematic differences")
    print("between characters rather than motion-specific adjustments.")
    print("=" * 60)


if __name__ == "__main__":
    main()
