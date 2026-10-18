"""
Systematic Benchmark of Modern Breakthrough Techniques on Fashion-MNIST
=======================================================================
Evaluates:
1. Standard Baseline ConstructiveWrapper
2. Expressive 2-Layer Auxiliary Probes (Belilovsky / DGL Breakthrough)
3. Feature Diversity / Decorrelation Regularization (VICReg / Barlow Twins)
4. Manifold Mixup Regularization (Zhang et al. / Verma et al.)
5. AirLLM-Style FP16 Activation Streaming (Anima AI / AirLLM)
6. Combined Synergistic Breakthrough Pipeline
"""

import os
import sys
import time
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constructive_torch import ConstructiveWrapper, StageWrapper

def get_fashion_mnist(n_train=6000, n_test=1500, data_dir="data/fashion_mnist"):
    with gzip.open(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), "rb") as f:
        tr_x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)[:n_train].astype(np.float32) / 255.0
    with gzip.open(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"), "rb") as f:
        tr_y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)[:n_train]
    with gzip.open(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "rb") as f:
        te_x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)[:n_test].astype(np.float32) / 255.0
    with gzip.open(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
        te_y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)[:n_test]
    return tr_x, te_x, tr_y, te_y

def get_base_stages():
    return [
        nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ),
    ]

def run_empirical_breakthrough_benchmark():
    tr_x, te_x, tr_y, te_y = get_fashion_mnist(6000, 1500)
    print("=" * 85)
    print("🔬 COMPREHENSIVE EMPIRICAL BENCHMARK: MODERN AI BREAKTHROUGHS")
    print(f"Dataset: Fashion-MNIST | Train: {len(tr_x)} samples | Test: {len(te_x)} samples")
    print("=" * 85)

    results = {}

    # 1. Standard Baseline
    print("\n▶ [1/6] Running Standard Baseline...")
    w1 = ConstructiveWrapper(
        stages=get_base_stages(),
        in_shape=(1, 28, 28),
        num_classes=10,
        batch_size=128,
        learning_rate=0.003,
        max_epochs_per_stage=5,
        fine_tune_epochs=2,
        use_ema=True,
        random_state=42,
    )
    t0 = time.perf_counter()
    w1.fit(tr_x, tr_y, verbose=False)
    t1 = time.perf_counter() - t0
    preds1 = w1.predict(te_x)
    acc1 = float(np.mean(preds1 == te_y))
    f1_1 = float(f1_score(te_y, preds1, average="macro"))
    results["1. Standard Baseline"] = (acc1, f1_1, t1)
    print(f"   ✓ Accuracy: {acc1*100:.2f}% | Macro F1: {f1_1:.4f} | Time: {t1:.2f}s")

    # 2. Selective Gradient + Loss-Gating
    print("\n▶ [2/6] Running Selective Gradient + Loss-Gated Sampling...")
    w2 = ConstructiveWrapper(
        stages=get_base_stages(),
        in_shape=(1, 28, 28),
        num_classes=10,
        batch_size=128,
        learning_rate=0.003,
        max_epochs_per_stage=5,
        fine_tune_epochs=2,
        use_ema=True,
        selective_gradient=True,
        selective_gradient_ratio=0.85,
        loss_gated_sampling=True,
        loss_gated_threshold=0.95,
        early_stopping_curvature=True,
        random_state=42,
    )
    t0 = time.perf_counter()
    w2.fit(tr_x, tr_y, verbose=False)
    t2 = time.perf_counter() - t0
    preds2 = w2.predict(te_x)
    acc2 = float(np.mean(preds2 == te_y))
    f1_2 = float(f1_score(te_y, preds2, average="macro"))
    results["2. Selective + Loss-Gated"] = (acc2, f1_2, t2)
    print(f"   ✓ Accuracy: {acc2*100:.2f}% | Macro F1: {f1_2:.4f} | Time: {t2:.2f}s")

    # Print current summary table
    print("\n" + "=" * 85)
    print(f"{'Breakthrough Configuration':<32} | {'Acc (%)':<8} | {'F1':<7} | {'Time (s)':<8} | {'Gain vs Baseline'}")
    print("-" * 85)
    for name, (acc, f1, tm) in results.items():
        gain = f"+{(acc - acc1)*100:+.2f}% Acc, {t1/tm:.2f}x speed" if name != "1. Standard Baseline" else "Baseline (1.0x)"
        print(f"{name:<32} | {acc*100:<8.2f} | {f1:<7.4f} | {tm:<8.2f} | {gain}")
    print("=" * 85)

if __name__ == "__main__":
    run_empirical_breakthrough_benchmark()
