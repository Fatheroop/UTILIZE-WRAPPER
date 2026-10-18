"""
End-to-End Evaluation of Modern AI Breakthroughs on Fashion-MNIST
Comparing Baseline vs Individual Breakthroughs vs Full Synergistic Suite
"""

import os
import sys
import time
import gzip
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constructive_torch import ConstructiveWrapper

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

def build_stages():
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

def evaluate_config(name, tr_x, te_x, tr_y, te_y, **wrapper_kwargs):
    w = ConstructiveWrapper(
        stages=build_stages(),
        in_shape=(1, 28, 28),
        num_classes=10,
        batch_size=128,
        learning_rate=0.003,
        max_epochs_per_stage=5,
        fine_tune_epochs=2,
        use_ema=True,
        random_state=42,
        **wrapper_kwargs,
    )
    t0 = time.perf_counter()
    w.fit(tr_x, tr_y, verbose=False)
    t_train = time.perf_counter() - t0
    acc = w.evaluate(te_x, te_y)
    
    # Test export parity
    std_model = w.get_standard_model()
    with torch.no_grad():
        t_te_x = torch.from_numpy(te_x[:100]).float()
        diff = torch.max(torch.abs(std_model(t_te_x) - w(t_te_x))).item()
    
    return acc, t_train, diff

def main():
    print("=" * 90)
    print("🏆 FINAL BENCHMARK: MODERN AI BREAKTHROUGH SUITE ON FASHION-MNIST")
    print("=" * 90)
    tr_x, te_x, tr_y, te_y = get_fashion_mnist(6000, 1500)

    configs = [
        ("Baseline ConstructiveWrapper", {}),
        ("AirLLM FP16 Activation Streaming", {"activation_precision": "float16"}),
        ("2-Layer Expressive Auxiliary Probes (DGL)", {"probe_head_depth": 2}),
        ("Feature Covariance Decorrelation", {"feature_decorrelation": True, "feature_decorrelation_weight": 0.01}),
        ("Manifold / Input Mixup (alpha=0.2)", {"mixup_alpha": 0.2}),
        ("Gradient Centralization (GC)", {"gradient_centralization": True}),
        ("Full Breakthrough Suite (All Enabled)", {
            "activation_precision": "float16",
            "probe_head_depth": 2,
            "feature_decorrelation": True,
            "feature_decorrelation_weight": 0.01,
            "gradient_centralization": True,
            "selective_gradient": True,
            "loss_gated_sampling": True,
        }),
    ]

    results = []
    print(f"{'Configuration':<42} | {'Accuracy':<10} | {'Train Time':<12} | {'Export Parity Diff':<18}")
    print("-" * 90)
    for name, kwargs in configs:
        acc, t_train, diff = evaluate_config(name, tr_x, te_x, tr_y, te_y, **kwargs)
        results.append((name, acc, t_train, diff))
        print(f"{name:<42} | {acc*100:>7.2f}%  | {t_train:>9.2f}s  | {diff:.10f}")

    print("=" * 90)
    print("✅ BENCHMARK COMPLETE!")

if __name__ == "__main__":
    main()
