"""
Test ConstructiveWrapper with Selective Gradient and Loss-Gated Optimization
=============================================================================
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constructive_torch import ConstructiveWrapper
from benchmark_four_datasets import get_fashion_mnist_dataset

def build_conv_stages():
    s1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
    s2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
    s3 = nn.Sequential(nn.Linear(64 * 4 * 4, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 10))
    return [s1, s2, s3]

def main():
    x_tr, x_te, y_tr, y_te = get_fashion_mnist_dataset(n_train=6000, n_test=1500)

    configs = [
        {"name": "Standard Wrapper (No Selective Gradient)", "selective_grad": False, "ratio": 1.0},
        {"name": "Wrapper + Selective Gradient (Ratio=0.85)", "selective_grad": True, "ratio": 0.85},
        {"name": "Wrapper + Selective Gradient (Ratio=0.75)", "selective_grad": True, "ratio": 0.75},
        {"name": "Wrapper + Selective Gradient (Ratio=0.65)", "selective_grad": True, "ratio": 0.65},
    ]

    print("=" * 85)
    print("🚀 BENCHMARKING SELECTIVE GRADIENT IN CONSTRUCTIVE WRAPPER")
    print("=" * 85)

    for cfg in configs:
        torch.manual_seed(42)
        np.random.seed(42)
        stages = build_conv_stages()
        wrapper = ConstructiveWrapper(
            stages=stages,
            in_shape=(1, 28, 28),
            num_classes=10,
            batch_size=128,
            learning_rate=0.003,
            max_epochs_per_stage=4,
            fine_tune_epochs=2,
            fine_tune_lr=0.001,
            auto_tune_hyperparams=False,
            use_ema=False,
            selective_gradient=cfg["selective_grad"],
            selective_gradient_ratio=cfg["ratio"],
        )

        t0 = time.time()
        wrapper.fit(x_tr, y_tr, verbose=False)
        dur = time.time() - t0

        preds = wrapper.predict(x_te)
        acc = accuracy_score(y_te, preds) * 100
        f1 = f1_score(y_te, preds, average="macro")
        print(f"   [{cfg['name']:<48}] Test Acc: {acc:.2f}% | Macro F1: {f1:.4f} | Time: {dur:.2f}s")

if __name__ == "__main__":
    main()
