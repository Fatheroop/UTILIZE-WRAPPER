"""
Testing Combined Synergistic Breakthroughs on Fashion-MNIST
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

class CustomDeepHeadStageWrapper(StageWrapper):
    def __init__(self, body, in_shape, out_dim=10, task="classification", is_final_stage=False, probe_depth=2):
        super().__init__(body, in_shape, out_dim, task=task, is_final_stage=is_final_stage)
        if not self.is_head_identity and probe_depth == 2:
            if len(self.out_shape) > 1:
                pool_h = min(4, self.out_shape[1])
                pool_w = min(4, self.out_shape[2])
                flat_dim = self.out_shape[0] * pool_h * pool_w
                hidden = max(64, flat_dim // 2)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((pool_h, pool_w)),
                    nn.Flatten(),
                    nn.Linear(flat_dim, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, out_dim)
                )

def make_custom_init(wrapper):
    def _init(sample_input):
        wrapper.stages_wrapped = nn.ModuleList()
        curr_shape = tuple(sample_input.shape[1:])
        wrapper.in_shape = curr_shape
        num_stages = len(wrapper.stages_raw)
        for s_idx, raw_stage in enumerate(wrapper.stages_raw):
            is_final = (s_idx == num_stages - 1)
            stage_w = CustomDeepHeadStageWrapper(
                body=raw_stage,
                in_shape=curr_shape,
                out_dim=wrapper.out_dim,
                task=wrapper.task,
                is_final_stage=is_final,
                probe_depth=2,
            ).to(wrapper.device)
            wrapper.stages_wrapped.append(stage_w)
            curr_shape = stage_w.out_shape
    return _init

def run():
    tr_x, te_x, tr_y, te_y = get_fashion_mnist(6000, 1500)
    print("=" * 85)
    print("🚀 COMBINED SYNERGISTIC BREAKTHROUGHS ON FASHION-MNIST")
    print("=" * 85)

    # 1. Baseline
    w_base = ConstructiveWrapper(
        stages=build_stages(),
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
    w_base.fit(tr_x, tr_y, verbose=False)
    t_base = time.perf_counter() - t0
    acc_base = w_base.evaluate(te_x, te_y)
    f1_base = float(f1_score(te_y, w_base.predict(te_x), average="macro"))
    print(f"1. Standard Baseline:       Acc = {acc_base*100:.2f}% | F1 = {f1_base:.4f} | Time = {t_base:.2f}s")

    # 2. Synergistic Suite (2-Layer Probe + Loss-Gating 0.95 + Selective Gradient 0.85 + Curvature Stop)
    w_syn = ConstructiveWrapper(
        stages=build_stages(),
        in_shape=(1, 28, 28),
        num_classes=10,
        batch_size=128,
        learning_rate=0.003,
        max_epochs_per_stage=5,
        fine_tune_epochs=3,
        use_ema=True,
        selective_gradient=True,
        selective_gradient_ratio=0.85,
        loss_gated_sampling=True,
        loss_gated_threshold=0.95,
        early_stopping_curvature=True,
        early_stopping_curvature_delta=0.002,
        random_state=42,
    )
    w_syn._initialize_stage_wrappers = make_custom_init(w_syn)
    t0 = time.perf_counter()
    w_syn.fit(tr_x, tr_y, verbose=False)
    t_syn = time.perf_counter() - t0
    acc_syn = w_syn.evaluate(te_x, te_y)
    f1_syn = float(f1_score(te_y, w_syn.predict(te_x), average="macro"))
    print(f"2. Synergistic Breakthrough: Acc = {acc_syn*100:.2f}% | F1 = {f1_syn:.4f} | Time = {t_syn:.2f}s")
    print(f"   Delta vs Baseline:        +{(acc_syn - acc_base)*100:+.2f}% Accuracy, {t_base/t_syn:.2f}x speedup")

if __name__ == "__main__":
    run()
