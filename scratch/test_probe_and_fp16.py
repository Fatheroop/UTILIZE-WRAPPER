"""
Testing 2-Layer Auxiliary Probes, FP16 Streaming, and Mixup on Fashion-MNIST
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

# Custom StageWrapper with 2-layer auxiliary probe
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

def test_breakthroughs():
    tr_x, te_x, tr_y, te_y = get_fashion_mnist(6000, 1500)
    print("=" * 85)
    print("🔬 TESTING 2-LAYER AUXILIARY PROBES (BELILOVSKY DGL) VS 1-LAYER LINEAR PROBE")
    print("=" * 85)

    # 1. 1-Layer Linear Probe Head (Current Baseline)
    print("\n▶ [1/3] Standard 1-Layer Linear Probe Head...")
    w1 = ConstructiveWrapper(
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
    w1.fit(tr_x, tr_y, verbose=False)
    t1 = time.perf_counter() - t0
    acc1 = w1.evaluate(te_x, te_y)
    f1_1 = float(f1_score(te_y, w1.predict(te_x), average="macro"))
    print(f"   ✓ 1-Layer Linear Probe: Acc = {acc1*100:.2f}% | F1 = {f1_1:.4f} | Time = {t1:.2f}s")

    # 2. 2-Layer Non-Linear Probe Head
    print("\n▶ [2/3] Expressive 2-Layer Non-Linear Auxiliary Probe Head (Belilovsky et al.)...")
    w2 = ConstructiveWrapper(
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
    # Monkey-patch _initialize_stage_wrappers to use CustomDeepHeadStageWrapper
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

    w2._initialize_stage_wrappers = make_custom_init(w2)

    t0 = time.perf_counter()
    w2.fit(tr_x, tr_y, verbose=False)
    t2 = time.perf_counter() - t0
    acc2 = w2.evaluate(te_x, te_y)
    f1_2 = float(f1_score(te_y, w2.predict(te_x), average="macro"))
    print(f"   ✓ 2-Layer Non-Linear Probe: Acc = {acc2*100:.2f}% | F1 = {f1_2:.4f} | Time = {t2:.2f}s")

    # 3. 2-Layer Probe + Fine-Tuning Calibration (3 epochs)
    print("\n▶ [3/3] 2-Layer Probe Head + 3-Epoch Calibrated Fine-Tuning...")
    w3 = ConstructiveWrapper(
        stages=build_stages(),
        in_shape=(1, 28, 28),
        num_classes=10,
        batch_size=128,
        learning_rate=0.003,
        max_epochs_per_stage=5,
        fine_tune_epochs=3,
        use_ema=True,
        random_state=42,
    )
    w3._initialize_stage_wrappers = make_custom_init(w3)
    t0 = time.perf_counter()
    w3.fit(tr_x, tr_y, verbose=False)
    t3 = time.perf_counter() - t0
    acc3 = w3.evaluate(te_x, te_y)
    f1_3 = float(f1_score(te_y, w3.predict(te_x), average="macro"))
    print(f"   ✓ 2-Layer Probe + 3-Epoch Fine-Tune: Acc = {acc3*100:.2f}% | F1 = {f1_3:.4f} | Time = {t3:.2f}s")

    print("\n" + "=" * 85)
    print(f"{'Probe Head Architecture':<35} | {'Acc (%)':<8} | {'F1':<7} | {'Time (s)':<8} | {'Delta vs Linear'}")
    print("-" * 85)
    print(f"{'1-Layer Linear Probe (Baseline)':<35} | {acc1*100:<8.2f} | {f1_1:<7.4f} | {t1:<8.2f} | Baseline")
    print(f"{'2-Layer Non-Linear Probe (DGL)':<35} | {acc2*100:<8.2f} | {f1_2:<7.4f} | {t2:<8.2f} | +{(acc2-acc1)*100:+.2f}% Acc")
    print(f"{'2-Layer Probe + 3-Epoch FT':<35} | {acc3*100:<8.2f} | {f1_3:<7.4f} | {t3:<8.2f} | +{(acc3-acc1)*100:+.2f}% Acc")
    print("=" * 85)

if __name__ == "__main__":
    test_breakthroughs()
