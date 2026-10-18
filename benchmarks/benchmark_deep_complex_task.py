"""
Deep Neural Network Benchmark on Complex Tasks
=============================================
Evaluates Deep Neural Architectures (7 to 8 progressive stages) comparing:
  - Network With Wrapper (ConstructiveWrapper)
  - Network Without Wrapper (Standard PyTorch End-to-End Baseline)

Workloads:
  1. Deep 7-Stage Convolutional Neural Network on Fashion-MNIST (10,000 train, 2,000 test, 10 classes)
     - 5 Conv + BatchNorm + Pooling Stages + 2 Dense Stages (14 total parameterized layers)
     - Tests deep spatial texture representation and vanishing gradient mitigation
  2. Deep 8-Stage Multi-Layer Perceptron on Complex 64-D Multi-Class Manifold (8,000 train, 2,000 test)
     - 8 Dense Stages with BatchNorm, GELU activations, and Dropout
     - Tests deep non-linear representation growth in high-dimensional feature spaces

Hardware Profiled:
  - Real-time OS process monitoring via `psutil` (User + System CPU-seconds, Peak RAM RSS)
  - Training throughput (samples/sec) and inference throughput (samples/sec)
  - Full deployment parity verification (.get_standard_model())
"""

import os
import gc
import time
from typing import Dict, Tuple, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_models.constructive_torch import ConstructiveWrapper
from utils.resource_monitor import ResourceTracker
from benchmarks.benchmark_four_datasets import (
    get_fashion_mnist_dataset,
    measure_latency_per_1k,
)


# =====================================================================
# ARCHITECTURE BUILDERS FOR DEEP NETWORKS
# =====================================================================

def build_deep_7stage_convnet() -> List[nn.Module]:
    """
    Build a Deep 7-Stage Convolutional Neural Network (14 parameterized layers).
    """
    s1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    )
    s2 = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )
    s3 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    s4 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )
    s5 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
    )
    s6 = nn.Sequential(
        nn.Linear(128 * 2 * 2, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
    )
    s7 = nn.Sequential(
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    return [s1, s2, s3, s4, s5, s6, s7]


def build_deep_8stage_mlp() -> List[nn.Module]:
    """
    Build a Deep 8-Stage Multi-Layer Perceptron (16 parameterized layers).
    """
    return [
        nn.Sequential(nn.Linear(64, 128), nn.BatchNorm1d(128), nn.GELU()),
        nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1)),
        nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU()),
        nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1)),
        nn.Sequential(nn.Linear(128, 96), nn.BatchNorm1d(96), nn.GELU()),
        nn.Sequential(nn.Linear(96, 96), nn.BatchNorm1d(96), nn.GELU(), nn.Dropout(0.1)),
        nn.Sequential(nn.Linear(96, 64), nn.BatchNorm1d(64), nn.GELU()),
        nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Linear(32, 10)),
    ]


def get_complex_manifold_dataset(
    n_samples: int = 10000,
    n_features: int = 64,
    n_classes: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate high-dimensional non-linear manifold classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=48,
        n_redundant=16,
        n_classes=n_classes,
        n_clusters_per_class=2,
        flip_y=0.03,
        class_sep=1.1,
        random_state=random_state,
    )
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    # Standardize features
    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0) + 1e-7
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std
    return x_tr.astype(np.float32), x_te.astype(np.float32), y_tr.astype(np.int64), y_te.astype(np.int64)


# =====================================================================
# BENCHMARK EXECUTION ENGINE
# =====================================================================

def run_deep_network_benchmark(
    workload_name: str,
    stages_factory,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    in_shape: Tuple[int, ...],
    num_classes: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.003,
    max_epochs_per_stage: int = 4,
    fine_tune_epochs: int = 3,
    fine_tune_lr: float = 0.001,
    std_epochs: int = 12,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute head-to-head deep neural network benchmark."""
    print("\n" + "=" * 85)
    print(f"🔥 EVALUATING DEEP NEURAL NETWORK ON COMPLEX TASK: {workload_name.upper()}")
    print(f"   Train samples: {len(X_train):,} | Test samples: {len(X_test):,} | Input Shape: {in_shape}")
    print("=" * 85)

    device = torch.device("cpu")
    n_train = len(X_train)

    # ------------------------------------------------------------------
    # 1. NETWORK WITH WRAPPER (ConstructiveWrapper)
    # ------------------------------------------------------------------
    print(f"\n>>> [1/2] Training DEEP NETWORK WITH WRAPPER on {workload_name}...")
    gc.collect()
    torch.manual_seed(42)
    np.random.seed(42)

    wrap_stages = stages_factory()
    n_stages = len(wrap_stages)
    print(f"   Architecture depth: {n_stages} progressive stages")

    wrapper = ConstructiveWrapper(
        stages=wrap_stages,
        in_shape=in_shape,
        num_classes=num_classes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs_per_stage=max_epochs_per_stage,
        fine_tune_epochs=fine_tune_epochs,
        fine_tune_lr=fine_tune_lr,
        auto_tune_hyperparams=False,
        use_ema=False,
        device=device,
        random_state=42,
    )

    with ResourceTracker(f"With Wrapper ({workload_name})", sample_interval_sec=0.01) as wrap_tracker:
        wrapper.fit(X_train, y_train, verbose=False)
        train_preds = wrapper.predict(X_train)
        test_preds = wrapper.predict(X_test)
        train_acc = float(accuracy_score(y_train, train_preds))
        test_acc = float(accuracy_score(y_test, test_preds))
        macro_f1 = float(f1_score(y_test, test_preds, average="macro"))

        lat_1k = measure_latency_per_1k(lambda x: wrapper.predict(x), X_test)
        total_effective_epochs = (n_stages * max_epochs_per_stage) + fine_tune_epochs
        wrap_tracker.log_results(
            train_acc=train_acc,
            test_acc=test_acc,
            macro_f1=macro_f1,
            inference_latency_ms=lat_1k,
            num_samples=n_train,
            num_epochs=total_effective_epochs,
        )

    # Verify standard model export parity
    std_exported = wrapper.get_standard_model()
    dummy_test = torch.from_numpy(X_test[:100]).float().to(device)
    with torch.inference_mode():
        exported_preds = torch.argmax(std_exported(dummy_test), dim=1).cpu().numpy()
    assert np.array_equal(exported_preds, test_preds[:100]), "Export parity check failed!"
    print("   ✓ Pure PyTorch export parity verified: 100.00% identical outputs!")

    wrap_summary = wrap_tracker.summary()
    print(f"   ✓ With Wrapper Finished: Acc={wrap_summary['test_accuracy_pct']:.2f}% | Macro F1={wrap_summary['macro_f1']:.4f} | Wall Time={wrap_summary['total_time_sec']:.2f}s | CPU Time={wrap_summary['total_cpu_time_sec']:.2f}s | Train Throughput={wrap_summary['train_throughput_samples_sec']:,.0f} s/s")

    # ------------------------------------------------------------------
    # 2. NETWORK WITHOUT WRAPPER (Standard PyTorch Baseline)
    # ------------------------------------------------------------------
    print(f"\n>>> [2/2] Training DEEP NETWORK WITHOUT WRAPPER (Standard Baseline) on {workload_name}...")
    gc.collect()
    torch.manual_seed(42)
    np.random.seed(42)

    std_stages = stages_factory()
    net_std = nn.Sequential(*std_stages).to(device)
    optimizer = torch.optim.Adam(net_std.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    tx = torch.from_numpy(X_train).float()
    ty = torch.from_numpy(y_train).long()
    train_loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True)

    with ResourceTracker(f"Without Wrapper ({workload_name})", sample_interval_sec=0.01) as std_tracker:
        for ep in range(std_epochs):
            net_std.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = net_std(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        net_std.eval()
        def std_predict(data: np.ndarray) -> np.ndarray:
            preds_list = []
            with torch.inference_mode():
                for i in range(0, len(data), 512):
                    end_i = min(i + 512, len(data))
                    b = torch.from_numpy(data[i:end_i]).float().to(device)
                    preds_list.append(torch.argmax(net_std(b), dim=1).cpu().numpy())
            return np.concatenate(preds_list)

        tr_preds_std = std_predict(X_train)
        te_preds_std = std_predict(X_test)
        std_train_acc = float(accuracy_score(y_train, tr_preds_std))
        std_test_acc = float(accuracy_score(y_test, te_preds_std))
        std_macro_f1 = float(f1_score(y_test, te_preds_std, average="macro"))

        std_lat_1k = measure_latency_per_1k(std_predict, X_test)
        std_tracker.log_results(
            train_acc=std_train_acc,
            test_acc=std_test_acc,
            macro_f1=std_macro_f1,
            inference_latency_ms=std_lat_1k,
            num_samples=n_train,
            num_epochs=std_epochs,
        )

    std_summary = std_tracker.summary()
    print(f"   ✓ Without Wrapper Finished: Acc={std_summary['test_accuracy_pct']:.2f}% | Macro F1={std_summary['macro_f1']:.4f} | Wall Time={std_summary['total_time_sec']:.2f}s | CPU Time={std_summary['total_cpu_time_sec']:.2f}s | Train Throughput={std_summary['train_throughput_samples_sec']:,.0f} s/s")

    return wrap_summary, std_summary


def generate_deep_benchmark_plot(
    results: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]],
    output_path: str = "plots/benchmark_deep_network_complex_task.png",
):
    """Generate high-resolution comparative charts for deep network benchmark."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Deep Neural Network Benchmark on Complex Tasks\nNetwork With Wrapper vs. Network Without Wrapper",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    workloads = list(results.keys())
    x = np.arange(len(workloads))
    width = 0.35

    color_wrap = "#107569"  # Emerald teal
    color_std = "#e05260"   # Coral crimson

    # Subplot 1: Test Accuracy (%)
    ax1 = axes[0, 0]
    wrap_acc = [results[w][0]["test_accuracy_pct"] for w in workloads]
    std_acc = [results[w][1]["test_accuracy_pct"] for w in workloads]
    b1 = ax1.bar(x - width/2, wrap_acc, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax1.bar(x + width/2, std_acc, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax1.set_title("Test Accuracy (%) [Higher is Better]", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads, fontweight="semibold")
    ax1.set_ylabel("Accuracy %")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")
    for b in b1 + b2:
        h = b.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot 2: Wall-Clock Training Duration (seconds)
    ax2 = axes[0, 1]
    wrap_time = [results[w][0]["total_time_sec"] for w in workloads]
    std_time = [results[w][1]["total_time_sec"] for w in workloads]
    b1 = ax2.bar(x - width/2, wrap_time, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax2.bar(x + width/2, std_time, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax2.set_title("Wall-Clock Training Time (Seconds) [Lower is Faster]", fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(workloads, fontweight="semibold")
    ax2.set_ylabel("Seconds")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left")
    for b in b1 + b2:
        h = b.get_height()
        ax2.annotate(f"{h:.2f}s", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot 3: Training Throughput (samples / second)
    ax3 = axes[1, 0]
    wrap_thru = [results[w][0]["train_throughput_samples_sec"] for w in workloads]
    std_thru = [results[w][1]["train_throughput_samples_sec"] for w in workloads]
    b1 = ax3.bar(x - width/2, wrap_thru, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax3.bar(x + width/2, std_thru, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax3.set_title("Training Throughput (Samples / Second) [Higher is Faster]", fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(workloads, fontweight="semibold")
    ax3.set_ylabel("Samples / Second")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right")
    for b in b1 + b2:
        h = b.get_height()
        ax3.annotate(f"{h:,.0f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 4: Total CPU Compute Time (Seconds = User + System)
    ax4 = axes[1, 1]
    wrap_cpu = [results[w][0]["total_cpu_time_sec"] for w in workloads]
    std_cpu = [results[w][1]["total_cpu_time_sec"] for w in workloads]
    b1 = ax4.bar(x - width/2, wrap_cpu, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax4.bar(x + width/2, std_cpu, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax4.set_title("Total CPU Compute Time (Seconds = User + System) [Lower is Better]", fontweight="bold", pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(workloads, fontweight="semibold")
    ax4.set_ylabel("CPU-Seconds")
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper left")
    for b in b1 + b2:
        h = b.get_height()
        ax4.annotate(f"{h:.1f}s", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Deep Network Benchmark visualization successfully saved to: {output_path}")


def main():
    results = {}

    # ------------------------------------------------------------------
    # WORKLOAD 1: Deep 7-Stage ConvNet on Fashion-MNIST (10,000 train, 2,000 test)
    # ------------------------------------------------------------------
    x_tr_fash, x_te_fash, y_tr_fash, y_te_fash = get_fashion_mnist_dataset(n_train=10000, n_test=2000)
    wrap_fash, std_fash = run_deep_network_benchmark(
        workload_name="Fashion-MNIST (7-Stage ConvNet)",
        stages_factory=build_deep_7stage_convnet,
        X_train=x_tr_fash,
        X_test=x_te_fash,
        y_train=y_tr_fash,
        y_test=y_te_fash,
        in_shape=(1, 28, 28),
        num_classes=10,
        batch_size=128,
        learning_rate=0.003,
        max_epochs_per_stage=4,
        fine_tune_epochs=3,
        fine_tune_lr=0.001,
        std_epochs=12,
    )
    results["Fashion-MNIST (7 Stages)"] = (wrap_fash, std_fash)

    # ------------------------------------------------------------------
    # WORKLOAD 2: Deep 8-Stage MLP on Complex 64-D Manifold (8,000 train, 2,000 test)
    # ------------------------------------------------------------------
    x_tr_man, x_te_man, y_tr_man, y_te_man = get_complex_manifold_dataset(
        n_samples=10000, n_features=64, n_classes=10, random_state=42
    )
    wrap_man, std_man = run_deep_network_benchmark(
        workload_name="64-D Manifold (8-Stage MLP)",
        stages_factory=build_deep_8stage_mlp,
        X_train=x_tr_man,
        X_test=x_te_man,
        y_train=y_tr_man,
        y_test=y_te_man,
        in_shape=(64,),
        num_classes=10,
        batch_size=64,
        learning_rate=0.005,
        max_epochs_per_stage=4,
        fine_tune_epochs=2,
        fine_tune_lr=0.001,
        std_epochs=16,
    )
    results["64-D Manifold (8 Stages)"] = (wrap_man, std_man)

    # ------------------------------------------------------------------
    # MASTER BENCHMARK SUMMARY TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("🏆 MASTER DEEP NEURAL NETWORK ON COMPLEX TASKS BENCHMARK SUMMARY")
    print("=" * 120)
    header = f"{'Workload':<28} | {'Approach':<18} | {'Test Acc':<9} | {'Macro F1':<9} | {'Wall Time':<10} | {'CPU-Sec':<9} | {'Train Thrpt':<13} | {'Acc / CPU-Sec':<13}"
    print(header)
    print("-" * 120)

    for w_name, (w_res, s_res) in results.items():
        print(f"{w_name:<28} | {'With Wrapper':<18} | {w_res['test_accuracy_pct']:>7.2f}% | {w_res['macro_f1']:>9.4f} | {w_res['total_time_sec']:>8.2f}s | {w_res['total_cpu_time_sec']:>7.2f}s | {w_res['train_throughput_samples_sec']:>10,.0f} s/s | {w_res['accuracy_per_cpu_sec']:>11.1f}")
        print(f"{'':<28} | {'Without Wrapper':<18} | {s_res['test_accuracy_pct']:>7.2f}% | {s_res['macro_f1']:>9.4f} | {s_res['total_time_sec']:>8.2f}s | {s_res['total_cpu_time_sec']:>7.2f}s | {s_res['train_throughput_samples_sec']:>10,.0f} s/s | {s_res['accuracy_per_cpu_sec']:>11.1f}")

        acc_diff = w_res['test_accuracy_pct'] - s_res['test_accuracy_pct']
        speedup = s_res['total_time_sec'] / max(0.01, w_res['total_time_sec'])
        train_gain = w_res['train_throughput_samples_sec'] / max(1.0, s_res['train_throughput_samples_sec'])
        sign = "+" if acc_diff >= 0 else ""
        print(f"{'':<28} | {'  -> Gain / Ratio':<18} | {sign}{acc_diff:>6.2f}% | {'':<9} | {speedup:>7.2f}x speed | {'':<9} | {train_gain:>9.2f}x thrpt |")
        print("-" * 120)

    print("=" * 120)

    generate_deep_benchmark_plot(results, "plots/benchmark_deep_network_complex_task.png")


if __name__ == "__main__":
    main()
