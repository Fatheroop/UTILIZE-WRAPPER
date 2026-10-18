"""
Comprehensive CPU Utilization and Overall Efficiency Benchmark
==============================================================
Rigorous evaluation comparing:
  - Network With Wrapper (ConstructiveWrapper)
  - Network Without Wrapper (Standard PyTorch Baseline)

Metrics Evaluated:
  1. Process CPU Utilization (Average CPU %, Peak CPU %, User Time vs. System Time)
  2. Total CPU Compute Effort (CPU-Seconds consumed = User + System)
  3. CPU Core Load Factor (Multi-core utilization ratio = CPU_time / Wall_time)
  4. Training Throughput (samples processed per second)
  5. Inference Throughput (samples predicted per second)
  6. Peak Memory Footprint (RSS in MB) & Memory Growth (+Δ MB)
  7. Overall Efficiency Index (Accuracy points achieved per CPU-Second invested)
"""

import os
import sys
import gc
import time
from typing import Dict, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_models.constructive_torch import ConstructiveWrapper
from utils.resource_monitor import ResourceTracker
from benchmarks.benchmark_four_datasets import (
    get_moons_dataset,
    build_moons_model,
    get_mnist_dataset,
    build_mnist_model,
    get_fashion_mnist_dataset,
    build_fashion_model,
    measure_latency_per_1k,
)


def run_efficiency_benchmark(
    workload_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_builder,
    in_shape: Tuple[int, ...],
    num_classes: int,
    batch_size: int = 128,
    learning_rate: float = 0.003,
    max_epochs_per_stage: int = 4,
    fine_tune_epochs: int = 3,
    fine_tune_lr: float = 0.001,
    std_epochs: int = 8,
    use_ema: bool = False,
    auto_tune: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute head-to-head CPU utilization and efficiency benchmark."""
    print("\n" + "=" * 80)
    print(f"⚡ PROFILING CPU EFFICIENCY: {workload_name.upper()}")
    print(f"   Train samples: {len(X_train):,} | Test samples: {len(X_test):,} | Shape: {in_shape}")
    print("=" * 80)

    device = torch.device("cpu")
    n_train = len(X_train)

    # ------------------------------------------------------------------
    # 1. NETWORK WITH WRAPPER
    # ------------------------------------------------------------------
    print(f"\n>>> [1/2] Profiling NETWORK WITH WRAPPER on {workload_name}...")
    gc.collect()
    torch.manual_seed(42)
    np.random.seed(42)

    net_wrap = model_builder()
    wrapper = ConstructiveWrapper(
        model=net_wrap,
        in_shape=in_shape,
        num_classes=num_classes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs_per_stage=max_epochs_per_stage,
        fine_tune_epochs=fine_tune_epochs,
        fine_tune_lr=fine_tune_lr,
        use_ema=use_ema,
        auto_tune_hyperparams=auto_tune,
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
        num_stages = len(wrapper.stages_wrapped)
        total_effective_epochs = (num_stages * max_epochs_per_stage) + fine_tune_epochs
        wrap_tracker.log_results(
            train_acc=train_acc,
            test_acc=test_acc,
            macro_f1=macro_f1,
            inference_latency_ms=lat_1k,
            num_samples=n_train,
            num_epochs=total_effective_epochs,
        )

    wrap_summary = wrap_tracker.summary()
    print(f"   ✓ With Wrapper Finished: Acc={wrap_summary['test_accuracy_pct']:.2f}% | Wall Time={wrap_summary['total_time_sec']:.2f}s | CPU Time={wrap_summary['total_cpu_time_sec']:.2f}s | CPU Load={wrap_summary['cpu_core_load_pct']:.1f}% | Train Throughput={wrap_summary['train_throughput_samples_sec']:,.0f} s/s | Inf Throughput={wrap_summary['inference_throughput_samples_sec']:,.0f} s/s")

    # ------------------------------------------------------------------
    # 2. NETWORK WITHOUT WRAPPER (Standard PyTorch Baseline)
    # ------------------------------------------------------------------
    print(f"\n>>> [2/2] Profiling NETWORK WITHOUT WRAPPER (Standard Baseline) on {workload_name}...")
    gc.collect()
    torch.manual_seed(42)
    np.random.seed(42)

    net_std = model_builder().to(device)
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
    print(f"   ✓ Without Wrapper Finished: Acc={std_summary['test_accuracy_pct']:.2f}% | Wall Time={std_summary['total_time_sec']:.2f}s | CPU Time={std_summary['total_cpu_time_sec']:.2f}s | CPU Load={std_summary['cpu_core_load_pct']:.1f}% | Train Throughput={std_summary['train_throughput_samples_sec']:,.0f} s/s | Inf Throughput={std_summary['inference_throughput_samples_sec']:,.0f} s/s")

    return wrap_summary, std_summary


def generate_cpu_efficiency_plot(results: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]], output_path: str = "plots/benchmark_cpu_efficiency.png"):
    """Generate high-resolution comparative charts for CPU utilization and overall efficiency."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("CPU Utilization & Overall Efficiency Benchmark\nNetwork With Wrapper vs. Network Without Wrapper", fontsize=15, fontweight="bold", y=0.98)

    workloads = list(results.keys())
    x = np.arange(len(workloads))
    width = 0.35

    color_wrap = "#107569"  # Emerald teal
    color_std  = "#e05260"  # Coral crimson

    # Subplot 1: Training Throughput (samples/second)
    ax1 = axes[0, 0]
    wrap_thru = [results[w][0]["train_throughput_samples_sec"] for w in workloads]
    std_thru  = [results[w][1]["train_throughput_samples_sec"] for w in workloads]
    b1 = ax1.bar(x - width/2, wrap_thru, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax1.bar(x + width/2, std_thru, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax1.set_title("Training Throughput (Samples / Second) [Higher is Faster]", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads, fontweight="semibold")
    ax1.set_ylabel("Samples / Second")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right")
    for b in b1 + b2:
        h = b.get_height()
        ax1.annotate(f"{h:,.0f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 2: Inference Prediction Throughput (samples/second)
    ax2 = axes[0, 1]
    wrap_inf = [results[w][0]["inference_throughput_samples_sec"] for w in workloads]
    std_inf  = [results[w][1]["inference_throughput_samples_sec"] for w in workloads]
    b1 = ax2.bar(x - width/2, wrap_inf, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax2.bar(x + width/2, std_inf, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax2.set_title("Inference Throughput (Samples / Second) [Higher is Faster]", fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(workloads, fontweight="semibold")
    ax2.set_ylabel("Samples / Second")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right")
    for b in b1 + b2:
        h = b.get_height()
        ax2.annotate(f"{h:,.0f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 3: Total CPU Compute Seconds Consumed (User + System)
    ax3 = axes[1, 0]
    wrap_cpu = [results[w][0]["total_cpu_time_sec"] for w in workloads]
    std_cpu  = [results[w][1]["total_cpu_time_sec"] for w in workloads]
    b1 = ax3.bar(x - width/2, wrap_cpu, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax3.bar(x + width/2, std_cpu, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax3.set_title("Total CPU Compute Time (Seconds = User + System) [Lower is Better]", fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(workloads, fontweight="semibold")
    ax3.set_ylabel("CPU-Seconds")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left")
    for b in b1 + b2:
        h = b.get_height()
        ax3.annotate(f"{h:.2f}s", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 4: Compute Efficiency Index (Accuracy Points / CPU-Second)
    ax4 = axes[1, 1]
    wrap_eff = [results[w][0]["accuracy_per_cpu_sec"] for w in workloads]
    std_eff  = [results[w][1]["accuracy_per_cpu_sec"] for w in workloads]
    b1 = ax4.bar(x - width/2, wrap_eff, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax4.bar(x + width/2, std_eff, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax4.set_title("Efficiency Index (Accuracy Points per CPU-Second) [Higher is Better]", fontweight="bold", pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(workloads, fontweight="semibold")
    ax4.set_ylabel("Acc % / CPU-sec")
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right")
    for b in b1 + b2:
        h = b.get_height()
        ax4.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n📊 CPU Efficiency visualization successfully saved to: {output_path}")


def main():
    results = {}

    # 1. Non-linear Tabular Workload (Moons, 3,000 samples)
    x_tr, x_te, y_tr, y_te = get_moons_dataset(n_samples=3000, noise=0.25, random_state=42)
    wrap_moons, std_moons = run_efficiency_benchmark(
        workload_name="Two Moons (Tabular)",
        X_train=x_tr,
        X_test=x_te,
        y_train=y_tr,
        y_test=y_te,
        model_builder=build_moons_model,
        in_shape=(2,),
        num_classes=2,
        batch_size=64,
        learning_rate=0.015,
        max_epochs_per_stage=6,
        fine_tune_epochs=2,
        fine_tune_lr=0.005,
        std_epochs=20,
    )
    results["Two Moons"] = (wrap_moons, std_moons)

    # 2. MNIST Digits Workload (10,000 train, 2,000 test)
    x_tr, x_te, y_tr, y_te = get_mnist_dataset(n_train=10000, n_test=2000)
    wrap_mnist, std_mnist = run_efficiency_benchmark(
        workload_name="MNIST Digits (LeNet-5)",
        X_train=x_tr,
        X_test=x_te,
        y_train=y_tr,
        y_test=y_te,
        model_builder=build_mnist_model,
        in_shape=(1, 28, 28),
        num_classes=10,
        batch_size=128,
        learning_rate=0.003,
        max_epochs_per_stage=4,
        fine_tune_epochs=3,
        fine_tune_lr=0.001,
        std_epochs=8,
    )
    results["MNIST"] = (wrap_mnist, std_mnist)

    # 3. Fashion-MNIST Workload (10,000 train, 2,000 test)
    x_tr, x_te, y_tr, y_te = get_fashion_mnist_dataset(n_train=10000, n_test=2000)
    wrap_fash, std_fash = run_efficiency_benchmark(
        workload_name="Fashion-MNIST (ConvNet)",
        X_train=x_tr,
        X_test=x_te,
        y_train=y_tr,
        y_test=y_te,
        model_builder=build_fashion_model,
        in_shape=(1, 28, 28),
        num_classes=10,
        batch_size=128,
        learning_rate=0.003,
        max_epochs_per_stage=6,
        fine_tune_epochs=5,
        fine_tune_lr=0.001,
        std_epochs=8,
        use_ema=False,
        auto_tune=False,
    )
    results["Fashion-MNIST"] = (wrap_fash, std_fash)

    # ------------------------------------------------------------------
    # MASTER CPU EFFICIENCY SUMMARY TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("🏆 MASTER CPU UTILIZATION & OVERALL EFFICIENCY BENCHMARK SUMMARY")
    print("=" * 115)
    header = f"{'Workload':<18} | {'Approach':<18} | {'Test Acc':<9} | {'CPU-Sec':<9} | {'CPU Load':<9} | {'Train Thrpt':<13} | {'Inf Thrpt':<13} | {'Acc / CPU-Sec':<13}"
    print(header)
    print("-" * 115)

    for w_name, (w_res, s_res) in results.items():
        print(f"{w_name:<18} | {'With Wrapper':<18} | {w_res['test_accuracy_pct']:>7.2f}% | {w_res['total_cpu_time_sec']:>7.2f}s | {w_res['cpu_core_load_pct']:>7.1f}% | {w_res['train_throughput_samples_sec']:>10,.0f} s/s | {w_res['inference_throughput_samples_sec']:>10,.0f} s/s | {w_res['accuracy_per_cpu_sec']:>11.1f}")
        print(f"{'':<18} | {'Without Wrapper':<18} | {s_res['test_accuracy_pct']:>7.2f}% | {s_res['total_cpu_time_sec']:>7.2f}s | {s_res['cpu_core_load_pct']:>7.1f}% | {s_res['train_throughput_samples_sec']:>10,.0f} s/s | {s_res['inference_throughput_samples_sec']:>10,.0f} s/s | {s_res['accuracy_per_cpu_sec']:>11.1f}")
        
        acc_diff = w_res['test_accuracy_pct'] - s_res['test_accuracy_pct']
        inf_gain = w_res['inference_throughput_samples_sec'] / max(1.0, s_res['inference_throughput_samples_sec'])
        train_gain = w_res['train_throughput_samples_sec'] / max(1.0, s_res['train_throughput_samples_sec'])
        sign = "+" if acc_diff >= 0 else ""
        print(f"{'':<18} | {'  -> Gain / Ratio':<18} | {sign}{acc_diff:>6.2f}% | {'':<9} | {'':<9} | {train_gain:>9.2f}x thrpt | {inf_gain:>9.2f}x thrpt |")
        print("-" * 115)

    print("=" * 115)

    generate_cpu_efficiency_plot(results, "plots/benchmark_cpu_efficiency.png")


if __name__ == "__main__":
    main()
