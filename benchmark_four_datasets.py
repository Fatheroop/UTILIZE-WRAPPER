"""
Comprehensive Benchmark: Network With Wrapper vs. Network Without Wrapper
========================================================================
Benchmarking across four canonical datasets:
  1. XOR (4-bit parity classification - non-linearly separable)
  2. Moons (make_moons non-linear interleaving boundary)
  3. MNIST (Handwritten digits - LeNet-5 ConvNet)
  4. Fashion-MNIST (Zalando articles - Modern ConvNet)

Evaluates and compares:
  - Test Accuracy (%) & Training Accuracy (%)
  - Macro F1-Score
  - Training Time (seconds)
  - Peak RAM Usage (RSS in MB) & RAM Delta (+Δ MB)
  - Inference Latency per 1,000 samples (ms)
  - Memory & computational efficiency
"""

import os
import gc
import gzip
import time
import argparse
from typing import Dict, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from constructive_torch import ConstructiveWrapper
from resource_monitor import ResourceTracker


# ======================================================================
# DATASET GENERATORS & LOADERS
# ======================================================================

def get_xor_dataset(n_samples: int = 800, noise: float = 0.05, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 4-bit parity XOR dataset with Gaussian noise."""
    np.random.seed(random_state)
    # 16 vertices of 4D hypercube
    vertices = np.array([
        [(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1]
        for i in range(16)
    ], dtype=np.float32)
    # Parity labels: 1 if odd number of 1s, else 0
    parity_labels = (np.sum(vertices, axis=1) % 2).astype(np.int64)

    repeats = int(np.ceil(n_samples / 16))
    X_rep = np.tile(vertices, (repeats, 1))[:n_samples]
    y_rep = np.tile(parity_labels, repeats)[:n_samples]

    # Add Gaussian jitter
    X = X_rep + np.random.normal(0, noise, size=X_rep.shape).astype(np.float32)
    y = y_rep

    return train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)


def get_moons_dataset(n_samples: int = 2000, noise: float = 0.25, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate two-moons non-linear dataset, standard scaled."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test


def get_mnist_dataset(n_train: int = 10000, n_test: int = 2000, data_dir: str = "data/mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load normalized MNIST dataset from local cache."""
    train_img_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_lbl_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_img_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_lbl_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    with gzip.open(train_img_path, "rb") as f:
        tr_x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)[:n_train].astype(np.float32) / 255.0
    with gzip.open(train_lbl_path, "rb") as f:
        tr_y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)[:n_train]
    with gzip.open(test_img_path, "rb") as f:
        te_x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)[:n_test].astype(np.float32) / 255.0
    with gzip.open(test_lbl_path, "rb") as f:
        te_y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)[:n_test]

    return tr_x, te_x, tr_y, te_y


def get_fashion_mnist_dataset(n_train: int = 10000, n_test: int = 2000, data_dir: str = "data/fashion_mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load normalized Fashion-MNIST dataset from local cache."""
    train_img_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_lbl_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_img_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_lbl_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    with gzip.open(train_img_path, "rb") as f:
        tr_x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)[:n_train].astype(np.float32) / 255.0
    with gzip.open(train_lbl_path, "rb") as f:
        tr_y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)[:n_train]
    with gzip.open(test_img_path, "rb") as f:
        te_x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)[:n_test].astype(np.float32) / 255.0
    with gzip.open(test_lbl_path, "rb") as f:
        te_y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)[:n_test]

    return tr_x, te_x, tr_y, te_y


# ======================================================================
# MODEL ARCHITECTURES (IDENTICAL PAIRS)
# ======================================================================

def build_xor_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )

def build_moons_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.GELU(),
        nn.Linear(64, 32),
        nn.GELU(),
        nn.Linear(32, 2),
    )

def build_mnist_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
    )

def build_fashion_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


# ======================================================================
# INFERENCE LATENCY MEASUREMENT
# ======================================================================

def measure_latency_per_1k(predict_fn, X_test: np.ndarray, num_runs: int = 5) -> float:
    """Measure inference latency in ms per 1,000 samples."""
    sample_size = min(1000, len(X_test))
    subset = X_test[:sample_size]

    # Warmup
    _ = predict_fn(subset[:min(50, sample_size)])

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = predict_fn(subset)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        lat_per_1k = (elapsed_ms / sample_size) * 1000.0
        latencies.append(lat_per_1k)

    return float(np.mean(latencies))


# ======================================================================
# BENCHMARK RUNNER
# ======================================================================

def run_single_comparison(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_builder,
    in_shape: Tuple[int, ...],
    num_classes: int,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    max_epochs_per_stage: int = 6,
    std_epochs: int = 25,
    fine_tune_epochs: Optional[int] = None,
    fine_tune_lr: Optional[float] = None,
    use_ema: bool = True,
    auto_tune_hyperparams: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute head-to-head comparison on a dataset."""
    print(f"\n{'='*75}")
    print(f"🔬 RUNNING BENCHMARK: {dataset_name.upper()}")
    print(f"   Train samples: {len(X_train):,} | Test samples: {len(X_test):,} | Shape: {in_shape} | Classes: {num_classes}")
    print(f"{'='*75}")

    device = torch.device("cpu")

    # -------------------------------------------------------------
    # 1. NETWORK WITH WRAPPER (ConstructiveWrapper)
    # -------------------------------------------------------------
    print(f"\n>>> [1/2] Training NETWORK WITH WRAPPER on {dataset_name}...")
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
        fine_tune_epochs=fine_tune_epochs if fine_tune_epochs is not None else 3,
        fine_tune_lr=fine_tune_lr,
        use_ema=use_ema,
        ema_decay=0.99,
        auto_tune_hyperparams=auto_tune_hyperparams,
        device=device,
        random_state=42,
    )

    with ResourceTracker(f"With Wrapper ({dataset_name})", sample_interval_sec=0.01) as wrap_tracker:
        wrapper.fit(X_train, y_train, verbose=False)
        train_preds = wrapper.predict(X_train, batch_size=512)
        test_preds = wrapper.predict(X_test, batch_size=512)
        train_acc = float(accuracy_score(y_train, train_preds))
        test_acc = float(accuracy_score(y_test, test_preds))
        macro_f1 = float(f1_score(y_test, test_preds, average="macro"))

        lat_1k = measure_latency_per_1k(lambda x: wrapper.predict(x, batch_size=512), X_test)
        wrap_tracker.log_results(train_acc=train_acc, test_acc=test_acc, macro_f1=macro_f1, inference_latency_ms=lat_1k)

    wrap_summary = wrap_tracker.summary()
    print(f"   ✓ With Wrapper Finished: Test Acc = {wrap_summary['test_accuracy_pct']:.2f}% | Train Time = {wrap_summary['total_time_sec']:.2f} s | Peak RAM = {wrap_summary['peak_ram_mb']:.1f} MB | Latency = {wrap_summary['inference_latency_ms']:.2f} ms/1k")

    # -------------------------------------------------------------
    # 2. NETWORK WITHOUT WRAPPER (Standard PyTorch Baseline)
    # -------------------------------------------------------------
    print(f"\n>>> [2/2] Training NETWORK WITHOUT WRAPPER (Standard PyTorch Baseline) on {dataset_name}...")
    gc.collect()
    torch.manual_seed(42)
    np.random.seed(42)

    net_std = model_builder().to(device)
    optimizer = torch.optim.Adam(net_std.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    tx = torch.from_numpy(X_train).float()
    ty = torch.from_numpy(y_train).long()
    train_loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True)

    with ResourceTracker(f"Without Wrapper ({dataset_name})", sample_interval_sec=0.01) as std_tracker:
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
        std_tracker.log_results(train_acc=std_train_acc, test_acc=std_test_acc, macro_f1=std_macro_f1, inference_latency_ms=std_lat_1k)

    std_summary = std_tracker.summary()
    print(f"   ✓ Without Wrapper Finished: Test Acc = {std_summary['test_accuracy_pct']:.2f}% | Train Time = {std_summary['total_time_sec']:.2f} s | Peak RAM = {std_summary['peak_ram_mb']:.1f} MB | Latency = {std_summary['inference_latency_ms']:.2f} ms/1k")

    return wrap_summary, std_summary


# ======================================================================
# VISUALIZATION GENERATOR
# ======================================================================

def generate_multi_dataset_plot(results: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]], output_path: str = "plots/benchmark_four_datasets.png"):
    """Generate high-resolution comparative bar chart across all 4 datasets."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Benchmark Comparison: Network With Wrapper vs. Network Without Wrapper\nAcross XOR, Moons, MNIST, and Fashion-MNIST Datasets", fontsize=15, fontweight="bold", y=0.98)

    datasets = list(results.keys())
    x = np.arange(len(datasets))
    width = 0.35

    color_wrap = "#1f77b4"  # Modern deep blue
    color_std  = "#ff7f0e"  # Vivid amber orange

    # Subplot 1: Test Accuracy (%)
    ax1 = axes[0, 0]
    wrap_acc = [results[d][0]["test_accuracy_pct"] for d in datasets]
    std_acc  = [results[d][1]["test_accuracy_pct"] for d in datasets]
    b1 = ax1.bar(x - width/2, wrap_acc, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax1.bar(x + width/2, std_acc, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax1.set_title("Test Accuracy (%) [Higher is Better]", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontweight="semibold")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim([max(0, min(wrap_acc + std_acc) - 10), 105])
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")
    for b in b1 + b2:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 2: Total Training Time (seconds)
    ax2 = axes[0, 1]
    wrap_time = [results[d][0]["total_time_sec"] for d in datasets]
    std_time  = [results[d][1]["total_time_sec"] for d in datasets]
    b1 = ax2.bar(x - width/2, wrap_time, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax2.bar(x + width/2, std_time, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax2.set_title("Training Duration (seconds) [Lower is Faster]", fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontweight="semibold")
    ax2.set_ylabel("Time (seconds)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left")
    for b in b1 + b2:
        h = b.get_height()
        ax2.annotate(f"{h:.2f}s", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 3: Peak RAM Memory Footprint (MB)
    ax3 = axes[1, 0]
    wrap_ram = [results[d][0]["peak_ram_mb"] for d in datasets]
    std_ram  = [results[d][1]["peak_ram_mb"] for d in datasets]
    b1 = ax3.bar(x - width/2, wrap_ram, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax3.bar(x + width/2, std_ram, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax3.set_title("Peak Process Memory (RSS in MB) [Lower is Better]", fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, fontweight="semibold")
    ax3.set_ylabel("Peak RAM (MB)")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left")
    for b in b1 + b2:
        h = b.get_height()
        ax3.annotate(f"{h:.0f}M", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 4: Inference Latency per 1k samples (ms)
    ax4 = axes[1, 1]
    wrap_lat = [results[d][0]["inference_latency_ms"] for d in datasets]
    std_lat  = [results[d][1]["inference_latency_ms"] for d in datasets]
    b1 = ax4.bar(x - width/2, wrap_lat, width, label="With Wrapper", color=color_wrap, edgecolor="black", alpha=0.9)
    b2 = ax4.bar(x + width/2, std_lat, width, label="Without Wrapper (Baseline)", color=color_std, edgecolor="black", alpha=0.9)
    ax4.set_title("Inference Latency per 1k Samples (ms) [Lower is Faster]", fontweight="bold", pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets, fontweight="semibold")
    ax4.set_ylabel("Latency (ms / 1k samples)")
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper left")
    for b in b1 + b2:
        h = b.get_height()
        ax4.annotate(f"{h:.2f}ms", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n📊 Multi-dataset benchmark visualization successfully saved to: {output_path}")


# ======================================================================
# MAIN EXECUTION
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark With vs Without Wrapper on 4 datasets.")
    parser.add_argument("--n_mnist_train", type=int, default=10000, help="Number of MNIST training samples")
    parser.add_argument("--n_mnist_test", type=int, default=2000, help="Number of MNIST test samples")
    parser.add_argument("--n_fashion_train", type=int, default=10000, help="Number of Fashion-MNIST training samples")
    parser.add_argument("--n_fashion_test", type=int, default=2000, help="Number of Fashion-MNIST test samples")
    args = parser.parse_args()

    results: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    # 1. 4-bit XOR Parity
    x_tr, x_te, y_tr, y_te = get_xor_dataset(n_samples=800, noise=0.04, random_state=42)
    wrap_xor, std_xor = run_single_comparison(
        dataset_name="4-Bit XOR",
        X_train=x_tr,
        X_test=x_te,
        y_train=y_tr,
        y_test=y_te,
        model_builder=build_xor_model,
        in_shape=(4,),
        num_classes=2,
        batch_size=32,
        learning_rate=0.02,
        max_epochs_per_stage=12,
        std_epochs=45,
    )
    results["4-Bit XOR"] = (wrap_xor, std_xor)

    # 2. Two Moons
    x_tr, x_te, y_tr, y_te = get_moons_dataset(n_samples=2000, noise=0.25, random_state=42)
    wrap_moons, std_moons = run_single_comparison(
        dataset_name="Two Moons",
        X_train=x_tr,
        X_test=x_te,
        y_train=y_tr,
        y_test=y_te,
        model_builder=build_moons_model,
        in_shape=(2,),
        num_classes=2,
        batch_size=64,
        learning_rate=0.015,
        max_epochs_per_stage=8,
        std_epochs=35,
    )
    results["Two Moons"] = (wrap_moons, std_moons)

    # 3. MNIST Digits
    x_tr, x_te, y_tr, y_te = get_mnist_dataset(n_train=args.n_mnist_train, n_test=args.n_mnist_test)
    wrap_mnist, std_mnist = run_single_comparison(
        dataset_name="MNIST Digits",
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
        std_epochs=8,
    )
    results["MNIST"] = (wrap_mnist, std_mnist)

    # 4. Fashion-MNIST
    x_tr, x_te, y_tr, y_te = get_fashion_mnist_dataset(n_train=args.n_fashion_train, n_test=args.n_fashion_test)
    wrap_fash, std_fash = run_single_comparison(
        dataset_name="Fashion-MNIST",
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
        use_ema=False,
        auto_tune_hyperparams=False,
        std_epochs=8,
    )
    results["Fashion-MNIST"] = (wrap_fash, std_fash)

    # -------------------------------------------------------------
    # PRINT MASTER COMPARISON TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 98)
    print("🏆 MASTER BENCHMARK SUMMARY: NETWORK WITH WRAPPER vs. NETWORK WITHOUT WRAPPER")
    print("=" * 98)
    header = f"{'Dataset':<16} | {'Approach':<20} | {'Test Acc':<10} | {'Macro F1':<10} | {'Train Time':<12} | {'Peak RAM':<12} | {'Latency':<10}"
    print(header)
    print("-" * 98)

    for d_name, (wrap_res, std_res) in results.items():
        print(f"{d_name:<16} | {'With Wrapper':<20} | {wrap_res['test_accuracy_pct']:>8.2f}% | {wrap_res['macro_f1']:>10.4f} | {wrap_res['total_time_sec']:>10.2f}s | {wrap_res['peak_ram_mb']:>10.1f}MB | {wrap_res['inference_latency_ms']:>6.2f}ms")
        print(f"{'':<16} | {'Without Wrapper':<20} | {std_res['test_accuracy_pct']:>8.2f}% | {std_res['macro_f1']:>10.4f} | {std_res['total_time_sec']:>10.2f}s | {std_res['peak_ram_mb']:>10.1f}MB | {std_res['inference_latency_ms']:>6.2f}ms")
        acc_diff = wrap_res['test_accuracy_pct'] - std_res['test_accuracy_pct']
        time_speedup = std_res['total_time_sec'] / max(0.01, wrap_res['total_time_sec'])
        sign = "+" if acc_diff >= 0 else ""
        print(f"{'':<16} | {'  -> Delta / Gain':<20} | {sign}{acc_diff:>7.2f}% | {'':<10} | {time_speedup:>9.2f}x | {'':<12} |")
        print("-" * 98)

    print("=" * 98)

    # Generate Chart
    generate_multi_dataset_plot(results, "plots/benchmark_four_datasets.png")


if __name__ == "__main__":
    main()
