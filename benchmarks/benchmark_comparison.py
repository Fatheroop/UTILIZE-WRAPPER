"""
Benchmark Comparison: Network With Wrapper vs. Network Without Wrapper (PyTorch Baseline)
========================================================================================

Monitors, benchmarks, and compares the EXACT SAME PyTorch network architecture:
  1. Network With Wrapper: ConstructiveWrapper (layer-by-layer progressive decoupling,
     zero-copy strided tensor slicing, ~2 MB double-buffered chunk prefetching,
     and 1st-order Recurrent EMA convergence tracking).
  2. Network Without Wrapper: Standard PyTorch end-to-end backpropagation through all layers
     simultaneously using standard PyTorch DataLoader and optim.Adam.

Tracks and reports:
  - System Resources: Peak RAM (RSS MB), RAM Increase (+Δ MB), Average RAM
  - Accuracy Dynamics: Training Accuracy, Test Accuracy, Macro F1
  - Execution Performance: Wall-Clock Training Duration, Inference Latency per 1k samples
  - Visualizations: 4-panel comparative analysis saved to plots/benchmark_comparison.png
"""

import sys
import os
import time
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mnist_dataset import get_normalized_mnist
from utils.resource_monitor import ResourceTracker
from core_models.constructive_torch import ConstructiveWrapper


def create_lenet5_network(in_shape=(1, 28, 28), num_classes=10) -> nn.Sequential:
    """Build standard PyTorch LeNet-5 architecture (identical for both benchmarks)."""
    padding = 2 if in_shape[1:] == (28, 28) else 0
    return nn.Sequential(
        nn.Conv2d(in_shape[0], 6, kernel_size=5, padding=padding),
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
        nn.Linear(84, num_classes),
    )


def measure_inference_latency(predict_fn, data: np.ndarray, num_runs: int = 3) -> float:
    """Measure inference latency per 1,000 samples in milliseconds."""
    sample_size = min(1000, len(data))
    eval_slice = data[:sample_size]

    # Warmup
    _ = predict_fn(eval_slice[:min(100, sample_size)])

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = predict_fn(eval_slice)
        elapsed = time.perf_counter() - t0
        lat_per_1k = (elapsed / sample_size) * 1000.0 * 1000.0  # ms per 1k
        latencies.append(lat_per_1k)
    return float(np.mean(latencies))


def print_comparison_table(wrap_summary: dict, std_summary: dict):
    """Print clean comparison table with consistent 2-decimal formatting."""
    print("\n" + "=" * 82)
    print("📊 BENCHMARK COMPARISON: NETWORK WITH WRAPPER vs. NETWORK WITHOUT WRAPPER")
    print("=" * 82)
    header = f"{'Metric':<30} | {'Network With Wrapper':<24} | {'Network Without Wrapper':<24}"
    print(header)
    print("-" * 82)

    rows = [
        ("Test Accuracy (%)", f"{wrap_summary['test_accuracy_pct']:.2f}%", f"{std_summary['test_accuracy_pct']:.2f}%"),
        ("Training Accuracy (%)", f"{wrap_summary['train_accuracy_pct']:.2f}%", f"{std_summary['train_accuracy_pct']:.2f}%"),
        ("Macro F1-Score", f"{wrap_summary['macro_f1']:.4f}", f"{std_summary['macro_f1']:.4f}"),
        ("Peak RAM (RSS MB)", f"{wrap_summary['peak_ram_mb']:.2f} MB", f"{std_summary['peak_ram_mb']:.2f} MB"),
        ("RAM Increase (+Δ MB)", f"+{wrap_summary['ram_increase_mb']:.2f} MB", f"+{std_summary['ram_increase_mb']:.2f} MB"),
        ("Average RAM (MB)", f"{wrap_summary['avg_ram_mb']:.2f} MB", f"{std_summary['avg_ram_mb']:.2f} MB"),
        ("Total Training Time (s)", f"{wrap_summary['total_time_sec']:.2f} s", f"{std_summary['total_time_sec']:.2f} s"),
        ("Auto-Tune Hyperparams", "Active (Zero-Epoch In-Situ)", "None (Static Baseline)"),
        ("Inference Latency (ms/1k)", f"{wrap_summary['inference_latency_ms']:.3f} ms", f"{std_summary['inference_latency_ms']:.3f} ms"),
        ("ACT Early Exit Latency", f"{wrap_summary.get('act_latency_ms', 0.0):.3f} ms" if wrap_summary.get('act_latency_ms') else "N/A", "N/A (No Early Exit)"),
        ("Disk Footprint (MB)", f"{wrap_summary['disk_usage_mb']:.2f} MB", f"{std_summary['disk_usage_mb']:.2f} MB"),
    ]

    for label, wrap_val, std_val in rows:
        print(f"{label:<30} | {wrap_val:<24} | {std_val:<24}")

    print("=" * 82)


def generate_comparison_plots(wrap_summary: dict, std_summary: dict, output_path: str):
    """Generate 4-panel visual comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    colors = ["#2b5c8f", "#e07a5f"]
    models = ["Network With Wrapper\n(Constructive)", "Network Without Wrapper\n(Standard PyTorch)"]

    # 1. Peak RAM & Memory Increase
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    w = 0.35
    b1 = ax1.bar(x - w / 2, [wrap_summary["peak_ram_mb"], std_summary["peak_ram_mb"]], width=w, label="Peak Total RAM", color="#2b5c8f")
    b2 = ax1.bar(x + w / 2, [wrap_summary["ram_increase_mb"], std_summary["ram_increase_mb"]], width=w, label="RAM Increase (+Δ)", color="#e07a5f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylabel("Memory (MB)")
    ax1.set_title("System Memory (RAM) Consumption", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f} MB", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax1.annotate(f"+{h:.1f} MB", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Accuracy Comparison
    ax2 = axes[0, 1]
    b_tr = ax2.bar(x - w / 2, [wrap_summary["train_accuracy_pct"], std_summary["train_accuracy_pct"]], width=w, label="Train Accuracy", color="#3d5a80")
    b_te = ax2.bar(x + w / 2, [wrap_summary["test_accuracy_pct"], std_summary["test_accuracy_pct"]], width=w, label="Test Accuracy", color="#2a9d8f")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim([85, 102])
    ax2.set_title("Training & Test Accuracy", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in b_tr:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b_te:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. Training Time Comparison
    ax3 = axes[1, 0]
    b_time = ax3.bar(models, [wrap_summary["total_time_sec"], std_summary["total_time_sec"]], color=colors, width=0.45, edgecolor="black")
    ax3.set_ylabel("Wall-Clock Time (seconds)")
    ax3.set_title("Training Execution Duration", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in b_time:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f} s", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 4. Summary Scorecard / Key Advantages
    ax4 = axes[1, 1]
    ax4.axis("off")

    test_diff = wrap_summary["test_accuracy_pct"] - std_summary["test_accuracy_pct"]
    sign = "+" if test_diff >= 0 else ""

    summary_text = (
        "[SUMMARY OF BENCHMARK FINDINGS]\n\n"
        f"• Test Accuracy:\n"
        f"  - Network With Wrapper:    {wrap_summary['test_accuracy_pct']:.2f}%\n"
        f"  - Network Without Wrapper: {std_summary['test_accuracy_pct']:.2f}%\n"
        f"  - Difference:              {sign}{test_diff:.2f}%\n\n"
        f"• Inference Latency:\n"
        f"  - With Wrapper:            {wrap_summary['inference_latency_ms']:.3f} ms / 1k\n"
        f"  - Without Wrapper:         {std_summary['inference_latency_ms']:.3f} ms / 1k\n\n"
        f"• Architectural Equivalence & Deployment:\n"
        f"  - Both networks use the EXACT SAME PyTorch architecture.\n"
        f"  - ConstructiveWrapper exports back a 100% pure standard\n"
        f"    PyTorch nn.Sequential with zero wrapper dependencies.\n"
        f"  - Exported model achieves identical test accuracy."
    )

    ax4.text(
        0.05,
        0.5,
        summary_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved benchmark comparison visualization to: {output_path}")


def _benchmark_worker_with_wrapper(
    in_shape, num_classes, sub_train_images, sub_train_labels,
    val_images, val_labels, test_images, test_labels, args, result_dict
):
    """Run Network With Wrapper in an isolated clean process."""
    print("\n>>> [1/2] Benchmarking NETWORK WITH WRAPPER (ConstructiveWrapper)...")
    storage_dir = "constructive_bench_cache"

    with ResourceTracker("Network With Wrapper") as tracker:
        # Create standard PyTorch network
        network = create_lenet5_network(in_shape, num_classes)

        # Wrap it with ConstructiveWrapper
        wrapper = ConstructiveWrapper(
            model=network,
            in_shape=in_shape,
            num_classes=num_classes,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            min_delta=args.min_delta,
            max_epochs_per_stage=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            feature_storage=args.feature_storage,
            storage_dir=storage_dir,
            auto_tune_hyperparams=args.auto_tune,
            random_state=42,
        )

        wrapper.fit(
            train_data=sub_train_images,
            train_y=sub_train_labels,
            val_data=val_images,
            val_y=val_labels,
            verbose=False,
        )

        test_preds = wrapper.predict(test_images)
        train_preds = wrapper.predict(sub_train_images)

        test_acc = accuracy_score(test_labels, test_preds)
        train_acc = accuracy_score(sub_train_labels, train_preds)
        f1 = f1_score(test_labels, test_preds, average="macro")

        latency = measure_inference_latency(wrapper.predict, test_images)
        predict_act = lambda x: wrapper.predict(x, early_exit_threshold=args.early_exit_threshold)
        act_latency = measure_inference_latency(predict_act, test_images)

        # Verify standard model export in batches
        std_model = wrapper.get_standard_model()
        std_model.eval()
        exp_preds_list = []
        with torch.inference_mode():
            for bi in range(0, len(test_images), 512):
                tb = torch.from_numpy(test_images[bi : bi + 512]).float().to(wrapper.device)
                exp_preds_list.append(torch.argmax(std_model(tb), dim=1).cpu().numpy())
        exp_preds = np.concatenate(exp_preds_list)
        assert np.array_equal(test_preds, exp_preds), "Exported standard model predictions must match wrapper 100%!"

        tracker.stop(disk_dir_to_measure=storage_dir)
        tracker.log_results(train_acc, test_acc, f1, latency)

    tracker.print_report()
    res = tracker.summary()
    res["act_latency_ms"] = act_latency
    tuned_info = wrapper.get_tuned_hyperparameters()
    res["tuned_count"] = len(tuned_info)
    if tuned_info:
        res["final_tuned_lr"] = tuned_info[-1]["chosen_lr"]
        res["final_tuned_wd"] = tuned_info[-1]["chosen_wd"]
    result_dict.update(res)


def _benchmark_worker_without_wrapper(
    in_shape, num_classes, sub_train_images, sub_train_labels,
    val_images, val_labels, test_images, test_labels, args, result_dict
):
    """Run Network Without Wrapper in an isolated clean process."""
    print("\n>>> [2/2] Benchmarking NETWORK WITHOUT WRAPPER (Standard PyTorch End-to-End)...")

    with ResourceTracker("Network Without Wrapper") as tracker:
        # Create identical standard PyTorch network
        network = create_lenet5_network(in_shape, num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-4, foreach=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Standard PyTorch DataLoader
        t_train_x = torch.from_numpy(sub_train_images).float()
        t_train_y = torch.from_numpy(sub_train_labels).long()
        train_loader = DataLoader(TensorDataset(t_train_x, t_train_y), batch_size=args.batch_size, shuffle=True)

        t_val_x = torch.from_numpy(val_images).float().to(device)
        t_val_y = torch.from_numpy(val_labels).long().to(device)

        best_val_acc = 0.0
        epochs_no_improve = 0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            network.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = network(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation check
            network.eval()
            with torch.inference_mode():
                val_out = network(t_val_x)
                val_acc = float((torch.argmax(val_out, dim=1) == t_val_y).float().mean().item())

            if val_acc > best_val_acc + args.min_delta:
                best_val_acc = val_acc
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                break

        if best_state is not None:
            network.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Evaluation
        network.eval()
        def predict_fn(data_np: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                xt = torch.from_numpy(data_np).float()
                preds = []
                for i in range(0, len(xt), 512):
                    bx = xt[i : i + 512].to(device)
                    out = network(bx)
                    preds.append(torch.argmax(out, dim=1).cpu().numpy())
                return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.int64)

        test_preds = predict_fn(test_images)
        train_preds = predict_fn(sub_train_images)

        test_acc = accuracy_score(test_labels, test_preds)
        train_acc = accuracy_score(sub_train_labels, train_preds)
        f1 = f1_score(test_labels, test_preds, average="macro")

        latency = measure_inference_latency(predict_fn, test_images)
        tracker.stop()
        tracker.log_results(train_acc, test_acc, f1, latency)

    tracker.print_report()
    res = tracker.summary()
    res["act_latency_ms"] = None
    result_dict.update(res)


def main():
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Benchmark Network With Wrapper vs Network Without Wrapper")
    parser.add_argument("--num_train", type=int, default=30000, help="Training samples for benchmark")
    parser.add_argument("--num_test", type=int, default=10000, help="Testing samples for benchmark")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per stage/model")
    parser.add_argument("--lr", type=float, default=0.0015, help="Learning rate for Adam optimizer")
    parser.add_argument("--patience", type=int, default=3, help="Saturation patience epochs")
    parser.add_argument("--min_delta", type=float, default=0.0001, help="Min delta improvement for saturation")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--fine_tune_epochs", type=int, default=1, help="Fast end-to-end fine-tuning epochs for wrapper")
    parser.add_argument("--feature_storage", type=str, default="auto", help="Feature storage: auto, ram, or chunked_disk")
    parser.add_argument("--early_exit_threshold", type=float, default=0.98, help="ACT dynamic early exit threshold")
    parser.add_argument("--auto_tune", action="store_true", default=True, help="Enable continuous in-situ hyperparameter auto-tuning without extra epochs")
    parser.add_argument("--no_auto_tune", dest="auto_tune", action="store_false", help="Disable in-situ hyperparameter auto-tuning")
    parser.add_argument("--output_dir", type=str, default="plots", help="Plots output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("🚀 BENCHMARK: NETWORK WITH WRAPPER vs. NETWORK WITHOUT WRAPPER")
    print(f"   Architecture: Standard LeNet-5 (Identical PyTorch Modules)")
    print(f"   Samples: {args.num_train} train | {args.num_test} test | Batch: {args.batch_size}")
    print("=" * 70)

    train_images, train_labels, test_images, test_labels = get_normalized_mnist(
        num_train=args.num_train,
        num_test=args.num_test,
    )

    val_size = int(len(train_images) * 0.1)
    val_images, val_labels = train_images[:val_size], train_labels[:val_size]
    sub_train_images, sub_train_labels = train_images[val_size:], train_labels[val_size:]

    in_shape = tuple(train_images.shape[1:])
    num_classes = len(np.unique(train_labels))

    # Process Isolation: execute each model in an independent forked process
    ctx = mp.get_context("fork")
    mgr = ctx.Manager()

    wrap_dict = mgr.dict()
    p_wrap = ctx.Process(
        target=_benchmark_worker_with_wrapper,
        args=(in_shape, num_classes, sub_train_images, sub_train_labels,
              val_images, val_labels, test_images, test_labels, args, wrap_dict)
    )
    p_wrap.start()
    p_wrap.join()
    wrap_summary = dict(wrap_dict)

    std_dict = mgr.dict()
    p_std = ctx.Process(
        target=_benchmark_worker_without_wrapper,
        args=(in_shape, num_classes, sub_train_images, sub_train_labels,
              val_images, val_labels, test_images, test_labels, args, std_dict)
    )
    p_std.start()
    p_std.join()
    std_summary = dict(std_dict)

    # Clean up benchmark disk cache
    if os.path.exists("constructive_bench_cache"):
        import shutil
        shutil.rmtree("constructive_bench_cache", ignore_errors=True)

    # Compare and visualize
    print_comparison_table(wrap_summary, std_summary)

    chart_path = os.path.join(args.output_dir, "benchmark_comparison.png")
    generate_comparison_plots(wrap_summary, std_summary, chart_path)


if __name__ == "__main__":
    main()
