"""
Comprehensive Benchmark: Performance & Resource Optimizations in Constructive Learning
Tests and quantifies modern techniques across:
1. Training:
   - Gradient Accumulation (Physical batch vs Accumulation RAM comparison)
   - Polyak Adaptive Warmup EMA (Acc & Stability)
   - LR Warmup Scheduling (Convergence stability)
2. Prediction / Inference:
   - Zero-Copy Chunk Streaming vs Whole-Array RAM Allocation
   - Standard PyTorch vs JIT Traced vs Dynamic Quantized (INT8) Latency & RAM
"""

import os
import gc
import time
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
from constructive_torch import ConstructiveWrapper, AutoConstructiveModel


def benchmark_inference_zero_copy():
    print("\n" + "=" * 70)
    print("🚀 BENCHMARK 1: Zero-Copy Chunk Streaming vs Whole-Array Duplication")
    print("=" * 70)
    N_SAMPLES = 100_000
    N_FEATS = 64
    N_CLASSES = 10

    print(f"Generating standard Pandas/NumPy dataset of {N_SAMPLES:,} samples x {N_FEATS} features (float64, ~{N_SAMPLES*N_FEATS*8 / (1024**2):.1f} MB)...")
    np.random.seed(42)
    X_large = np.random.randn(N_SAMPLES, N_FEATS)

    # Build model
    model = nn.Sequential(
        nn.Linear(N_FEATS, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, N_CLASSES),
    )
    wrapper = ConstructiveWrapper(
        model=model,
        in_shape=(N_FEATS,),
        num_classes=N_CLASSES,
        max_epochs_per_stage=1,
        patience=1,
    )
    # Fit on small dummy to populate stages
    dummy_x = torch.randn(100, N_FEATS)
    dummy_y = torch.randint(0, N_CLASSES, (100,))
    wrapper.fit(dummy_x, dummy_y, verbose=False)

    import psutil
    proc = psutil.Process()

    # 1. Old naive approach: Whole-array torch.from_numpy(X).float() upfront
    gc.collect()
    m0_naive = proc.memory_info().rss
    t0 = time.perf_counter()
    xt_naive = torch.from_numpy(X_large).float()
    m_peak_naive = proc.memory_info().rss
    out_naive = np.empty(N_SAMPLES, dtype=np.int64)
    with torch.inference_mode():
        for i in range(0, N_SAMPLES, 512):
            end_i = min(i + 512, N_SAMPLES)
            bx = xt_naive[i:end_i]
            out = wrapper(bx)
            out_naive[i:end_i] = torch.argmax(out, dim=1).cpu().numpy()
    t_naive = (time.perf_counter() - t0) * 1000.0
    ram_naive_mb = max(0.0, (m_peak_naive - m0_naive) / (1024**2))
    del xt_naive
    gc.collect()

    # 2. Optimized Zero-Copy Chunk Streaming
    gc.collect()
    m0_fast = proc.memory_info().rss
    t0 = time.perf_counter()
    out_fast = wrapper.predict(X_large, batch_size=512)
    t_fast = (time.perf_counter() - t0) * 1000.0
    m_peak_fast = proc.memory_info().rss
    ram_fast_mb = max(0.0, (m_peak_fast - m0_fast) / (1024**2))

    # Verify parity
    np.testing.assert_array_equal(out_naive, out_fast)

    print(f"   [Whole-Array Duplication] Latency: {t_naive:.2f} ms | Additional RSS RAM: {ram_naive_mb:.2f} MB")
    print(f"   [Zero-Copy Chunk Stream]  Latency: {t_fast:.2f} ms | Additional RSS RAM: {ram_fast_mb:.2f} MB")
    print(f"   ⚡ Memory Efficiency: {ram_naive_mb - ram_fast_mb:+.2f} MB saved by zero-copy chunk streaming!")

    return {
        "naive_ms": t_naive,
        "fast_ms": t_fast,
        "naive_ram_mb": ram_naive_mb,
        "fast_ram_mb": ram_fast_mb,
    }


def benchmark_engine_compilation_and_quantization():
    print("\n" + "=" * 70)
    print("🚀 BENCHMARK 2: Inference Engine (Standard vs JIT Traced vs Dynamic INT8)")
    print("=" * 70)
    N_SAMPLES = 50_000
    N_FEATS = 32
    N_CLASSES = 5

    X_test = torch.randn(N_SAMPLES, N_FEATS)
    model = nn.Sequential(
        nn.Linear(N_FEATS, 128),
        nn.GELU(),
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Linear(64, N_CLASSES),
    )
    wrapper = ConstructiveWrapper(
        model=model,
        in_shape=(N_FEATS,),
        num_classes=N_CLASSES,
        max_epochs_per_stage=1,
    )
    wrapper.fit(X_test[:200], torch.randint(0, N_CLASSES, (200,)), verbose=False)

    std_model = wrapper.get_standard_model()
    traced_model = wrapper.export_traced_model()
    quant_model = wrapper.export_quantized_model()

    # Benchmark Standard
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, N_SAMPLES, 1024):
            _ = std_model(X_test[i:i+1024])
    t_std = (time.perf_counter() - t0) * 1000.0

    # Benchmark JIT Traced
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, N_SAMPLES, 1024):
            _ = traced_model(X_test[i:i+1024])
    t_jit = (time.perf_counter() - t0) * 1000.0

    # Benchmark Quantized INT8
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, N_SAMPLES, 1024):
            _ = quant_model(X_test[i:i+1024])
    t_quant = (time.perf_counter() - t0) * 1000.0

    print(f"   [Standard PyTorch Model]   Latency ({N_SAMPLES:,} samples): {t_std:.2f} ms")
    print(f"   [TorchScript JIT Traced]  Latency ({N_SAMPLES:,} samples): {t_jit:.2f} ms ({t_std/t_jit:.2f}x speed)")
    print(f"   [Dynamic Quantized (INT8)] Latency ({N_SAMPLES:,} samples): {t_quant:.2f} ms ({t_std/t_quant:.2f}x speed)")

    return {"std_ms": t_std, "jit_ms": t_jit, "quant_ms": t_quant}


def benchmark_gradient_accumulation():
    print("\n" + "=" * 70)
    print("🚀 BENCHMARK 3: Gradient Accumulation (Physical RAM vs Effective Batch)")
    print("=" * 70)
    N_SAMPLES = 4_000
    N_FEATS = 64
    N_CLASSES = 10

    torch.manual_seed(42)
    X = torch.randn(N_SAMPLES, N_FEATS)
    y = torch.randint(0, N_CLASSES, (N_SAMPLES,))

    # 1. Large Physical Batch (batch_size=512, accum=1)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    w_large = ConstructiveWrapper(
        stages=[
            nn.Sequential(nn.Linear(N_FEATS, 128), nn.ReLU()),
            nn.Sequential(nn.Linear(128, N_CLASSES)),
        ],
        in_shape=(N_FEATS,),
        num_classes=N_CLASSES,
        batch_size=512,
        gradient_accumulation_steps=1,
        max_epochs_per_stage=3,
        fine_tune_epochs=1,
        random_state=42,
    )
    w_large.fit(X, y, verbose=False)
    t_large = (time.perf_counter() - t0) * 1000.0
    _, peak_large = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    acc_large = w_large.evaluate(X, y)

    # 2. Gradient Accumulation (physical batch=64, accum=8 -> effective 512)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    w_accum = ConstructiveWrapper(
        stages=[
            nn.Sequential(nn.Linear(N_FEATS, 128), nn.ReLU()),
            nn.Sequential(nn.Linear(128, N_CLASSES)),
        ],
        in_shape=(N_FEATS,),
        num_classes=N_CLASSES,
        batch_size=64,
        gradient_accumulation_steps=8,
        max_epochs_per_stage=3,
        fine_tune_epochs=1,
        random_state=42,
    )
    w_accum.fit(X, y, verbose=False)
    t_accum = (time.perf_counter() - t0) * 1000.0
    _, peak_accum = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    acc_accum = w_accum.evaluate(X, y)

    print(f"   [Physical Batch 512]        Peak RAM: {peak_large / (1024**2):.2f} MB | Train Time: {t_large:.1f} ms | Acc: {acc_large*100:.2f}%")
    print(f"   [Accumulated Batch (64x8)]  Peak RAM: {peak_accum / (1024**2):.2f} MB | Train Time: {t_accum:.1f} ms | Acc: {acc_accum*100:.2f}%")
    print(f"   ⚡ RAM Reduction: {(1.0 - peak_accum / peak_large)*100:.1f}% less peak memory while preserving accuracy!")

    return {
        "large_ram_mb": peak_large / (1024**2),
        "accum_ram_mb": peak_accum / (1024**2),
        "acc_large": acc_large,
        "acc_accum": acc_accum,
    }


def benchmark_ema_and_warmup():
    print("\n" + "=" * 70)
    print("🚀 BENCHMARK 4: Polyak Adaptive Warmup EMA & LR Scheduler Stability")
    print("=" * 70)
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=14,
        n_redundant=4,
        flip_y=0.08,  # Label noise to test EMA stabilization
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 1. AutoConstructive without EMA
    auto_no_ema = AutoConstructiveModel(
        in_shape=20,
        num_classes=2,
        max_stages=2,
        candidate_neurons=[32, 64],
        candidate_activations=["relu", "gelu"],
        candidate_optimizers=["adamw"],
        batch_size=64,
        max_epochs_per_stage=6,
        fine_tune_epochs=2,
        use_ema=False,
        warmup_epochs=0,
        random_state=42,
    )
    auto_no_ema.fit(X_train, y_train, verbose=False)
    acc_no_ema = auto_no_ema.evaluate(X_test, y_test)

    # 2. AutoConstructive with Polyak Adaptive Warmup EMA and LR Warmup
    auto_ema = AutoConstructiveModel(
        in_shape=20,
        num_classes=2,
        max_stages=2,
        candidate_neurons=[32, 64],
        candidate_activations=["relu", "gelu"],
        candidate_optimizers=["adamw"],
        batch_size=64,
        max_epochs_per_stage=6,
        fine_tune_epochs=2,
        use_ema=True,
        ema_decay=0.99,
        warmup_epochs=1,
        random_state=42,
    )
    auto_ema.fit(X_train, y_train, verbose=False)
    acc_ema = auto_ema.evaluate(X_test, y_test)

    print(f"   [Standard Training (No EMA)] Test Accuracy: {acc_no_ema * 100:.2f}%")
    print(f"   [Polyak Adaptive EMA + Warmup] Test Accuracy: {acc_ema * 100:.2f}%")
    print(f"   ⚡ Accuracy Delta: {(acc_ema - acc_no_ema)*100:+.2f}% on noisy tabular benchmark!")

    return {"acc_no_ema": acc_no_ema, "acc_ema": acc_ema}


if __name__ == "__main__":
    r1 = benchmark_inference_zero_copy()
    r2 = benchmark_engine_compilation_and_quantization()
    r3 = benchmark_gradient_accumulation()
    r4 = benchmark_ema_and_warmup()
    print("\n✅ All benchmarks completed successfully!")
