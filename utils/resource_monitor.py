"""
Resource and Accuracy Monitoring Module
Tracks:
  - System Resources: Peak RAM (RSS in MB), Average RAM, Peak VRAM (CUDA), Disk usage.
  - Training Dynamics: Loss progression, Training Accuracy, Validation/Test Accuracy.
  - Timing: Wall-clock training duration and per-sample inference latency.
  - Generates comparison tables and visualizations.
"""

import os
import time
import threading
import psutil
from typing import Dict, Any, List, Optional
import numpy as np
import torch


class ResourceTracker:
    """
    Thread-safe context manager and live tracker for system resource utilization and accuracy metrics.
    Runs a lightweight background daemon thread to sample peak memory (RSS).
    """

    def __init__(self, name: str, sample_interval_sec: float = 0.05):
        self.name = name
        self.sample_interval_sec = sample_interval_sec
        self.process = psutil.Process(os.getpid())
        self._lock = threading.Lock()

        # Resource metrics
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.total_time_sec: float = 0.0

        self.initial_ram_mb: float = 0.0
        self.peak_ram_mb: float = 0.0
        self.ram_samples: List[float] = []

        self.peak_vram_mb: float = 0.0
        self.disk_usage_mb: float = 0.0

        # Model accuracy metrics
        self.train_accuracy: float = 0.0
        self.test_accuracy: float = 0.0
        self.macro_f1: float = 0.0
        self.inference_latency_ms: float = 0.0

        self.epoch_losses: List[float] = []
        self.epoch_train_accs: List[float] = []
        self.epoch_val_accs: List[float] = []

        # Background monitoring thread
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        # CPU metrics
        self.cpu_samples: List[float] = []
        self.peak_cpu_percent: float = 0.0
        self.initial_cpu_times: Optional[Any] = None
        self.final_cpu_times: Optional[Any] = None
        self.total_cpu_time_sec: float = 0.0
        self.user_cpu_time_sec: float = 0.0
        self.system_cpu_time_sec: float = 0.0
        self.num_samples: int = 0
        self.num_epochs: int = 1

    def _sample_loop(self):
        """Background thread polling memory and CPU utilization at high frequency."""
        try:
            self.process.cpu_percent()
        except Exception:
            pass
        while not self._stop_event.is_set():
            try:
                rss_mb = self.process.memory_info().rss / (1024 * 1024)
                cpu_pct = self.process.cpu_percent()
                with self._lock:
                    self.ram_samples.append(rss_mb)
                    if rss_mb > self.peak_ram_mb:
                        self.peak_ram_mb = rss_mb
                    if cpu_pct > 0.0:
                        self.cpu_samples.append(cpu_pct)
                        if cpu_pct > self.peak_cpu_percent:
                            self.peak_cpu_percent = cpu_pct
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.sample_interval_sec)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """Begin tracking resources."""
        current_rss = self.process.memory_info().rss / (1024 * 1024)
        with self._lock:
            self.initial_ram_mb = current_rss
            self.peak_ram_mb = current_rss
            self.ram_samples = [current_rss]
            self.cpu_samples = []
            self.peak_cpu_percent = 0.0

        try:
            self.initial_cpu_times = self.process.cpu_times()
            self.process.cpu_percent()
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.start_time = time.perf_counter()
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self, disk_dir_to_measure: Optional[str] = None):
        """Stop tracking resources and finalize metrics."""
        if not self._stop_event.is_set():
            self.end_time = time.perf_counter()
            self.total_time_sec = self.end_time - self.start_time
            self._stop_event.set()
            if self._monitor_thread is not None:
                self._monitor_thread.join(timeout=1.0)

        try:
            self.final_cpu_times = self.process.cpu_times()
            if self.initial_cpu_times and self.final_cpu_times:
                self.user_cpu_time_sec = max(0.0, self.final_cpu_times.user - self.initial_cpu_times.user)
                self.system_cpu_time_sec = max(0.0, self.final_cpu_times.system - self.initial_cpu_times.system)
                self.total_cpu_time_sec = self.user_cpu_time_sec + self.system_cpu_time_sec
        except Exception:
            pass

        # Measure peak VRAM if available
        if torch.cuda.is_available():
            self.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Safely measure disk footprint if directory provided
        if disk_dir_to_measure and os.path.exists(disk_dir_to_measure):
            total_bytes = 0
            try:
                for root, _, files in os.walk(disk_dir_to_measure):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            total_bytes += os.path.getsize(fp)
                        except OSError:
                            pass
            except OSError:
                pass
            self.disk_usage_mb = total_bytes / (1024 * 1024)

    def log_results(
        self,
        train_acc: float,
        test_acc: float,
        macro_f1: float = 0.0,
        inference_latency_ms: float = 0.0,
        num_samples: int = 0,
        num_epochs: int = 1,
    ):
        """Attach accuracy and throughput metrics."""
        self.train_accuracy = train_acc
        self.test_accuracy = test_acc
        self.macro_f1 = macro_f1
        self.inference_latency_ms = inference_latency_ms
        self.num_samples = num_samples
        self.num_epochs = num_epochs

    @property
    def avg_ram_mb(self) -> float:
        with self._lock:
            return float(np.mean(self.ram_samples)) if self.ram_samples else self.initial_ram_mb

    @property
    def ram_delta_mb(self) -> float:
        """Net increase in RAM from starting point."""
        with self._lock:
            return max(0.0, self.peak_ram_mb - self.initial_ram_mb)

    @property
    def avg_cpu_pct(self) -> float:
        with self._lock:
            return float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0

    @property
    def cpu_core_load_pct(self) -> float:
        """Total CPU compute time as percentage of wall-clock time (100% = 1 full core)."""
        return (self.total_cpu_time_sec / max(0.001, self.total_time_sec)) * 100.0

    @property
    def train_throughput(self) -> float:
        """Samples processed per second during training."""
        if self.num_samples > 0:
            return (self.num_samples * self.num_epochs) / max(0.001, self.total_time_sec)
        return 0.0

    @property
    def inference_throughput(self) -> float:
        """Samples predicted per second during inference."""
        if self.inference_latency_ms > 0:
            return 1000.0 / (self.inference_latency_ms / 1000.0)
        return 0.0

    @property
    def accuracy_per_cpu_sec(self) -> float:
        """Accuracy points achieved per CPU second invested."""
        return (self.test_accuracy * 100.0) / max(0.01, self.total_cpu_time_sec)

    def summary(self) -> Dict[str, Any]:
        """Return full metrics summary dictionary."""
        with self._lock:
            peak = self.peak_ram_mb
            avg = float(np.mean(self.ram_samples)) if self.ram_samples else self.initial_ram_mb
            delta = max(0.0, peak - self.initial_ram_mb)
            avg_cpu = float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
            peak_cpu = self.peak_cpu_percent

        return {
            "name": self.name,
            "train_accuracy_pct": round(self.train_accuracy * 100, 2),
            "test_accuracy_pct": round(self.test_accuracy * 100, 2),
            "macro_f1": round(self.macro_f1, 4),
            "total_time_sec": round(self.total_time_sec, 2),
            "total_cpu_time_sec": round(self.total_cpu_time_sec, 2),
            "user_cpu_time_sec": round(self.user_cpu_time_sec, 2),
            "system_cpu_time_sec": round(self.system_cpu_time_sec, 2),
            "avg_cpu_pct": round(avg_cpu, 1),
            "peak_cpu_pct": round(peak_cpu, 1),
            "cpu_core_load_pct": round(self.cpu_core_load_pct, 1),
            "peak_ram_mb": round(peak, 2),
            "avg_ram_mb": round(avg, 2),
            "ram_increase_mb": round(delta, 2),
            "peak_vram_mb": round(self.peak_vram_mb, 2),
            "disk_usage_mb": round(self.disk_usage_mb, 2),
            "inference_latency_ms": round(self.inference_latency_ms, 3),
            "train_throughput_samples_sec": round(self.train_throughput, 1),
            "inference_throughput_samples_sec": round(self.inference_throughput, 1),
            "accuracy_per_cpu_sec": round(self.accuracy_per_cpu_sec, 2),
        }

    def print_report(self):
        """Print a formatted report for this tracker."""
        s = self.summary()
        print("\n" + "=" * 65)
        print(f"📋 Resource & Efficiency Report: {s['name']}")
        print("=" * 65)
        print(f"  • Training Accuracy:       {s['train_accuracy_pct']:.2f}%")
        print(f"  • Test Accuracy:           {s['test_accuracy_pct']:.2f}%")
        print(f"  • Macro F1-Score:          {s['macro_f1']:.4f}")
        print(f"  • Wall Time:               {s['total_time_sec']:.2f} seconds")
        print(f"  • Total CPU Time:          {s['total_cpu_time_sec']:.2f} seconds (User: {s['user_cpu_time_sec']:.2f}s, Sys: {s['system_cpu_time_sec']:.2f}s)")
        print(f"  • CPU Utilization:         Avg: {s['avg_cpu_pct']:.1f}% | Peak: {s['peak_cpu_pct']:.1f}% | Load: {s['cpu_core_load_pct']:.1f}%")
        print(f"  • Peak RAM (RSS):          {s['peak_ram_mb']:.2f} MB (Net Increase: +{s['ram_increase_mb']:.2f} MB)")
        print(f"  • Average RAM (RSS):       {s['avg_ram_mb']:.2f} MB")
        if s['train_throughput_samples_sec'] > 0:
            print(f"  • Training Throughput:     {s['train_throughput_samples_sec']:,.1f} samples/second")
        if s['inference_throughput_samples_sec'] > 0:
            print(f"  • Inference Throughput:    {s['inference_throughput_samples_sec']:,.1f} samples/second ({s['inference_latency_ms']:.2f} ms/1k)")
        print(f"  • Compute Efficiency:      {s['accuracy_per_cpu_sec']:.2f} Acc points / CPU-sec")
        if torch.cuda.is_available():
            print(f"  • Peak VRAM:               {s['peak_vram_mb']:.2f} MB")
        if s['disk_usage_mb'] > 0:
            print(f"  • Disk Footprint:          {s['disk_usage_mb']:.2f} MB")
        print("=" * 65 + "\n")
