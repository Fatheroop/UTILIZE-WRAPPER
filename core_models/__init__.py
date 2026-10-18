"""
Core Models Package
===================
Provides modular, layer-by-layer constructive neural network architectures and standard baselines:

1. `constructive_torch`:
   - `ConstructiveWrapper`: Universal progressive layer-by-layer wrapper for any standard PyTorch architecture.
   - `AutoConstructiveModel`: Autonomous layer-by-layer architecture search and tournament tuning.
   - `InSituHyperparameterTuner`: Online continuous hyperparameter auto-tuner optimized for maximum validation accuracy.
   - `ArchitecturalAutoTuner`: Micro-batch candidate tournament optimizer for stage configurations.
   - `ModelEMA`: Polyak adaptive warmup exponential moving average shadow weights.
   - `StageWrapper`: Stage partitioning and auxiliary probe attachment.

2. `core_model`:
   - `ConstructiveLayerwiseModel`: Constructive progressive MLP implementation for vector/tabular tasks.

3. `lenet5_disk_model`:
   - `DiskConstructiveLeNet5`: Disk-streamed chunked LeNet-5 with asynchronous prefetching.
   - `ChunkedFeatureMapStore`: Bounded ~2 MB binary serialization manager.
   - `PrefetchedChunkStreamer`: Double-buffered background chunk prefetching engine.

4. `standard_lenet5`:
   - `StandardLeNet5`: Standard end-to-end LeNet-5 baseline.
   - `StandardLeNet5Trainer`: Backpropagation trainer baseline.

5. `standard_mlp`:
   - `StandardMLPTrainer`: Standard end-to-end MLP trainer baseline.
"""

from core_models.constructive_torch import (
    ConstructiveWrapper,
    AutoConstructiveModel,
    StageWrapper,
    InSituHyperparameterTuner,
    ArchitecturalAutoTuner,
    ModelEMA,
    PrefetchedChunkStreamer as WrapperPrefetchedChunkStreamer,
    ChunkedFeatureStore,
)
from core_models.core_model import (
    ConstructiveLayerwiseModel,
    SingleConstructiveLayer,
)
from core_models.lenet5_disk_model import (
    DiskConstructiveLeNet5,
    ChunkedFeatureMapStore,
    PrefetchedChunkStreamer,
)
from core_models.standard_lenet5 import (
    StandardLeNet5,
    StandardLeNet5Trainer,
)
from core_models.standard_mlp import (
    StandardMLPTrainer,
    StandardMLP,
)

__all__ = [
    "ConstructiveWrapper",
    "AutoConstructiveModel",
    "StageWrapper",
    "InSituHyperparameterTuner",
    "ArchitecturalAutoTuner",
    "ModelEMA",
    "ConstructiveLayerwiseModel",
    "SingleConstructiveLayer",
    "DiskConstructiveLeNet5",
    "ChunkedFeatureMapStore",
    "PrefetchedChunkStreamer",
    "StandardLeNet5",
    "StandardLeNet5Trainer",
    "StandardMLPTrainer",
    "StandardMLP",
]
