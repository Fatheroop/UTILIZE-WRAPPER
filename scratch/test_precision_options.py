import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
import torch.nn as nn
from constructive_torch import ConstructiveWrapper, AutoConstructiveModel

def test_precision_options():
    print("Testing Precision Options (float32, float16, bfloat16, amp_dtype)...")
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
    
    # 1. Test precision='float16'
    w_fp16 = ConstructiveWrapper(model=model, in_shape=(10,), num_classes=2, precision="float16")
    assert w_fp16.activation_dtype == torch.float16, f"Expected float16, got {w_fp16.activation_dtype}"
    assert w_fp16.activation_precision == "float16"
    print("✅ precision='float16' verified.")
    
    # 2. Test precision='bfloat16'
    w_bf16 = ConstructiveWrapper(model=model, in_shape=(10,), num_classes=2, precision="bfloat16")
    assert w_bf16.activation_dtype == torch.bfloat16, f"Expected bfloat16, got {w_bf16.activation_dtype}"
    assert w_bf16.activation_precision == "bfloat16"
    print("✅ precision='bfloat16' verified.")

    # 3. Test amp_dtype='bfloat16'
    w_amp = ConstructiveWrapper(model=model, in_shape=(10,), num_classes=2, use_amp=True, amp_dtype="bfloat16")
    assert w_amp.amp_dtype_val == torch.bfloat16
    print("✅ amp_dtype='bfloat16' verified.")

    # 4. Test AutoConstructiveModel precision forwarding
    auto_m = AutoConstructiveModel(in_shape=(10,), num_classes=2, precision="float16", auto_tune_objective="max_accuracy")
    assert auto_m.activation_precision == "float16"
    assert auto_m.auto_tune_objective == "max_accuracy"
    print("✅ AutoConstructiveModel precision and max_accuracy forwarding verified.")

    # 5. Fit a quick micro-batch with precision='float16' and max_accuracy tuner
    w_fit = ConstructiveWrapper(model=model, in_shape=(10,), num_classes=2, precision="float16", max_epochs_per_stage=1, fine_tune_epochs=0)
    X = torch.randn(40, 10)
    y = torch.randint(0, 2, (40,))
    w_fit.fit(X, y, verbose=False)
    preds = w_fit.predict(X)
    assert len(preds) == 40
    print("✅ fit() with precision='float16' and max_accuracy tuner succeeded!")

if __name__ == "__main__":
    test_precision_options()
