import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
import torch.nn as nn
from constructive_torch import ConstructiveWrapper, InSituHyperparameterTuner

def test_tuner_accuracy_metric():
    print("Testing InSituHyperparameterTuner with accuracy-first objective...")
    # Create simple classifier
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 5))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    tuner = InSituHyperparameterTuner(
        module=model,
        optimizer=opt,
        criterion=crit,
        task="classification",
        stage_idx=0,
        objective="max_accuracy",
    )
    
    # Generate test probe batches
    torch.manual_seed(42)
    tx = torch.randn(64, 20)
    ty = torch.randint(0, 5, (64,))
    vx = torch.randn(64, 20)
    vy = torch.randint(0, 5, (64,))
    
    rec = tuner.step_epoch(
        epoch=1,
        probe_train_x=tx,
        probe_train_y=ty,
        probe_val_x=vx,
        probe_val_y=vy,
        verbose=True
    )
    print("Step 1 record:", rec)
    assert "chosen_lr" in rec
    print("✅ InSituHyperparameterTuner max_accuracy test passed successfully!")

if __name__ == "__main__":
    test_tuner_accuracy_metric()
