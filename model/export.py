import torch
import torch.nn as nn
from torchvision import models
import os

DEVICE    = torch.device("cpu")
MODEL_IN  = "model.pth"
MODEL_OUT = "model.onnx"

# Rebuild exact same architecture as train.py
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_IN, map_location=DEVICE))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

# Use the modern export API for PyTorch 2.x
torch.onnx.export(
    model,
    dummy_input,
    MODEL_OUT,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=18,       # use 18 instead of 11
    dynamo=False            # disable dynamo, use legacy exporter
)

size_mb = round(os.path.getsize(MODEL_OUT) / 1024 / 1024, 1)
print(f"Model exported to {MODEL_OUT}")
print(f"File size: {size_mb} MB")

if size_mb < 5:
    print("WARNING: File seems too small, something may have gone wrong")
else:
    print("Export looks good!")