import torch
import coremltools as ct
from ultralytics import YOLO
import copy

# 1. Load Model
print("Loading model...")
yolo = YOLO("yolov8n-face.pt")
model = copy.deepcopy(yolo.model)
model.eval()
model.fuse()

# 2. Define Wrapper to Insert Dummy Class
class PaddedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
            
        # out shape: (1, 20, 8400)
        # Structure: Box(4) | Conf(1) | Kpts(15)
        # We want: Box(4) | Conf(1) | Dummy(1) | Kpts(15) -> (1, 21, 8400)
        
        # Split
        part1 = out[:, :5, :] # Box + Conf
        part2 = out[:, 5:, :] # Kpts
        
        # Dummy channel (-inf to ensure not selected)
        dummy = torch.zeros_like(out[:, 0:1, :]) - 1e9
        
        return torch.cat([part1, dummy, part2], dim=1)

# 3. Trace
print("Tracing model...")
wrapper = PaddedModel(model)
wrapper.eval()
dummy_input = torch.zeros(1, 3, 640, 640)
traced_model = torch.jit.trace(wrapper, dummy_input, strict=False, check_trace=False)

# 4. Convert
print("Converting to CoreML...")
# Metadata is crucial for Ultralytics AutoBackend to detect task/stride/names
metadata = {
    "task": "pose",
    "stride": "32",
    "names": "{0: 'face', 1: 'dummy'}",
    "kpt_shape": "[5, 3]",
    "imgsz": "[640, 640]",
    "nc": "2",
    "batch": "1"
    # Additional metadata usually added by Exporter
}

model_ct = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 640, 640), scale=1/255.0)],
    outputs=[ct.TensorType(name="output0")], # Output (1, 21, 8400)
    convert_to="mlprogram"
)

# 5. Add Metadata
model_ct.user_defined_metadata.update(metadata)
model_ct.short_description = "YOLOv8n-Face Padded for CoreML"

# 6. Save
output_path = "yolov8n-face-custom.mlpackage"
model_ct.save(output_path)
print(f"Saved to {output_path}")
