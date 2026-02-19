from ultralytics import YOLO

# Load trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Export to ONNX
model.export(format="onnx", imgsz=224)

print("✅ ONNX model exported successfully!")
