from ultralytics import YOLO

# Load trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Validate model
metrics = model.val()

print("📊 Evaluation Metrics:")
print(metrics)

print("✅ Testing Completed!")
