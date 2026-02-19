from ultralytics import YOLO

# Dataset directory (update if needed)
DATASET_PATH = "yolo_weather"   # Folder containing train/val/test

# Load YOLOv8 Nano Classification Model
model = YOLO("yolov8n-cls.pt")

# Train the model
model.train(
    data=DATASET_PATH,
    epochs=10,
    imgsz=224,
    batch=16,
    device="cpu"   # Change to 0 if GPU available
)

print("✅ Training Completed!")
