from ultralytics import YOLO

def load_model(weights_path="yolov8n-cls.pt"):
    """
    Loads YOLOv8 classification model.
    """
    model = YOLO(weights_path)
    return model


if __name__ == "__main__":
    model = load_model()
    print("YOLOv8 Nano Classification Model Loaded Successfully!")
