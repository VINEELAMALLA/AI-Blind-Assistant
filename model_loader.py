import torch
from ultralytics import YOLO
from models.zero_dce import ZeroDCE

# Load YOLO model for object detection
def load_yolo():
    return YOLO("models/yolov8n.pt")

# Load Zero-DCE model for low-light enhancement
def load_zero_dce():
    model = ZeroDCE()
    model.load_state_dict(torch.load("models/zero_dce.pth", map_location="cpu"))
    model.eval()
    return model

# Load Face Analysis model for age, gender, and emotion detection
def load_face_model():
    model = torch.load("models/face_model.pth", map_location="cpu")
    model.eval()
    return model
