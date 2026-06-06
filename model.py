import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2

# ================= CONFIG =================
SEQUENCE_LEN = 16
MODEL_PATH = "model_output/accident_model.pth"

CLASS_NAMES = {
    0: "NO ACCIDENT",
    1: "ACCIDENT"
}

DEVICE = torch.device("cpu")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= MODEL =================
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.cnn = backbone

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        features = self.cnn(x)
        features = features.view(b, t, 512)

        lstm_out, _ = self.lstm(features)
        out = lstm_out[:, -1, :]

        return self.classifier(out)


# ================= LOAD MODEL =================
model = CNNLSTM().to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model Loaded Successfully")


# ================= IMAGE PREDICT =================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    frames = torch.stack([image] * SEQUENCE_LEN)
    frames = frames.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(frames)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return CLASS_NAMES[pred.item()], float(conf.item() * 100)


# ================= VIDEO PREDICT =================
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return "ERROR", 0.0

    if len(frames) >= SEQUENCE_LEN:
        frames = frames[-SEQUENCE_LEN:]
    else:
        last = frames[-1]
        while len(frames) < SEQUENCE_LEN:
            frames.append(last)

    frames = torch.stack(frames).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(frames)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return CLASS_NAMES[pred.item()], float(conf.item() * 100)