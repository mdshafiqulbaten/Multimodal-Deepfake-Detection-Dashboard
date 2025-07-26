import cv2
import numpy as np
import tempfile
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Load a pre-trained ResNet (you can replace with better deepfake models later)
model = resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classifier
model.eval()

# Dummy weights for illustration (replace with trained model)
# torch.load('face_deepfake_model.pth') # optional if you train

# Face detection (Haar cascade or Dlib could be used too)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Transform for input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def detect_video(video_file):
    score_sum = 0
    frame_count = 0

    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Sample every 15th frame
        if frame_count % 15 != 0:
            continue

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_tensor = transform(face).unsqueeze(0)
            with torch.no_grad():
                output = model(face_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                deepfake_prob = probs[0][1].item()
                score_sum += deepfake_prob

        if frame_count > 150:  # Limit frames for speed
            break

    cap.release()
    if frame_count == 0:
        return 0.0
    return round(score_sum / (frame_count // 15), 2)
