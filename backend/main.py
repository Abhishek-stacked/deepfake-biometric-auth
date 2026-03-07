import os
from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "frontend", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "..", "frontend", "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 2)

model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "face_anti_spoof", "deepfake_model.pth"),
        map_location="cpu"
    )
)

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["video"]
    filename = secure_filename(file.filename)

    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    fake_probs = []
    frame_paths = []

    count = 0
    max_frames = 30

    while count < max_frames:

        success, frame = cap.read()

        if not success:
            break

        # center crop for better face focus
        h, w, _ = frame.shape
        crop = frame[h//4:3*h//4, w//4:3*w//4]

        frame_name = f"{filename}_frame_{count}.jpg"
        frame_path = os.path.join(UPLOAD_FOLDER, frame_name)

        cv2.imwrite(frame_path, crop)

        frame_paths.append("/static/uploads/" + frame_name)

        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0)

        with torch.no_grad():

            output = model(image)
            prob = torch.softmax(output, dim=1)

            fake_prob = prob[0][1].item()
            fake_probs.append(fake_prob)

        count += 1

    cap.release()

    if len(fake_probs) == 0:
        return "Could not process video"

    avg_fake_prob = sum(fake_probs) / len(fake_probs)
    real_prob = 1 - avg_fake_prob

    if avg_fake_prob > 0.5:
        result = "FAKE"
        confidence = avg_fake_prob
    else:
        result = "REAL"
        confidence = real_prob

    return render_template(
        "index.html",
        prediction=result,
        fake_prob=round(avg_fake_prob * 100, 2),
        real_prob=round(real_prob * 100, 2),
        confidence=round(confidence * 100, 2),
        frames=frame_paths
    )


if __name__ == "__main__":
    app.run(debug=True)