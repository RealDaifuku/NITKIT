from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load YOLO model
model = YOLO("runs/detect/train-12/weights/best.pt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    # Remove header
    image_data = data.split(",")[1]

    # Decode base64 image
    image_bytes = base64.b64decode(image_data)

    # Convert to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # YOLO prediction
    results = model(frame)

    # Draw boxes
    annotated_frame = results[0].plot()

    # Encode back to jpg
    _, buffer = cv2.imencode(".jpg", annotated_frame)

    # Convert to base64
    processed_image = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "image": processed_image
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)