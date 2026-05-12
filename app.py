from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Load model
model = YOLO("runs/detect/train-12/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()

        if not success:
            break

        # YOLO prediction
        results = model(frame)

        # Draw boxes
        annotated_frame = results[0].plot()

        # Encode frame as jpg
        ret, buffer = cv2.imencode('.jpg', annotated_frame)

        frame_bytes = buffer.tobytes()

        # Stream frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)