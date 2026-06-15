from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from ollama import chat
import cv2
import numpy as np
import base64

app = Flask(__name__)


#train12 for daifuku
#train16 for 3Dprints
model = YOLO("runs/detect/train-16/weights/best.pt")


# HOME PAGE
@app.route("/")
def home():
    return render_template("home.html")


# # REALTIME PAGE
# @app.route("/realtime")
# def realtime():
#     return render_template("realtime.html")


# PHOTO PAGE
@app.route("/photo")
def photo():
    return render_template("photo.html")


# # PHOTO PAGE
# @app.route("/counter")
# def counter():
#     return render_template("counter.html")

@app.route("/chat", methods=["POST"])
def chat_with_darek():

    prompt = request.json["prompt"]

    response = chat(
        model="darek:latest",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return jsonify({
        "response":
        response["message"]["content"]
    })


# YOLO DETECTION API
@app.route("/detect", methods=["POST"])
def detect():

    data = request.json["image"]

    image_data = data.split(",")[1]

    image_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame)

    # CHECK DETECTION
    detected = len(results[0].boxes) > 0

    label = "No Match"
    llm_response = ""

    if detected:

        cls_id = int(results[0].boxes[0].cls[0])

        label = model.names[cls_id]

        if label == "crack":

            label = "crack"

            prompt = """
            Crack was detected in the 3D printed object

            React like an AI assistant and give trouble shooting advice to fix cracks in 3D prints.
            Suggest troubleshooting steps, printer configuration parameters.

            Keep the response under 5 sentences
            """

        elif label == "spaghetti":

            label = "spaghetti"

            prompt = """
            Spaghetti was detected in the 3D printed object

            React like an AI assistant and give trouble shooting advice to fix spaghetti in 3D prints.
            Suggest troubleshooting steps, printer configuration parameters.

            Keep the response under 5 sentences
            """

        elif label == "stringing":

            label = "stringing"

            prompt = """
            Stringing was detected in the 3D printed object

            React like an AI assistant and give trouble shooting advice to fix stringing in 3D prints.
            Suggest troubleshooting steps, printer configuration parameters.

            Keep the response under 5 sentences
            """

    else:

        prompt = """
        No flaws were detected in the 3D printed object.

        Suggest the user that no flaws were detected.

        Keep the response under 2 setences
        """

    response = chat(
        model="darek:latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    llm_response = response["message"]["content"]

    # DRAW YOLO BOXES
    annotated = results[0].plot()

    # CONVERT TO JPG
    _, buffer = cv2.imencode(".jpg", annotated)

    # CONVERT TO BASE64
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "image": jpg_as_text,
        "detected": detected,
        "label": label,
        "llm_response": llm_response
    })

@app.route("/detect_upload", methods=["POST"])
def detect_upload():

    file = request.files["image"]

    image_bytes = file.read()

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    results = model(frame)

    detected = len(results[0].boxes) > 0

    label = "No Match"
    llm_response = ""

    if detected:

        labels = []

        for box in results[0].boxes:

            cls_id = int(box.cls[0])

            labels.append(
                model.names[cls_id]
            )

        print("Detected labels:", labels)

        if "crack" in labels:

            label = "crack"

            prompt = """
            Crack was detected in the 3D printed object

            React like an AI assistant and give trouble shooting advice to fix cracks in 3D prints.
            Suggest troubleshooting steps, printer configuration parameters.

            Keep the response under 5 sentences
            """

        elif "spaghetti" in labels:

            label = "spaghetti"

            prompt = """
            Spaghetti was detected in the 3D printed object

            React like an AI assistant and give trouble shooting advice to fix spaghetti in 3D prints.
            Suggest troubleshooting steps, printer configuration parameters.

            Keep the response under 5 sentences
            """

        elif "stringing" in labels:

            label = "stringing"

            prompt = """
            Stringing was detected in the 3D printed object

            React like an AI assistant and give trouble shooting advice to fix stringing in 3D prints.
            Suggest troubleshooting steps, printer configuration parameters.

            Keep the response under 5 sentences
            """

    else:

        prompt = """
        No flaws were detected in the 3D printed object.

        Suggest the user that no flaws were detected.

        Keep the response under 2 setences
        """

    try:

        response = chat(
            model="darek:latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        llm_response = response["message"]["content"]

    except Exception as e:

        print("OLLAMA ERROR:", e)

        llm_response = f"Ollama error: {str(e)}"

    # DRAW YOLO BOXES
    annotated = results[0].plot()

    # CONVERT IMAGE TO JPG
    _, buffer = cv2.imencode(
        ".jpg",
        annotated
    )

    # CONVERT TO BASE64
    jpg_as_text = base64.b64encode(
        buffer
    ).decode("utf-8")

    print("Label:", label)
    print("AI:", llm_response)

    return jsonify({
        "image": jpg_as_text,
        "detected": detected,
        "label": label,
        "llm_response": llm_response
    })


# COUNTER VARIABLES
object_count = 0
previous_x = None


# COUNTER API
@app.route("/detect_counter", methods=["POST"])
def detect_counter():

    global object_count
    global previous_x

    data = request.json["image"]

    image_data = data.split(",")[1]

    image_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame)

    annotated = frame.copy()

    h, w, _ = frame.shape

    # COUNTING LINE
    line_x = w // 2

    cv2.line(
        annotated,
        (line_x, 0),
        (line_x, h),
        (0,255,255),
        3
    )

    boxes = results[0].boxes

    for box in boxes:

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # DRAW BOX
        cv2.rectangle(
            annotated,
            (x1,y1),
            (x2,y2),
            (255,0,0),
            2
        )

        # DRAW CENTER
        cv2.circle(
            annotated,
            (center_x, center_y),
            5,
            (0,0,255),
            -1
        )

        # COUNT CROSSING
        if previous_x is not None:

            if previous_x < line_x and center_x >= line_x:
                object_count += 1

        previous_x = center_x

    # DISPLAY COUNT
    cv2.putText(
        annotated,
        f"Count: {object_count}",
        (30,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255,0,0),
        4
    )

    _, buffer = cv2.imencode(".jpg", annotated)

    jpg_as_text = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "image": jpg_as_text,
        "count": object_count
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)