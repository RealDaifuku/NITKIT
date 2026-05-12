from ultralytics import YOLO

if __name__ == "__main__":
    # Load model
    model = YOLO("yolo26n.pt")

    # Train
    train_results = model.train(
        data="data.yaml",
        epochs=500,
        imgsz=640,
        device=0,
    )