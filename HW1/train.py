from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # pass any model type
    results = model.train(data='train.yaml', epochs=1, imgsz=640)

