from ultralytics import YOLO

model = YOLO('yolov8l.pt')


results = model.train(data='datasets/yolo_gestures/sensoryart.yaml', epochs=100, imgsz=384)