from ultralytics import YOLO

# Load a pre-trained YOLO model (adjust model type as needed)
model = YOLO("runs/detect/yolov11_obstacle_train2/weights/best.pt")  # n, s, m, l, x versions available

# Perform object detection on an image
results = model.predict(source="image_87_87.jpg",save=True)  # Can also use video, directory, URL, etc.

# Display the results
results[0].show()  # Show the first image results