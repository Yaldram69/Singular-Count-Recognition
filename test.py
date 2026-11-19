from ultralytics import YOLO
import cv2
from collections import Counter
import matplotlib.pyplot as plt




model_path='yolov8m.pt'

def process_image(image_path, model_path, output_resolution=(640, 480)):
    """
    Detect objects in an image, count classes, and display the annotated image.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Load and resize the image
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, output_resolution)

    # Run detection
    results = model.predict(image_resized, conf=0.5)

    # Count detected classes
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
    class_counts = Counter(detected_classes)

    # Annotate image
    annotated_image = results[0].plot()
    start_x, start_y = 10, 30
    for i, (cls, count) in enumerate(class_counts.items()):
        text = f"{cls} = {count}"
        position = (start_x, start_y + i * 20)
        cv2.putText(
            annotated_image, text, position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )

    # Display annotated image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return class_counts


def process_video(video_path, output_path, model_path, output_resolution=(640, 480)):
    """
    Detect objects in a video, count classes, and save the annotated video.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Define output video writer
    frame_width, frame_height = output_resolution
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame_resized = cv2.resize(frame, (frame_width, frame_height))

        # Run detection
        results = model.predict(frame_resized, conf=0.5)

        # Count detected classes
        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
        class_counts = Counter(detected_classes)

        # Annotate frame
        annotated_frame = results[0].plot()
        start_x, start_y = 10, 30
        for i, (cls, count) in enumerate(class_counts.items()):
            text = f"{cls} = {count}"
            position = (start_x, start_y + i * 20)
            cv2.putText(
                annotated_frame, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )

        # Write frame to output video
        out.write(annotated_frame)

        # Optionally display the frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing complete. Saved to:", output_path)

image_counts = process_image('obj.jpg', model_path, output_resolution=(640, 480))
print("Image Class Counts:", image_counts)