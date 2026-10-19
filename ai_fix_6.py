import cv2
import numpy as np

def label_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Frame', frame)

        # Wait for a key press
        key = cv2.waitKey(0) & 0xFF

        if key == ord('a'):
            # Adopt model's boxes (placeholder for actual model boxes)
            model_boxes = get_model_boxes(frame)
            draw_boxes(frame, model_boxes)

        elif key == ord('d'):
            # Draw labels manually
            draw_boxes(frame, [])

        # Save the frame with labels (placeholder for actual saving logic)
        save_frame_with_labels(frame)

    cap.release()
    cv2.destroyAllWindows()

def get_model_boxes(frame):
    # Placeholder function to get model boxes
    # Replace with actual model inference logic
    return []

def draw_boxes(frame, boxes):
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

def save_frame_with_labels(frame):
    # Placeholder function to save frame with labels
    # Replace with actual saving logic
    pass

# Example usage
video_path = 'path_to_video.mp4'
num_frames = 250
label_frames(video_path, num_frames)