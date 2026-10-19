import os
import cv2
import numpy as np

# Function to load model predictions
def load_model_predictions(frame_path):
    # Placeholder function to load model predictions
    # Replace this with actual model prediction loading logic
    return np.array([[100, 100, 200, 200], [300, 300, 400, 400]])

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
    return image

# Function to save labels
def save_labels(frame_path, labels):
    # Placeholder function to save labels
    # Replace this with actual label saving logic
    with open(frame_path.replace('.jpg', '.txt'), 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

# Main function to label frames
def label_frames(start_frame, end_frame):
    for frame_number in range(start_frame, end_frame + 1):
        frame_path = f"path/to/frames/frame_{frame_number:04d}.jpg"
        image = cv2.imread(frame_path)
        model_boxes = load_model_predictions(frame_path)
        labeled_image = draw_bounding_boxes(image, model_boxes)
        cv2.imshow('Label Frame', labeled_image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'):
            labels = model_boxes.tolist()
        elif key == ord('d'):
            labels = []
            while True:
                cv2.imshow('Label Frame', labeled_image)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s'):
                    break
                elif key == ord('r'):
                    labeled_image = draw_bounding_boxes(image, model_boxes)
        save_labels(frame_path, labels)
    cv2.destroyAllWindows()

# Example usage
label_frames(1, 250)