import os
import cv2
import numpy as np

def load_model_boxes(frame_path):
    # Placeholder function to load model boxes
    # Replace with actual implementation
    return np.array([[100, 100, 200, 200], [300, 300, 400, 400]])

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return image

def label_frames(frame_paths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        model_boxes = load_model_boxes(frame_path)
        labeled_frame = draw_boxes(frame.copy(), model_boxes)

        cv2.imshow('Labeled Frame', labeled_frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('a'):
            # Adopt model boxes and manually adjust
            # Placeholder for manual adjustment logic
            pass
        elif key == ord('d'):
            # Draw labels manually
            # Placeholder for manual drawing logic
            pass

        cv2.imwrite(os.path.join(output_dir, os.path.basename(frame_path)), labeled_frame)

if __name__ == "__main__":
    frame_paths = [f"path/to/frame_{i}.jpg" for i in range(250)]
    output_dir = "labeled_frames"
    label_frames(frame_paths, output_dir)