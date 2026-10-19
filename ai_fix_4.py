import os
import sys
import cv2
import numpy as np

def load_model_boxes(model_path):
    # Placeholder function to load model boxes
    # Replace with actual implementation
    return np.array([[100, 100, 200, 200], [300, 300, 400, 400]])

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return image

def label_frames(dataset_path, model_boxes_path, output_path):
    model_boxes = load_model_boxes(model_boxes_path)
    frames = os.listdir(dataset_path)
    frames.sort()

    for frame_name in frames:
        frame_path = os.path.join(dataset_path, frame_name)
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Error: Unable to read frame {frame_name}")
            continue

        labeled_image = draw_boxes(image, model_boxes)
        cv2.imshow('Label Frame', labeled_image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('a'):
            # Adopt all model boxes
            pass
        elif key == ord('d'):
            # Draw new boxes
            pass
        elif key == ord('s'):
            # Save the labeled image
            output_frame_path = os.path.join(output_path, frame_name)
            cv2.imwrite(output_frame_path, labeled_image)
            print(f"Saved {output_frame_path}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python label_frames.py <dataset_path> <model_boxes_path> <output_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    model_boxes_path = sys.argv[2]
    output_path = sys.argv[3]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    label_frames(dataset_path, model_boxes_path, output_path)