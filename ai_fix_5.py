import cv2
import numpy as np

def load_model_boxes(frame_path):
    # Placeholder function to load model boxes
    # Replace this with actual logic to load model boxes
    return np.array([[100, 100, 200, 200], [300, 300, 400, 400]])

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return image

def draw_labels(image, labels, color=(0, 0, 255), thickness=2):
    for label in labels:
        cv2.putText(image, label, (label[0], label[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
    return image

def main():
    frame_path = 'path_to_frame.jpg'  # Replace with actual frame path
    model_boxes = load_model_boxes(frame_path)
    image = cv2.imread(frame_path)
    image_with_boxes = draw_boxes(image.copy(), model_boxes)

    cv2.imshow('Frame with Model Boxes', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Placeholder for manual label drawing logic
    manual_labels = []  # Replace with actual manual labels
    image_with_labels = draw_labels(image_with_boxes, manual_labels)

    cv2.imshow('Frame with Labels', image_with_labels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()