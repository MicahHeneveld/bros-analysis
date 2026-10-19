import os
import cv2
import numpy as np

def load_model_boxes(model_path):
    # Placeholder function to load model's bounding boxes
    # Replace with actual implementation
    return np.array([[100, 100, 200, 200], [300, 300, 400, 400]])

def draw_boxes(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return image

def label_frames(dataset_path, model_boxes, output_path):
    frames = os.listdir(dataset_path)
    for frame in frames:
        frame_path = os.path.join(dataset_path, frame)
        image = cv2.imread(frame_path)
        if image is None:
            continue
        
        labeled_image = draw_boxes(image, model_boxes)
        output_frame_path = os.path.join(output_path, frame)
        cv2.imwrite(output_frame_path, labeled_image)

if __name__ == "__main__":
    dataset_path = "path/to/ultrank_11_20"
    model_boxes = load_model_boxes("path/to/model_boxes.npy")
    output_path = "path/to/labeled_frames"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    label_frames(dataset_path, model_boxes, output_path)