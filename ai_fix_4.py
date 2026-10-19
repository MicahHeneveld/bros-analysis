import cv2
import numpy as np

def load_model_boxes(frame_path):
    # Placeholder function to load model boxes
    # Replace this with actual model box loading logic
    return np.array([[100, 100, 200, 200], [300, 300, 400, 400]])

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return image

def label_frame(frame_path, model_boxes):
    frame = cv2.imread(frame_path)
    frame_with_boxes = draw_boxes(frame.copy(), model_boxes)
    cv2.imshow('Label Frame', frame_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    num_frames = 250
    for i in range(num_frames):
        frame_path = f'path/to/frame_{i}.jpg'
        model_boxes = load_model_boxes(frame_path)
        label_frame(frame_path, model_boxes)
    print("All frames labeled.")

if __name__ == "__main__":
    main()