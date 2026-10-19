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
    frame = draw_boxes(frame, model_boxes)
    cv2.imshow('Label Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return frame

def main():
    dataset_path = 'path_to_dataset'
    num_frames = 250
    for i in range(num_frames):
        frame_path = f'{dataset_path}/frame_{i}.jpg'
        model_boxes = load_model_boxes(frame_path)
        labeled_frame = label_frame(frame_path, model_boxes)
        cv2.imwrite(f'{dataset_path}/labeled_frame_{i}.jpg', labeled_frame)

if __name__ == '__main__':
    main()