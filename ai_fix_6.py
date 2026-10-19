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

def label_frame(frame_path, model_boxes):
    frame = cv2.imread(frame_path)
    frame = draw_boxes(frame, model_boxes)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(0)
    if key == ord('a'):
        return model_boxes
    elif key == ord('d'):
        return draw_boxes(frame, [], color=(0, 0, 255), thickness=2)
    return None

def main():
    dataset_path = 'path_to_dataset'
    frames = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    for frame in frames:
        frame_path = os.path.join(dataset_path, frame)
        model_boxes = load_model_boxes(frame_path)
        labeled_boxes = label_frame(frame_path, model_boxes)
        if labeled_boxes is not None:
            # Save the labeled boxes to a file or database
            pass

if __name__ == '__main__':
    main()