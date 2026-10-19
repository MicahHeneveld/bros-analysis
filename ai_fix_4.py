import cv2
import numpy as np

def load_model_boxes(frame_path):
    # Placeholder function to load model boxes
    # Replace this with actual logic to load model boxes
    return np.array([[100, 100, 200, 200], [300, 300, 400, 400]])

def draw_boxes(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return image

def label_frame(frame_path, model_boxes):
    image = cv2.imread(frame_path)
    image_with_boxes = draw_boxes(image, model_boxes)
    cv2.imshow('Label Frame', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image_with_boxes

def main():
    dataset_path = 'path_to_dataset'
    frames = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    for frame in frames[:250]:
        frame_path = os.path.join(dataset_path, frame)
        model_boxes = load_model_boxes(frame_path)
        labeled_frame = label_frame(frame_path, model_boxes)
        # Save the labeled frame
        cv2.imwrite(os.path.join(dataset_path, f'labeled_{frame}'), labeled_frame)

if __name__ == '__main__':
    main()