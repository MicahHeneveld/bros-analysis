import cv2
import numpy as np

def load_model_boxes(frame_path):
    # Placeholder function to load model boxes
    # Replace this with actual model inference code
    return np.array([[100, 100, 200, 200], [300, 300, 400, 400]])

def draw_boxes(frame, boxes):
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return frame

def label_frame(frame_path):
    frame = cv2.imread(frame_path)
    model_boxes = load_model_boxes(frame_path)
    frame = draw_boxes(frame, model_boxes)
    
    cv2.imshow('Label Frame', frame)
    key = cv2.waitKey(0)
    
    if key == ord('a'):
        # Adopt all model boxes
        pass
    elif key == ord('d'):
        # Draw new boxes
        pass
    
    cv2.destroyAllWindows()
    return frame

def main():
    for i in range(250):
        frame_path = f'ultrank_11_20/frame_{i}.jpg'
        labeled_frame = label_frame(frame_path)
        cv2.imwrite(f'ultrank_11_20/labeled_frame_{i}.jpg', labeled_frame)

if __name__ == '__main__':
    main()