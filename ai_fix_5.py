import cv2
import numpy as np

def load_model_boxes(frame_path):
    # Placeholder function to load model boxes
    # Replace this with actual logic to load model boxes
    return np.array([[100, 100, 200, 200], [300, 300, 400, 400]])

def draw_boxes(frame, boxes):
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return frame

def label_frame(frame_path, model_boxes):
    frame = cv2.imread(frame_path)
    frame = draw_boxes(frame, model_boxes)
    cv2.imshow('Label Frame', frame)
    key = cv2.waitKey(0)
    if key == ord('a'):
        # Adopt model boxes and fix inaccuracies
        pass
    elif key == ord('d'):
        # Draw labels directly
        pass
    cv2.destroyAllWindows()
    return frame

def main():
    num_frames = 250
    for i in range(num_frames):
        frame_path = f'frames/frame_{i}.jpg'
        model_boxes = load_model_boxes(frame_path)
        labeled_frame = label_frame(frame_path, model_boxes)
        cv2.imwrite(f'labeled_frames/labeled_frame_{i}.jpg', labeled_frame)
    print("All frames labeled.")

if __name__ == "__main__":
    main()