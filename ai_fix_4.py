import cv2
import os

# Constants
DATASET_PATH = 'path/to/ultrank_11_20'
LABELS_PATH = 'path/to/labels'
MODEL_BOXES_PATH = 'path/to/model_boxes'
KEY_A = ord('a')
KEY_D = ord('d')
KEY_N = ord('n')
KEY_ENTER = ord('\r')

# Load model boxes
def load_model_boxes():
    with open(MODEL_BOXES_PATH, 'r') as file:
        model_boxes = [line.strip().split() for line in file]
    return model_boxes

# Save labels
def save_labels(frame_index, labels):
    label_file = os.path.join(LABELS_PATH, f'frame_{frame_index}.txt')
    with open(label_file, 'w') as file:
        for label in labels:
            file.write(' '.join(label) + '\n')

# Main labeling function
def label_frames(num_frames):
    model_boxes = load_model_boxes()
    for frame_index in range(num_frames):
        frame_path = os.path.join(DATASET_PATH, f'frame_{frame_index}.jpg')
        frame = cv2.imread(frame_path)
        labels = []
        
        while True:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0) & 0xFF
            
            if key == KEY_A:
                for box in model_boxes:
                    labels.append(box)
            elif key == KEY_D:
                # Draw new box logic here
                pass
            elif key == KEY_N or key == KEY_ENTER:
                save_labels(frame_index, labels)
                break
        
        cv2.destroyAllWindows()

# Run the labeling process
if __name__ == '__main__':
    label_frames(250)