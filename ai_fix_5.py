import cv2
import os

# Constants
DATASET_PATH = 'path/to/ultrank_11_20'
LABELS_PATH = 'path/to/labels'
MODEL_BBOXES_PATH = 'path/to/model_bboxes'
NUM_FRAMES = 250

# Load model bounding boxes
def load_model_bboxes():
    model_bboxes = {}
    with open(MODEL_BBOXES_PATH, 'r') as f:
        for line in f:
            frame, bbox = line.strip().split(',')
            model_bboxes[frame] = eval(bbox)
    return model_bboxes

# Save labels
def save_labels(frame, labels):
    with open(os.path.join(LABELS_PATH, f'{frame}.txt'), 'w') as f:
        for label in labels:
            f.write(f'{label}\n')

# Main labeling function
def label_frames():
    model_bboxes = load_model_bboxes()
    for i in range(1, NUM_FRAMES + 1):
        frame_path = os.path.join(DATASET_PATH, f'frame_{i}.jpg')
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f'Frame {i} not found')
            continue
        
        labels = []
        bbox = model_bboxes.get(f'frame_{i}', [])
        for b in bbox:
            x, y, w, h = b
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Frame', frame)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):
                for b in bbox:
                    labels.append(f'{b[0]},{b[1]},{b[2]},{b[3]}')
                break
            elif key == ord('d'):
                x, y, w, h = cv2.selectROI('Frame', frame, fromCenter=False, showCrosshair=True)
                labels.append(f'{x},{y},{w},{h}')
                cv2.destroyWindow('ROI selector')
            elif key == ord('q'):
                break
        
        save_labels(f'frame_{i}', labels)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    label_frames()