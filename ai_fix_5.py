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
    for frame in range(1, NUM_FRAMES + 1):
        frame_path = os.path.join(DATASET_PATH, f'frame_{frame}.jpg')
        if not os.path.exists(frame_path):
            print(f'Frame {frame} not found')
            continue
        
        img = cv2.imread(frame_path)
        if frame in model_bboxes:
            for bbox in model_bboxes[frame]:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        cv2.imshow('Label Frame', img)
        labels = []
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):
                # Adopt model's bounding boxes
                for bbox in model_bboxes[frame]:
                    labels.append(f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}')
                break
            elif key == ord('d'):
                # Draw new bounding boxes
                bbox = cv2.selectROI('Label Frame', img, fromCenter=False, showCrosshair=True)
                labels.append(f'{int(bbox[0])},{int(bbox[1])},{int(bbox[0] + bbox[2])},{int(bbox[1] + bbox[3])}')
                cv2.destroyROI('Label Frame')
                break
            elif key == ord('q'):
                # Quit without saving
                cv2.destroyAllWindows()
                return
        
        save_labels(frame, labels)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    label_frames()