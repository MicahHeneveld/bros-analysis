import os
import sys
import cv2
import numpy as np

def claim_bounty(number):
    with open('bounty.txt', 'w') as f:
        f.write(f'Claimed {number} frames for labeling.')

def start_labeling(number):
    claim_bounty(number)
    for i in range(number):
        frame_path = f'frames/frame_{i}.jpg'
        if not os.path.exists(frame_path):
            print(f'Frame {i} does not exist.')
            continue
        frame = cv2.imread(frame_path)
        model_boxes = get_model_boxes(frame)  # Replace with actual model box retrieval logic
        cv2.imshow('Frame', frame)
        while True:
            key = cv2.waitKey(0)
            if key == ord('a'):
                label_frame(frame, model_boxes)
                break
            elif key == ord('d'):
                label_frame(frame, draw_boxes=True)
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
        cv2.imwrite(f'labels/label_{i}.jpg', frame)

def get_model_boxes(frame):
    # Replace with actual model box retrieval logic
    return []

def label_frame(frame, model_boxes=None, draw_boxes=False):
    if draw_boxes:
        for box in model_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    else:
        # Replace with actual label drawing logic
        pass

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python scripts/bounty.py start <number>')
        sys.exit(1)
    number = int(sys.argv[1])
    start_labeling(number)