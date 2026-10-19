import os
import cv2
import numpy as np

def adopt_model_boxes(frame, model_boxes):
    for box in model_boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def draw_new_boxes(frame):
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('frame', draw_rectangle)
    drawing = False
    ix, iy = -1, -1
    boxes = []

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, boxes
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
            boxes.append((ix, iy, x - ix, y - iy))

    while True:
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('d'):
            frame = draw_new_boxes(frame)
        elif k == ord('s'):
            break
    cv2.destroyAllWindows()
    return boxes

def label_frames(start_frame, end_frame):
    for frame_number in range(start_frame, end_frame + 1):
        frame_path = f'dataset/ultrank_11_20/frame_{frame_number}.jpg'
        frame = cv2.imread(frame_path)
        model_boxes = get_model_boxes(frame_number)  # Replace with actual model box retrieval logic
        if model_boxes:
            frame = adopt_model_boxes(frame, model_boxes)
        else:
            frame = draw_new_boxes(frame)
        cv2.imwrite(f'dataset/ultrank_11_20/labeled/frame_{frame_number}.jpg', frame)

def get_model_boxes(frame_number):
    # Replace with actual model box retrieval logic
    return [(100, 100, 50, 50), (200, 200, 30, 30)]  # Example model boxes

if __name__ == '__main__':
    start_frame = 1
    end_frame = 250
    label_frames(start_frame, end_frame)