import cv2
import numpy as np

def load_model():
    # Placeholder for loading the model
    return None

def draw_bounding_boxes(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    return image

def label_frame(frame, model_boxes):
    cv2.imshow('Frame', draw_bounding_boxes(frame, model_boxes))
    key = cv2.waitKey(0)
    if key == ord('a'):
        return True, model_boxes
    elif key == ord('d'):
        new_box = cv2.selectROI('Frame', frame, fromCenter=False, showCrosshair=True)
        return True, np.vstack((model_boxes, new_box))
    return False, model_boxes

def main():
    model = load_model()
    cap = cv2.VideoCapture('ultrank_11_20.mp4')
    frame_count = 0
    all_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 250 == 0:
            model_boxes = model.predict(frame)
            continue
        adopt, boxes = label_frame(frame, model_boxes)
        if adopt:
            all_boxes.append(boxes)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return all_boxes

if __name__ == "__main__":
    boxes = main()
    # Save the boxes to a file or database
    np.save('labeled_boxes.npy', boxes)