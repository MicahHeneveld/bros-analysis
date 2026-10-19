import cv2
import os

def adopt_and_fix_bounding_boxes(image_path, model_boxes):
    image = cv2.imread(image_path)
    for box in model_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_new_bounding_boxes(image_path):
    image = cv2.imread(image_path)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Image', draw_rectangle)
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
                cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Image', image)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
            boxes.append((ix, iy, x - ix, y - iy))
            cv2.imshow('Image', image)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return boxes

def label_frames(dataset_path, num_frames):
    for i in range(num_frames):
        image_path = os.path.join(dataset_path, f'frame_{i}.jpg')
        model_boxes = get_model_boxes(image_path)  # Replace with actual model box retrieval logic
        if model_boxes:
            adopt_and_fix_bounding_boxes(image_path, model_boxes)
        else:
            boxes = draw_new_bounding_boxes(image_path)
            save_labels(image_path, boxes)  # Replace with actual label saving logic

def get_model_boxes(image_path):
    # Replace with actual model box retrieval logic
    return [(100, 100, 50, 50), (200, 200, 30, 30)]  # Example model boxes

def save_labels(image_path, boxes):
    # Replace with actual label saving logic
    with open(image_path.replace('.jpg', '.txt'), 'w') as f:
        for box in boxes:
            f.write(f'{box[0]} {box[1]} {box[2]} {box[3]}\n')

if __name__ == '__main__':
    dataset_path = 'path_to_dataset'
    num_frames = 250
    label_frames(dataset_path, num_frames)