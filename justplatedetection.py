import os
import cv2
import numpy as np
from anpr import ObjectDetectionProcessor

processor = ObjectDetectionProcessor()
image_folder_path = 'C:/Users/Onur/Desktop/ANPR/Tensorflow/workspace/images/trr'

for image_filename in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_filename)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error reading image: {image_filename}")
        continue

    image_np = np.array(img)

    target_size = (800, 600)
    resized_image_np = cv2.resize(image_np, target_size)

    detections = processor.detect_objects(resized_image_np)
    print("All detections:", detections)


    for i in range(len(detections['detection_boxes'])):
        box = detections['detection_boxes'][i]
        class_id = detections['detection_classes'][i]
        confidence = detections['detection_scores'][i]

        if confidence > 0.6:
            h, w, _ = resized_image_np.shape
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)
            cv2.rectangle(resized_image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Displaying bounding box coordinates, class scores, class labels, and detection confidence
            text = f'Bounding Boxes: ({xmin}, {ymin}), ({xmax}, {ymax}), Score: {confidence:.2f}'
            cv2.putText(resized_image_np, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Plate Detection Results', resized_image_np)
    cv2.waitKey(0)

cv2.destroyAllWindows()
