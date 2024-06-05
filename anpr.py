import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import tensorflow as tf
import easyocr
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Model and label map configurations
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
LABEL_MAP_NAME = 'label_map.pbtxt'
DETECTION_THRESHOLD = 0.2
REGION_THRESHOLD = 0.1
OCR_CONFIDENCE_THRESHOLD = 0.2

# File paths and configurations
ANNOTATION_PATH = os.path.join('Tensorflow', 'workspace', 'annotations')
CHECKPOINT_PATH = os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME)
PIPELINE_CONFIG = os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config')
LABELMAP = os.path.join(ANNOTATION_PATH, LABEL_MAP_NAME)

# Object Detection Processing class
class ObjectDetectionProcessor:

    def __init__(self):
        # Set TensorFlow configurations
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # Start ThreadPoolExecutor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=8)
        # Load model configurations from pipeline config file
        configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG)
        # Build the detection model
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        # Restore model weights from checkpoint
        ckpt = tf.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-28')).expect_partial()
        # Initialize EasyOCR reader for text detection and recognition
        self.reader = easyocr.Reader(['en', 'tr'], detector='dbnet18')

    # Method to detect objects in an image
    def detect_objects(self, image_np):
        # Perform object detection on the input image
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        # Process detections to include only relevant information
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        return detections

    # Method to perform object detection using the model
    def detect_fn(self, image):
        # Preprocess the image using the model
        image, shapes = self.detection_model.preprocess(image)
        # Predict using the model
        prediction_dict = self.detection_model.predict(image, shapes)
        # Postprocess the predictions
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections


    def apply_ocr_to_detection(self, image_np_with_detections, detections):
        # Resimdeki tespit edilen nesnelere OCR uygula
        scores = list(filter(lambda x: x > DETECTION_THRESHOLD, detections['detection_scores']))
        boxes = detections['detection_boxes'][:len(scores)]

        if len(boxes) == 0:
            print("NO DETECTION")
            return None, None, None, None, None  # Tespit yoksa OCR güvenliğini None olarak döndür

        box = boxes[0]
        width, height = image_np_with_detections.shape[1], image_np_with_detections.shape[0]
        roi = box * [height, width, height, width]
        car_region = image_np_with_detections[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])].copy()
        cv2.rectangle(image_np_with_detections, (int(roi[1]), int(roi[0])), (int(roi[3]), int(roi[2])), (0, 255, 0), 2)
        confidence = scores[0]

        ocr_future = self.executor.submit(self.reader.readtext, car_region, batch_size=5)
        ocr_result = ocr_future.result()
        confident_ocr_results = [result for result in ocr_result if result[2] > OCR_CONFIDENCE_THRESHOLD]
        text = self.filter_text(car_region, confident_ocr_results, REGION_THRESHOLD)

        if confident_ocr_results:
            ocr_confidence = confident_ocr_results[0][2]
            if ocr_confidence > 0.1:
                display_text = f"OCR Sonucu: {text}, Confidence: {confidence:.2f}, OCR Confidence: {ocr_confidence:.2f}"
                cv2.putText(image_np_with_detections, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                print(display_text)
                return text, car_region, confidence, ocr_confidence, car_region  # Return the plate image as well

        return None, None, None, None, None

    def filter_text(self, region, ocr_result, region_threshold):
        # Bölge boyutu eşiği baz alınarak OCR sonuçlarını filtrele
        rectangle_size = region.shape[0] * region.shape[1]
        plate = []

        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))

            if length * height / rectangle_size > region_threshold:
                plate.append(result[1])

        return plate
