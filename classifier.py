import os

import keras
from keras import preprocessing
import numpy as np
import time
import cv2
from PIL import Image
import face_recognition
from mtcnn import MTCNN

original_dir = "files/original"
cropped_dir = "files/cropped"

detector = MTCNN()


def predict(image_name, model):
    # Step 1: Load the image
    img_path = f"{cropped_dir}/{image_name}"
    img = preprocessing.image.load_img(img_path, target_size=(299, 299))

    # Step 2: Preprocess the image
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the image

    # Step 4: Make prediction
    start_time = time.time()
    predictions = model.predict(img_array)
    end_time = time.time()
    inference_time = end_time - start_time

    predicted_class = np.argmax(predictions[0])

    # Step 5: Interpret the prediction
    class_labels = ['Perempuan', 'Laki-laki']  # Replace with your class labels
    predicted_label = class_labels[predicted_class]
    predictions = predictions.astype(str)
    score = predictions[0][predicted_class]
    return predicted_label, score, inference_time


def crop_face(image_name):
    path = f"{original_dir}/{image_name}"
    image = cv2.imread(path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale (required for face detection)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Increase the box size by multiply
    # lying width and height of the bounding box
    # Let's multiply by a factor of 1.5 here
    filenames = []
    for idx, (x, y, w, h) in enumerate(faces):
        new_w = int(w * 1.5)
        new_h = int(h * 1.5)
        # Calculate the new (x, y) coordinates to keep the face centered
        new_x = x - (new_w - w) // 2
        new_y = y - (new_h - h) // 2

        # Write to file
        face_roi = image[new_y:new_y + new_h, new_x:new_x + new_w]
        cropped = f"c{idx+1}_{image_name}"
        filenames.append(cropped)
        cv2.imwrite(f"files/{cropped}", face_roi)

    return filenames


def crop_faces_mtcnn(image_name, scale_factor=1.6):
    path = f"{original_dir}/{image_name}"
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img)
    min_conf = 0.9
    results = []
    for i, det in enumerate(detections):
        if det['confidence'] >= min_conf:
            x, y, w, h = det['box']

            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            # Calculate the new (x, y) coordinates to keep the face centered
            new_x = x - (new_w - w) // 2
            new_y = y - (new_h - h) // 2

            face = img[new_y:new_y + new_h, new_x:new_x + new_w]
            cropped = f"cf{i}_{image_name}"
            cv2.imwrite(f"{cropped_dir}/{cropped}", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            results.append(cropped)
    return results


def crop_faces(image_name, scale_factor=2):
    # Load the image
    image = face_recognition.load_image_file(f"{original_dir}/{image_name}")

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)

    results = []

    # Loop through each face location and crop the face
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location

        # Calculate the dimensions of the original face region
        face_width = right - left
        face_height = bottom - top

        # Calculate the new dimensions after scaling
        new_face_width = int(face_width * scale_factor)
        new_face_height = int(face_height * scale_factor)

        # Calculate the top-left corner of the new cropping region to keep the face centered
        new_left = max(0, left - int((new_face_width - face_width) // 2))
        new_top = max(0, top - int((new_face_height - face_height) // 2))

        # Calculate the bottom-right corner of the new cropping region
        new_right = min(image.shape[1], new_left + new_face_width)
        new_bottom = min(image.shape[0], new_top + new_face_height)

        # Crop the face from the original image
        face_image = image[new_top:new_bottom, new_left:new_right]

        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(face_image)
        # Save the cropped face to the output directory
        cropped_face = f"cf{i}_{image_name}"
        results.append(cropped_face)
        output_path = f"{cropped_dir}/{cropped_face}"
        pil_image.save(output_path)
    return results

