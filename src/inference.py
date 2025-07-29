import cv2
from change_detection import compute_change_map
from analyze_change import compute_disaster_probability
import tensorflow as tf
import numpy as np

def run_inference(pre_img_path, post_img_path, classifier_path):
    pre_img = cv2.imread(pre_img_path)
    post_img = cv2.imread(post_img_path)

    # Change detection
    change_map = compute_change_map(pre_img, post_img)
    probability = compute_disaster_probability(change_map)
    print(f"Disaster Probability: {probability:.2f}%")

    # Classification
    model = tf.keras.models.load_model(classifier_path)
    post_img_resized = cv2.resize(post_img, (224, 224)) / 255.0
    pred = model.predict(np.expand_dims(post_img_resized, axis=0))[0][0]
    label = "Hurricane" if pred > 0.5 else "Earthquake"
    print(f"Predicted Disaster Type: {label}")
