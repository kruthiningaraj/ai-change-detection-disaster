import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np

def evaluate_classifier(model_path, test_gen):
    model = tf.keras.models.load_model(model_path)
    preds = (model.predict(test_gen) > 0.5).astype(int)
    acc = accuracy_score(test_gen.classes, preds)
    print(f"Classification Accuracy: {acc:.4f}")
