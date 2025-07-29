import cv2
import numpy as np
import tensorflow as tf
from skimage.filters import threshold_otsu

def extract_features_vgg19(image):
    base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv4').output)
    image_resized = cv2.resize(image, (224, 224))
    img_array = tf.keras.applications.vgg19.preprocess_input(np.expand_dims(image_resized, axis=0))
    return model.predict(img_array)[0]

def compute_change_map(pre_img, post_img):
    feat_pre = extract_features_vgg19(pre_img)
    feat_post = extract_features_vgg19(post_img)
    rss_map = np.sum((feat_post - feat_pre) ** 2, axis=-1)
    rss_map_resized = cv2.resize(rss_map, (256, 256))
    thresh = threshold_otsu(rss_map_resized)
    change_map = (rss_map_resized > thresh).astype(np.uint8) * 255
    return change_map
