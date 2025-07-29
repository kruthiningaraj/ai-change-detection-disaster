import os
import cv2
import numpy as np

def load_image_pairs(pre_dir, post_dir, target_size=(256, 256)):
    pairs = []
    pre_images = sorted([f for f in os.listdir(pre_dir) if f.endswith(('png','jpg'))])
    post_images = sorted([f for f in os.listdir(post_dir) if f.endswith(('png','jpg'))])
    for pre, post in zip(pre_images, post_images):
        pre_img = cv2.imread(os.path.join(pre_dir, pre))
        post_img = cv2.imread(os.path.join(post_dir, post))
        pre_img = cv2.resize(pre_img, target_size) / 255.0
        post_img = cv2.resize(post_img, target_size) / 255.0
        pairs.append((pre_img, post_img))
    return pairs
