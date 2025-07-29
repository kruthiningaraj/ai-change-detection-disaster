import numpy as np

def compute_disaster_probability(change_map):
    white_pixels = np.sum(change_map == 255)
    total_pixels = change_map.size
    probability = (white_pixels / total_pixels) * 100
    return probability
