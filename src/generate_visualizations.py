import os
import cv2
import matplotlib.pyplot as plt
from change_detection import compute_change_map
from analyze_change import compute_disaster_probability

def generate_sample_visualizations(pre_img_path, post_img_path, output_dir="outputs/"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Read images
    pre_img = cv2.imread(pre_img_path)
    post_img = cv2.imread(post_img_path)

    # Compute change map
    change_map = compute_change_map(pre_img, post_img)
    probability = compute_disaster_probability(change_map)

    # Convert BGR to RGB for plotting
    pre_rgb = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_rgb = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

    # Plot results
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow(pre_rgb); plt.title("Pre-Disaster Image"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(post_rgb); plt.title("Post-Disaster Image"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(change_map, cmap='gray'); plt.title(f"Change Map\nProb: {probability:.2f}%"); plt.axis('off')

    # Save visualization
    output_path = os.path.join(output_dir, "change_visualization.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # Example usage: Replace with actual paths after dataset download
    sample_pre = "data/xBD/train/pre_disaster/sample_image.png"
    sample_post = "data/xBD/train/post_disaster/sample_image.png"
    generate_sample_visualizations(sample_pre, sample_post)
