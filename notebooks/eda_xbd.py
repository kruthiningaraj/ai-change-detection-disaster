import os
import cv2
import matplotlib.pyplot as plt

# EDA for xBD Dataset
def eda_xbd(pre_dir, post_dir, label_dir=None):
    pre_images = sorted([f for f in os.listdir(pre_dir) if f.endswith(('png','jpg'))])
    post_images = sorted([f for f in os.listdir(post_dir) if f.endswith(('png','jpg'))])

    print(f"Total pre-disaster images: {len(pre_images)}")
    print(f"Total post-disaster images: {len(post_images)}")

    # Display sample pre/post image pairs
    for i in range(3):
        pre_img = cv2.imread(os.path.join(pre_dir, pre_images[i]))
        post_img = cv2.imread(os.path.join(post_dir, post_images[i]))
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1); plt.imshow(pre_img); plt.title('Pre-Disaster'); plt.axis('off')
        plt.subplot(1,2,2); plt.imshow(post_img); plt.title('Post-Disaster'); plt.axis('off')
        plt.show()

    # Display labels if available
    if label_dir and os.path.exists(label_dir):
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(('png','jpg'))])
        if label_files:
            label_img = cv2.imread(os.path.join(label_dir, label_files[0]), cv2.IMREAD_GRAYSCALE)
            plt.figure(figsize=(6,6))
            plt.imshow(label_img, cmap='gray')
            plt.title('Building Labels')
            plt.axis('off')
            plt.show()
        else:
            print("No labels found in label directory.")

if __name__ == "__main__":
    # Example usage (adjust paths as needed)
    pre_dir = "data/xBD/train/pre_disaster"
    post_dir = "data/xBD/train/post_disaster"
    label_dir = "data/xBD/train/labels"
    eda_xbd(pre_dir, post_dir, label_dir)
