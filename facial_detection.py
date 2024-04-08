from facenet_pytorch import MTCNN
import torch
from PIL import Image
import os
import pandas as pd

# Directory containing the original dataset
root_dir = '/scratch/gilbreth/kim2903/test'
# Directory to save the cropped faces
cropped_dir = '/scratch/gilbreth/kim2903/test_cropped'

# Ensure the save directory exists
os.makedirs(cropped_dir, exist_ok=True)

# Initialize MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Function to process and save cropped faces
def process_and_save_images(image_path):
    img = Image.open(image_path).convert('RGB')
    # Detect and save face
    face, prob = mtcnn(img, save_path=os.path.join(cropped_dir, os.path.basename(image_path)), return_prob=True)
    # Check if a face was detected and the probability is not None
    if prob is not None and prob < 0.9:  # Confidence threshold
        print(f"Face not detected with high confidence: {image_path}, prob: {prob}")
    elif prob is None:
        print(f"No face detected in the image: {image_path}")
    return face, prob


# Iterate over images in the dataset
for img_file in os.listdir(root_dir):
    img_path = os.path.join(root_dir, img_file)
    process_and_save_images(img_path)

print("Preprocessing complete.")
