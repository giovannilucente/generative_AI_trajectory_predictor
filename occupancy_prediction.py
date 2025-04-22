"""
Author: Giovanni Lucente

This script loads the multi step occupancy prediction unet model and tests its performance with respect to the test datset

"""

import torch
import segmentation_models_pytorch as smp
import numpy as np
import torch.nn as nn
import cv2
import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from models.Transformer import Transformer
from models.Autoencoder import Autoencoder
import argparse
import matplotlib.pyplot as plt
from trajectory_dataset_multistep import *
from scipy.ndimage import gaussian_filter
from collections import defaultdict

IMAGE_DIR = 'output_images_cv2'  
MODEL_PATH = 'output/unet_multistep_balanced_weighted_l1_loss/best_model.pth'
IMAGE_SIZE = 256  # Change according to model input size
TEST_FRACTION = 0.001

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for image prediction models.')
    
    # Model selection
    parser.add_argument('--model_path', type=str, default='output/unet_multistep_balanced_weighted_l1_loss/best_model.pth', 
                        help='Model path to evaluate, example: "output/unet_multistep_balanced_weighted_l1_loss/best_model.pth"')

    parser.add_argument('--datset_directory', type=str, default='output_images_cv2', 
                        help='Image dataset directory, example: "output_images_cv2"')

    args = parser.parse_args()
    return args

def load_unet(model_path, encoder_name='resnet34', encoder_weights='imagenet', classes=1):
    """Load the trained Unet model with weights."""
    model = smp.Unet(
        encoder_name=encoder_name, 
        encoder_weights=encoder_weights, 
        in_channels=3,                 
        classes=3
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image_path, transform):
    """Load and preprocess an image for the model."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image)  # Apply transformation
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def denormalize(img):
    return ((img.cpu().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()

def postprocess_image(image):
    """ Postprocess the image """
    image = image.squeeze(0)  # Remove batch dimension (B, C, H, W) -> (C, H, W)
    image = denormalize(image)
    return image

def probability_map(rgb_image):
    """
    Create a probability map of traffic participant presence from an RGB image.

    Args:
        rgb_image (np.ndarray): RGB image of shape (H, W, 3), dtype float32 or uint8.

    Returns:
        np.ndarray: Single-channel probability map (H, W), float32, values in [0, 1].
    """
    # Ensure image is float32 and in [0, 1]
    if rgb_image.dtype == np.uint8:
        rgb_image = rgb_image.astype(np.float32) / 255.0
    elif rgb_image.dtype == np.float64:
        rgb_image = rgb_image.astype(np.float32)

    # Use max channel as a soft proxy for color intensity
    max_rgb = np.max(rgb_image, axis=2)  # shape: (H, W)

    # Optional: normalize and blur
    prob_map = np.clip(max_rgb, 0.0, 1.0)

    return prob_map

def pixel_classification_metrics(predicted_map, target_map, threshold=0.5, eps=1e-8):
    """
    Compute pixel-wise classification metrics between predicted and target occupancy maps.
    the metrics are:
    TP: number of pixels that are true positive
    FP: number of pixels that are false positive
    FN: number of pixels that are false negative
    precision: TP / (TP + FP)
    recall: TP / (TP + FN)
    F1: 2 * precision * recall / (precision + recall)
    IoU: TP / (TP + FP + FN)

    Args:
        predicted_map (np.ndarray): Model output (H, W), values in [0, 1]
        target_map (np.ndarray): Ground truth (H, W), values in [0, 1]
        threshold (float): Threshold for binarization
        eps (float): Small value to avoid division by zero

    Returns:
        dict: Dictionary containing TP, FP, FN, precision, recall, F1, IoU
    """
    # Binarize maps
    pred_bin = (predicted_map >= threshold).astype(np.uint8)
    target_bin = (target_map >= threshold).astype(np.uint8)

    # Compute pixel-level confusion elements
    tp = np.logical_and(pred_bin == 1, target_bin == 1).sum()
    fp = np.logical_and(pred_bin == 1, target_bin == 0).sum()
    fn = np.logical_and(pred_bin == 0, target_bin == 1).sum()

    # Compute metrics
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
    }

if __name__ == "__main__":
    args = parse_args()
    IMAGE_DIR = args.datset_directory
    MODEL_PATH = args.model_path
    num_steps = 5
    test_path = "test"
    os.makedirs(test_path, exist_ok=True)

    # Transform 
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Dataset 
    dataset = TrajectoryDatasetMultistep(image_dir=IMAGE_DIR, transform=transform, num_steps=num_steps)
    test_set_size = int(TEST_FRACTION * len(dataset))
    test_set, _ = random_split(dataset, [test_set_size, len(dataset) - test_set_size])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Load Model 
    model = load_unet(MODEL_PATH)
    model.eval()

    # Metrics accumulator: one for each step
    metrics_accumulator = [defaultdict(float) for _ in range(num_steps)]
    count_per_step = [0] * (num_steps)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images = [img for img in batch]
            input_img = images[0]
            
            for step in range(num_steps):
                target_img = images[step + 1]
                output_img = model(input_img)

                output_prob_map = probability_map(postprocess_image(output_img))
                target_prob_map = probability_map(postprocess_image(target_img))

                metrics = pixel_classification_metrics(output_prob_map, target_prob_map, threshold=0.5)

                # Accumulate metrics
                for k, v in metrics.items():
                    metrics_accumulator[step][k] += v
                count_per_step[step] += 1

                # Save the output and target with updated naming
                output_filename = os.path.join(test_path, f"{idx}_{step}_output.png")
                Image.fromarray(postprocess_image(output_img)).save(output_filename)
                target_filename = os.path.join(test_path, f"{idx}_{step}_target.png")
                Image.fromarray(postprocess_image(target_img)).save(target_filename)

                #for k, v in metrics.items():
                #    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

                #plt.imshow(output_prob_map, cmap="gray")
                #plt.title("Soft Probability Map (from color magnitude)")
                #plt.axis("off")
                #plt.show()
                
                input_img = output_img  # Use output as input for next step
    
    # Compute average metrics
    for step in range(num_steps):
        print(f"\nStep {step + 1} average metrics:")
        for k, total in metrics_accumulator[step].items():
            avg = total / count_per_step[step]
            print(f"  {k}: {avg:.4f}" if isinstance(avg, float) else f"  {k}: {avg}")
                