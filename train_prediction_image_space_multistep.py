"""
Author: Giovanni Lucente

This script trains the unet model to predict the trajectories of traffic participants. The predicted trajectories are 
encoded in an image, where they are displayed as colored lines on a black background. 
The model receives the image of the trajectories from the previous second as input (or a conditioning image)
and predicts the next frame (as image).

The model is trained to receive its output as input, to predict also the next seconds.

The model loads the pretrained unet model for one second inference and trains it to predict the next steps.

"""

import numpy as np
from loss_functions import *
from trajectory_dataset_multistep import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import segmentation_models_pytorch as smp
import pytorch_ssim
import shutil
import types

# Configuration Parameters
UNET_PATH = "output/Unet_weighted_l1_loss/best_model.pth"
IMAGE_DIR = 'output_images_cv2'                             # Directory containing the images
IMAGE_DIR_TRACKING = 'output_tracking_images_cv2'           # Directory containing the tracking images
OUTPUT_DIR = 'output'                                       # Directory to save the trained models and plots
OUTPUT_DIR_TRACKING = 'output_tracking'                     # Directory to save the trained models and plots
TIME_STEPS = 1000                                           # number of steps in the noising and denoising process
LEARNING_RATE = 1e-4
DATASET_FRACTION = 1.0                                      # = 1 if using the whole dataset
VALIDATION_SPLIT = 0.2
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for image prediction models.')
    
    # Loss function selection
    parser.add_argument('--loss', type=str, default='balanced_weighted_l1', choices=['mse', 'weighted_mse', 'weighted_l1', 'balanced_weighted_l1', 'ssim', 'dice', 'weighted_dice', 'weighted_ssim', 'perceptual', 'edge', 'color', 'wavelet', 'l1'],
                        help='Loss function to use: "mse" (Mean Squared Error), "weighted_mse" (Weighted Mean Squared Error), "ssim" (SSIM), "dice" (Dice Loss), "weighted_dice", "weighted_ssim", "perceptual", "edge", "color", "wavelet", "weighted_l1", "balanced_weighted_l1" or "l1"')

    # Trajectory prediction or occupancy prediction
    parser.add_argument('--prediction', type=str, default='occupancy', choices=['occupancy', 'trajectories'], help='Prediction to do: "occupancy" to predict occupancy, "trajectories" to predict trajectories (different datasets)')

    # Number of prediction steps
    parser.add_argument('--steps', type=int, default=3, help='Prediction steps in the future')
    
    # Batch size selection
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    
    # Number of epochs
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')

    args = parser.parse_args()
    return args

def extract(a, t, x_shape):
    """Extract values at index t from tensor a and reshape to match x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out

# train function for multistep prediction
def train(model, train_loader, val_loader, optimizer, epochs, device, output_dir, loss_fn, num_steps=3):
    model = model.to(device)
    model.train()
    
    model_name = "unet_multistep"
    loss_fn_name = loss_fn.__name__ if hasattr(loss_fn, '__name__') else loss_fn.__class__.__name__
    model_save_dir = os.path.join(output_dir, f"{model_name}_{loss_fn_name}")
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "best_model.pth")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0  
        
        for batch in train_loader:
            images = [img.to(device) for img in batch]  # Unpack images from dataset
            optimizer.zero_grad()
            
            input_img = images[0]  # Start with the first image
            loss_sum = 0
            
            for step in range(1, num_steps + 1):
                target_img = images[step]  # The next frame is the target
                output_img = model(input_img)  # Predict next frame
                
                loss = loss_fn(output_img, target_img)
                loss_sum += loss
                
                input_img = output_img  # Use output as input for next step
            
            loss_sum.backward()
            optimizer.step()
            total_loss += loss_sum.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")
        validation_output_dir = os.path.join(output_dir, f"validation_plots_multistep_unet_{loss_fn_name}")
        best_val_loss = validate(model, val_loader, device, validation_output_dir, epoch + 1, loss_fn, best_model_path, best_val_loss, num_steps)


def validate(model, dataloader, device, output_dir, epoch_number, loss_fn, best_model_path, best_val_loss, num_steps=3):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Clean the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    epoch_folder = os.path.join(output_dir, f"epoch_{epoch_number}")
    os.makedirs(epoch_folder, exist_ok=True)
    
    def denormalize(img):
        return ((img.cpu().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            images = [img.to(device) for img in batch]
            input_img = images[0]
            loss_sum = 0
            
            for step in range(1, num_steps + 1):
                target_img = images[step]
                output_img = model(input_img)
                
                loss = loss_fn(output_img, target_img)
                loss_sum += loss.item()
                
                input_img = output_img  # Use output as input for next step
                
                Image.fromarray(denormalize(output_img[0])).save(os.path.join(epoch_folder, f"{idx}_step{step}_output.png"))
                Image.fromarray(denormalize(target_img[0])).save(os.path.join(epoch_folder, f"{idx}_step{step}_ground_truth.png"))

            total_loss += loss_sum
            num_batches += 1
    
    avg_val_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    print(f"Validation Loss (Epoch {epoch_number}): {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated at Epoch {epoch_number} with validation loss: {best_val_loss:.4f}")
    
    return best_val_loss

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

def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0

def main():
    args = parse_args()
    output_dir = OUTPUT_DIR
    image_dir = IMAGE_DIR

    # Number of prediction steps:
    num_steps = args.steps
    
    # Create output directory if it doesn't exist
    if args.prediction.lower() == 'occupancy':
        output_dir = OUTPUT_DIR
        image_dir = IMAGE_DIR
    elif args.prediction.lower() == 'trajectories':
        output_dir = OUTPUT_DIR_TRACKING
        image_dir = IMAGE_DIR_TRACKING
    else:
        raise ValueError(f"Unknown prediction: {args.prediction}")

    os.makedirs(output_dir, exist_ok=True)

    # Parameters
    timesteps = TIME_STEPS
    batch_size = args.batch      
    image_size = 256
    epochs = args.epochs
    device = DEVICE

    # Model loading 
    if is_folder_empty(UNET_PATH):
        model = smp.Unet(
            encoder_name='resnet34', 
            encoder_weights='imagenet', 
            in_channels=3,                 
            classes=3
        )
    else:
        model = load_unet(UNET_PATH)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loss function selection 
    if args.loss.lower() == 'mse':
        loss_fn = F.mse_loss
    elif args.loss.lower() == 'weighted_mse':
        loss_fn = weighted_l2_loss
    elif args.loss.lower() == 'weighted_l1':
        loss_fn = weighted_l1_loss
    elif args.loss.lower() == 'ssim':
        loss_fn = ssim_loss
    elif args.loss.lower() == 'dice':
        loss_fn = dice_loss
    elif args.loss.lower() == 'weighted_dice':
        loss_fn = weighted_dice_loss
    elif args.loss.lower() == 'weighted_ssim':
        loss_fn = weighted_ssim_loss
    elif args.loss.lower() == 'perceptual':
        loss_fn = PerceptualLoss(device = device)
    elif args.loss.lower() == 'edge':
        loss_fn = edge_loss 
    elif args.loss.lower() == 'color':
        loss_fn = color_loss
    elif args.loss.lower() == 'wavelet':
        loss_fn = WaveletLoss(base_loss=ssim_loss, device = device)
    elif args.loss.lower() == 'l1':
        loss_fn = F.l1_loss
    elif args.loss.lower() == 'balanced_weighted_l1':
        loss_fn = balanced_weighted_l1_loss
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    print(f"Using model: pretrained unet")
    print(f"Using loss function: {args.loss.lower()}")
    print(f"Predicting: {args.prediction.lower()}")
    print(f"Prediction Steps: {args.steps}")
    print(f"Batch size: {args.batch}")
    print(f"Number of epochs: {args.epochs}")

    # Assuming raw_dataset is a list of (cond, x_0) PIL image pairs
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])

    # Initialize dataset
    dataset = TrajectoryDatasetMultistep(image_dir=image_dir, transform=transform, num_steps=num_steps)
    
    # Calculate the size of the subset to use and the training/validation split
    subset_size = int(DATASET_FRACTION * len(dataset))  # 40% of the dataset
    train_size = int((1 - VALIDATION_SPLIT) * subset_size)  # 80% of the subset for training
    val_size = subset_size - train_size  # 20% of the subset for validation

    # Randomly sample a subset of the dataset
    subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

    # Split the subset into training and validation sets
    train_dataset, val_dataset = random_split(subset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # Train the model
    train(model, train_loader, val_loader, optimizer, epochs, device, output_dir, loss_fn, num_steps)
    
if __name__ == '__main__':
    main()