"""
Author: Marko Mizdrak

This script processes the DLR UT dataset, transforming abstract trajectory data into images. 
Specifically, it divides the dataset into images that represent one-second trajectories, 
visualized as colored lines on a black background.

"""

import pandas as pd
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import os
import shutil

def get_dominant_class(row):
    """
    Determines the dominant classification class for a traffic participant.
    Returns the class name with the highest classification score.
    If all classification scores are NaN, returns 'unknown'.
    """
    classes = ['pedestrian', 'bicycle', 'motorbike', 'car', 'van', 'truck']
    scores = row[['classifications_pedestrian', 'classifications_bicycle', 
                 'classifications_motorbike', 'classifications_car', 
                 'classifications_van', 'classifications_truck']]
    
    # Check if all scores are NaN
    if scores.isnull().all():
        return 'unknown'
    
    # Replace NaN with 0 to allow idxmax to work
    scores_filled = scores.fillna(0)
    dominant_class = scores_filled.idxmax().replace('classifications_', '')
    
    return dominant_class

def map_class_to_hue(dominant_class):
    """
    Maps a traffic class to a specific hue value in HSV color space.
    'unknown' class is mapped to gray.
    """
    class_hue_map = {
        'pedestrian': 0,        # Red
        'bicycle': 30,          # Orange
        'motorbike': 60,        # Yellow
        'car': 120,             # Green
        'van': 150,             # Cyan
        'truck': 180,           # Blue
        'unknown': 0            # Will set to gray later
    }
    return class_hue_map.get(dominant_class, 0)  # Default to Red if class not found

def get_global_bounds(df, fixed_bounds=None):
    """
    Determines the global minimum and maximum for easting and northing across the entire dataset.
    Allows for fixed bounds to be set manually.
    
    Parameters:
    - df: pandas DataFrame containing 'center_easting' and 'center_northing'
    - fixed_bounds: dict with keys 'min_easting', 'max_easting', 'min_northing', 'max_northing'
    
    Returns:
    - min_easting, max_easting, min_northing, max_northing
    """
    if fixed_bounds:
        min_easting = fixed_bounds.get('min_easting', df['center_easting'].min())
        max_easting = fixed_bounds.get('max_easting', df['center_easting'].max())
        min_northing = fixed_bounds.get('min_northing', df['center_northing'].min())
        max_northing = fixed_bounds.get('max_northing', df['center_northing'].max())
    else:
        min_easting = df['center_easting'].min()
        max_easting = df['center_easting'].max()
        min_northing = df['center_northing'].min()
        max_northing = df['center_northing'].max()
        
        # Add a buffer to ensure trajectories are not at the very edges
        buffer_easting = (max_easting - min_easting) * 0.05 if max_easting != min_easting else 1
        buffer_northing = (max_northing - min_northing) * 0.05 if max_northing != min_northing else 1
        
        min_easting -= buffer_easting
        max_easting += buffer_easting
        min_northing -= buffer_northing
        max_northing += buffer_northing
    
    return min_easting, max_easting, min_northing, max_northing

def normalize_coordinates(easting, northing, min_easting, max_easting, min_northing, max_northing, img_width, img_height, margin):
    """
    Normalizes easting and northing to image pixel coordinates.
    """
    x = ((easting - min_easting) / (max_easting - min_easting)) * (img_width - 2 * margin) + margin
    y = ((northing - min_northing) / (max_northing - min_northing)) * (img_height - 2 * margin) + margin
    # Invert y-axis for image coordinates
    y = img_height - y
    return x.astype(int), y.astype(int)

def rasterize_trajectories_cv2(df, window_start, window_end, image_size=(800, 800), margin=50, 
                              output_dir='output_images', 
                              global_bounds=None, 
                              min_luminance=50, max_luminance=255):
    """
    Rasterizes trajectories of traffic participants within a specified time window onto an image using OpenCV.
    Incorporates a time-based luminance gradient along each trajectory.
    
    Parameters:
    - df: pandas DataFrame with MultiIndex (timestamp, id)
    - window_start: pd.Timestamp, start of the 5-second window
    - window_end: pd.Timestamp, end of the 5-second window
    - image_size: tuple, size of the output image in pixels (width, height)
    - margin: int, margin in pixels around the trajectories
    - output_dir: str, directory path to save the rasterized images
    - global_bounds: tuple, (min_easting, max_easting, min_northing, max_northing)
                     If None, compute based on the window data.
    - min_luminance: int, minimum brightness value for the gradient
    - max_luminance: int, maximum brightness value for the gradient
    """
    # Filter data within the time window
    window_df = df.loc[window_start:window_end]
    
    if window_df.empty:
        # print(f"No data in the time window {window_start} to {window_end}. Skipping.")
        return
    
    # Reset index to access 'timestamp' and 'id'
    window_df = window_df.reset_index()
    
    # Determine the dominant class for each participant
    window_df['dominant_class'] = window_df.apply(get_dominant_class, axis=1)
    
    # Map class to hue
    window_df['hue'] = window_df['dominant_class'].apply(map_class_to_hue)
    
    # Normalize coordinates
    if global_bounds:
        min_easting, max_easting, min_northing, max_northing = global_bounds
    else:
        min_easting, max_easting, min_northing, max_northing = get_global_bounds(window_df)
    
    img_width, img_height = image_size
    
    # Initialize a blank white image
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)  # Black background
    
    # Prepare participants grouped by ID
    participants = window_df.groupby('id')
    
    for participant_id, participant_data in participants:
        # Sort by timestamp to ensure correct trajectory
        participant_data = participant_data.sort_values('timestamp')
        
        # Get normalized coordinates
        x, y = normalize_coordinates(participant_data['center_easting'], 
                                     participant_data['center_northing'], 
                                     min_easting, max_easting, 
                                     min_northing, max_northing, 
                                     img_width, img_height, margin)
        
        # Combine x and y into points
        points = np.vstack((x, y)).T.reshape((-1, 1, 2))
        
        # Compute relative time for each point within the window
        time_normalized = (participant_data['timestamp'] - window_start) / (window_end - window_start)
        time_normalized = time_normalized.clip(lower=0, upper=1)  # Ensure values are within [0,1]
        
        # Iterate through consecutive point pairs to draw segments with gradient luminance
        for i in range(len(points) - 1):
            pt1 = points[i][0]
            pt2 = points[i + 1][0]
            
            # Compute average time for the segment
            time_segment = (time_normalized.iloc[i] + time_normalized.iloc[i + 1]) / 2
            luminance = int(time_segment * (max_luminance - min_luminance) + min_luminance)
            luminance = np.clip(luminance, min_luminance, max_luminance)  # Ensure visibility
            
            # Get hue based on classification
            dominant_class = participant_data['dominant_class'].iloc[i]
            hue = participant_data['hue'].iloc[i]
            
            if dominant_class == 'unknown':
                # Assign gray color for unknown classes
                color_bgr = (128, 128, 128)  # Gray in BGR
            else:
                # Convert hue to OpenCV hue (0-179)
                hue_cv = int((hue / 360) * 180)
                
                # Set saturation and value
                saturation = 255
                value = luminance
                
                # Convert HSV to BGR
                color_hsv = np.uint8([[[hue_cv, saturation, value]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
            
            # Determine line thickness based on participant's width
            # Assume dimension_width is in meters and map it to pixels
            # Define a scale factor (pixels per meter). Adjust as needed.
            scale_factor = (img_width - 2 * margin) / (max_easting - min_easting)
            width_meters = participant_data['dimension_width'].iloc[i]
            line_thickness = max(1, int(width_meters * scale_factor ))  # Adjust multiplier as needed
            
            # Draw the line segment
            cv2.line(img, tuple(pt1), tuple(pt2), color_bgr, thickness=line_thickness, lineType=cv2.LINE_AA)
    
    # Define the output path with window start time
    window_str = window_start.strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'trajectories_{window_str}.png')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the image
    cv2.imwrite(output_path, img)
    # print(f"Rasterized image saved to {output_path}")

def main():
    # Load the data
    # Path to the directory containing trajectory files
    trajectories_dir = "./trajectories/"
    
    # Find all CSV files in the directory
    csv_files = sorted(glob(os.path.join(trajectories_dir, "trajectories_*.csv")))
    
    if not csv_files:
        print("No trajectory files found in the directory.")
        return

    # Load all CSV files into a single DataFrame
    df_list = [pd.read_csv(file, parse_dates=['timestamp']) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)  # Combine all files
    
    # Define the classification columns
    classification_columns = [
        'classifications_pedestrian',
        'classifications_bicycle',
        'classifications_motorbike',
        'classifications_car',
        'classifications_van',
        'classifications_truck'
    ]
    
    # Convert classification columns to numeric, coercing errors to NaN
    df[classification_columns] = df[classification_columns].apply(pd.to_numeric, errors='coerce')
    
    # Ensure the data is sorted by timestamp
    df.sort_values('timestamp', inplace=True)
    
    # Set MultiIndex for efficient querying
    df.set_index(['timestamp', 'id'], inplace=True)
    
    # Reset index to compute global bounds
    df_reset = df.reset_index()
    
    # Compute global bounds for consistent scaling
    min_easting, max_easting, min_northing, max_northing = get_global_bounds(df_reset)
    global_bounds = (min_easting, max_easting, min_northing, max_northing)
    
    # print("Global bounds established:")
    # print(f"  Easting: {min_easting} to {max_easting}")
    # print(f"  Northing: {min_northing} to {max_northing}")
    
    # Extract the earliest and latest timestamps
    start_time = df_reset['timestamp'].min().floor('S')  # Floor to the nearest second
    end_time = df_reset['timestamp'].max().ceil('S')     # Ceil to the nearest second
    
    # Define window size
    window_size = pd.Timedelta(seconds=1)

    # Define window step size (e.g., 5 seconds for non-overlapping, 1 second for overlapping)
    window_step = pd.Timedelta(seconds=1)  # Change to desired step size, e.g., 1 for sliding windows
    #window_step = window_size

    # Generate all window start times
    window_starts = pd.date_range(start=start_time, end=end_time - window_size, freq=window_step)
    
    # Define output directory
    output_dir = 'output_images_cv2'

    # Clean the output directory before saving new images
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove the directory and all its contents
    
    # Iterate over each window and rasterize
    for window_start in tqdm(window_starts, desc="Processing Windows"):
        window_end = window_start + window_size
        rasterize_trajectories_cv2(df, window_start, window_end, image_size=(800, 800), margin=50, 
                                  output_dir=output_dir, global_bounds=global_bounds)
    
    print("All windows have been processed.")

if __name__ == "__main__":
    main()
