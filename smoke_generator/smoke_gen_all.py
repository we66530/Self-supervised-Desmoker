import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import os

# Specify the root folder containing subfolders with image series
root_folder = r"C:\Users\User\Desktop\pure_clear series"

# Parameters
width = 1920
height = 1080
center_x = width // 2
center_y = height // 2

# Iterate through all subfolders
for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)
    
    # Check if it's a directory
    if not os.path.isdir(subfolder_path):
        continue
        
    print(f"Processing subfolder: {subfolder_path}")
    
    # Get all JPG files in the subfolder and sort them
    image_files = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith('.jpg')])
    if not image_files:  # Skip empty folders or folders with no JPGs
        print(f"No JPG files found in {subfolder_path}")
        continue
        
    n_frames = len(image_files)  # Number of frames based on actual images

    for frame, image_file in enumerate(image_files):
        # Load the background image
        image_path = os.path.join(subfolder_path, image_file)
        background = np.array(Image.open(image_path).resize((width, height)))

        # First frame should not have blobs
        if frame == 0:
            # Save original image without modification
            result = background
        else:
            # Create canvas and mask for each frame
            canvas = np.zeros((height, width))
            blob_mask = np.zeros((height, width), dtype=bool)
            
            # Scale spread and radius with frame number (starting from frame 1)
            frame_idx = frame - 1  # Adjust index since we skip first frame
            max_idx = n_frames - 2  # Maximum index for scaling (excluding first frame)
            
            # Progressive scaling based on position in sequence
            spread_x = 150 + (frame_idx * 400 / max_idx) if max_idx > 0 else 150
            spread_y = 80 + (frame_idx * 200 / max_idx) if max_idx > 0 else 80
            min_radius = 40 + (frame_idx * 80 / max_idx) if max_idx > 0 else 40
            max_radius = 120 + (frame_idx * 160 / max_idx) if max_idx > 0 else 120
            
            # Generate blobs
            n_blobs = 20
            for _ in range(n_blobs):
                offset_x = int(np.random.normal(0, spread_x))
                offset_y = int(np.random.normal(0, spread_y))
                x = center_x + offset_x
                y = center_y + offset_y
                x = np.clip(x, 50, width-50)
                y = np.clip(y, 50, height-50)
                r = np.random.randint(min_radius, max_radius)
                y_grid, x_grid = np.ogrid[:height, :width]
                mask = ((x_grid - x)**2 + (y_grid - y)**2) <= r**2
                canvas[mask] += np.random.random() * 0.5 + 0.5
                blob_mask |= mask

            # Smooth the edges
            canvas = gaussian_filter(canvas, sigma=30)
            canvas = np.clip(canvas, 0, 1)

            # Add salt-and-pepper noise
            noise_density = 0.05
            noise = np.random.random(canvas.shape)
            salt = (noise > 1 - noise_density/2) & blob_mask
            pepper = (noise < noise_density/2) & blob_mask
            canvas[salt] = 1
            canvas[pepper] = 0

            # Create RGBA mask (white)
            mask_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            mask_rgba[..., 0] = 255  # Red (white)
            mask_rgba[..., 1] = 255  # Green (white)
            mask_rgba[..., 2] = 255  # Blue (white)
            mask_rgba[..., 3] = (canvas * 120).astype(np.uint8)

            # Combine with background
            bg_rgba = np.dstack((background, np.full((height, width), 255, dtype=np.uint8)))
            alpha_mask = mask_rgba[..., 3] / 255.0
            alpha_bg = 1.0 - alpha_mask
            result = np.zeros_like(bg_rgba, dtype=np.float32)
            for channel in range(3):
                result[..., channel] = (bg_rgba[..., channel] * alpha_bg + 
                                      mask_rgba[..., channel] * alpha_mask)
            result[..., 3] = 255
            result = result.astype(np.uint8)

        # Convert to PIL Image and save with original filename
        result_image = Image.fromarray(result[..., :3])  # Remove alpha channel for saving
        output_path = os.path.join(subfolder_path, image_file)
        result_image.save(output_path)
        print(f"Saved: {output_path}")

print("Processing of all subfolders complete!")