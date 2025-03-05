import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

# Load the background image
image_path = r"D:\Desmoking Dataset\LH_frames\1970_01_01_010226_LH_S6\smoke_ahead_series\00002\frames_0000206.jpg"
background = np.array(Image.open(image_path).resize((1920, 1080)))

# Parameters
width = 1920
height = 1080
center_x = width // 2
center_y = height // 2
n_frames = 5  # Number of frames in the series

# Create figure for subplots
fig, axes = plt.subplots(1, n_frames, figsize=(19.2 * n_frames / 2, 10.8))

for frame in range(n_frames):
    # Create canvas and mask for each frame
    canvas = np.zeros((height, width))
    blob_mask = np.zeros((height, width), dtype=bool)
    
    # Scale spread and radius with frame number
    spread_x = 150 + frame * 100  # Increases from 150 to 550
    spread_y = 80 + frame * 50    # Increases from 80 to 280
    min_radius = 40 + frame * 20  # Increases from 40 to 120
    max_radius = 120 + frame * 40 # Increases from 120 to 280
    
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

    # Display in subplot
    axes[frame].imshow(result)
    axes[frame].axis('off')
    axes[frame].set_title(f'Frame {frame + 1}')

plt.tight_layout()
plt.show()