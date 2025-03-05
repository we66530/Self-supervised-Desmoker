import os
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

def display_images(images=[], titles=[], rows=1, cols=None):
    if cols is None:
        cols = len(images)  # Automatically adjust columns based on images
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
    axes = np.atleast_1d(axes)  # Ensure axes is an iterable, even for 1 image
    for i, ax in enumerate(axes.flat[:len(images)]):
        ax.imshow(images[i])
        ax.axis('off')
        ax.set_title(titles[i] if i < len(titles) else f"Image {i+1}")
    plt.tight_layout()
    plt.show()

# Load input image
img = Image.open("D:\Desmoking Dataset\LH_frames\1970_01_01_010226_LH_S6\smoke_ahead_series\00002\frames_0000206.jpg").convert("RGB")

# Load pipeline
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")

# Perform depth estimation
depth_result = pipe(img)

# Extract the depth map and normalize it
depth = np.array(depth_result["depth"])
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to 0–1 range

# Display input and depth images
display_images([img, depth_normalized], titles=["Input Image", "Depth Map"])



# reduce saturation
enhancer = ImageEnhance.Color(img)
img_2 = enhancer.enhance(0.5)

# reduce brightness
enhancer2 = ImageEnhance.Brightness(img_2)
img_2 = enhancer2.enhance(0.7)

# increase contrast
enhancer3 = ImageEnhance.Contrast(img_2)
img_2 = enhancer3.enhance(2.2)

display_images([img, img_2])



def overlay_transparent_layer(rgb_image, grayscale_image, fog_thickness=0.5):
    # Clamp fog_thickness between 0 and 1
    fog_thickness = np.clip(fog_thickness, 0, 1)
    # Create a white layer with the same size as the input images
    white_layer = Image.new('RGBA', rgb_image.size, (216, 216, 216, 0))  # Initial transparency = 0
    # Convert images to numpy arrays for easier manipulation
    rgb_array = np.array(rgb_image)
    grayscale_array = np.array(grayscale_image)
    white_array = np.array(white_layer)
    # Calculate alpha values, invert grayscale values
    alpha = (255 - grayscale_array) * fog_thickness  # Scale by fog_thickness
    # Set the alpha channel of the white layer
    white_array[:, :, 3] = alpha.astype(np.uint8)  # Ensure integer values
    # Convert back to PIL Image
    white_layer_transparent = Image.fromarray(white_array, 'RGBA')
    # Composite the images
    result = Image.alpha_composite(rgb_image.convert('RGBA'), white_layer_transparent)
    return result


# Example usage
result_img = overlay_transparent_layer(img_2, depth, fog_thickness=0.9)  # Adjust fog_thickness here
display_images([img, result_img], titles=["Original Image", "Foggy Image"])


result_img.save("foggy_img.png", format="PNG")

print("Images saved successfully!")