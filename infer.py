import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import time
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import tkinter as tk
from tkinter import filedialog

# Assuming all your model classes (FlowGuidedDCN, OpticalFlowOffsetEstimator, etc.) 
# are defined above as in your training code
# I'll include only the necessary parts for prediction

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(16)  # Utilize 16 CPU threads for operations that run on CPU



class FlowGuidedDCN(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, padding=1):
        super(FlowGuidedDCN, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dcn = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x, offsets):
        # Ensure input has correct shape: (N, C, H, W)
        if x.shape[-1] == 3:  # If shape is (1, H, W, 3), convert it
            x = x.permute(0, 3, 1, 2)  # Convert (1, H, W, 3) -> (1, 3, H, W)

        # Ensure offsets are in correct shape (N, 18, H, W)
        if offsets.shape[1] != 2 * self.kernel_size * self.kernel_size:
            raise ValueError(f"Offsets should have {2 * self.kernel_size * self.kernel_size} channels, but got {offsets.shape[1]}.")

        # Convert to float32
        x = x.float()  # Convert image to float32
        offsets = offsets.float()  # Convert offsets to float32

        # Apply deformable convolution
        return deform_conv2d(x, offsets, self.dcn.weight, bias=self.dcn.bias, padding=self.padding)



class OpticalFlowOffsetEstimator(nn.Module):
    def __init__(self, flow_channels=2, kernel_size=3):
        super(OpticalFlowOffsetEstimator, self).__init__()
        self.offset_conv = nn.Conv2d(flow_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
    
    def forward(self, flow):
        return self.offset_conv(flow)

class ImageEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super(ImageEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class PastFramePropagation(nn.Module):
    def __init__(self, channels):
        super(PastFramePropagation, self).__init__()
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, past_feat, curr_feat):
        return self.fusion(torch.cat([past_feat, curr_feat], dim=1))

class Upsampler(nn.Module):
    def __init__(self, in_channels=16, out_channels=3):
        super(Upsampler, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.upsample(x)





def extract_channel(img: torch.Tensor, index: int) -> torch.Tensor:
    output = torch.zeros_like(img)  # Create a tensor of zeros with the same shape
    output[..., index] = img[..., index]  # Copy only the selected channel
    return output


def estimate_optical_flow(prev_img: torch.Tensor, next_img: torch.Tensor, device='cuda'):
    # Convert from torch tensor to NumPy (remove batch dimension and ensure proper format)
    prev_np = prev_img.squeeze(0).cpu().numpy().astype(np.uint8)
    next_np = next_img.squeeze(0).cpu().numpy().astype(np.uint8)

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_np, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_np, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to PyTorch tensor (format: [1, 2, H, W])
    flow_tensor = torch.tensor(np.transpose(flow, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)

    return flow_tensor.to(device)  # Move to GPU if needed



def warp_image(image: torch.Tensor, flow: torch.Tensor, device='cuda'):
    """
    Args:
        image (torch.Tensor): Image tensor of shape (1, H, W, 3) with dtype uint8
        flow (torch.Tensor): Optical flow tensor of shape (1, 2, H, W) with dtype float32
    Returns:
        torch.Tensor: Warped image of shape (1, H, W, 3) with dtype float32
    """
    B, H, W, C = image.shape  # B = 1, H = 1080, W = 1920, C = 3

    # Convert image to float32 and normalize to [0,1] range
    image = image.float() / 255.0  # (1, H, W, 3) → (float32)

    # Rearrange image to (B, C, H, W) for grid_sample
    image = image.permute(0, 3, 1, 2)  # (1, 3, H, W)

    # Create mesh grid
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    grid = torch.stack((x, y), dim=0).float()  # (2, H, W)

    # Apply flow displacement
    grid = grid + flow.squeeze(0)  # (2, H, W)

    # Normalize grid to [-1, 1] range
    grid[0] = 2.0 * grid[0] / (W - 1) - 1.0  # Normalize X-coords
    grid[1] = 2.0 * grid[1] / (H - 1) - 1.0  # Normalize Y-coords
    grid = grid.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 2)

    # Warp image using grid_sample
    warped_image = F.grid_sample(image, grid, mode="bilinear", align_corners=True)  # (1, 3, H, W)

    # Convert back to (1, H, W, 3)
    return warped_image.permute(0, 2, 3, 1)  # (1, H, W, 3)


# Convert torch tensor [1, 1080, 1920, 3] to numpy image [1080, 1920, 3]
def tensor_to_image(tensor):
    """
    Args:
        tensor (torch.Tensor): Tensor of shape (1, H, W, 3), values in [0,1] (float32)
    Returns:
        np.ndarray: Numpy image of shape (H, W, 3), values in [0,255] (uint8)
    """
    tensor = tensor.squeeze(0)  # Remove batch dimension → (H, W, 3)
    tensor = tensor.cpu().numpy()  # Convert to NumPy
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)  # Scale & convert to uint8
    return tensor

# Rescale to [0, 255]
def rescale_tensor(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).clamp(0, 255)







# Function to compute the dark channel
def dark_channel(image, patch_size=15):
    # Ensure the image is in RGB format
    if image.shape[3] != 3:  # PyTorch tensor is in format [1, H, W, C]
        raise ValueError("Input image must have 3 channels (RGB)")
    
    # Convert from [1, H, W, C] to [H, W, C]
    image = image.squeeze(0).cpu().numpy()  # Convert to NumPy array (H, W, C)

    # Convert to minimum across RGB channels
    min_channel = np.min(image, axis=2)

    # Apply minimum filter using OpenCV's erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    return torch.tensor(dark_channel).unsqueeze(0)  # Convert back to torch tensor and add batch dimension


# Function to apply Gaussian blur
def apply_gaussian_blur(image, kernel_size=(11, 11)):
    # Ensure the input image is of the correct shape [1, H, W]
    if image.ndimension() != 3 or image.shape[0] != 1:
        raise ValueError("Input image must have shape [1, H, W]")
    
    # Convert the image to NumPy array from PyTorch tensor (squeeze batch dimension)
    image = image.squeeze(0).cpu().numpy()  # Convert to [H, W] (NumPy array)

    # Apply Gaussian blur using OpenCV
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    # Convert back to torch tensor and add the batch dimension back
    return torch.tensor(blurred_image).unsqueeze(0)

# Function to split an RGB image into patches
def split_into_patches(image, patch_size):
    patches = [
        image[i:i + patch_size, j:j + patch_size]
        for i in range(0, image.shape[0], patch_size)
        for j in range(0, image.shape[1], patch_size)
    ]
    return patches




## Compute GAN Loss
class Discriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64):
        super(Discriminator, self).__init__()
        
        def conv_block(in_channels, out_channels, normalize=True):
            """Creates a convolutional block with optional batch normalization."""
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Input: (B, 3, H, W) → (B, 64, H/2, W/2)
        self.model = nn.Sequential(
            conv_block(input_channels, num_filters, normalize=False),  # (B, 3, H, W) -> (B, 64, H/2, W/2)
            conv_block(num_filters, num_filters * 2),  # (B, 64, H/2, W/2) -> (B, 128, H/4, W/4)
            conv_block(num_filters * 2, num_filters * 4),  # (B, 128, H/4, W/4) -> (B, 256, H/8, W/8)
            conv_block(num_filters * 4, num_filters * 8),  # (B, 256, H/8, W/8) -> (B, 512, H/16, W/16)
            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)  # (B, 512, H/16, W/16) -> (B, 1, H/16, W/16)
        )

    def forward(self, x):
        return self.model(x)
    


class vanilla_GANLoss(nn.Module):
    """Defines the loss function for GANs.

    Specifically, this implementation supports 'vanilla' GAN loss using BCEWithLogitsLoss.

    Args:
        real_label_val (float): The value for real labels. Default: 1.0.
        fake_label_val (float): The value for fake labels. Default: 0.0.
        loss_weight (float): The weight for generator loss. Default: 1.0.
    """
    def __init__(self, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super().__init__()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight
        self.loss = nn.BCEWithLogitsLoss()  # Vanilla GAN loss function

    def get_target_label(self, input, target_is_real):
        """Generates target labels.

        Args:
            input (Tensor): The input tensor (discriminator output).
            target_is_real (bool): Whether the target label is for real or fake data.

        Returns:
            Tensor: A tensor filled with the target label value.
        """
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_full(input.size(), target_val)

    def forward(self, input, target_is_real, is_disc=False):
        """
        Computes the GAN loss.

        Args:
            input (Tensor): Predictions from the discriminator.
            target_is_real (bool): Whether the target is real or fake.
            is_disc (bool): Whether the loss is for the discriminator or generator. Default: False.

        Returns:
            Tensor: The calculated loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss if is_disc else loss * self.loss_weight



# Define the RBSR_X model class (same as in your training code)
class RBSR_X(nn.Module):
    def __init__(self):
        super(RBSR_X, self).__init__()
        self.image_encoder = ImageEncoder().to(device)
        self.offset_estimator = OpticalFlowOffsetEstimator().to(device)
        self.flow_guided_dcn = FlowGuidedDCN(in_channels=3, out_channels=16).to(device)
        self.fusion_model = PastFramePropagation(channels=16).to(device)
        self.upsampler = Upsampler().to(device)
        self.discriminator = Discriminator().to(device)

    def forward(self, curr_frame, past_frame):
        first_img_a, first_img_b = past_frame.to(device), curr_frame.to(device)

        red_A, green_A, blue_A = [extract_channel(first_img_a, i) for i in range(3)]
        red_B, green_B, blue_B = [extract_channel(first_img_b, i) for i in range(3)]

        flow_R, flow_G, flow_B = [estimate_optical_flow(a, b) for a, b in zip([red_A, green_A, blue_A], [red_B, green_B, blue_B])]

        warped_A_R = warp_image(red_A, flow_R)
        warped_A_G = warp_image(green_A, flow_G)
        warped_A_B = warp_image(blue_A, flow_B)

        warped_red_np = tensor_to_image(warped_A_R)
        warped_green_np = tensor_to_image(warped_A_G)
        warped_blue_np = tensor_to_image(warped_A_B)

        warped_A = np.stack([warped_red_np[:, :, 0], warped_green_np[:, :, 1], warped_blue_np[:, :, 2]], axis=-1)

        estimated_offsets_R, estimated_offsets_G, estimated_offsets_B = [self.offset_estimator(flow) for flow in [flow_R, flow_G, flow_B]]

        red_B = red_B.permute(0, 3, 1, 2).float() if red_B.shape[-1] == 3 else red_B
        green_B = green_B.permute(0, 3, 1, 2).float() if green_B.shape[-1] == 3 else green_B
        blue_B = blue_B.permute(0, 3, 1, 2).float() if blue_B.shape[-1] == 3 else blue_B

        output_R, output_G, output_B = [
            self.flow_guided_dcn(img, offsets) for img, offsets in zip(
                [red_B, green_B, blue_B],
                [estimated_offsets_R, estimated_offsets_G, estimated_offsets_B]
            )
        ]

        warped_A_R = warped_A_R.permute(0, 3, 1, 2)
        warped_A_G = warped_A_G.permute(0, 3, 1, 2)
        warped_A_B = warped_A_B.permute(0, 3, 1, 2)

        encoded_red_A, encoded_green_A, encoded_blue_A = [self.image_encoder(img) for img in [warped_A_R, warped_A_G, warped_A_B]]

        output_feat_R, output_feat_G, output_feat_B = [self.fusion_model(past, curr) for past, curr in zip([encoded_red_A, encoded_green_A, encoded_blue_A], [output_R, output_G, output_B])]

        upsampled_output_R, upsampled_output_G, upsampled_output_B = [self.upsampler(feat) for feat in [output_feat_R, output_feat_G, output_feat_B]]

        upscale_factor = 2

        red_A_temp = red_A.permute(0, 3, 1, 2)
        red_A_temp = red_A_temp.repeat(1, 4, 1, 1)
        green_A_temp = green_A.permute(0, 3, 1, 2)
        green_A_temp = green_A_temp.repeat(1, 4, 1, 1)
        blue_A_temp = blue_A.permute(0, 3, 1, 2)
        blue_A_temp = blue_A_temp.repeat(1, 4, 1, 1)

        A_upscaled_R = F.pixel_shuffle(red_A_temp, upscale_factor)
        A_upscaled_G = F.pixel_shuffle(green_A_temp, upscale_factor)
        A_upscaled_B = F.pixel_shuffle(blue_A_temp, upscale_factor)

        sum_upsampled_output_A_upscaled_R = upsampled_output_R + A_upscaled_R
        sum_upsampled_output_A_upscaled_G = upsampled_output_G + A_upscaled_G
        sum_upsampled_output_A_upscaled_B = upsampled_output_B + A_upscaled_B

        sum_upsampled_output_A_upscaled_R, sum_upsampled_output_A_upscaled_G, sum_upsampled_output_A_upscaled_B = [rescale_tensor(img) for img in [sum_upsampled_output_A_upscaled_R, sum_upsampled_output_A_upscaled_G, sum_upsampled_output_A_upscaled_B]]

        red_channel, green_channel, blue_channel = [img[0, i, :, :] for i, img in enumerate([sum_upsampled_output_A_upscaled_R, sum_upsampled_output_A_upscaled_G, sum_upsampled_output_A_upscaled_B])]

        target_shape = red_channel.shape
        green_channel_resized = F.interpolate(green_channel.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        blue_channel_resized = F.interpolate(blue_channel.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        synthetic_tensor = torch.stack([red_channel, green_channel_resized, blue_channel_resized], dim=0)
        
        return synthetic_tensor  # Return only the synthetic tensor for prediction

# Load and preprocess images
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.0  # Normalize to [0, 1]
    img = img.unsqueeze(0)  # Add batch dimension: [1, H, W, 3]
    return img

# Paths to your images
guide_path = r"D:\Desmoking Dataset\LH_frames\1970_01_01_010226_LH_S6\smoke_ahead_series\00002\frames_0000206.jpg"  # Image A (guide)
input_path = r"D:\Desmoking Dataset\LH_frames\1970_01_01_010226_LH_S6\smoke_ahead_series\00002\frames_0000208.jpg"  # Image B (input)
model_path = r"C:\Users\User\Desktop\常用程式\rbsr_model_epoch_1.pth"

# Load images
guide_img = load_image(guide_path).to(device)
input_img = load_image(input_path).to(device)

# Initialize model
model = RBSR_X().to(device)

# Load pre-trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to evaluation mode

# Make prediction
with torch.no_grad():
    synthetic_tensor = model(input_img, guide_img)  # curr_frame = Image B, past_frame = Image A

# Post-process the output
synthetic_tensor = synthetic_tensor.squeeze(0)  # Remove batch dimension if present
synthetic_tensor = synthetic_tensor.cpu().numpy()  # Convert to numpy
synthetic_tensor = np.transpose(synthetic_tensor, (1, 2, 0))  # Change from [C, H, W] to [H, W, C]
synthetic_image = ((synthetic_tensor - synthetic_tensor.min()) / (synthetic_tensor.max() - synthetic_tensor.min()) * 255).astype(np.uint8)

# Convert to PIL and show
image_pil = Image.fromarray(synthetic_image)
image_pil.show()

# Optional: Save the image using PIL
# output_path = r"C:\Users\User\Desktop\predicted_synthetic_image_B.jpg"
# image_pil.save(output_path)

# print(f"Predicted synthetic image saved at: {output_path}")


