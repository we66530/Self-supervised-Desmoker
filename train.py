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
from PIL import Image
import time
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

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












# Define a dataset class for loading image pairs from subfolders
class ImagePairDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.subfolders = [os.path.join(root_dir, subfolder) for subfolder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subfolder))]
        self.image_pairs = []
        
        # Load image pairs from each subfolder
        for subfolder in self.subfolders:
            images = sorted([os.path.join(subfolder, img) for img in os.listdir(subfolder) if img.endswith('.png')])
            for i in range(1, len(images)):
                self.image_pairs.append((images[0], images[i]))  # Pair the first image (Image_A) with the others (Image_B)
                
    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_a_path, img_b_path = self.image_pairs[idx]

        # Load images using OpenCV
        img_a = cv2.imread(img_a_path)
        img_b = cv2.imread(img_b_path)
        
        # Convert images to RGB
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
        
        return img_a, img_b






# # Initialize the dataset and dataloader
# root_dir = r"D:\Self Supervised Video Desmoking for Laparoscopic Surgery\LSVD_train"
# dataset = ImagePairDataset(root_dir)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



# # Get the first batch from the dataloader
# first_img_a, first_img_b = next(iter(dataloader))
# first_img_a, first_img_b = first_img_a.to(device), first_img_b.to(device)


# # Debug: Print shape or show the images
# print(f"First batch img_a shape: {first_img_a.shape}")
# print(f"First batch img_b shape: {first_img_b.shape}")

# red_A, green_A, blue_A = [extract_channel(first_img_a, i) for i in range(3)]
# red_B, green_B, blue_B = [extract_channel(first_img_b, i) for i in range(3)]

# print(f"red_A shape: {red_A.shape}")
# print(f"red_B shape: {red_B.shape}")

# # Compute optical flow
# flow_R, flow_G, flow_B = [estimate_optical_flow(a, b) for a, b in zip([red_A, green_A, blue_A], [red_B, green_B, blue_B])]

# print(f"flow_R shape: {flow_R.shape}")


# # Warp images using respective flows
# warped_A_R = warp_image(red_A, flow_R)
# warped_A_G = warp_image(green_A, flow_G)
# warped_A_B = warp_image(blue_A, flow_B)

# print(f"warped_A_R shape: {warped_A_R.shape}")

# # Convert warped tensors to numpy
# warped_red_np = tensor_to_image(warped_A_R)   # (1080, 1920, 3)
# warped_green_np = tensor_to_image(warped_A_G) # (1080, 1920, 3)
# warped_blue_np = tensor_to_image(warped_A_B)  # (1080, 1920, 3)

# # Merge warped channels correctly
# warped_A = np.stack([
#     warped_red_np[:, :, 0],  # Red channel
#     warped_green_np[:, :, 1],  # Green channel
#     warped_blue_np[:, :, 2]   # Blue channel
# ], axis=-1)  # Shape: (1080, 1920, 3)

# print(f"warped_A shape: {warped_A.shape}")


# # Initialize models on GPU
# offset_estimator = OpticalFlowOffsetEstimator().to(device)
# flow_guided_dcn = FlowGuidedDCN(in_channels=3, out_channels=16).to(device)
# image_encoder = ImageEncoder().to(device)
# fusion_model = PastFramePropagation(16).to(device)
# upsampler = Upsampler().to(device)


# # Predict offsets and move tensors to GPU
# estimated_offsets_R, estimated_offsets_G, estimated_offsets_B = [offset_estimator(flow) for flow in [flow_R, flow_G, flow_B]]

# print(f"estimated_offsets_R shape: {estimated_offsets_R.shape}")

# # Convert img to (1, 3, H, W) format if needed
# red_B = red_B.permute(0, 3, 1, 2).float() if red_B.shape[-1] == 3 else red_B
# green_B = green_B.permute(0, 3, 1, 2).float() if green_B.shape[-1] == 3 else green_B
# blue_B = blue_B.permute(0, 3, 1, 2).float() if blue_B.shape[-1] == 3 else blue_B

# print(f"red_B shape: {red_B.shape}")

# # Apply flow-guided deformable convolution
# output_R, output_G, output_B = [
#     flow_guided_dcn(img, offsets) for img, offsets in zip(
#         [red_B, green_B, blue_B],
#         [estimated_offsets_R, estimated_offsets_G, estimated_offsets_B]
#     )
# ]
# print(f"output_R shape: {output_R.shape}")


# # Convert from (1, 1080, 1920, 3) to (1, 3, 1080, 1920) by permuting the axes
# warped_A_R = warped_A_R.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)
# warped_A_G = warped_A_G.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)
# warped_A_B = warped_A_B.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)


# # Encode past frame (image_A)
# encoded_red_A, encoded_green_A, encoded_blue_A = [image_encoder(img) for img in [warped_A_R, warped_A_G, warped_A_B]]
# print(f"encoded_red_A shape: {encoded_red_A.shape}")


# # Fusion of past and current features
# output_feat_R, output_feat_G, output_feat_B = [fusion_model(past, curr) for past, curr in zip([encoded_red_A, encoded_green_A, encoded_blue_A], [output_R, output_G, output_B])]
# print(f"output_feat_R shape: {output_feat_R.shape}")

# # Upsample the results
# upsampled_output_R, upsampled_output_G, upsampled_output_B = [upsampler(feat) for feat in [output_feat_R, output_feat_G, output_feat_B]]
# print(f"upsampled_output_R shape: {upsampled_output_R.shape}")


# # Define the upscale factor
# upscale_factor = 2

# # Prepare the input tensors by permuting and repeating the channels to make them divisible by 4
# red_A_temp = red_A.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)
# red_A_temp = red_A_temp.repeat(1, 4, 1, 1)  # Repeat the channels to make the channels 12 (3 * 4)

# green_A_temp = green_A.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)
# green_A_temp = green_A_temp.repeat(1, 4, 1, 1)  # Repeat the channels to make the channels 12

# blue_A_temp = blue_A.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)
# blue_A_temp = blue_A_temp.repeat(1, 4, 1, 1)  # Repeat the channels to make the channels 12

# # Print the shape after preparation
# print(f"red_A_temp shape: {red_A_temp.shape}")

# # Apply PixelShuffle to upscale the images
# A_upscaled_R = F.pixel_shuffle(red_A_temp, upscale_factor)  # Output shape: (1, 3, 2160, 3840)
# A_upscaled_G = F.pixel_shuffle(green_A_temp, upscale_factor)  # Output shape: (1, 3, 2160, 3840)
# A_upscaled_B = F.pixel_shuffle(blue_A_temp, upscale_factor)  # Output shape: (1, 3, 2160, 3840)

# # Print the shape of the upscaled images
# print(f"A_upscaled_R shape: {A_upscaled_R.shape}")
# print(f"upsampled_output_R shape: {upsampled_output_R.shape}")

# sum_upsampled_output_A_upscaled_R = upsampled_output_R + A_upscaled_R
# sum_upsampled_output_A_upscaled_G = upsampled_output_G + A_upscaled_G
# sum_upsampled_output_A_upscaled_B = upsampled_output_B + A_upscaled_B


# sum_upsampled_output_A_upscaled_R, sum_upsampled_output_A_upscaled_G, sum_upsampled_output_A_upscaled_B = [rescale_tensor(img) for img in [sum_upsampled_output_A_upscaled_R, sum_upsampled_output_A_upscaled_G, sum_upsampled_output_A_upscaled_B]]
# print(f"A_upscaled_rescaled_R shape: {sum_upsampled_output_A_upscaled_R.shape}")

# # Extract color channels
# red_channel, green_channel, blue_channel = [img[0, i, :, :] for i, img in enumerate([sum_upsampled_output_A_upscaled_R, sum_upsampled_output_A_upscaled_G, sum_upsampled_output_A_upscaled_B])]
# print(f"red_channel shape: {red_channel.shape}")


# # Resize each channel to match the size of the largest one (e.g., red_channel's shape)
# target_shape = red_channel.shape  # Use red_channel's shape

# # Resize all channels to the same shape
# green_channel_resized = F.interpolate(green_channel.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
# blue_channel_resized = F.interpolate(blue_channel.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

# # Stack the channels and keep it as a torch tensor
# synthetic_tensor = torch.stack([red_channel, green_channel_resized, blue_channel_resized], dim=0)
# print(f"synthetic_tensor shape: {synthetic_tensor.shape}")




# ## Mask Generation
# # Load images with alpha channel (BGRA format) or grayscale
# # Convert warped_A from NumPy to a PyTorch tensor
# warped_A_temp = torch.tensor(warped_A)  # Make sure it's a float32 tensor if needed


# image_S_ref_to_i = warped_A_temp.unsqueeze(0)
# image_S_i = first_img_b

# print(f"image_S_ref_to_i shape: {image_S_ref_to_i.shape}")


# # Apply dark channel to the image
# dark_channel_image_S_ref_to_i = dark_channel(image_S_ref_to_i)
# dark_channel_image_S_i = dark_channel(image_S_i)

# print(f"dark_channel_image_S_ref_to_i shape: {dark_channel_image_S_ref_to_i.shape}")

# # Apply Gaussian blur to the dark channel result
# blurred_dark_channel_image_S_ref_to_i = apply_gaussian_blur(dark_channel_image_S_ref_to_i)
# blurred_dark_channel_image_S_i = apply_gaussian_blur(dark_channel_image_S_i)

# # Convert to NumPy array and remove the singleton dimension
# blurred_dark_channel_image_S_ref_to_i = blurred_dark_channel_image_S_ref_to_i.squeeze(0).cpu().numpy()
# blurred_dark_channel_image_S_i = blurred_dark_channel_image_S_i.squeeze(0).cpu().numpy()

# print(f"blurred_dark_channel_image_S_ref_to_i shape: {blurred_dark_channel_image_S_ref_to_i.shape}")


# # Normalize image to [0, 1]
# image_S_ref_to_i = blurred_dark_channel_image_S_ref_to_i / np.max(blurred_dark_channel_image_S_ref_to_i)
# image_S_i = blurred_dark_channel_image_S_i / np.max(blurred_dark_channel_image_S_i)

# print(f"image_S_ref_to_i shape: {image_S_ref_to_i.shape}")

# # Parameters
# patch_size = 8
# epsilon = 0.92  # Threshold


# # Split images into patches for each channel
# patches_S_ref_to_i = split_into_patches(image_S_ref_to_i, patch_size)
# patches_S_i = split_into_patches(image_S_i, patch_size)


# # Calculate Mi^p for each patch (RGB 3 channels)
# Mi_p = [
#     np.sign(max(0, np.mean([ssim(ref_patch, input_patch, data_range=1.0)]) - epsilon))
#     for ref_patch, input_patch in zip(patches_S_ref_to_i, patches_S_i)
# ]

# # Convert Mi^p into a full-sized mask (1080 x 1920)
# mask = np.zeros_like(image_S_ref_to_i)
# idx = 0
# for i in range(0, image_S_ref_to_i.shape[0], patch_size):
#     for j in range(0, image_S_ref_to_i.shape[1], patch_size):
#         mask[i:i + patch_size, j:j + patch_size] = Mi_p[idx]
#         idx += 1

# # Output confirmation
# print(f"Mask shape: {mask.shape}")


# # Normalize the mask to fit into a displayable range if needed
# mask_normalized = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize to 0-1


# # Assuming mask_normalized and warped_A are numpy arrays of the same shape
# # Expand mask_normalized to match the 3 channels of warped_A
# mask_normalized_expanded = np.expand_dims(mask_normalized, axis=-1)

# # Now perform pixel-wise multiplication
# output_image = warped_A * mask_normalized_expanded

# # Normalize output_image to the range [0, 255] if it's not already
# output_image = np.clip(output_image, 0, 255)


# ## Compute L1 regularization loss
# l1_loss = np.sum(np.abs(output_image)) / 10000000
# print("L1 Regularization Loss:", l1_loss)






# ## Compute Reconstruction Loss
# # Ensure correct slicing of dimensions
# flow_combined = estimate_optical_flow(first_img_a, first_img_b)
# flow_combined = flow_combined.squeeze(0)
# print(f"flow_combined shape: {flow_combined.shape}")


# # Create the all-1 matrix (shape = (1080, 1920))
# H, W = flow_combined.shape[1:]  # Get height and width from flow
# all_ones = torch.ones(1, 1, H, W, device=device)

# # Prepare the grid for warping
# grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
# grid = torch.stack((grid_x, grid_y), dim=0).float().to(device)  # Ensure grid is on the same device

# grid = grid + flow_combined  # Now both are on the same device
# grid[0] = 2.0 * grid[0] / (W - 1) - 1.0  # Normalize x to [-1, 1]
# grid[1] = 2.0 * grid[1] / (H - 1) - 1.0  # Normalize y to [-1, 1]
# grid = grid.permute(1, 2, 0)  # (H, W, 2)

# # Warp the matrix using grid_sample
# grid = grid.unsqueeze(0)  # Add batch dimension (1, H, W, 2)
# warped_matrix = F.grid_sample(all_ones, grid, mode="bilinear", align_corners=True)

# # Process the warped matrix with the specified formula
# tau = 0.999
# processed_matrix = torch.sign(torch.clamp(warped_matrix - tau, min=0))
# processed_matrix = processed_matrix.squeeze(0)
# print(f"processed_matrix.shape: {processed_matrix.shape}")



# # Warp images using respective flows
# flow_reverse = estimate_optical_flow(first_img_b, first_img_a)
# print(f"first_img_b.shape: {first_img_b.shape}")
# print(f"flow_reverse.shape: {flow_reverse.shape}")


# warped_B = warp_image(first_img_b, flow_reverse)
# print(f"warped_B.shape: {warped_B.shape}")

# # Expand processed_matrix to match the shape of warped_B
# expanded_processed_matrix = processed_matrix.unsqueeze(-1)  # Shape: [1, 1080, 1920, 1]
# expanded_processed_matrix = expanded_processed_matrix.repeat(1, 1, 1, 3)  # Shape: [1, 1080, 1920, 3]

# # Perform pixel-wise multiplication
# result = warped_B * expanded_processed_matrix
# print(f"result.shape: {result.shape}")


# # If you need a numeric scalar result (e.g., the sum of all pixel values)
# Rec_Loss = result.sum().item()/1000000

# # Print the result tensor shape and numeric result
# print(f"Reconstruction Loss: {Rec_Loss}")



# # Instantiate the discriminator
# discriminator = Discriminator().to(device)
# # Initialize GANLoss
# gan_loss = vanilla_GANLoss(real_label_val=1.0, fake_label_val=0.0)


# # Transpose the image to (1, 3, 1080, 1920)
# image_A_transposed = first_img_a.permute(0, 3, 1, 2).float()  # Shape becomes (1, 3, 1080, 1920)

# # Resize the image using interpolate (bicubic by default, can change mode if needed)
# image_A_resized = F.interpolate(image_A_transposed, size=(2160, 3840), mode='bilinear', align_corners=False)

# # Ensure the tensor is float (if not already)
# image_A_resized = image_A_resized.float()
# synthetic_tensor_temp = synthetic_tensor.unsqueeze(0).float()

# print(f"image_A_resized.shape: {image_A_resized.shape}")
# print(f"synthetic_tensor_temp.shape: {synthetic_tensor_temp.shape}")

# # Move the input tensor to the same device
# image_A_resized = image_A_resized.to(device)
# synthetic_tensor_temp = image_A_resized.to(device)


# D_real = discriminator(image_A_resized)  # Discriminator output for real images
# D_fake = discriminator(synthetic_tensor_temp)  # Discriminator output for fake images

# # Compute discriminator loss (real + fake)
# loss_D_real = gan_loss(D_real, target_is_real=True, is_disc=True)  # Real loss
# loss_D_fake = gan_loss(D_fake, target_is_real=False, is_disc=True)  # Fake loss
# loss_D = (loss_D_real + loss_D_fake) * 0.5  # Total discriminator loss

# # Compute generator loss (goal: fool the discriminator)
# loss_G = gan_loss(D_fake, target_is_real=True, is_disc=False)  # Generator loss


# print(f"loss_D:{loss_D}")
# print(f"loss_G:{loss_G}")



class RBSR_X(nn.Module):
    def __init__(self):
        super(RBSR_X, self).__init__()
        
        # Initialize components
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

        # Compute optical flow
        flow_R, flow_G, flow_B = [estimate_optical_flow(a, b) for a, b in zip([red_A, green_A, blue_A], [red_B, green_B, blue_B])]

        # Warp images using respective flows
        warped_A_R = warp_image(red_A, flow_R)
        warped_A_G = warp_image(green_A, flow_G)
        warped_A_B = warp_image(blue_A, flow_B)

        # Convert warped tensors to numpy
        warped_red_np = tensor_to_image(warped_A_R)   # (1080, 1920, 3)
        warped_green_np = tensor_to_image(warped_A_G) # (1080, 1920, 3)
        warped_blue_np = tensor_to_image(warped_A_B)  # (1080, 1920, 3)

        # Merge warped channels correctly
        warped_A = np.stack([
            warped_red_np[:, :, 0],  # Red channel
            warped_green_np[:, :, 1],  # Green channel
            warped_blue_np[:, :, 2]   # Blue channel
        ], axis=-1)  # Shape: (1080, 1920, 3)

        # Predict offsets and move tensors to GPU
        estimated_offsets_R, estimated_offsets_G, estimated_offsets_B = [self.offset_estimator(flow) for flow in [flow_R, flow_G, flow_B]]

        # Convert img to (1, 3, H, W) format if needed
        red_B = red_B.permute(0, 3, 1, 2).float() if red_B.shape[-1] == 3 else red_B
        green_B = green_B.permute(0, 3, 1, 2).float() if green_B.shape[-1] == 3 else green_B
        blue_B = blue_B.permute(0, 3, 1, 2).float() if blue_B.shape[-1] == 3 else blue_B

        # Apply flow-guided deformable convolution
        output_R, output_G, output_B = [
            self.flow_guided_dcn(img, offsets) for img, offsets in zip(
                [red_B, green_B, blue_B],
                [estimated_offsets_R, estimated_offsets_G, estimated_offsets_B]
            )
        ]

        # Convert from (1, 1080, 1920, 3) to (1, 3, 1080, 1920) by permuting the axes
        warped_A_R = warped_A_R.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)
        warped_A_G = warped_A_G.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)
        warped_A_B = warped_A_B.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)

        # Encode past frame (image_A)
        encoded_red_A, encoded_green_A, encoded_blue_A = [self.image_encoder(img) for img in [warped_A_R, warped_A_G, warped_A_B]]

        # Fusion of past and current features
        output_feat_R, output_feat_G, output_feat_B = [self.fusion_model(past, curr) for past, curr in zip([encoded_red_A, encoded_green_A, encoded_blue_A], [output_R, output_G, output_B])]

        # Upsample the results
        upsampled_output_R, upsampled_output_G, upsampled_output_B = [self.upsampler(feat) for feat in [output_feat_R, output_feat_G, output_feat_B]]

        # Define the upscale factor
        upscale_factor = 2

        # Prepare the input tensors by permuting and repeating the channels to make them divisible by 4
        red_A_temp = red_A.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)
        red_A_temp = red_A_temp.repeat(1, 4, 1, 1)  # Repeat the channels to make the channels 12 (3 * 4)

        green_A_temp = green_A.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)
        green_A_temp = green_A_temp.repeat(1, 4, 1, 1)  # Repeat the channels to make the channels 12

        blue_A_temp = blue_A.permute(0, 3, 1, 2)  # Change shape to (1, 3, 1080, 1920)
        blue_A_temp = blue_A_temp.repeat(1, 4, 1, 1)  # Repeat the channels to make the channels 12

        # Apply PixelShuffle to upscale the images
        A_upscaled_R = F.pixel_shuffle(red_A_temp, upscale_factor)  # Output shape: (1, 3, 2160, 3840)
        A_upscaled_G = F.pixel_shuffle(green_A_temp, upscale_factor)  # Output shape: (1, 3, 2160, 3840)
        A_upscaled_B = F.pixel_shuffle(blue_A_temp, upscale_factor)  # Output shape: (1, 3, 2160, 3840)

        sum_upsampled_output_A_upscaled_R = upsampled_output_R + A_upscaled_R
        sum_upsampled_output_A_upscaled_G = upsampled_output_G + A_upscaled_G
        sum_upsampled_output_A_upscaled_B = upsampled_output_B + A_upscaled_B

        sum_upsampled_output_A_upscaled_R, sum_upsampled_output_A_upscaled_G, sum_upsampled_output_A_upscaled_B = [rescale_tensor(img) for img in [sum_upsampled_output_A_upscaled_R, sum_upsampled_output_A_upscaled_G, sum_upsampled_output_A_upscaled_B]]

        # Extract color channels
        red_channel, green_channel, blue_channel = [img[0, i, :, :] for i, img in enumerate([sum_upsampled_output_A_upscaled_R, sum_upsampled_output_A_upscaled_G, sum_upsampled_output_A_upscaled_B])]

        # Resize each channel to match the size of the largest one (e.g., red_channel's shape)
        target_shape = red_channel.shape  # Use red_channel's shape

        # Resize all channels to the same shape
        green_channel_resized = F.interpolate(green_channel.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        blue_channel_resized = F.interpolate(blue_channel.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Stack the channels and keep it as a torch tensor
        synthetic_tensor = torch.stack([red_channel, green_channel_resized, blue_channel_resized], dim=0)


        ## Mask Generation
        # Load images with alpha channel (BGRA format) or grayscale
        # Convert warped_A from NumPy to a PyTorch tensor
        warped_A_temp = torch.tensor(warped_A)  # Make sure it's a float32 tensor if needed

        image_S_ref_to_i = warped_A_temp.unsqueeze(0)
        image_S_i = first_img_b

        # Apply dark channel to the image
        dark_channel_image_S_ref_to_i = dark_channel(image_S_ref_to_i)
        dark_channel_image_S_i = dark_channel(image_S_i)

        # Apply Gaussian blur to the dark channel result
        blurred_dark_channel_image_S_ref_to_i = apply_gaussian_blur(dark_channel_image_S_ref_to_i)
        blurred_dark_channel_image_S_i = apply_gaussian_blur(dark_channel_image_S_i)

        # Convert to NumPy array and remove the singleton dimension
        blurred_dark_channel_image_S_ref_to_i = blurred_dark_channel_image_S_ref_to_i.squeeze(0).cpu().numpy()
        blurred_dark_channel_image_S_i = blurred_dark_channel_image_S_i.squeeze(0).cpu().numpy()

        # Normalize image to [0, 1]
        image_S_ref_to_i = blurred_dark_channel_image_S_ref_to_i / np.max(blurred_dark_channel_image_S_ref_to_i)
        image_S_i = blurred_dark_channel_image_S_i / np.max(blurred_dark_channel_image_S_i)

        # Parameters
        patch_size = 8
        epsilon = 0.92  # Threshold

        # Split images into patches for each channel
        patches_S_ref_to_i = split_into_patches(image_S_ref_to_i, patch_size)
        patches_S_i = split_into_patches(image_S_i, patch_size)


        # Calculate Mi^p for each patch (RGB 3 channels)
        Mi_p = [
            np.sign(max(0, np.mean([ssim(ref_patch, input_patch, data_range=1.0)]) - epsilon))
            for ref_patch, input_patch in zip(patches_S_ref_to_i, patches_S_i)
        ]

        # Convert Mi^p into a full-sized mask (1080 x 1920)
        mask = np.zeros_like(image_S_ref_to_i)
        idx = 0
        for i in range(0, image_S_ref_to_i.shape[0], patch_size):
            for j in range(0, image_S_ref_to_i.shape[1], patch_size):
                mask[i:i + patch_size, j:j + patch_size] = Mi_p[idx]
                idx += 1

        # Normalize the mask to fit into a displayable range if needed
        mask_normalized = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize to 0-1

        # Assuming mask_normalized and warped_A are numpy arrays of the same shape
        # Expand mask_normalized to match the 3 channels of warped_A
        mask_normalized_expanded = np.expand_dims(mask_normalized, axis=-1)

        # Now perform pixel-wise multiplication
        output_image = warped_A * mask_normalized_expanded

        # Normalize output_image to the range [0, 255] if it's not already
        output_image = np.clip(output_image, 0, 255)

        ## Compute L1 regularization loss
        l1_loss = np.sum(np.abs(output_image)) / 10000000



        ## Compute Reconstruction Loss
        # Ensure correct slicing of dimensions
        flow_combined = estimate_optical_flow(first_img_a, first_img_b)
        flow_combined = flow_combined.squeeze(0)

        # Create the all-1 matrix (shape = (1080, 1920))
        H, W = flow_combined.shape[1:]  # Get height and width from flow
        all_ones = torch.ones(1, 1, H, W, device=device)

        # Prepare the grid for warping
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(device)  # Ensure grid is on the same device

        grid = grid + flow_combined  # Now both are on the same device
        grid[0] = 2.0 * grid[0] / (W - 1) - 1.0  # Normalize x to [-1, 1]
        grid[1] = 2.0 * grid[1] / (H - 1) - 1.0  # Normalize y to [-1, 1]
        grid = grid.permute(1, 2, 0)  # (H, W, 2)

        # Warp the matrix using grid_sample
        grid = grid.unsqueeze(0)  # Add batch dimension (1, H, W, 2)
        warped_matrix = F.grid_sample(all_ones, grid, mode="bilinear", align_corners=True)

        # Process the warped matrix with the specified formula
        tau = 0.999
        processed_matrix = torch.sign(torch.clamp(warped_matrix - tau, min=0))
        processed_matrix = processed_matrix.squeeze(0)

        # Warp images using respective flows
        flow_reverse = estimate_optical_flow(first_img_b, first_img_a)

        warped_B = warp_image(first_img_b, flow_reverse)

        # Expand processed_matrix to match the shape of warped_B
        expanded_processed_matrix = processed_matrix.unsqueeze(-1)  # Shape: [1, 1080, 1920, 1]
        expanded_processed_matrix = expanded_processed_matrix.repeat(1, 1, 1, 3)  # Shape: [1, 1080, 1920, 3]

        # Perform pixel-wise multiplication
        result = warped_B * expanded_processed_matrix

        # If you need a numeric scalar result (e.g., the sum of all pixel values)
        Rec_Loss = result.sum().item()/1000000



        # Initialize GANLoss
        gan_loss = vanilla_GANLoss(real_label_val=1.0, fake_label_val=0.0)

        # Transpose the image to (1, 3, 1080, 1920)
        image_A_transposed = first_img_a.permute(0, 3, 1, 2).float()  # Shape becomes (1, 3, 1080, 1920)

        # Resize the image using interpolate (bicubic by default, can change mode if needed)
        image_A_resized = F.interpolate(image_A_transposed, size=(2160, 3840), mode='bilinear', align_corners=False)

        # Ensure the tensor is float (if not already)
        image_A_resized = image_A_resized.float()
        synthetic_tensor_temp = synthetic_tensor.unsqueeze(0).float()

        # Move the input tensor to the same device
        image_A_resized = image_A_resized.to(device)
        synthetic_tensor_temp = image_A_resized.to(device)

        D_real = self.discriminator(image_A_resized)  # Discriminator output for real images
        D_fake = self.discriminator(synthetic_tensor_temp)  # Discriminator output for fake images

        # Compute discriminator loss (real + fake)
        loss_D_real = gan_loss(D_real, target_is_real=True, is_disc=True)  # Real loss
        loss_D_fake = gan_loss(D_fake, target_is_real=False, is_disc=True)  # Fake loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5  # Total discriminator loss

        # Compute generator loss (goal: fool the discriminator)
        loss_G = gan_loss(D_fake, target_is_real=True, is_disc=False)  # Generator loss

        return synthetic_tensor, l1_loss, Rec_Loss, loss_D, loss_G
    




if __name__ == '__main__':  # ✅ Add this to fix the multiprocessing error
    # Initialize dataset and dataloader
    root_dir = r"D:\Self Supervised Video Desmoking for Laparoscopic Surgery\LSVD_train"
    dataset = ImagePairDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)

    # Initialize model and optimizer
    model = RBSR_X().to(device)  # Ensure you use the appropriate device (GPU/CPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()

        # Initialize tqdm for progress tracking
        pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

        running_loss = 0.0
        for img_a, img_b in pbar:
            img_a, img_b = img_a.to(device), img_b.to(device)

            optimizer.zero_grad()

            # Forward pass
            synthetic_tensor, l1_loss, Rec_Loss, loss_D, loss_G = model(img_a, img_b)

            # Total loss
            total_loss = l1_loss + Rec_Loss + loss_D + loss_G
            running_loss += total_loss.item()

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            # Update the progress bar
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}, Time: {epoch_duration:.2f}s")

        # Save model periodically
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f"rbsr_model_epoch_{epoch+1}.pth")

    # Final model saving
    torch.save(model.state_dict(), "rbsr_model_final.pth")