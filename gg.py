import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import deform_conv2d
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(16)  # Utilize 16 CPU threads for operations that run on CPU

class FlowGuidedDCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(FlowGuidedDCN, self).__init__()
        self.dcn = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    
    def forward(self, x, offsets):
        return deform_conv2d(x, offsets, self.dcn.weight, bias=self.dcn.bias, padding=1)

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

def estimate_optical_flow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = np.transpose(flow, (2, 0, 1))
    return torch.tensor(flow).unsqueeze(0).float().to(device)  # Move to GPU


def warp_image(image, flow):
    """
    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W)
        flow (torch.Tensor): Optical flow tensor of shape (1, 2, H, W)
    Returns:
        torch.Tensor: Warped image of shape (C, H, W)
    """
    _, H, W = image.shape

    # Create mesh grid
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid = torch.stack((x, y), dim=0).float().to(device)  # (2, H, W)

    # Normalize to [-1, 1] range for grid_sample
    flow = flow.squeeze(0)  # Remove batch dimension (2, H, W)
    grid = grid + flow  # Apply flow displacement
    grid[0] = 2.0 * grid[0] / (W - 1) - 1.0  # Normalize X-coords
    grid[1] = 2.0 * grid[1] / (H - 1) - 1.0  # Normalize Y-coords
    grid = grid.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 2)

    # Prepare image tensor for grid_sample
    image = image.unsqueeze(0)  # Add batch dimension (1, C, H, W)

    # Warp image using grid sampling
    warped_image = F.grid_sample(image, grid, mode="bilinear", align_corners=True)

    return warped_image.squeeze(0)  # Remove batch dimension (C, H, W)

# Convert images to tensors and normalize
def image_to_tensor(image):
    return torch.tensor(image.transpose(2, 0, 1)).float().to(device) / 255.0



# Load images
image_A = cv2.imread("D:\\Desmoking Dataset\\LH_frames\\1970_01_01_010226_LH_S6\\smoke_ahead_series\\00001\\frames_0000193.jpg")
image_B = cv2.imread("D:\\Desmoking Dataset\\LH_frames\\1970_01_01_010226_LH_S6\\smoke_ahead_series\\00001\\frames_0000194.jpg")

# Convert images to RGB and extract color channels
image_A, image_B = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in [image_A, image_B]]

def extract_channel(img, index):
    output = np.zeros_like(img)
    output[:, :, index] = img[:, :, index]
    return output

red_A, green_A, blue_A = [extract_channel(image_A, i) for i in range(3)]
red_B, green_B, blue_B = [extract_channel(image_B, i) for i in range(3)]

# Compute optical flow
flow_R, flow_G, flow_B = [estimate_optical_flow(a, b) for a, b in zip([red_A, green_A, blue_A], [red_B, green_B, blue_B])]



red_A_t, green_A_t, blue_A_t = [image_to_tensor(img) for img in [red_A, green_A, blue_A]]

# Warp images using respective flows
warped_A_R = warp_image(red_A_t, flow_R)
warped_A_G = warp_image(green_A_t, flow_G)
warped_A_B = warp_image(blue_A_t, flow_B)

# Convert back to numpy for visualization
def tensor_to_image(tensor):
    return (tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

warped_red_np = tensor_to_image(warped_A_R)
warped_green_np = tensor_to_image(warped_A_G)
warped_blue_np = tensor_to_image(warped_A_B)

# Merge warped channels
warped_A = np.stack([warped_red_np[:, :, 0], warped_green_np[:, :, 1], warped_blue_np[:, :, 2]], axis=-1)




# Initialize models on GPU
offset_estimator = OpticalFlowOffsetEstimator().to(device)
flow_guided_dcn = FlowGuidedDCN(in_channels=3, out_channels=16).to(device)
image_encoder = ImageEncoder().to(device)
fusion_model = PastFramePropagation(16).to(device)
upsampler = Upsampler().to(device)

# Predict offsets and move tensors to GPU
estimated_offsets_R, estimated_offsets_G, estimated_offsets_B = [offset_estimator(flow) for flow in [flow_R, flow_G, flow_B]]

def prepare_tensor(img):
    if isinstance(img, torch.Tensor):  # If already a tensor, ensure it's in the right shape and device
        return img.unsqueeze(0).to(device)
    img = torch.tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0).float().to(device)
    return img

red_B, green_B, blue_B = [prepare_tensor(img) for img in [red_B, green_B, blue_B]]

# Perform flow-guided deformable convolution
output_R, output_G, output_B = [flow_guided_dcn(img, offsets) for img, offsets in zip([red_B, green_B, blue_B], [estimated_offsets_R, estimated_offsets_G, estimated_offsets_B])]


# Encode past frame (image_A)
red_A, green_A, blue_A = [prepare_tensor(img) for img in [warped_A_R, warped_A_G, warped_A_B]]
encoded_red_A, encoded_green_A, encoded_blue_A = [image_encoder(img) for img in [red_A, green_A, blue_A]]




# Fusion of past and current features
output_feat_R, output_feat_G, output_feat_B = [fusion_model(past, curr) for past, curr in zip([encoded_red_A, encoded_green_A, encoded_blue_A], [output_R, output_G, output_B])]

# Upsample the results
upsampled_output_R, upsampled_output_G, upsampled_output_B = [upsampler(feat) for feat in [output_feat_R, output_feat_G, output_feat_B]]

# Prepare for PixelShuffle
upscale_factor = 2
red_A_temp, green_A_temp, blue_B_temp = [img.repeat(1, 4, 1, 1) for img in [red_A, green_A, blue_B]]

A_upscaled_R, A_upscaled_G, A_upscaled_B = [F.pixel_shuffle(img, upscale_factor) for img in [red_A_temp, green_A_temp, blue_B_temp]]

# Rescale to [0, 255]
def rescale_tensor(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).clamp(0, 255)

A_upscaled_rescaled_R, A_upscaled_rescaled_G, A_upscaled_rescaled_B = [rescale_tensor(img) for img in [A_upscaled_R, A_upscaled_G, A_upscaled_B]]

# Extract color channels
red_channel, green_channel, blue_channel = [img[0, i, :, :] for i, img in enumerate([A_upscaled_rescaled_R, A_upscaled_rescaled_G, A_upscaled_rescaled_B])]

# Stack and convert to NumPy
tensor_rgb = torch.stack([red_channel, green_channel, blue_channel], dim=0).cpu().numpy().astype(np.uint8)
image_np = np.transpose(tensor_rgb, (1, 2, 0))

# Convert to PIL and show
# image_pil = Image.fromarray(image_np)
# image_pil.show()



## Mask Generation
# Function to compute the dark channel
def dark_channel(image, patch_size=15):
    # Ensure the image is in RGB format
    if image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (RGB)")

    # Convert to minimum across RGB channels
    min_channel = np.min(image, axis=2)

    # Apply minimum filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    return dark_channel

# Function to apply Gaussian blur
def apply_gaussian_blur(image, kernel_size=(11, 11)):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    return blurred_image


# Load images with alpha channel (BGRA format) or grayscale
image_S_ref_to_i = warped_A
image_S_i = image_B


# Check if images are grayscale (single channel) and convert to RGB (3 channels)
if len(image_S_ref_to_i.shape) == 2:  # Grayscale image (2D array)
    image_S_ref_to_i = cv2.cvtColor(image_S_ref_to_i, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB
elif image_S_ref_to_i.shape[2] == 4:  # BGRA image (4 channels)
    image_S_ref_to_i = image_S_ref_to_i[..., :3]  # Discard the alpha channel (use only RGB)

if len(image_S_i.shape) == 2:  # Grayscale image (2D array)
    image_S_i = cv2.cvtColor(image_S_i, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB
elif image_S_i.shape[2] == 4:  # BGRA image (4 channels)
    image_S_i = image_S_i[..., :3]  # Discard the alpha channel (use only RGB)



# Apply dark channel to the image
dark_channel_image_S_ref_to_i = dark_channel(image_S_ref_to_i)
dark_channel_image_S_i = dark_channel(image_S_i)


# Apply Gaussian blur to the dark channel result
blurred_dark_channel_image_S_ref_to_i = apply_gaussian_blur(dark_channel_image_S_ref_to_i)
blurred_dark_channel_image_S_i = apply_gaussian_blur(dark_channel_image_S_i)


# Normalize image to [0, 1]
image_S_ref_to_i = blurred_dark_channel_image_S_ref_to_i / np.max(blurred_dark_channel_image_S_ref_to_i)
image_S_i = blurred_dark_channel_image_S_i / np.max(blurred_dark_channel_image_S_i) 


# Parameters
patch_size = 8
epsilon = 0.92  # Threshold


# Function to split an RGB image into patches
def split_into_patches(image, patch_size):
    patches = [
        image[i:i + patch_size, j:j + patch_size]
        for i in range(0, image.shape[0], patch_size)
        for j in range(0, image.shape[1], patch_size)
    ]
    return patches

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


# Output confirmation
print(f"Mask saved as 'dark_mask.png', shape: {mask.shape}")


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
l1_loss = np.sum(np.abs(output_image))
print("L1 Regularization Loss:", l1_loss)



## Compute Reconstruction Loss
# Ensure correct slicing of dimensions
flow_combined = estimate_optical_flow(image_A, image_B)
flow_combined = flow_combined.squeeze(0)


# Ensure the flow tensor shape matches (2, 1080, 1920)
if flow_combined.shape != (2, 1080, 1920):
    raise ValueError(f"Loaded flow has shape {flow_combined.shape}, expected (2, 1080, 1920)")


device = flow_combined.device  # Get the device of flow_combined
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
print(f"processed_matrix.shape: {processed_matrix.shape}")


# Warp images using respective flows
flow_reverse = estimate_optical_flow(image_B, image_A)
flow_reverse = flow_reverse.to(device)  # Ensure flow is also on the same device
image_B = torch.tensor(image_B).permute(2, 0, 1).float().to(device)  # Convert to CHW format and move to device
warped_B = warp_image(image_B, flow_reverse)

# Perform pixel-wise multiplication
result = warped_B * processed_matrix

# If you need a numeric scalar result (e.g., the sum of all pixel values)
Rec_Loss = result.sum().item()

# Print the result tensor shape and numeric result
print(f"Reconstruction Loss: {Rec_Loss}")



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


# Instantiate the discriminator
discriminator = Discriminator()
# Initialize GANLoss
gan_loss = vanilla_GANLoss(real_label_val=1.0, fake_label_val=0.0)


image_A_transposed = np.transpose(image_A, (2, 0, 1))
image_A_resized = np.array([cv2.resize(image_A_transposed[i], (3840, 2160)) for i in range(3)])


# Convert to PyTorch tensors (if they're currently NumPy arrays)
image_A_resized = torch.from_numpy(image_A_resized).float()  # Convert to float tensor
tensor_rgb_temp = torch.from_numpy(tensor_rgb).float()  # Convert to float tensor

# Add a batch dimension (e.g., for a batch of 1 image)
image_A_resized = image_A_resized.unsqueeze(0)  # Shape becomes (1, 3, 2160, 3840)
tensor_rgb_temp = tensor_rgb_temp.unsqueeze(0)  # Shape becomes (1, 3, 2160, 3840)



D_real = discriminator(image_A_resized)  # Discriminator output for real images
D_fake = discriminator(tensor_rgb_temp)  # Discriminator output for fake images

# Compute discriminator loss (real + fake)
loss_D_real = gan_loss(D_real, target_is_real=True, is_disc=True)  # Real loss
loss_D_fake = gan_loss(D_fake, target_is_real=False, is_disc=True)  # Fake loss
loss_D = (loss_D_real + loss_D_fake) * 0.5  # Total discriminator loss

# Compute generator loss (goal: fool the discriminator)
loss_G = gan_loss(D_fake, target_is_real=True, is_disc=False)  # Generator loss


print(f"loss_D:{loss_D}")
print(f"loss_G:{loss_G}")


