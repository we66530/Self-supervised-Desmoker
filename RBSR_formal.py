import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import deform_conv2d
from PIL import Image

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

# Initialize models on GPU
offset_estimator = OpticalFlowOffsetEstimator().to(device)
flow_guided_dcn = FlowGuidedDCN(in_channels=3, out_channels=16).to(device)
image_encoder = ImageEncoder().to(device)
fusion_model = PastFramePropagation(16).to(device)
upsampler = Upsampler().to(device)

# Predict offsets and move tensors to GPU
estimated_offsets_R, estimated_offsets_G, estimated_offsets_B = [offset_estimator(flow) for flow in [flow_R, flow_G, flow_B]]

# Convert images to tensors and move to GPU
def prepare_tensor(img):
    img = torch.tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0).float().to(device)
    return img

red_B, green_B, blue_B = [prepare_tensor(img) for img in [red_B, green_B, blue_B]]

# Perform flow-guided deformable convolution
output_R, output_G, output_B = [flow_guided_dcn(img, offsets) for img, offsets in zip([red_B, green_B, blue_B], [estimated_offsets_R, estimated_offsets_G, estimated_offsets_B])]

# Encode past frame (image_A)
red_A, green_A, blue_A = [prepare_tensor(img) for img in [red_A, green_A, blue_A]]
encoded_red_A, encoded_green_A, encoded_blue_A = [image_encoder(img) for img in [red_A, green_A, blue_A]]

# Fusion of past and current features
output_feat_R, output_feat_G, output_feat_B = [fusion_model(past, curr) for past, curr in zip([encoded_red_A, encoded_green_A, encoded_blue_A], [output_R, output_G, output_B])]

# Upsample the results
upsampled_output_R, upsampled_output_G, upsampled_output_B = [upsampler(feat) for feat in [output_feat_R, output_feat_G, output_feat_B]]

# Prepare for PixelShuffle
upscale_factor = 2
red_A, green_A, blue_B = [img.repeat(1, 4, 1, 1) for img in [red_A, green_A, blue_B]]

A_upscaled_R, A_upscaled_G, A_upscaled_B = [F.pixel_shuffle(img, upscale_factor) for img in [red_A, green_A, blue_B]]

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
image_pil = Image.fromarray(image_np)
image_pil.show()
