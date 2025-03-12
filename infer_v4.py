import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F  # For F.interpolate
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def calculate_and_apply_optical_flow(img_a_path, img_b_path, output_path):
    # Read the images
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)

    if img_a is None or img_b is None:
        raise ValueError("Failed to load one or both input images")

    # Convert to grayscale
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0
    )

    # Get image dimensions
    h, w = flow.shape[:2]

    # Create coordinate grid
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Add flow to coordinates to get new positions
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    map_x = x + flow_x
    map_y = y + flow_y

    # Warp the original image using the flow
    warped_img = cv2.remap(
        img_a,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Save the warped result
    cv2.imwrite(output_path, warped_img)

    # Convert flow to HSV for visualization
    hsv = np.zeros_like(img_a)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save flow visualization
    cv2.imwrite('flow_visualization_new.jpg', flow_rgb)

    return warped_img, flow_rgb, img_b

# Define the DeepLabV3-based FusionNet class (same as training)
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        
        # Load DeepLabV3 with ResNet50 backbone (pretrained)
        deeplab = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        
        # Modify the input layer to accept 6 channels (warped + img_b)
        self.backbone = nn.Sequential(*list(deeplab.backbone.children()))
        self.backbone[0] = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Replace first conv
        
        # Extract intermediate features from DeepLabV3 backbone
        self.encoder = deeplab.backbone
        
        # Decoder layers (corrected channel counts)
        self.dec4 = nn.Conv2d(2048, 512, 3, padding=1)         # From layer4 (2048 channels)
        self.dec3 = nn.Conv2d(512 + 1024, 256, 3, padding=1)   # Concat with layer3 (1024 channels)
        self.dec2 = nn.Conv2d(256 + 512, 128, 3, padding=1)    # Concat with layer2 (512 channels)
        self.dec1 = nn.Conv2d(128 + 256, 64, 3, padding=1)     # Concat with layer1 (256 channels)
        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)       # Final output
        
        self.relu = nn.ReLU()

    def forward(self, warped, img_b):
        # Concatenate inputs
        x = torch.cat((warped, img_b), dim=1)  # [batch, 6, h, w]
        
        # Encoder (DeepLabV3 ResNet50)
        features = {}
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # bn1
        x = self.backbone[2](x)  # relu
        x = self.backbone[3](x)  # maxpool
        features['layer1'] = self.backbone[4](x)  # [batch, 256, h/4, w/4]
        features['layer2'] = self.backbone[5](features['layer1'])  # [batch, 512, h/8, w/8]
        features['layer3'] = self.backbone[6](features['layer2'])  # [batch, 1024, h/16, w/16]
        x = self.backbone[7](features['layer3'])  # [batch, 2048, h/32, w/32]
        
        # Decoder with skip connections
        d4 = self.relu(self.dec4(x))  # [batch, 512, h/32, w/32]
        d4 = F.interpolate(d4, size=features['layer3'].shape[2:], mode='bilinear', align_corners=True)  # Match layer3 size
        
        d3 = torch.cat((d4, features['layer3']), dim=1)  # [batch, 512+1024, h/16, w/16]
        d3 = self.relu(self.dec3(d3))                    # [batch, 256, h/16, w/16]
        d3 = F.interpolate(d3, size=features['layer2'].shape[2:], mode='bilinear', align_corners=True)  # Match layer2 size
        
        d2 = torch.cat((d3, features['layer2']), dim=1)  # [batch, 256+512, h/8, w/8]
        d2 = self.relu(self.dec2(d2))                    # [batch, 128, h/8, w/8]
        d2 = F.interpolate(d2, size=features['layer1'].shape[2:], mode='bilinear', align_corners=True)  # Match layer1 size
        
        d1 = torch.cat((d2, features['layer1']), dim=1)  # [batch, 128+256, h/4, w/4]
        d1 = self.relu(self.dec1(d1))                    # [batch, 64, h/4, w/4]
        d1 = F.interpolate(d1, size=(warped.shape[2], warped.shape[3]), mode='bilinear', align_corners=True)  # Match input size
        
        output = self.final_conv(d1)                     # [batch, 3, h, w]
        
        return torch.sigmoid(output)  # Output in [0,1] range

# Main execution
if __name__ == "__main__":
    # New image paths
    img_a_path = r"D:\Self Supervised Video Desmoking for Laparoscopic Surgery\LSVD_test\0007\00000931.png"
    img_b_path = r"D:\Self Supervised Video Desmoking for Laparoscopic Surgery\LSVD_test\0007\00000966.png"
    output_path = r"C:\Users\User\Self_SVDwarped_image_new.jpg"
    fused_output_path = r"C:\Users\User\Self_SVDfused_image_new.jpg"
    model_load_path = r"C:\Users\User\Self_SVD\models\fusion_model_deeplab_full.pth" # Updated to DeepLab model

    try:
        # Process new images with optical flow
        warped_result, flow_vis, img_b = calculate_and_apply_optical_flow(
            img_a_path,
            img_b_path,
            output_path
        )
        print("Optical flow processing complete. Results saved.")

        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load the trained model
        model = FusionNet().to(device)
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        model.eval()
        print(f"Loaded trained model from: {model_load_path}")

        # Convert new images to PyTorch tensors
        warped_tensor = torch.from_numpy(warped_result.transpose(2, 0, 1)).float() / 255.0
        img_b_tensor = torch.from_numpy(img_b.transpose(2, 0, 1)).float() / 255.0
        
        # Add batch dimension and move to device
        warped_tensor = warped_tensor.unsqueeze(0).to(device)
        img_b_tensor = img_b_tensor.unsqueeze(0).to(device)

        # Generate fused image using the loaded model
        with torch.no_grad():
            fused_tensor = model(warped_tensor, img_b_tensor)
        fused_img = (fused_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        fused_img_bgr = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)

        # Save the fused image
        cv2.imwrite(fused_output_path, fused_img_bgr)
        print(f"Fused image saved to: {fused_output_path}")

        # Convert images for display
        warped_result_rgb = cv2.cvtColor(warped_result, cv2.COLOR_BGR2RGB)
        flow_vis_rgb = cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB)
        img_a_rgb = cv2.cvtColor(cv2.imread(img_a_path), cv2.COLOR_BGR2RGB)
        img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
        
        # Switch blue and red channels for fused image display
        fused_img_rgb = cv2.cvtColor(fused_img_bgr, cv2.COLOR_BGR2RGB)
        fused_img_rgb_swapped = fused_img_rgb.copy()
        fused_img_rgb_swapped[:, :, [0, 2]] = fused_img_rgb_swapped[:, :, [2, 0]]  # Swap R and B

        # Display with Matplotlib
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Image Processing Results (New Set)', fontsize=16)

        axs[0, 0].imshow(img_a_rgb)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(img_b_rgb)
        axs[0, 1].set_title('Target Image')
        axs[0, 1].axis('off')

        axs[0, 2].imshow(warped_result_rgb)
        axs[0, 2].set_title('Warped Image')
        axs[0, 2].axis('off')

        axs[1, 0].imshow(flow_vis_rgb)
        axs[1, 0].set_title('Flow Visualization')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(fused_img_rgb_swapped)
        axs[1, 1].set_title('Fused Image (DeepLabV3, R/B Swapped)')
        axs[1, 1].axis('off')

        # Leave the last subplot empty since we don't have ground truth
        axs[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {str(e)}")