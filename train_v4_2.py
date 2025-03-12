import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import os
from torch.amp import GradScaler, autocast

def calculate_and_apply_optical_flow(img_a, img_b, resize_dim=(512, 512)):
    img_a_resized = cv2.resize(img_a, resize_dim)
    img_b_resized = cv2.resize(img_b, resize_dim)
    gray_a = cv2.cvtColor(img_a_resized, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_resized, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    map_x, map_y = x + flow_x, y + flow_y
    warped_img = cv2.remap(img_a_resized, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    hsv = np.zeros_like(img_a_resized)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 1], hsv[..., 0], hsv[..., 2] = 255, ang * 180 / np.pi / 2, cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return warped_img, flow_rgb, img_b_resized

def ssim_loss(pred, target, window_size=11, size_average=True):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    window = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
    window = window.repeat(3, 1, 1, 1).to(pred.device).type_as(pred)
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=3)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=3) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return (1 - ssim_map.mean()) if size_average else (1 - ssim_map.mean(dim=(1, 2, 3)))

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        deeplab = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(deeplab.backbone.children()))
        self.backbone[0] = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = deeplab.backbone
        self.dec4 = nn.Conv2d(2048, 512, 3, padding=1)
        self.dec3 = nn.Conv2d(512 + 1024, 256, 3, padding=1)
        self.dec2 = nn.Conv2d(256 + 512, 128, 3, padding=1)
        self.dec1 = nn.Conv2d(128 + 256, 64, 3, padding=1)
        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, warped, img_b):
        x = torch.cat((warped, img_b), dim=1)
        features = {}
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        x = self.backbone[2](x)
        x = self.backbone[3](x)
        features['layer1'] = self.backbone[4](x)
        features['layer2'] = self.backbone[5](features['layer1'])
        features['layer3'] = self.backbone[6](features['layer2'])
        x = self.backbone[7](features['layer3'])
        d4 = self.relu(self.dec4(x))
        d4 = F.interpolate(d4, size=features['layer3'].shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat((d4, features['layer3']), dim=1)
        d3 = self.relu(self.dec3(d3))
        d3 = F.interpolate(d3, size=features['layer2'].shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((d3, features['layer2']), dim=1)
        d2 = self.relu(self.dec2(d2))
        d2 = F.interpolate(d2, size=features['layer1'].shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat((d2, features['layer1']), dim=1)
        d1 = self.relu(self.dec1(d1))
        d1 = F.interpolate(d1, size=(warped.shape[2], warped.shape[3]), mode='bilinear', align_corners=True)
        output = self.final_conv(d1)
        return torch.sigmoid(output)

def get_first_image(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    if not image_files:
        return None
    image_numbers = [int(os.path.splitext(f)[0]) for f in image_files]
    first_image = image_files[image_numbers.index(min(image_numbers))]
    return first_image

if __name__ == "__main__":
    base_masked_folder = r"D:\Desmoking_Dataset\LH_frames_smoky"
    base_unmasked_folder = r"D:\Desmoking_Dataset\LH_frames\pure_clear_series"
    output_folder = r"D:\Desmoking_Dataset\output"  # Not used for saving
    model_save_path = r"C:\Users\User\Self_SVD\models\fusion_model_deeplab_full.pth"

    try:
        masked_subfolders = [f for f in os.listdir(base_masked_folder) if os.path.isdir(os.path.join(base_masked_folder, f))]
        unmasked_subfolders = [f for f in os.listdir(base_unmasked_folder) if os.path.isdir(os.path.join(base_unmasked_folder, f))]
        common_subfolders = sorted(set(unmasked_subfolders) & set(masked_subfolders))
        
        if not common_subfolders:
            raise ValueError("No matching subfolders found between masked and unmasked directories")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = FusionNet().to(device)
        mse_loss = nn.MSELoss()
        alpha = 0.5
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler('cuda')
        num_epochs = 20

        # Preload initial images
        initial_images = {}
        total_pairs = 0
        for subfolder in common_subfolders:
            masked_subfolder = os.path.join(base_masked_folder, subfolder)
            unmasked_subfolder = os.path.join(base_unmasked_folder, subfolder)
            sub_subfolders_masked = [f for f in os.listdir(masked_subfolder) if os.path.isdir(os.path.join(masked_subfolder, f))]
            sub_subfolders_unmasked = [f for f in os.listdir(unmasked_subfolder) if os.path.isdir(os.path.join(unmasked_subfolder, f))]
            common_sub_subfolders = sorted(set(sub_subfolders_masked) & set(sub_subfolders_unmasked))
            
            for sub_subfolder in common_sub_subfolders:
                masked_path = os.path.join(masked_subfolder, sub_subfolder)
                first_image = get_first_image(masked_path)
                if first_image:
                    initial_img_path = os.path.join(masked_path, first_image)
                    img = cv2.imread(initial_img_path)
                    if img is not None:
                        initial_images[f"{subfolder}/{sub_subfolder}"] = img
                        # Count pairs (excluding the first image)
                        image_files = sorted([f for f in os.listdir(masked_path) if f.endswith('.jpg')], 
                                           key=lambda x: int(os.path.splitext(x)[0]))
                        total_pairs += len(image_files) - 1
                    else:
                        print(f"Warning: Could not load initial image from {initial_img_path}")
                else:
                    print(f"Warning: No images found in {masked_path}")

        # Training loop with progress bar for image pairs
        pair_count = 0
        with tqdm(total=total_pairs * num_epochs, desc="Processing Image Pairs") as pbar:
            for epoch in range(num_epochs):
                model.train()
                img_a_current_dict = {key: img.copy() for key, img in initial_images.items()}

                for key in initial_images.keys():
                    subfolder, sub_subfolder = key.split('/')
                    masked_folder = os.path.join(base_masked_folder, subfolder, sub_subfolder)
                    unmasked_folder = os.path.join(base_unmasked_folder, subfolder, sub_subfolder)
                    image_files = sorted([f for f in os.listdir(masked_folder) if f.endswith('.jpg')], 
                                       key=lambda x: int(os.path.splitext(x)[0]))
                    first_image = get_first_image(masked_folder)
                    if first_image in image_files:
                        image_files.remove(first_image)
                    
                    img_a_current = img_a_current_dict[key]

                    for img_name in image_files:
                        optimizer.zero_grad()

                        img_b_path = os.path.join(masked_folder, img_name)
                        ground_truth_path = os.path.join(unmasked_folder, img_name)

                        img_b = cv2.imread(img_b_path)
                        ground_truth = cv2.imread(ground_truth_path)
                        if img_b is None or ground_truth is None:
                            print(f"Skipping {img_name} in {key}: Could not load images")
                            continue

                        warped_img, _, img_b_resized = calculate_and_apply_optical_flow(img_a_current, img_b, resize_dim=(512, 512))
                        ground_truth_resized = cv2.resize(ground_truth, (512, 512))

                        warped_tensor = torch.from_numpy(warped_img.transpose(2, 0, 1)).float() / 255.0
                        img_b_tensor = torch.from_numpy(img_b_resized.transpose(2, 0, 1)).float() / 255.0
                        ground_truth_tensor = torch.from_numpy(ground_truth_resized.transpose(2, 0, 1)).float() / 255.0

                        warped_tensor = warped_tensor.unsqueeze(0).to(device)
                        img_b_tensor = img_b_tensor.unsqueeze(0).to(device)
                        ground_truth_tensor = ground_truth_tensor.unsqueeze(0).to(device)

                        with autocast('cuda'):
                            fused_tensor = model(warped_tensor, img_b_tensor)
                            mse = mse_loss(fused_tensor, ground_truth_tensor)
                            ssim = ssim_loss(fused_tensor, ground_truth_tensor)
                            loss = alpha * mse + (1 - alpha) * ssim

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        fused_img = (fused_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        img_a_current = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
                        img_a_current_dict[key] = img_a_current

                        pair_count += 1
                        pbar.update(1)
                        pbar.set_description(f"Processing Image Pairs [{epoch+1}/{num_epochs}]")

                        del warped_tensor, img_b_tensor, ground_truth_tensor, fused_tensor, loss
                        torch.cuda.empty_cache()

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] completed")

        torch.save(model.state_dict(), model_save_path)
        print(f"Trained model saved to: {model_save_path}")

        # Visualization for the last image of the last sub-sub folder
        with torch.no_grad():
            last_key = list(initial_images.keys())[-1]
            subfolder, sub_subfolder = last_key.split('/')
            masked_folder = os.path.join(base_masked_folder, subfolder, sub_subfolder)
            unmasked_folder = os.path.join(base_unmasked_folder, subfolder, sub_subfolder)
            image_files = sorted([f for f in os.listdir(masked_folder) if f.endswith('.jpg')], 
                               key=lambda x: int(os.path.splitext(x)[0]))
            last_image = image_files[-1]

            img_b_path = os.path.join(masked_folder, last_image)
            ground_truth_path = os.path.join(unmasked_folder, last_image)

            img_b_original = cv2.imread(img_b_path)
            ground_truth_original = cv2.imread(ground_truth_path)
            original_size = (img_b_original.shape[1], img_b_original.shape[0])

            img_a_current = img_a_current_dict[last_key]
            warped_img, _, img_b_resized = calculate_and_apply_optical_flow(img_a_current, img_b_original)
            warped_tensor = torch.from_numpy(warped_img.transpose(2, 0, 1)).float() / 255.0
            img_b_tensor = torch.from_numpy(img_b_resized.transpose(2, 0, 1)).float() / 255.0
            warped_tensor = warped_tensor.unsqueeze(0).to(device)
            img_b_tensor = img_b_tensor.unsqueeze(0).to(device)

            fused_tensor = model(warped_tensor, img_b_tensor)
            fused_img = (fused_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            warped_img_original_size = cv2.resize(warped_img, original_size)
            fused_img_original_size = cv2.resize(fused_img, original_size)

            img_b_rgb = cv2.cvtColor(img_b_original, cv2.COLOR_BGR2RGB)
            ground_truth_rgb = cv2.cvtColor(ground_truth_original, cv2.COLOR_BGR2RGB)
            warped_result_rgb = cv2.cvtColor(warped_img_original_size, cv2.COLOR_BGR2RGB)
            fused_img_rgb = cv2.cvtColor(fused_img_original_size, cv2.COLOR_BGR2RGB)

            warped_result_rgb_swapped = warped_result_rgb.copy()
            warped_result_rgb_swapped[:, :, [0, 2]] = warped_result_rgb_swapped[:, :, [2, 0]]
            fused_img_rgb_swapped = fused_img_rgb.copy()

            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle(f'Final Results for {last_key}/{last_image} (Original Size)', fontsize=16)
            axs[0, 0].imshow(img_b_rgb); axs[0, 0].set_title('Target Image'); axs[0, 0].axis('off')
            axs[0, 1].imshow(warped_result_rgb_swapped); axs[0, 1].set_title('Warped Image (R/B Swapped)'); axs[0, 1].axis('off')
            axs[1, 0].imshow(fused_img_rgb_swapped); axs[1, 0].set_title('Fused Image'); axs[1, 0].axis('off')
            axs[1, 1].imshow(ground_truth_rgb); axs[1, 1].set_title('Ground Truth'); axs[1, 1].axis('off')
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error: {str(e)}")