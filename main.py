import os
import sys
import warnings
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

# Add external directories to system path
generate_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'feature_matching'))
segment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter'))
sys.path.append(segment_path)
sys.path.append(generate_path)

# Importing modules
#from openpyxl import Workbook, load_workbook
from feature_matching.generate_points import generate, loading_dino
from feature_matching.generate_points_matrics import generate_indices
from segmenter.segment import seg_main, loading_seg
from automatic_prompt_engineering import APE

# Set up warnings filter
warnings.filterwarnings("ignore")


# Function to optimize prompts
def prompt_optimize(features, positive_prompt, negative_prompt, hard_prompt, dis_sp_p, dis_sp_n, dis_sp_hard, dis_ex,
                    device, image, max_contour, model_seg):
    positive_prompt_opt, negative_prompt_opt = APE(positive_prompt, negative_prompt, hard_prompt, dis_sp_p, dis_sp_n,
                                                   dis_sp_hard, dis_ex, True)
    return positive_prompt_opt, negative_prompt_opt, features


# Main function
def main():
    """
    Main execution of the script, which processes the dataset and performs prompt optimization and segmentation.
    """
    # Set device for computation (GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    model_dino = loading_dino(device)
    model_seg = loading_seg('vit_l', device)

    # Dataset and paths
    dataset_name = 'DATASET_NAME'
    image_size = 560
    image_prompt_dir = os.path.join(r'./data', dataset_name, 'reference_images')
    mask_path = os.path.join(r'./data', dataset_name, 'reference_masks')
    image_dir = os.path.join(r'./data', dataset_name, 'target_images')
    save_dir = os.path.join(r'./results', dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # Set distance for each type of prompt (scaled by image size)
    dis_sp_p, dis_sp_n, dis_ex = 0.125, 0.125, 0.125
    dis_sp_p, dis_sp_n, dis_ex = (np.array([dis_sp_p, dis_sp_n, dis_ex])) * image_size
    dis_sp_hard = 1  # Fixed hard prompt distance

    # List files in directories
    reference_list = os.listdir(image_prompt_dir)
    imglist = os.listdir(image_dir)

    # Loop over each reference image
    for reference in reference_list:
        image_prompt = Image.open(os.path.join(image_prompt_dir, reference)).resize((image_size, image_size))
        gt_mask = Image.open(os.path.join(mask_path, reference)).resize((image_size, image_size))


        # Loop over each image in the dataset
        for name in tqdm(imglist):
            image = Image.open(os.path.join(image_dir, name)).resize((image_size, image_size))

            # Generate features and prompts using 'generate' function
            features, positive_prompt, negative_prompt, hard_prompt = generate(gt_mask, [image_prompt, image],
                                                                               device, model_dino, image_size)

            # Optimize prompts
            positive_prompt_opt, negative_prompt_opt, _ = prompt_optimize(features, positive_prompt,
                                                                          negative_prompt, hard_prompt,
                                                                          dis_sp_p, dis_sp_n, dis_sp_hard, dis_ex,
                                                                          device, image, max_contour=None,
                                                                          model_seg=model_seg)

            # Perform segmentation using optimized prompts
            mask_final = seg_main(image, positive_prompt_opt, negative_prompt_opt, device, max_contour=None,
                                  seg_model=model_seg)

            # Save the final mask
            cv2.imwrite(os.path.join(save_dir, name), mask_final)


# Execute the script if this file is being run directly
if __name__ == "__main__":
    main()
