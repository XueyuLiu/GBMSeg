import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def prepare_input(ps_points, ng_points):
    # Combine positive and negative points (if available)
    if ps_points is not None and ng_points is not None:
        ps_points = np.array(ps_points)
        ng_points = np.array(ng_points)
        input_point = np.vstack((ps_points, ng_points))
        ps_label = np.ones(ps_points.shape[0])
        ng_label = np.zeros(ng_points.shape[0])
        input_label = np.concatenate((ps_label, ng_label))
    # Use only positive points (if available)
    else: 
        ps_points = np.array(ps_points)
        input_point = ps_points
        input_label = np.ones(ps_points.shape[0])
    
    return input_point, input_label

def save_max_contour_area(result_mask_path,mask):
    """
    Finds and saves the largest contour in an image. Can save either the image with drawn contour 
    or the cropped image of the largest contour area.

    :param image_path: Path to the input image.
    :param save_cropped: If True, saves the cropped area of the largest contour. 
                         Otherwise, saves the image with the drawn largest contour.
    """

    # image = cv2.imread(mask, 0)


    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(mask)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        cv2.drawContours(filled_image, [max_contour], -1, 255, thickness=cv2.FILLED)
        cv2.imwrite(result_mask_path, filled_image)
    return filled_image

# Define a function to process each image
def process_image(image_path, ps_points, ng_points, sam, save_path, mask_save_path,fix_mask_save_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize SamPredictor
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    # Get input_point and input_label based on available points
    input_point, input_label = prepare_input(ps_points, ng_points)
    
    # Predict masks
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    
    # Save masks
    for i, mask in enumerate(masks):
        result_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}.png"
        result_mask_path = os.path.join(mask_save_path, result_filename)
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(result_mask_path, mask_image)
        result_fix_mask_path = os.path.join(fix_mask_save_path, result_filename)
        mask=save_max_contour_area(result_fix_mask_path,mask_image)/255
        
        # Plot and save the image with masks
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        show_mask(mask_image/255, ax)
        show_points(input_point, input_label, ax)
        ax.axis('off')
        result_path = os.path.join(save_path, result_filename)
        plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)



def main():
    print('---------Start segmenting---------')
    # Define your image_folder, output_file, and other parameters here

    # Load the Sam model
    #sam_checkpoint = "./checkpoint/sam_vit_b_01ec64.pth"
    #sam_checkpoint = "./checkpoint/sam_vit_l_0b3195.pth"
    sam_checkpoint = "./checkpoint/sam_vit_h_4b8939.pth"

    model_type = "vit_l"

    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)


    image_folder='../data/target_image'
    image_file = '../data/prompt_final.txt'
    dataset_name = 'ViT_L'
    print(dataset_name)
    save_path=os.path.join('../results','seg_'+dataset_name)
    mask_save_path=os.path.join('../results','mask_'+dataset_name)
    fix_mask_save_path=os.path.join('../results','maxContour_mask_'+dataset_name)

    os.makedirs(mask_save_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(fix_mask_save_path, exist_ok=True)

    # Process each image in the dataset
    with open(image_file, 'r') as file:
        lines = file.readlines()

        for line in tqdm(lines):
            parts = line.strip().split(';')
            if len(parts) == 3:
                image_name = parts[0]
                ps_points = eval(parts[1])
                ng_points = eval(parts[2])
                image_path = os.path.join(image_folder, image_name)
                #print(image_path)
                try:
                    process_image(image_path, ps_points, ng_points, sam, save_path, mask_save_path,fix_mask_save_path)
                except:
                    continue

if __name__ == "__main__":
    main()
